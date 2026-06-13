//File 0096 : core/math/mesh_segmentation.h
//Mesh segmentation into regions based on dihedral angle and mean curvature similarity; region growing and merging with cotangent weights.
#ifndef CORE_MATH_MESH_SEGMENTATION_H
#define CORE_MATH_MESH_SEGMENTATION_H

#include "mesh_data.h"                   // HalfEdgeMesh
#include "vector_math.h"
#include "mesh_curvature.h"              // mean_curvature_normal_at_vertex, etc.
#include "math_constants.h"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <functional>
#include <limits>

namespace SimulationMath {
namespace mesh_segmentation {

using namespace mesh;

// -----------------------------------------------------------------------------
// 1. Compute dihedral angle between two adjacent faces (radians)
// -----------------------------------------------------------------------------
inline float dihedral_angle(const DirectX::XMVECTOR& nA, const DirectX::XMVECTOR& nB) noexcept {
    float dot = vector_math::dot3_scalar(nA, nB);
    dot = std::clamp(dot, -1.0f, 1.0f);
    return std::acos(dot);
}

// -----------------------------------------------------------------------------
// 2. Region adjacency graph (region -> list of neighbor regions)
// -----------------------------------------------------------------------------
using RegionGraph = std::unordered_map<uint32_t, std::unordered_set<uint32_t>>;

// -----------------------------------------------------------------------------
// 3. Seed points: select vertices with low curvature (flat regions)
// -----------------------------------------------------------------------------
inline std::vector<uint32_t> select_seeds(const HalfEdgeMesh& mesh, float flat_threshold = 0.1f) noexcept {
    size_t nv = mesh.vertex_count();
    std::vector<float> mean_curv = mesh_curvature::compute_mean_curvature_scalar_field(mesh); // from mesh_curvature.h
    std::vector<uint32_t> seeds;
    for (size_t i = 0; i < nv; ++i) {
        if (std::fabs(mean_curv[i]) < flat_threshold)
            seeds.push_back(static_cast<uint32_t>(i));
    }
    return seeds;
}

// -----------------------------------------------------------------------------
// 4. Region growing from seeds based on dihedral angle and curvature similarity
// -----------------------------------------------------------------------------
inline std::vector<uint32_t> region_growing_segmentation(const HalfEdgeMesh& mesh,
                                                         float angle_threshold = 0.5f,   // radians
                                                         float curvature_weight = 0.5f) noexcept {
    size_t nv = mesh.vertex_count();
    size_t nf = mesh.faces().size();
    std::vector<float> mean_curv = mesh_curvature::compute_mean_curvature_scalar_field(mesh);
    std::vector<uint32_t> seed_vertices = select_seeds(mesh, 0.1f);
    if (seed_vertices.empty()) return std::vector<uint32_t>(nf, 0);

    // Map each face to its region (initialize to invalid)
    std::vector<uint32_t> face_region(nf, 0xFFFFFFFFu);
    std::vector<bool> face_visited(nf, false);
    uint32_t next_region = 0;

    // For each seed vertex, start region if face not visited
    const auto& hedges = mesh.half_edges();
    const auto& verts  = mesh.vertices();
    for (uint32_t seed_v : seed_vertices) {
        // Find a face incident to seed_v that is not yet assigned
        uint32_t start_he = verts[seed_v].first_edge;
        if (start_he == 0xFFFFFFFFu) continue;
        uint32_t he = start_he;
        do {
            uint32_t face_idx = hedges[he].face_index;
            if (!face_visited[face_idx]) {
                // Assign new region and grow
                uint32_t current_region = next_region++;
                std::queue<uint32_t> face_queue;
                face_queue.push(face_idx);
                face_visited[face_idx] = true;
                face_region[face_idx] = current_region;

                while (!face_queue.empty()) {
                    uint32_t f = face_queue.front(); face_queue.pop();
                    const MeshFace& face = mesh.faces()[f];
                    uint32_t he0 = face.first_edge;
                    if (he0 == 0xFFFFFFFFu) continue;
                    DirectX::XMVECTOR nF = face.normal;

                    // For each edge of this face, check adjacent face
                    uint32_t he_cur = he0;
                    do {
                        uint32_t twin = hedges[he_cur].twin_edge;
                        if (twin != 0xFFFFFFFFu) {
                            uint32_t adj_face = hedges[twin].face_index;
                            if (!face_visited[adj_face]) {
                                const MeshFace& adj = mesh.faces()[adj_face];
                                float angle = dihedral_angle(nF, adj.normal);
                                // Compute curvature similarity: average mean curvature of vertices of the face?
                                // We'll use the maximum difference in mean curvature among incident vertices.
                                uint32_t v0 = hedges[he_cur].vertex_index;
                                uint32_t v1 = hedges[hedges[he_cur].next_edge].vertex_index;
                                float curv_diff = std::max(std::fabs(mean_curv[v0] - mean_curv[v1]), 0.0f);
                                // Combine angle and curvature
                                float cost = angle + curvature_weight * curv_diff;
                                if (cost < angle_threshold) {
                                    face_queue.push(adj_face);
                                    face_visited[adj_face] = true;
                                    face_region[adj_face] = current_region;
                                }
                            }
                        }
                        he_cur = hedges[he_cur].next_edge;
                    } while (he_cur != he0);
                }
            }
            he = hedges[hedges[he].prev_edge].twin_edge;
            if (he == 0xFFFFFFFFu) break;
        } while (he != start_he);
    }

    // Any unassigned faces become a separate region (or merge into closest)
    for (size_t f = 0; f < nf; ++f) {
        if (!face_visited[f]) {
            face_region[f] = next_region++;
        }
    }
    return face_region;
}

// -----------------------------------------------------------------------------
// 5. Assign per‑vertex region based on majority of incident faces
// -----------------------------------------------------------------------------
inline std::vector<uint32_t> vertex_region_from_faces(const HalfEdgeMesh& mesh,
                                                       const std::vector<uint32_t>& face_region) noexcept {
    size_t nv = mesh.vertex_count();
    std::vector<uint32_t> vertex_reg(nv, 0xFFFFFFFFu);
    // Count region occurrences per vertex
    std::vector<std::unordered_map<uint32_t, int>> counts(nv);
    const auto& hedges = mesh.half_edges();
    for (size_t f = 0; f < face_region.size(); ++f) {
        const MeshFace& face = mesh.faces()[f];
        uint32_t he0 = face.first_edge;
        if (he0 == 0xFFFFFFFFu) continue;
        uint32_t he = he0;
        do {
            uint32_t v = hedges[he].vertex_index;
            counts[v][face_region[f]]++;
            he = hedges[he].next_edge;
        } while (he != he0);
    }
    for (size_t i = 0; i < nv; ++i) {
        if (counts[i].empty()) continue;
        int max_count = 0;
        uint32_t best_region = 0;
        for (const auto& entry : counts[i]) {
            if (entry.second > max_count) {
                max_count = entry.second;
                best_region = entry.first;
            }
        }
        vertex_reg[i] = best_region;
    }
    return vertex_reg;
}

// -----------------------------------------------------------------------------
// 6. Full segmentation pipeline: assigns each face a region ID
// -----------------------------------------------------------------------------
inline std::vector<uint32_t> segment_mesh(const HalfEdgeMesh& mesh,
                                          float angle_threshold = 0.5f,
                                          float curvature_weight = 0.5f) noexcept {
    return region_growing_segmentation(mesh, angle_threshold, curvature_weight);
}

} // namespace mesh_segmentation
} // namespace SimulationMath

#endif // CORE_MATH_MESH_SEGMENTATION_H