//11/40
//File 0091 : core/math/mesh_repair.h
//Mesh repair utilities: duplicate vertex removal, face orientation normalization, non‑manifold edge detection and removal, and hole closing via advancing front triangulation.
#ifndef CORE_MATH_MESH_REPAIR_H
#define CORE_MATH_MESH_REPAIR_H

#include "mesh_data.h"               // HalfEdgeMesh for output
#include "vector_math.h"
#include "math_constants.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <set>

namespace SimulationMath {
namespace mesh_repair {

using namespace mesh;

// -----------------------------------------------------------------------------
// 1. Remove duplicate vertices (merge by distance threshold)
// -----------------------------------------------------------------------------
inline void remove_duplicate_vertices(std::vector<DirectX::XMVECTOR>& vertices,
                                      std::vector<uint32_t>& indices,
                                      float merge_epsilon = 1e-6f) noexcept {
    if (vertices.empty() || indices.empty()) return;

    // spatial hash grid for fast proximity search (simple O(n^2) for small meshes)
    size_t nv = vertices.size();
    std::vector<int> remap(nv, -1);
    std::vector<DirectX::XMVECTOR> unique_verts;

    for (size_t i = 0; i < nv; ++i) {
        if (remap[i] != -1) continue;
        remap[i] = static_cast<int>(unique_verts.size());
        unique_verts.push_back(vertices[i]);

        for (size_t j = i + 1; j < nv; ++j) {
            if (remap[j] != -1) continue;
            DirectX::XMVECTOR diff = DirectX::XMVectorSubtract(vertices[i], vertices[j]);
            float dist_sq = vector_math::length_sq3_scalar(diff);
            if (dist_sq <= merge_epsilon * merge_epsilon) {
                remap[j] = remap[i];
            }
        }
    }

    // update indices
    for (uint32_t& idx : indices) {
        if (remap[idx] >= 0)
            idx = static_cast<uint32_t>(remap[idx]);
    }
    vertices.swap(unique_verts);
}

// -----------------------------------------------------------------------------
// 2. Ensure consistent face orientations (all outward)
//    Assumes triangles are originally oriented but may be inconsistent.
//    Uses the method of propagating a consistent orientation across the mesh via a half‑edge structure.
//    If the mesh is non‑orientable, some faces may be flipped.
// -----------------------------------------------------------------------------
inline void make_orientation_consistent(std::vector<uint32_t>& indices,
                                         const std::vector<DirectX::XMVECTOR>& vertices) noexcept {
    if (indices.size() < 3) return;
    size_t num_triangles = indices.size() / 3;

    // Build adjacency: for each edge, list the triangle(s) that share it and the index of the edge within the triangle.
    struct Edge { uint32_t a, b; };
    auto edge_hash = [](Edge e) -> uint64_t {
        if (e.a < e.b) return (static_cast<uint64_t>(e.a) << 32) | e.b;
        else return (static_cast<uint64_t>(e.b) << 32) | e.a;
    };
    auto edge_eq = [](Edge e1, Edge e2) { return (e1.a==e2.a && e1.b==e2.b) || (e1.a==e2.b && e1.b==e2.a); };
    std::unordered_map<uint64_t, std::vector<std::pair<size_t,int>>> edge_to_tri; // key -> (tri_idx, local_edge_idx)

    for (size_t t = 0; t < num_triangles; ++t) {
        size_t base = t * 3;
        for (int i = 0; i < 3; ++i) {
            uint32_t a = indices[base + i];
            uint32_t b = indices[base + (i+1)%3];
            Edge e{a,b};
            uint64_t key = edge_hash(e);
            edge_to_tri[key].emplace_back(t, i);
        }
    }

    // BFS to propagate consistent orientation
    std::vector<bool> visited(num_triangles, false);
    std::vector<size_t> stack;
    // Start from triangle 0 (if exists)
    if (num_triangles > 0) {
        stack.push_back(0);
        visited[0] = true;
    }

    while (!stack.empty()) {
        size_t tri = stack.back();
        stack.pop_back();
        size_t base = tri * 3;

        // For each edge of this triangle, find the adjacent triangle(s)
        for (int i = 0; i < 3; ++i) {
            uint32_t a = indices[base + i];
            uint32_t b = indices[base + (i+1)%3];
            Edge e{a,b};
            uint64_t key = edge_hash(e);
            const auto& neighbors = edge_to_tri[key];
            for (const auto& nb : neighbors) {
                size_t other_tri = nb.first;
                if (other_tri == tri) continue;
                if (visited[other_tri]) continue;

                // Determine if the edge orientation in the other triangle needs to be flipped.
                // In this triangle, edge is a->b. The neighbor shares the same undirected edge.
                // In the neighbor triangle, the edge may appear as (a,b) or (b,a). If it appears as (a,b), then the neighbor's edge orientation is the same direction (a->b), which means the neighbor's normal would be opposite if we don't flip? Actually for a closed orientable surface, adjacent triangles should have opposite edge orientations to ensure consistent normals. So if this triangle has edge (a,b), the neighbor should have (b,a) to be consistent. So if the neighbor has (a,b), it is inconsistent and needs to be flipped.
                size_t other_base = other_tri * 3;
                uint32_t other_a = indices[other_base + nb.second];
                uint32_t other_b = indices[other_base + (nb.second+1)%3];
                bool needs_flip = (other_a == a && other_b == b); // same orientation -> inconsistent

                if (needs_flip) {
                    // Flip the triangle by swapping two vertices (e.g., swap B and C)
                    std::swap(indices[other_base + 1], indices[other_base + 2]);
                }
                visited[other_tri] = true;
                stack.push_back(other_tri);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 3. Detect and remove non‑manifold edges (edges shared by >2 triangles) by splitting vertices
// -----------------------------------------------------------------------------
inline void resolve_non_manifold_edges(std::vector<DirectX::XMVECTOR>& vertices,
                                        std::vector<uint32_t>& indices) noexcept {
    std::unordered_map<uint64_t, std::vector<size_t>> edge_triangles;
    auto edge_key = [](uint32_t a, uint32_t b) -> uint64_t {
        if (a < b) return ((uint64_t)a << 32) | b;
        else return ((uint64_t)b << 32) | a;
    };

    for (size_t t = 0; t < indices.size(); t += 3) {
        for (int i = 0; i < 3; ++i) {
            uint32_t a = indices[t+i];
            uint32_t b = indices[t+(i+1)%3];
            edge_key(a,b);
            edge_triangles[edge_key(a,b)].push_back(t);
        }
    }

    // For each edge with >2 incident triangles, duplicate the edge's vertices for the extra triangles.
    // This simple approach duplicates vertices for all triangles after the first two, effectively disconnecting them.
    for (auto& entry : edge_triangles) {
        if (entry.second.size() <= 2) continue;
        // Keep first two triangles as is, duplicate vertices for the rest.
        const auto& tri_indices = entry.second;
        uint32_t a = 0, b = 0;
        // We need to extract a,b from key, but easier: from first triangle.
        size_t t0 = tri_indices[0];
        for (int i = 0; i < 3; ++i) {
            uint32_t va = indices[t0+i];
            uint32_t vb = indices[t0+(i+1)%3];
            uint64_t k = edge_key(va,vb);
            if (k == entry.first) { a = va; b = vb; break; }
        }
        // For the extra triangles (third onward), replace a and b with newly created vertices.
        for (size_t idx = 2; idx < tri_indices.size(); ++idx) {
            size_t t = tri_indices[idx];
            // duplicate a and b
            DirectX::XMVECTOR pa = vertices[a];
            DirectX::XMVECTOR pb = vertices[b];
            uint32_t new_a = (uint32_t)vertices.size();
            vertices.push_back(pa);
            uint32_t new_b = (uint32_t)vertices.size();
            vertices.push_back(pb);
            // replace a and b in this triangle
            for (int j = 0; j < 3; ++j) {
                if (indices[t+j] == a) indices[t+j] = new_a;
                else if (indices[t+j] == b) indices[t+j] = new_b;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 4. Close holes (simple triangulation of boundary loops using advancing front)
//    Works for single boundary loops; fills with a fan from a central point (not optimal but simple).
// -----------------------------------------------------------------------------
inline void close_holes(std::vector<DirectX::XMVECTOR>& vertices,
                         std::vector<uint32_t>& indices) noexcept {
    // Build half‑edge to find boundary loops
    // Use a temporary map: undirected edge -> count of incident triangles
    std::unordered_map<uint64_t, int> edge_count;
    auto edge_key = [](uint32_t a, uint32_t b) -> uint64_t {
        if (a < b) return ((uint64_t)a << 32) | b;
        else return ((uint64_t)b << 32) | a;
    };

    for (size_t t = 0; t < indices.size(); t += 3) {
        for (int i = 0; i < 3; ++i) {
            uint32_t a = indices[t+i];
            uint32_t b = indices[t+(i+1)%3];
            edge_count[edge_key(a,b)]++;
        }
    }

    // Identify boundary edges (count == 1)
    std::vector<std::pair<uint32_t,uint32_t>> boundary_edges;
    for (const auto& ec : edge_count) {
        if (ec.second == 1) {
            uint32_t a = ec.first >> 32;
            uint32_t b = ec.first & 0xFFFFFFFF;
            boundary_edges.emplace_back(a, b);
        }
    }

    // Organize into loops
    std::vector<std::vector<uint32_t>> loops;
    std::unordered_set<uint32_t> used_vertices;
    while (!boundary_edges.empty()) {
        // Start a new loop
        std::vector<uint32_t> loop;
        uint32_t start = boundary_edges.back().first;
        uint32_t cur = start;
        do {
            loop.push_back(cur);
            // Find the next boundary edge containing cur
            bool found = false;
            for (auto it = boundary_edges.begin(); it != boundary_edges.end(); ++it) {
                if (it->first == cur) {
                    cur = it->second;
                    boundary_edges.erase(it);
                    found = true;
                    break;
                } else if (it->second == cur) {
                    cur = it->first;
                    boundary_edges.erase(it);
                    found = true;
                    break;
                }
            }
            if (!found) break;
        } while (cur != start && !boundary_edges.empty());

        if (loop.size() >= 3) loops.push_back(loop);
    }

    // Triangulate each loop by adding a central vertex and fanning
    for (const auto& loop : loops) {
        if (loop.size() < 3) continue;

        // Compute centroid of boundary vertices
        DirectX::XMVECTOR centroid = DirectX::XMVectorZero();
        for (uint32_t vi : loop) centroid = DirectX::XMVectorAdd(centroid, vertices[vi]);
        centroid = DirectX::XMVectorScale(centroid, 1.0f / loop.size());

        uint32_t new_center = (uint32_t)vertices.size();
        vertices.push_back(centroid);

        // Create triangles (new_center, loop[i], loop[i+1]) orientated outward?
        // We assume the boundary loop is oriented such that the hole is to be filled with outward normals.
        // We'll add triangles with orientation that keeps the normal pointing outward, but without a global orientation we just add.
        for (size_t i = 0; i < loop.size(); ++i) {
            uint32_t a = new_center;
            uint32_t b = loop[i];
            uint32_t c = loop[(i+1) % loop.size()];
            // Add triangle (a,b,c)
            indices.push_back(a); indices.push_back(b); indices.push_back(c);
        }
    }
}

// -----------------------------------------------------------------------------
// 5. Convert repaired vertex/index representation to HalfEdgeMesh
// -----------------------------------------------------------------------------
inline HalfEdgeMesh build_halfedge_from_vertices_indices(
    const std::vector<DirectX::XMVECTOR>& vertices,
    const std::vector<uint32_t>& indices) noexcept {

    HalfEdgeMesh result;
    for (const auto& v : vertices) result.add_vertex(v);
    for (size_t i = 0; i < indices.size(); i += 3) {
        std::vector<uint32_t> face = { indices[i], indices[i+1], indices[i+2] };
        result.add_face(face);
    }
    result.link_twins();
    result.compute_face_normals();
    result.compute_vertex_normals();
    return result;
}

} // namespace mesh_repair
} // namespace SimulationMath

#endif // CORE_MATH_MESH_REPAIR_H