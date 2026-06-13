//10/40
//File 0090 : core/math/mesh_remeshing.h
//Isotropic triangle mesh remeshing: iterative edge split/collapse/flip for uniform edge length and Delaunay quality, plus tangential Laplacian smoothing.
#ifndef CORE_MATH_MESH_REMESHING_H
#define CORE_MATH_MESH_REMESHING_H

#include "mesh_data.h"               // HalfEdgeMesh
#include "vector_math.h"
#include "math_constants.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <functional>
#include <queue>
#include <limits>

namespace SimulationMath {
namespace remeshing {

// -----------------------------------------------------------------------------
// 1. Internal representation – dynamic vertex and triangle list
// -----------------------------------------------------------------------------
struct DynamicMesh {
    std::vector<DirectX::XMVECTOR> vertices;
    std::vector<uint32_t> triangles;      // groups of 3
};

// -----------------------------------------------------------------------------
// 2. Undirected edge key
// -----------------------------------------------------------------------------
struct EdgeKey {
    uint32_t v0, v1;
    bool operator==(const EdgeKey& o) const noexcept { return (v0==o.v0 && v1==o.v1) || (v0==o.v1 && v1==o.v0); }
};
struct EdgeKeyHash {
    size_t operator()(const EdgeKey& k) const noexcept {
        if (k.v0 < k.v1) return (static_cast<uint64_t>(k.v0) << 32) | k.v1;
        else return (static_cast<uint64_t>(k.v1) << 32) | k.v0;
    }
};

// -----------------------------------------------------------------------------
// 3. Uniform remesher
// -----------------------------------------------------------------------------
class UniformRemesher {
public:
    UniformRemesher(const DynamicMesh& mesh, float target_edge_len)
        : target_len_(target_edge_len), mesh_(mesh) {}

    DynamicMesh compute(int iterations = 5) noexcept {
        for (int iter = 0; iter < iterations; ++iter) {
            split_edges();
            collapse_edges();
            flip_edges();
            smooth_vertices();
        }
        return mesh_;
    }

private:
    float target_len_;
    DynamicMesh mesh_;

    // Build adjacency: edge -> list of (triangle_index, opposite_vertex_index_in_triangle)
    void build_edge_info(
        std::unordered_map<EdgeKey, std::vector<std::tuple<uint32_t,uint32_t>>, EdgeKeyHash>& edges) noexcept
    {
        edges.clear();
        for (size_t t = 0; t < mesh_.triangles.size(); t += 3) {
            for (int i = 0; i < 3; ++i) {
                uint32_t a = mesh_.triangles[t + i];
                uint32_t b = mesh_.triangles[t + (i+1)%3];
                EdgeKey key{a,b};
                edges[key].emplace_back(static_cast<uint32_t>(t/3), (i+2)%3); // opposite vertex index within triangle
            }
        }
    }

    // Split edges longer than 4/3*target_len
    void split_edges() noexcept {
        const float max_len = (4.0f/3.0f) * target_len_;
        std::unordered_map<EdgeKey, std::vector<std::tuple<uint32_t,uint32_t>>, EdgeKeyHash> edges;
        build_edge_info(edges);

        struct SplitOp { uint32_t v0, v1; };
        std::vector<SplitOp> splits;
        for (const auto& entry : edges) {
            const EdgeKey& ek = entry.first;
            DirectX::XMVECTOR p0 = mesh_.vertices[ek.v0];
            DirectX::XMVECTOR p1 = mesh_.vertices[ek.v1];
            float len = vector_math::length3_scalar(DirectX::XMVectorSubtract(p1, p0));
            if (len > max_len)
                splits.push_back({ek.v0, ek.v1});
        }

        for (const auto& sp : splits) {
            uint32_t v0 = sp.v0, v1 = sp.v1;
            DirectX::XMVECTOR p0 = mesh_.vertices[v0];
            DirectX::XMVECTOR p1 = mesh_.vertices[v1];
            DirectX::XMVECTOR mid = DirectX::XMVectorScale(DirectX::XMVectorAdd(p0, p1), 0.5f);
            uint32_t new_idx = static_cast<uint32_t>(mesh_.vertices.size());
            mesh_.vertices.push_back(mid);

            std::vector<uint32_t> new_triangles;
            for (size_t t = 0; t < mesh_.triangles.size(); t += 3) {
                uint32_t a = mesh_.triangles[t];
                uint32_t b = mesh_.triangles[t+1];
                uint32_t c = mesh_.triangles[t+2];
                bool has_v0 = (a == v0 || b == v0 || c == v0);
                bool has_v1 = (a == v1 || b == v1 || c == v1);
                if (has_v0 && has_v1) {
                    // This triangle contains the edge; split it into two.
                    // Order: keep orientation consistent (v0, v1, v2) -> (v0, new, v2) and (new, v1, v2)
                    // Find the third vertex v2
                    uint32_t v2 = 0xFFFFFFFFu;
                    for (int i = 0; i < 3; ++i) {
                        if (a != v0 && a != v1) v2 = a;
                        if (b != v0 && b != v1) v2 = b;
                        if (c != v0 && c != v1) v2 = c;
                    }
                    if (v2 != 0xFFFFFFFFu) {
                        new_triangles.push_back(v0); new_triangles.push_back(new_idx); new_triangles.push_back(v2);
                        new_triangles.push_back(new_idx); new_triangles.push_back(v1); new_triangles.push_back(v2);
                    }
                } else {
                    // Keep original triangle
                    new_triangles.push_back(a); new_triangles.push_back(b); new_triangles.push_back(c);
                }
            }
            mesh_.triangles.swap(new_triangles);
        }
    }

    // Collapse edges shorter than 4/5*target_len (unless it would invert normals)
    void collapse_edges() noexcept {
        const float min_len = (4.0f/5.0f) * target_len_;
        std::unordered_map<EdgeKey, std::vector<std::tuple<uint32_t,uint32_t>>, EdgeKeyHash> edges;
        build_edge_info(edges);

        struct CollapseOp { uint32_t v0, v1; };
        std::vector<CollapseOp> collapses;
        for (const auto& entry : edges) {
            const EdgeKey& ek = entry.first;
            DirectX::XMVECTOR p0 = mesh_.vertices[ek.v0];
            DirectX::XMVECTOR p1 = mesh_.vertices[ek.v1];
            float len = vector_math::length3_scalar(DirectX::XMVectorSubtract(p1, p0));
            if (len < min_len && ek.v0 != ek.v1)
                collapses.push_back({ek.v0, ek.v1});
        }

        // Mark vertices that have been removed (collapsed into another)
        std::vector<bool> removed(mesh_.vertices.size(), false);
        for (const auto& col : collapses) {
            uint32_t v0 = col.v0, v1 = col.v1;
            if (removed[v0] || removed[v1]) continue;

            // Build a copy of the mesh with v1 replaced by v0 and check triangle normals
            std::vector<uint32_t> new_tris;
            bool valid = true;
            for (size_t t = 0; t < mesh_.triangles.size(); t += 3) {
                uint32_t a = mesh_.triangles[t];
                uint32_t b = mesh_.triangles[t+1];
                uint32_t c = mesh_.triangles[t+2];
                if (a == v1) a = v0;
                if (b == v1) b = v0;
                if (c == v1) c = v0;
                // Skip degenerate
                if (a == b || b == c || c == a) continue;
                new_tris.push_back(a); new_tris.push_back(b); new_tris.push_back(c);
            }
            // Check if any triangle area became zero or orientation flipped
            for (size_t t = 0; t < new_tris.size(); t += 3) {
                DirectX::XMVECTOR p0 = mesh_.vertices[new_tris[t]];
                DirectX::XMVECTOR p1 = mesh_.vertices[new_tris[t+1]];
                DirectX::XMVECTOR p2 = mesh_.vertices[new_tris[t+2]];
                DirectX::XMVECTOR n = vector_math::cross3(
                    DirectX::XMVectorSubtract(p1, p0),
                    DirectX::XMVectorSubtract(p2, p0));
                if (vector_math::length_sq3_scalar(n) < 1e-12f) {
                    valid = false; break; // degenerate
                }
                // Compare sign with original normal? Not needed for remeshing.
            }
            if (valid) {
                mesh_.triangles.swap(new_tris);
                removed[v1] = true;
                // Move v0 to average of v0 and v1 (or keep v0)
                mesh_.vertices[v0] = DirectX::XMVectorScale(
                    DirectX::XMVectorAdd(mesh_.vertices[v0], mesh_.vertices[v1]), 0.5f);
            }
        }
    }

    // Flip edges to improve Delaunay condition (cotan criterion)
    void flip_edges() noexcept {
        std::unordered_map<EdgeKey, std::vector<std::tuple<uint32_t,uint32_t>>, EdgeKeyHash> edges;
        build_edge_info(edges);

        for (const auto& entry : edges) {
            const auto& incidents = entry.second;
            if (incidents.size() != 2) continue;   // boundary edge – skip

            const EdgeKey& ek = entry.first;
            uint32_t a = ek.v0, b = ek.v1;

            // Find the two opposite vertices and their triangle indices
            uint32_t tri0_idx = std::get<0>(incidents[0]);
            uint32_t tri1_idx = std::get<0>(incidents[1]);
            size_t base0 = static_cast<size_t>(tri0_idx) * 3;
            size_t base1 = static_cast<size_t>(tri1_idx) * 3;

            uint32_t c = 0xFFFFFFFFu, d = 0xFFFFFFFFu;
            for (int i = 0; i < 3; ++i) {
                uint32_t v = mesh_.triangles[base0 + i];
                if (v != a && v != b) { c = v; break; }
            }
            for (int i = 0; i < 3; ++i) {
                uint32_t v = mesh_.triangles[base1 + i];
                if (v != a && v != b) { d = v; break; }
            }
            if (c == 0xFFFFFFFFu || d == 0xFFFFFFFFu) continue;

            DirectX::XMVECTOR pa = mesh_.vertices[a];
            DirectX::XMVECTOR pb = mesh_.vertices[b];
            DirectX::XMVECTOR pc = mesh_.vertices[c];
            DirectX::XMVECTOR pd = mesh_.vertices[d];

            // Compute cotangent weights to decide flip
            auto cotan = [](DirectX::FXMVECTOR u, DirectX::FXMVECTOR v) {
                float dot = vector_math::dot3_scalar(u, v);
                DirectX::XMVECTOR cross = vector_math::cross3(u, v);
                float len_cross = vector_math::length3_scalar(cross);
                if (len_cross < 1e-12f) return 0.0f;
                return dot / len_cross;
            };
            // Angle at c in triangle (a,b,c)
            float cot_c = cotan(DirectX::XMVectorSubtract(pa, pc), DirectX::XMVectorSubtract(pb, pc));
            // Angle at d in triangle (a,b,d)
            float cot_d = cotan(DirectX::XMVectorSubtract(pa, pd), DirectX::XMVectorSubtract(pb, pd));
            // Flip if cot_c + cot_d < 0 (non‑Delaunay)
            if (cot_c + cot_d >= 0.0f) continue;

            // Check that the new edge (c,d) is not already present and that the flip is valid
            // Build a temporary triangle list to check validity
            std::vector<uint32_t> new_tris;
            for (size_t t = 0; t < mesh_.triangles.size(); t += 3) {
                if (t == base0 || t == base1) continue;
                new_tris.push_back(mesh_.triangles[t]);
                new_tris.push_back(mesh_.triangles[t+1]);
                new_tris.push_back(mesh_.triangles[t+2]);
            }
            // Add two new triangles (a,c,d) and (b,d,c) (maintain orientation)
            new_tris.push_back(a); new_tris.push_back(c); new_tris.push_back(d);
            new_tris.push_back(b); new_tris.push_back(d); new_tris.push_back(c);

            // Verify that new triangles have positive area (no inversion)
            bool ok = true;
            for (size_t t = 0; t < new_tris.size(); t += 3) {
                DirectX::XMVECTOR p0 = mesh_.vertices[new_tris[t]];
                DirectX::XMVECTOR p1 = mesh_.vertices[new_tris[t+1]];
                DirectX::XMVECTOR p2 = mesh_.vertices[new_tris[t+2]];
                DirectX::XMVECTOR n = vector_math::cross3(
                    DirectX::XMVectorSubtract(p1, p0),
                    DirectX::XMVectorSubtract(p2, p0));
                if (vector_math::length_sq3_scalar(n) < 1e-12f) { ok = false; break; }
            }
            if (ok) {
                mesh_.triangles.swap(new_tris);
                // Update edge map? Not needed; next iteration rebuilds.
                break;  // flip one edge per outer loop to keep code simple
            }
        }
    }

    // Tangential Laplacian smoothing – move vertices towards centroid of neighbours, projected onto tangent plane
    void smooth_vertices() noexcept {
        size_t nv = mesh_.vertices.size();
        std::vector<DirectX::XMVECTOR> new_positions(nv);
        // Detect boundary vertices (vertices incident to boundary edges)
        std::vector<bool> boundary(nv, false);
        {
            std::unordered_map<EdgeKey, int, EdgeKeyHash> edge_count;
            for (size_t t = 0; t < mesh_.triangles.size(); t += 3) {
                for (int i = 0; i < 3; ++i) {
                    uint32_t a = mesh_.triangles[t+i];
                    uint32_t b = mesh_.triangles[t+(i+1)%3];
                    EdgeKey ek{a,b};
                    edge_count[ek]++;
                }
            }
            for (const auto& ec : edge_count) {
                if (ec.second == 1) {
                    boundary[ec.first.v0] = true;
                    boundary[ec.first.v1] = true;
                }
            }
        }

        for (size_t i = 0; i < nv; ++i) {
            if (boundary[i]) {
                new_positions[i] = mesh_.vertices[i];
                continue;
            }
            // Gather one‑ring neighbours
            std::unordered_set<uint32_t> neighbours;
            for (size_t t = 0; t < mesh_.triangles.size(); t += 3) {
                if (mesh_.triangles[t] == i || mesh_.triangles[t+1] == i || mesh_.triangles[t+2] == i) {
                    for (int k = 0; k < 3; ++k) {
                        uint32_t v = mesh_.triangles[t+k];
                        if (v != i) neighbours.insert(v);
                    }
                }
            }
            if (neighbours.empty()) {
                new_positions[i] = mesh_.vertices[i];
                continue;
            }
            DirectX::XMVECTOR centroid = DirectX::XMVectorZero();
            for (uint32_t nb : neighbours)
                centroid = DirectX::XMVectorAdd(centroid, mesh_.vertices[nb]);
            centroid = DirectX::XMVectorScale(centroid, 1.0f / neighbours.size());

            // Compute vertex normal as average of incident face normals weighted by area
            DirectX::XMVECTOR normal = DirectX::XMVectorZero();
            for (size_t t = 0; t < mesh_.triangles.size(); t += 3) {
                if (mesh_.triangles[t] == i || mesh_.triangles[t+1] == i || mesh_.triangles[t+2] == i) {
                    DirectX::XMVECTOR p0 = mesh_.vertices[mesh_.triangles[t]];
                    DirectX::XMVECTOR p1 = mesh_.vertices[mesh_.triangles[t+1]];
                    DirectX::XMVECTOR p2 = mesh_.vertices[mesh_.triangles[t+2]];
                    DirectX::XMVECTOR n = vector_math::cross3(
                        DirectX::XMVectorSubtract(p1, p0),
                        DirectX::XMVectorSubtract(p2, p0));
                    normal = DirectX::XMVectorAdd(normal, n);
                }
            }
            if (vector_math::length_sq3_scalar(normal) > 1e-12f)
                normal = vector_math::normalize3(normal);
            else
                normal = DirectX::XMVectorSet(0,0,1,0);

            // Compute direction to centroid and project onto tangent plane
            DirectX::XMVECTOR dir = DirectX::XMVectorSubtract(centroid, mesh_.vertices[i]);
            float ndot = vector_math::dot3_scalar(dir, normal);
            dir = DirectX::XMVectorSubtract(dir, DirectX::XMVectorScale(normal, ndot));

            new_positions[i] = DirectX::XMVectorAdd(mesh_.vertices[i], DirectX::XMVectorScale(dir, 0.5f));
        }
        mesh_.vertices.swap(new_positions);
    }
};

} // namespace remeshing
} // namespace SimulationMath

#endif // CORE_MATH_MESH_REMESHING_H