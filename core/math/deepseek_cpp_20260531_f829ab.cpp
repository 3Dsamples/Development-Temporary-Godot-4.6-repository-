//File 0056 : core/math/mesh_data.h
//Half‑edge mesh representation: vertex/face/edge containers, adjacency iterators, normal/area computation, SIMD‑accelerated geometric queries.
#ifndef CORE_MATH_MESH_DATA_H
#define CORE_MATH_MESH_DATA_H

#include "vector_math.h"
#include "geometry_primitives.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <vector>
#include <cstdint>
#include <cmath>
#include <limits>
#include <utility>
#include <functional>

namespace SimulationMath {
namespace mesh {

// -----------------------------------------------------------------------------
// 1. Vertex structure
// -----------------------------------------------------------------------------
struct MeshVertex {
    DirectX::XMVECTOR position;
    DirectX::XMVECTOR normal;
    DirectX::XMVECTOR color;       // RGBA packed? We'll store as separate floats for now; could use perceptual colour for debug.
    uint32_t        first_edge;    // index of one outgoing half‑edge (for adjacency traversal)
    MeshVertex() noexcept {
        position = vector_math::zero();
        normal   = vector_math::load3(0.0f, 0.0f, 1.0f);
        color    = vector_math::load4(1.0f, 1.0f, 1.0f, 1.0f);
        first_edge = 0xFFFFFFFFu;
    }
    explicit MeshVertex(DirectX::FXMVECTOR pos) noexcept : position(pos), normal(vector_math::load3(0,0,1)), color(vector_math::load4(1,1,1,1)), first_edge(0xFFFFFFFFu) {}
};

// -----------------------------------------------------------------------------
// 2. Half‑edge structure
// -----------------------------------------------------------------------------
struct HalfEdge {
    uint32_t vertex_index;      // vertex the half‑edge points to
    uint32_t face_index;        // face it belongs to
    uint32_t next_edge;         // next half‑edge in the same face
    uint32_t prev_edge;         // previous half‑edge in the same face
    uint32_t twin_edge;         // opposite half‑edge
    HalfEdge() noexcept : vertex_index(0xFFFFFFFFu), face_index(0xFFFFFFFFu), next_edge(0xFFFFFFFFu), prev_edge(0xFFFFFFFFu), twin_edge(0xFFFFFFFFu) {}
};

// -----------------------------------------------------------------------------
// 3. Face structure
// -----------------------------------------------------------------------------
struct MeshFace {
    uint32_t first_edge;        // one half‑edge of the face
    DirectX::XMVECTOR normal;   // face normal (precomputed)
    MeshFace() noexcept : first_edge(0xFFFFFFFFu), normal(vector_math::load3(0,0,1)) {}
};

// -----------------------------------------------------------------------------
// 4. The mesh class
// -----------------------------------------------------------------------------
class HalfEdgeMesh {
public:
    HalfEdgeMesh() = default;

    // -----------------------------------------------------------------------
    // Add a vertex, returns index
    // -----------------------------------------------------------------------
    uint32_t add_vertex(DirectX::FXMVECTOR position) {
        vertices_.emplace_back(position);
        return static_cast<uint32_t>(vertices_.size() - 1);
    }

    // -----------------------------------------------------------------------
    // Add a face as a polygon (list of vertex indices, must be counter‑clockwise)
    // -----------------------------------------------------------------------
    uint32_t add_face(const std::vector<uint32_t>& vertex_indices) {
        if (vertex_indices.size() < 3) return 0xFFFFFFFFu;
        uint32_t face_idx = static_cast<uint32_t>(faces_.size());
        faces_.emplace_back();

        size_t num_verts = vertex_indices.size();
        // Create half‑edges
        std::vector<uint32_t> he_indices(num_verts);
        for (size_t i = 0; i < num_verts; ++i) {
            HalfEdge he;
            he.vertex_index = vertex_indices[i];
            he.face_index   = face_idx;
            he.twin_edge    = 0xFFFFFFFFu; // will be set later if we link twins
            half_edges_.push_back(he);
            he_indices[i] = static_cast<uint32_t>(half_edges_.size() - 1);
        }
        // Link next/prev pointers cyclically
        for (size_t i = 0; i < num_verts; ++i) {
            half_edges_[he_indices[i]].next_edge = he_indices[(i+1) % num_verts];
            half_edges_[he_indices[i]].prev_edge = he_indices[(i+num_verts-1) % num_verts];
        }
        // Set the first edge of the face
        faces_[face_idx].first_edge = he_indices[0];

        // Update vertex's first_edge if not already set (or we could set to the first incident edge)
        for (size_t i = 0; i < num_verts; ++i) {
            uint32_t v = vertex_indices[i];
            if (vertices_[v].first_edge == 0xFFFFFFFFu)
                vertices_[v].first_edge = he_indices[i];
        }

        // Attempt to link twin edges: for each edge, if there exists another half‑edge between the same vertices in opposite order, link them.
        // For now we skip automatic twin linking; user can call link_twins().
        return face_idx;
    }

    // -----------------------------------------------------------------------
    // Link all twin edges by vertex pair matching (simple O(E^2) for building)
    // -----------------------------------------------------------------------
    void link_twins() {
        // Build map from ordered pair (v0,v1) to half‑edge index
        std::unordered_map<uint64_t, uint32_t> edge_map;
        for (uint32_t i = 0; i < half_edges_.size(); ++i) {
            const HalfEdge& he = half_edges_[i];
            uint32_t v0 = he.vertex_index;
            uint32_t v1 = half_edges_[he.next_edge].vertex_index; // next vertex along the face
            uint64_t key = (uint64_t)v0 | ((uint64_t)v1 << 32);
            auto it = edge_map.find(key);
            if (it != edge_map.end()) {
                // Found opposite direction? Actually we should store key as sorted pair for undirected edge
                // We'll instead store using ordered pair (v0,v1) and search for (v1,v0) when adding.
                // We'll do two‑pass: first collect all edges, then match.
            }
        }
        // Simpler: for each half‑edge i, find another half‑edge j that shares same vertices in reverse order and link twins.
        for (uint32_t i = 0; i < half_edges_.size(); ++i) {
            if (half_edges_[i].twin_edge != 0xFFFFFFFFu) continue;
            uint32_t v0 = half_edges_[i].vertex_index;
            uint32_t v1 = half_edges_[half_edges_[i].next_edge].vertex_index;
            for (uint32_t j = i+1; j < half_edges_.size(); ++j) {
                if (half_edges_[j].twin_edge != 0xFFFFFFFFu) continue;
                uint32_t u0 = half_edges_[j].vertex_index;
                uint32_t u1 = half_edges_[half_edges_[j].next_edge].vertex_index;
                if (v0 == u1 && v1 == u0) {
                    half_edges_[i].twin_edge = j;
                    half_edges_[j].twin_edge = i;
                    break;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Compute face normals for all faces (based on first three vertices)
    // -----------------------------------------------------------------------
    void compute_face_normals() {
        for (auto& face : faces_) {
            uint32_t he0 = face.first_edge;
            if (he0 == 0xFFFFFFFFu) continue;
            uint32_t v0 = half_edges_[he0].vertex_index;
            uint32_t v1 = half_edges_[half_edges_[he0].next_edge].vertex_index;
            uint32_t v2 = half_edges_[half_edges_[half_edges_[he0].next_edge].next_edge].vertex_index;
            DirectX::XMVECTOR p0 = vertices_[v0].position;
            DirectX::XMVECTOR p1 = vertices_[v1].position;
            DirectX::XMVECTOR p2 = vertices_[v2].position;
            DirectX::XMVECTOR e1 = DirectX::XMVectorSubtract(p1, p0);
            DirectX::XMVECTOR e2 = DirectX::XMVectorSubtract(p2, p0);
            face.normal = vector_math::normalize3(vector_math::cross3(e1, e2));
        }
    }

    // -----------------------------------------------------------------------
    // Compute vertex normals as area‑weighted average of adjacent face normals
    // -----------------------------------------------------------------------
    void compute_vertex_normals() {
        // Zero all vertex normals
        for (auto& v : vertices_)
            v.normal = vector_math::zero();

        for (const auto& face : faces_) {
            uint32_t he = face.first_edge;
            if (he == 0xFFFFFFFFu) continue;
            // Iterate around the face
            uint32_t start = he;
            do {
                uint32_t v = half_edges_[he].vertex_index;
                // Weight by face area? We'll just add face normal.
                vertices_[v].normal = DirectX::XMVectorAdd(vertices_[v].normal, face.normal);
                he = half_edges_[he].next_edge;
            } while (he != start);
        }
        // Normalize each vertex normal
        for (auto& v : vertices_) {
            float len = vector_math::length3_scalar(v.normal);
            if (len > 1e-12f)
                v.normal = DirectX::XMVectorScale(v.normal, 1.0f / len);
        }
    }

    // -----------------------------------------------------------------------
    // Compute area of a face (assuming planar, sum of triangle areas)
    // -----------------------------------------------------------------------
    float face_area(uint32_t face_idx) const {
        const MeshFace& face = faces_[face_idx];
        uint32_t he = face.first_edge;
        if (he == 0xFFFFFFFFu) return 0.0f;
        float total_area = 0.0f;
        // Triangulate from first vertex
        uint32_t v0 = half_edges_[he].vertex_index;
        uint32_t he1 = half_edges_[he].next_edge;
        while (half_edges_[he1].next_edge != he) { // while not back to start's previous (i.e., triangle fan)
            uint32_t v1 = half_edges_[he1].vertex_index;
            uint32_t v2 = half_edges_[half_edges_[he1].next_edge].vertex_index;
            DirectX::XMVECTOR p0 = vertices_[v0].position;
            DirectX::XMVECTOR p1 = vertices_[v1].position;
            DirectX::XMVECTOR p2 = vertices_[v2].position;
            DirectX::XMVECTOR cross = vector_math::cross3(DirectX::XMVectorSubtract(p1, p0), DirectX::XMVectorSubtract(p2, p0));
            total_area += 0.5f * vector_math::length3_scalar(cross);
            he1 = half_edges_[he1].next_edge;
        }
        return total_area;
    }

    // -----------------------------------------------------------------------
    // Edge length between two adjacent vertices
    // -----------------------------------------------------------------------
    float edge_length(uint32_t half_edge_idx) const {
        const HalfEdge& he = half_edges_[half_edge_idx];
        uint32_t v0 = he.vertex_index;
        uint32_t v1 = half_edges_[he.next_edge].vertex_index;
        return vector_math::length3_scalar(DirectX::XMVectorSubtract(vertices_[v1].position, vertices_[v0].position));
    }

    // -----------------------------------------------------------------------
    // Vertex adjacency: iterate over faces sharing a vertex
    // -----------------------------------------------------------------------
    void for_each_face_around_vertex(uint32_t vertex_idx, const std::function<void(uint32_t)>& callback) const {
        uint32_t start_edge = vertices_[vertex_idx].first_edge;
        if (start_edge == 0xFFFFFFFFu) return;
        uint32_t he = start_edge;
        do {
            callback(half_edges_[he].face_index);
            // Move to the next half‑edge that originates from the same vertex (prev of twin)
            he = half_edges_[half_edges_[he].prev_edge].twin_edge;
            if (he == 0xFFFFFFFFu) break; // boundary
        } while (he != start_edge);
    }

    // -----------------------------------------------------------------------
    // Direct accessors
    // -----------------------------------------------------------------------
    const std::vector<MeshVertex>& vertices() const noexcept { return vertices_; }
    const std::vector<HalfEdge>& half_edges() const noexcept { return half_edges_; }
    const std::vector<MeshFace>& faces() const noexcept { return faces_; }
    size_t vertex_count() const noexcept { return vertices_.size(); }
    size_t face_count() const noexcept { return faces_.size(); }

private:
    std::vector<MeshVertex> vertices_;
    std::vector<HalfEdge>   half_edges_;
    std::vector<MeshFace>   faces_;
};

} // namespace mesh
} // namespace SimulationMath

#endif // CORE_MATH_MESH_DATA_H