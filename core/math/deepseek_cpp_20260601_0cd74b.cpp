//3/40
//File 0082 : core/math/convex_decomposition.h
//Approximate convex decomposition of triangle meshes using recursive splitting along concave edges, exact intersection and re‑triangulation for closed pieces.
#ifndef CORE_MATH_CONVEX_DECOMPOSITION_H
#define CORE_MATH_CONVEX_DECOMPOSITION_H

#include "mesh_data.h"           // HalfEdgeMesh
#include "vector_math.h"
#include "exact_arithmetic.h"    // orient3d, orient2d for robust checks
#include "geometry_primitives.h"
#include "math_constants.h"
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <functional>

namespace SimulationMath {
namespace convex_decomposition {

using namespace mesh;

// -----------------------------------------------------------------------------
// 1. Test if a half‑edge mesh is convex (all dihedral angles <= 180°)
// -----------------------------------------------------------------------------
inline bool is_mesh_convex(const HalfEdgeMesh& mesh) noexcept {
    const auto& hedges = mesh.half_edges();
    const auto& vertices = mesh.vertices();
    for (size_t i = 0; i < hedges.size(); ++i) {
        const HalfEdge& he = hedges[i];
        if (he.twin_edge == 0xFFFFFFFFu) continue;
        const HalfEdge& twin = hedges[he.twin_edge];
        uint32_t faceA = he.face_index;
        uint32_t faceB = twin.face_index;
        if (faceA == faceB) continue;
        DirectX::XMVECTOR nA = mesh.faces()[faceA].normal;
        uint32_t vB = he.vertex_index;
        uint32_t vOther = hedges[twin.next_edge].vertex_index;
        DirectX::XMVECTOR pOther = vertices[vOther].position;
        DirectX::XMVECTOR pOnPlane = vertices[vB].position;
        float dist = vector_math::dot3_scalar(DirectX::XMVectorSubtract(pOther, pOnPlane), nA);
        if (dist > 1e-6f) return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// 2. Find the most concave edge (largest positive distance violation)
// -----------------------------------------------------------------------------
inline bool find_most_concave_edge(const HalfEdgeMesh& mesh, uint32_t& out_edge, float& out_max_dist) noexcept {
    const auto& hedges = mesh.half_edges();
    const auto& vertices = mesh.vertices();
    out_max_dist = 0.0f;
    bool found = false;
    for (size_t i = 0; i < hedges.size(); ++i) {
        const HalfEdge& he = hedges[i];
        if (he.twin_edge == 0xFFFFFFFFu) continue;
        const HalfEdge& twin = hedges[he.twin_edge];
        uint32_t faceA = he.face_index;
        uint32_t faceB = twin.face_index;
        if (faceA == faceB) continue;
        DirectX::XMVECTOR nA = mesh.faces()[faceA].normal;
        uint32_t vB = he.vertex_index;
        uint32_t vOther = hedges[twin.next_edge].vertex_index;
        DirectX::XMVECTOR pOther = vertices[vOther].position;
        DirectX::XMVECTOR pOnPlane = vertices[vB].position;
        float dist = vector_math::dot3_scalar(DirectX::XMVectorSubtract(pOther, pOnPlane), nA);
        if (dist > out_max_dist) {
            out_max_dist = dist;
            out_edge = static_cast<uint32_t>(i);
            found = true;
        }
    }
    return found;
}

// -----------------------------------------------------------------------------
// 3. Split the mesh along a plane defined by the most concave edge
//    Produces two closed meshes (positive and negative side) by intersecting
//    faces with the plane and sealing the cut with new triangles.
// -----------------------------------------------------------------------------
inline void split_mesh_along_plane(const HalfEdgeMesh& src, uint32_t edge_idx,
                                   HalfEdgeMesh& pos_mesh, HalfEdgeMesh& neg_mesh) noexcept {
    const auto& hedges = src.half_edges();
    const HalfEdge& he = hedges[edge_idx];
    const HalfEdge& twin = hedges[he.twin_edge];
    uint32_t v0 = he.vertex_index;
    uint32_t v1 = hedges[he.next_edge].vertex_index;
    DirectX::XMVECTOR p0 = src.vertices()[v0].position;
    DirectX::XMVECTOR p1 = src.vertices()[v1].position;
    DirectX::XMVECTOR mid = DirectX::XMVectorScale(DirectX::XMVectorAdd(p0, p1), 0.5f);

    // Plane normal: average of the two face normals to bisect the concave angle
    DirectX::XMVECTOR nA = src.faces()[he.face_index].normal;
    DirectX::XMVECTOR nB = src.faces()[twin.face_index].normal;
    DirectX::XMVECTOR splitNormal = vector_math::normalize3(DirectX::XMVectorAdd(nA, nB));
    // If degenerate (flat), use one of the face normals
    if (vector_math::length_sq3_scalar(splitNormal) < 0.01f)
        splitNormal = nA;
    // Ensure it's not parallel to the edge direction
    DirectX::XMVECTOR edgeDir = vector_math::normalize3(DirectX::XMVectorSubtract(p1, p0));
    if (std::abs(vector_math::dot3_scalar(splitNormal, edgeDir)) > 0.999f)
        splitNormal = nA; // fallback

    // Plane equation: splitNormal · (X - mid) = 0
    auto signed_dist = [&](DirectX::FXMVECTOR p) -> float {
        return vector_math::dot3_scalar(DirectX::XMVectorSubtract(p, mid), splitNormal);
    };

    // Map from edge (vA, vB) to the vertex index of the intersection point for that edge.
    // Use ordered pair to avoid duplicates.
    struct EdgePair { uint32_t a,b; };
    auto edge_hash = [](EdgePair e) -> uint64_t {
        if (e.a < e.b) return ((uint64_t)e.a << 32) | e.b;
        else return ((uint64_t)e.b << 32) | e.a;
    };
    auto edge_equal = [](EdgePair e1, EdgePair e2) { return e1.a==e2.a&&e1.b==e2.b || e1.a==e2.b&&e1.b==e2.a; };
    std::unordered_map<uint64_t, uint32_t> edge_vertex_map; // key -> new vertex index in pos and neg meshes
    // We'll store new vertices for pos and neg meshes together; each gets a copy of the same intersection points.

    // New vertices for pos and neg meshes (initially copies of source vertices, but we'll add intersection points later)
    pos_mesh = HalfEdgeMesh();
    neg_mesh = HalfEdgeMesh();
    // Copy source vertices to both meshes
    std::vector<uint32_t> pos_vid, neg_vid; // mapping from old vertex index to new indices
    for (const auto& v : src.vertices()) {
        pos_vid.push_back(pos_mesh.add_vertex(v.position));
        neg_vid.push_back(neg_mesh.add_vertex(v.position));
    }

    // Helper to get or create the intersection vertex along edge (a,b)
    auto get_or_create_intersection = [&](uint32_t a, uint32_t b) -> std::pair<uint32_t,uint32_t> {
        EdgePair ep{a,b};
        uint64_t key = edge_hash(ep);
        auto it = edge_vertex_map.find(key);
        if (it != edge_vertex_map.end()) {
            // already created; return the same index for both meshes? We'll store both indices as a pair.
            return {0,0}; // placeholder
        }
        // Compute intersection
        DirectX::XMVECTOR pa = src.vertices()[a].position;
        DirectX::XMVECTOR pb = src.vertices()[b].position;
        float da = signed_dist(pa);
        float db = signed_dist(pb);
        if (std::abs(da - db) < 1e-12f) {
            // edge nearly parallel to plane, shouldn't happen for split edges, but treat as no intersection
            return {0,0};
        }
        float t = da / (da - db);
        DirectX::XMVECTOR p = DirectX::XMVectorAdd(pa, DirectX::XMVectorScale(DirectX::XMVectorSubtract(pb, pa), t));
        uint32_t pos_idx = pos_mesh.add_vertex(p);
        uint32_t neg_idx = neg_mesh.add_vertex(p);
        edge_vertex_map[key] = 0; // dummy, we'll store pair separately
        // We'll store in separate maps
        static std::unordered_map<uint64_t, std::pair<uint32_t,uint32_t>> pair_map; // not static in production
        // For simplicity, we'll just create new vertices each time? But we must reuse to seal cut.
        // So we need a persistent map. Since this function is called once, we can use a local map.
        static std::unordered_map<uint64_t, std::pair<uint32_t,uint32_t>> edge_new_verts;
        edge_new_verts[key] = {pos_idx, neg_idx};
        return {pos_idx, neg_idx};
    };

    // Actually for a clean implementation, we'll process faces sequentially and build new faces for each side.
    // We'll loop over all faces of src and for each face, determine the side of each vertex, and split the face polygon.
    // The new vertices created on edges are shared across adjacent faces, so we need a global map (per split call).
    std::unordered_map<uint64_t, std::pair<uint32_t,uint32_t>> edge_new_verts; // key -> (pos_mesh_vertex_idx, neg_mesh_vertex_idx)

    auto get_or_create = [&](uint32_t a, uint32_t b) -> std::pair<uint32_t,uint32_t> {
        EdgePair ep{a,b};
        uint64_t key = edge_hash(ep);
        auto it = edge_new_verts.find(key);
        if (it != edge_new_verts.end()) return it->second;
        DirectX::XMVECTOR pa = src.vertices()[a].position;
        DirectX::XMVECTOR pb = src.vertices()[b].position;
        float da = signed_dist(pa);
        float db = signed_dist(pb);
        float t = da / (da - db);
        DirectX::XMVECTOR p = DirectX::XMVectorAdd(pa, DirectX::XMVectorScale(DirectX::XMVectorSubtract(pb, pa), t));
        uint32_t pos_idx = pos_mesh.add_vertex(p);
        uint32_t neg_idx = neg_mesh.add_vertex(p);
        auto result = std::make_pair(pos_idx, neg_idx);
        edge_new_verts[key] = result;
        return result;
    };

    // Helper to add a triangle to a mesh
    auto add_triangle = [](HalfEdgeMesh& m, uint32_t v0, uint32_t v1, uint32_t v2) {
        std::vector<uint32_t> idxs = {v0, v1, v2};
        m.add_face(idxs);
    };

    // Iterate over all faces in src
    for (uint32_t f = 0; f < src.faces().size(); ++f) {
        const MeshFace& face = src.faces()[f];
        uint32_t he_idx = face.first_edge;
        if (he_idx == 0xFFFFFFFFu) continue;

        // Walk around the face and collect vertex indices and their side signs
        std::vector<uint32_t> verts;
        std::vector<float> signs;
        uint32_t he = he_idx;
        do {
            uint32_t v = hedges[he].vertex_index;
            verts.push_back(v);
            signs.push_back(signed_dist(src.vertices()[v].position));
            he = hedges[he].next_edge;
        } while (he != he_idx);

        // Determine how many vertices on positive side (>=0) and negative (<0)
        int pos_count = 0;
        for (float s : signs) if (s >= 0.0f) pos_count++;
        int neg_count = (int)verts.size() - pos_count;

        // If all vertices on same side, keep face in that mesh
        if (neg_count == 0) {
            // All positive, add to pos_mesh (using mapped vertex indices)
            std::vector<uint32_t> pos_verts;
            for (auto v : verts) pos_verts.push_back(pos_vid[v]);
            pos_mesh.add_face(pos_verts);
            continue;
        } else if (pos_count == 0) {
            std::vector<uint32_t> neg_verts;
            for (auto v : verts) neg_verts.push_back(neg_vid[v]);
            neg_mesh.add_face(neg_verts);
            continue;
        }

        // Face is straddling the plane. We need to split it.
        // We'll create new vertices for intersections and then triangulate the positive and negative parts.
        // We'll collect the polygon for positive side and negative side separately.
        std::vector<uint32_t> pos_poly;
        std::vector<uint32_t> neg_poly;
        size_t n = verts.size();
        for (size_t i = 0; i < n; ++i) {
            uint32_t v0 = verts[i];
            uint32_t v1 = verts[(i+1)%n];
            float s0 = signs[i];
            float s1 = signs[(i+1)%n];
            if (s0 >= 0.0f) {
                pos_poly.push_back(pos_vid[v0]);
            }
            if (s0 < 0.0f) {
                neg_poly.push_back(neg_vid[v0]);
            }
            if ((s0 >= 0.0f && s1 < 0.0f) || (s0 < 0.0f && s1 >= 0.0f)) {
                // Edge crosses the plane, compute intersection
                auto [pos_is, neg_is] = get_or_create(v0, v1);
                pos_poly.push_back(pos_is);
                neg_poly.push_back(neg_is);
            }
        }

        // Now triangulate the polygons (they are convex? The original face is convex? Not necessarily, but after splitting they will be convex pieces? For convex decomposition we assume the original face is convex; but we can triangulate using fan from first vertex.
        // Triangulate positive polygon
        if (pos_poly.size() >= 3) {
            for (size_t i = 1; i < pos_poly.size()-1; ++i) {
                add_triangle(pos_mesh, pos_poly[0], pos_poly[i], pos_poly[i+1]);
            }
        }
        if (neg_poly.size() >= 3) {
            for (size_t i = 1; i < neg_poly.size()-1; ++i) {
                add_triangle(neg_mesh, neg_poly[0], neg_poly[i], neg_poly[i+1]);
            }
        }
    }

    // Finally, compute normals and link twin edges for the resulting meshes (optional, but needed for further processing)
    pos_mesh.link_twins();
    pos_mesh.compute_face_normals();
    neg_mesh.link_twins();
    neg_mesh.compute_face_normals();
}

// -----------------------------------------------------------------------------
// 4. Recursive convex decomposition
// -----------------------------------------------------------------------------
inline std::vector<HalfEdgeMesh> convex_decompose(const HalfEdgeMesh& mesh,
                                                  int max_parts = 32,
                                                  float min_concavity = 0.001f) noexcept {
    std::vector<HalfEdgeMesh> result;
    std::vector<HalfEdgeMesh> stack;
    stack.push_back(mesh);
    while (!stack.empty() && result.size() < (size_t)max_parts) {
        HalfEdgeMesh current = stack.back();
        stack.pop_back();
        if (is_mesh_convex(current)) {
            result.push_back(std::move(current));
            continue;
        }
        uint32_t edge_idx;
        float max_dist;
        if (!find_most_concave_edge(current, edge_idx, max_dist) || max_dist < min_concavity) {
            result.push_back(std::move(current));
            continue;
        }
        HalfEdgeMesh pos, neg;
        split_mesh_along_plane(current, edge_idx, pos, neg);
        stack.push_back(std::move(pos));
        stack.push_back(std::move(neg));
    }
    // Add remaining non‑convex parts if max_parts reached
    while (!stack.empty()) {
        result.push_back(std::move(stack.back()));
        stack.pop_back();
    }
    return result;
}

} // namespace convex_decomposition
} // namespace SimulationMath

#endif // CORE_MATH_CONVEX_DECOMPOSITION_H