//File 0102 : core/math/mesh_raycasting.h
//Accelerated ray‑triangle intersection for triangle meshes using bounding‑volume hierarchy (BVH) and stack‑based traversal; supports closest‑hit queries.
#ifndef CORE_MATH_MESH_RAYCASTING_H
#define CORE_MATH_MESH_RAYCASTING_H

#include "mesh_data.h"               // HalfEdgeMesh
#include "vector_math.h"
#include "geometry_primitives.h"     // AABB, Ray, Triangle
#include "math_constants.h"
#include <vector>
#include <stack>
#include <algorithm>
#include <limits>
#include <cstdint>

namespace SimulationMath {
namespace mesh_raycasting {

// -----------------------------------------------------------------------------
// 1. BVH node structure (binary tree, stored in array)
// -----------------------------------------------------------------------------
struct BVHNode {
    geometry::AABB bounds;
    uint32_t left_or_first;   // if leaf: first triangle index, else left child index
    uint32_t right_or_count;  // if leaf: triangle count, else right child index
    bool is_leaf() const noexcept { return (right_or_count & 0x80000000u) != 0; } // high bit set for leaf
    uint32_t count() const noexcept { return right_or_count & 0x7FFFFFFFu; }
};

// -----------------------------------------------------------------------------
// 2. Triangle storage: indices and positions (packed for cache)
// -----------------------------------------------------------------------------
struct PackedTriangle {
    uint32_t v0, v1, v2;
    DirectX::XMVECTOR p0, p1, p2;
    PackedTriangle() noexcept = default;
    PackedTriangle(uint32_t a, uint32_t b, uint32_t c,
                   DirectX::FXMVECTOR q0, DirectX::FXMVECTOR q1, DirectX::FXMVECTOR q2) noexcept
        : v0(a), v1(b), v2(c), p0(q0), p1(q1), p2(q2) {}
};

// -----------------------------------------------------------------------------
// 3. Ray‑mesh intersection result
// -----------------------------------------------------------------------------
struct RayMeshHit {
    float t;                     // distance along ray to intersection
    uint32_t triangle_index;     // which triangle
    float u, v;                  // barycentric coordinates (u, v, w=1-u-v)
};

// -----------------------------------------------------------------------------
// 4. Ray casting BVH for a triangle mesh
// -----------------------------------------------------------------------------
class MeshRaycaster {
public:
    MeshRaycaster() noexcept = default;

    // Build the BVH from a HalfEdgeMesh (extracts triangles)
    void build(const HalfEdgeMesh& mesh) noexcept {
        const auto& verts = mesh.vertices();
        const auto& hedges = mesh.half_edges();
        const auto& faces = mesh.faces();

        // Extract all triangles into packed array
        triangles_.clear();
        for (size_t f = 0; f < faces.size(); ++f) {
            const MeshFace& face = faces[f];
            uint32_t he0 = face.first_edge;
            if (he0 == 0xFFFFFFFFu) continue;
            uint32_t v0 = hedges[he0].vertex_index;
            uint32_t v1 = hedges[hedges[he0].next_edge].vertex_index;
            uint32_t v2 = hedges[hedges[he0].next_edge].next_edge;
            v2 = hedges[v2].vertex_index;
            triangles_.emplace_back(v0, v1, v2,
                                    verts[v0].position,
                                    verts[v1].position,
                                    verts[v2].position);
        }
        if (triangles_.empty()) return;

        // Build leaf indices array for recursive construction
        std::vector<uint32_t> indices(triangles_.size());
        for (size_t i = 0; i < indices.size(); ++i) indices[i] = static_cast<uint32_t>(i);

        // Recursively build BVH nodes
        nodes_.clear();
        nodes_.reserve(2 * triangles_.size());
        uint32_t root = build_recursive(indices.data(), 0, static_cast<uint32_t>(indices.size()), 0);
        (void)root; // root is always 0 if we push nodes sequentially.
    }

    // Closest‑hit ray intersection: returns true and fills hit if any triangle is hit
    bool closest_hit(const geometry::Ray& ray, RayMeshHit& out_hit) const noexcept {
        if (nodes_.empty()) return false;
        float t_min = ray.t_min;
        float t_max = ray.t_max;
        bool hit = false;
        // Stack for traversal (max depth 64)
        uint32_t stack[64];
        int stack_ptr = 0;
        stack[stack_ptr++] = 0; // root

        // Precompute inverse direction for AABB test
        float inv_dir_x = 1.0f / vector_math::get_x(ray.direction);
        float inv_dir_y = 1.0f / vector_math::get_y(ray.direction);
        float inv_dir_z = 1.0f / vector_math::get_z(ray.direction);

        while (stack_ptr > 0) {
            uint32_t idx = stack[--stack_ptr];
            const BVHNode& node = nodes_[idx];

            // Ray‑AABB intersection test (slabs method)
            if (!ray_aabb_intersect(ray, node.bounds, inv_dir_x, inv_dir_y, inv_dir_z, t_max))
                continue;

            if (node.is_leaf()) {
                uint32_t first = node.left_or_first;
                uint32_t count = node.count();
                for (uint32_t i = 0; i < count; ++i) {
                    const PackedTriangle& tri = triangles_[first + i];
                    float t, u, v;
                    if (ray_triangle_intersect(ray, tri.p0, tri.p1, tri.p2, t, u, v)) {
                        if (t >= t_min && t < t_max) {
                            t_max = t;
                            out_hit.t = t;
                            out_hit.u = u;
                            out_hit.v = v;
                            out_hit.triangle_index = first + i;
                            hit = true;
                        }
                    }
                }
            } else {
                // Push both children (traverse front‑to‑back? not implemented; just push both)
                uint32_t left  = node.left_or_first;
                uint32_t right = node.right_or_count;
                // Optional: could push closer child first based on ray direction sign, but simple stack works.
                if (stack_ptr + 2 <= 64) {
                    stack[stack_ptr++] = right;
                    stack[stack_ptr++] = left;
                }
            }
        }
        return hit;
    }

private:
    std::vector<BVHNode> nodes_;
    std::vector<PackedTriangle> triangles_;

    // Recursive BVH construction (returns node index)
    uint32_t build_recursive(uint32_t* indices, uint32_t start, uint32_t end, uint32_t depth) {
        uint32_t count = end - start;
        // Compute AABB of the range
        geometry::AABB box;
        for (uint32_t i = start; i < end; ++i) {
            const PackedTriangle& tri = triangles_[indices[i]];
            box.extend(tri.p0);
            box.extend(tri.p1);
            box.extend(tri.p2);
        }

        // Create leaf node if few triangles or max depth
        const uint32_t MAX_LEAF_TRIS = 4;
        if (count <= MAX_LEAF_TRIS || depth > 24) {
            BVHNode leaf;
            leaf.bounds = box;
            leaf.left_or_first = start; // first index in original indices (but we need to copy them to a contiguous array)
            leaf.right_or_count = count | 0x80000000u; // leaf flag
            nodes_.push_back(leaf);
            // copy indices to leaf storage? For simplicity, we assume indices are in the original triangles_ order; we'll need a persistent array.
            // Since our construction algorithm reorders the indices array, the leaf must store a pointer to a segment of the global triangle list. But we are building from a temporary indices array. We'll instead store the indices directly into a separate leaf_triangles_ vector and store start/count.
            // But the code above doesn't copy; we'll modify: we'll create a vector leafTriangles that holds the actual triangle indices for leaves.
            // For this implementation, we'll store the leaf's triangle indices as a contiguous sub‑list in a separate array leaf_triangles_. The left_or_first will be the start in that array.
            // This design is incomplete without that array. Since we must provide complete code, I'll implement a proper data structure: a single vector `leaf_indices_` that stores the triangle indices for all leaves in order, and leaf nodes point into it.
            // We'll modify the build to accumulate leaf indices.
            // To keep it simple, I'll use a different approach: store triangle data directly in leaves? No.
            // Instead, I'll restructure: the BVH will own an array of `PackedTriangle` and an array of indices (leaf_indices_). For internal nodes, left_or_first is left child index, right_or_count is right child index. For leaves, left_or_first is first index in leaf_indices_, right_or_count has high bit set and count. We'll fill leaf_indices_ during build.
            // But the current build works on a temporary `indices` array that is reordered. The leaf must capture the indices at its moment. So we need to copy the relevant segment of `indices` into a persistent `leaf_indices_` array and record the offset. I'll implement that.
            // I'll declare leaf_indices_ as a member and modify accordingly.
        }

        // Internal node: split along longest axis
        uint32_t axis = 0;
        float extent_x = vector_math::get_x(box.max) - vector_math::get_x(box.min);
        float extent_y = vector_math::get_y(box.max) - vector_math::get_y(box.min);
        float extent_z = vector_math::get_z(box.max) - vector_math::get_z(box.min);
        if (extent_y > extent_x && extent_y > extent_z) axis = 1;
        if (extent_z > extent_x && extent_z > extent_y) axis = 2;

        // Median split based on triangle centroids
        uint32_t mid = start + count / 2;
        std::nth_element(indices + start, indices + mid, indices + end,
            [&](uint32_t a, uint32_t b) {
                float ca = get_centroid_component(triangles_[a], axis);
                float cb = get_centroid_component(triangles_[b], axis);
                return ca < cb;
            });

        BVHNode internal;
        internal.bounds = box;
        // push placeholder, children will be set later
        uint32_t my_idx = static_cast<uint32_t>(nodes_.size());
        nodes_.push_back(internal);
        uint32_t left = build_recursive(indices, start, mid, depth + 1);
        uint32_t right = build_recursive(indices, mid, end, depth + 1);
        nodes_[my_idx].left_or_first = left;
        nodes_[my_idx].right_or_count = right;
        return my_idx;
    }

    // Get centroid coordinate along axis for a triangle
    static float get_centroid_component(const PackedTriangle& tri, uint32_t axis) noexcept {
        float cx = (vector_math::get_x(tri.p0) + vector_math::get_x(tri.p1) + vector_math::get_x(tri.p2)) / 3.0f;
        float cy = (vector_math::get_y(tri.p0) + vector_math::get_y(tri.p1) + vector_math::get_y(tri.p2)) / 3.0f;
        float cz = (vector_math::get_z(tri.p0) + vector_math::get_z(tri.p1) + vector_math::get_z(tri.p2)) / 3.0f;
        return (axis == 0) ? cx : (axis == 1) ? cy : cz;
    }

    // Ray‑AABB intersection test (slabs)
    static bool ray_aabb_intersect(const geometry::Ray& ray, const geometry::AABB& box,
                                   float inv_dir_x, float inv_dir_y, float inv_dir_z,
                                   float t_max) noexcept {
        float t0 = (vector_math::get_x(box.min) - vector_math::get_x(ray.origin)) * inv_dir_x;
        float t1 = (vector_math::get_x(box.max) - vector_math::get_x(ray.origin)) * inv_dir_x;
        float tx_min = std::min(t0, t1);
        float tx_max = std::max(t0, t1);

        t0 = (vector_math::get_y(box.min) - vector_math::get_y(ray.origin)) * inv_dir_y;
        t1 = (vector_math::get_y(box.max) - vector_math::get_y(ray.origin)) * inv_dir_y;
        float ty_min = std::min(t0, t1);
        float ty_max = std::max(t0, t1);

        t0 = (vector_math::get_z(box.min) - vector_math::get_z(ray.origin)) * inv_dir_z;
        t1 = (vector_math::get_z(box.max) - vector_math::get_z(ray.origin)) * inv_dir_z;
        float tz_min = std::min(t0, t1);
        float tz_max = std::max(t0, t1);

        float t_enter = std::max({tx_min, ty_min, tz_min, ray.t_min});
        float t_exit  = std::min({tx_max, ty_max, tz_max, t_max});
        return t_enter <= t_exit;
    }

    // Ray‑triangle intersection (Möller‑Trumbore) – returns barycentric (u,v) and t
    static bool ray_triangle_intersect(const geometry::Ray& ray,
                                       DirectX::FXMVECTOR v0, DirectX::FXMVECTOR v1, DirectX::FXMVECTOR v2,
                                       float& t, float& u, float& v) noexcept {
        DirectX::XMVECTOR e1 = DirectX::XMVectorSubtract(v1, v0);
        DirectX::XMVECTOR e2 = DirectX::XMVectorSubtract(v2, v0);
        DirectX::XMVECTOR h = vector_math::cross3(ray.direction, e2);
        float a = vector_math::dot3_scalar(e1, h);
        if (std::fabs(a) < 1e-12f) return false;
        float f = 1.0f / a;
        DirectX::XMVECTOR s = DirectX::XMVectorSubtract(ray.origin, v0);
        u = f * vector_math::dot3_scalar(s, h);
        if (u < 0.0f || u > 1.0f) return false;
        DirectX::XMVECTOR q = vector_math::cross3(s, e1);
        v = f * vector_math::dot3_scalar(ray.direction, q);
        if (v < 0.0f || u + v > 1.0f) return false;
        t = f * vector_math::dot3_scalar(e2, q);
        return t >= ray.t_min;
    }
};

} // namespace mesh_raycasting
} // namespace SimulationMath

#endif // CORE_MATH_MESH_RAYCASTING_H
``` The BVH construction above has a missing `leaf_indices_` storage for leaves. I'll need to fix that to provide complete code. I'll adjust the class to include `std::vector<uint32_t> leaf_indices_` and modify build to fill it. I'll rewrite the build method to properly store leaf triangle indices and use them in traversal. I'll do that now.```cpp
//File 0102 : core/math/mesh_raycasting.h
//Accelerated ray‑triangle intersection for triangle meshes using bounding‑volume hierarchy (BVH) and stack‑based traversal; supports closest‑hit queries.
#ifndef CORE_MATH_MESH_RAYCASTING_H
#define CORE_MATH_MESH_RAYCASTING_H

#include "mesh_data.h"               // HalfEdgeMesh
#include "vector_math.h"
#include "geometry_primitives.h"     // AABB, Ray, Triangle
#include "math_constants.h"
#include <vector>
#include <algorithm>
#include <limits>
#include <cstdint>

namespace SimulationMath {
namespace mesh_raycasting {

// -----------------------------------------------------------------------------
// 1. BVH node structure (binary tree, stored in array)
// -----------------------------------------------------------------------------
struct BVHNode {
    geometry::AABB bounds;
    uint32_t left_or_first;   // if leaf: first triangle index in leaf_indices_, else left child node index
    uint32_t right_or_count;  // if leaf: triangle count (with high bit set), else right child node index
    bool is_leaf() const noexcept { return (right_or_count & 0x80000000u) != 0; }
    uint32_t count() const noexcept { return right_or_count & 0x7FFFFFFFu; }
};

// -----------------------------------------------------------------------------
// 2. Triangle storage: indices and positions (packed for cache)
// -----------------------------------------------------------------------------
struct PackedTriangle {
    DirectX::XMVECTOR p0, p1, p2;
    PackedTriangle() noexcept = default;
    PackedTriangle(DirectX::FXMVECTOR q0, DirectX::FXMVECTOR q1, DirectX::FXMVECTOR q2) noexcept
        : p0(q0), p1(q1), p2(q2) {}
};

// -----------------------------------------------------------------------------
// 3. Ray‑mesh intersection result
// -----------------------------------------------------------------------------
struct RayMeshHit {
    float t;                     // distance along ray to intersection
    uint32_t triangle_index;     // which triangle in the original mesh (face index)
    float u, v;                  // barycentric coordinates (u, v, w=1-u-v)
};

// -----------------------------------------------------------------------------
// 4. Ray casting BVH for a triangle mesh
// -----------------------------------------------------------------------------
class MeshRaycaster {
public:
    MeshRaycaster() noexcept = default;

    // Build the BVH from a HalfEdgeMesh (extracts triangles and stores face indices)
    void build(const HalfEdgeMesh& mesh) noexcept {
        const auto& verts = mesh.vertices();
        const auto& hedges = mesh.half_edges();
        const auto& faces = mesh.faces();

        // Extract all triangles into packed array
        triangles_.clear();
        face_to_original_.clear();
        for (size_t f = 0; f < faces.size(); ++f) {
            const MeshFace& face = faces[f];
            uint32_t he0 = face.first_edge;
            if (he0 == 0xFFFFFFFFu) continue;
            uint32_t v0 = hedges[he0].vertex_index;
            uint32_t v1 = hedges[hedges[he0].next_edge].vertex_index;
            uint32_t v2 = hedges[hedges[he0].next_edge].next_edge;
            v2 = hedges[v2].vertex_index;
            triangles_.emplace_back(verts[v0].position,
                                    verts[v1].position,
                                    verts[v2].position);
            face_to_original_.push_back(static_cast<uint32_t>(f));
        }
        if (triangles_.empty()) return;

        // Build leaf indices array for recursive construction
        std::vector<uint32_t> temp_indices(triangles_.size());
        for (size_t i = 0; i < temp_indices.size(); ++i) temp_indices[i] = static_cast<uint32_t>(i);

        leaf_indices_.clear();
        nodes_.clear();
        nodes_.reserve(2 * triangles_.size());
        uint32_t root = build_recursive(temp_indices.data(), 0, static_cast<uint32_t>(temp_indices.size()), 0);
        (void)root; // root is always index 0
    }

    // Closest‑hit ray intersection: returns true and fills hit if any triangle is hit
    bool closest_hit(const geometry::Ray& ray, RayMeshHit& out_hit) const noexcept {
        if (nodes_.empty()) return false;
        float t_min = ray.t_min;
        float t_max = ray.t_max;
        bool hit = false;
        uint32_t stack[64];
        int stack_ptr = 0;
        stack[stack_ptr++] = 0; // root

        // Precompute inverse direction for AABB test
        float inv_dir_x = 1.0f / vector_math::get_x(ray.direction);
        float inv_dir_y = 1.0f / vector_math::get_y(ray.direction);
        float inv_dir_z = 1.0f / vector_math::get_z(ray.direction);

        while (stack_ptr > 0) {
            uint32_t idx = stack[--stack_ptr];
            const BVHNode& node = nodes_[idx];

            // Ray‑AABB intersection test
            if (!ray_aabb_intersect(ray, node.bounds, inv_dir_x, inv_dir_y, inv_dir_z, t_max))
                continue;

            if (node.is_leaf()) {
                uint32_t first = node.left_or_first;
                uint32_t count = node.count();
                for (uint32_t i = 0; i < count; ++i) {
                    uint32_t tri_idx = leaf_indices_[first + i];
                    const PackedTriangle& tri = triangles_[tri_idx];
                    float t, u, v;
                    if (ray_triangle_intersect(ray, tri.p0, tri.p1, tri.p2, t, u, v)) {
                        if (t >= t_min && t < t_max) {
                            t_max = t;
                            out_hit.t = t;
                            out_hit.u = u;
                            out_hit.v = v;
                            out_hit.triangle_index = face_to_original_[tri_idx];
                            hit = true;
                        }
                    }
                }
            } else {
                uint32_t left  = node.left_or_first;
                uint32_t right = node.right_or_count;
                if (stack_ptr + 2 <= 64) {
                    stack[stack_ptr++] = right;
                    stack[stack_ptr++] = left;
                }
            }
        }
        return hit;
    }

private:
    std::vector<BVHNode> nodes_;
    std::vector<PackedTriangle> triangles_;            // all triangles
    std::vector<uint32_t> face_to_original_;           // maps triangle index → original face index
    std::vector<uint32_t> leaf_indices_;               // per‑leaf triangle indices (contiguous)

    // Recursive BVH construction (returns node index)
    uint32_t build_recursive(uint32_t* indices, uint32_t start, uint32_t end, uint32_t depth) {
        uint32_t count = end - start;
        // Compute AABB of the range
        geometry::AABB box;
        for (uint32_t i = start; i < end; ++i) {
            const PackedTriangle& tri = triangles_[indices[i]];
            box.extend(tri.p0);
            box.extend(tri.p1);
            box.extend(tri.p2);
        }

        const uint32_t MAX_LEAF_TRIS = 4;
        if (count <= MAX_LEAF_TRIS || depth > 24) {
            BVHNode leaf;
            leaf.bounds = box;
            leaf.left_or_first = static_cast<uint32_t>(leaf_indices_.size()); // start in leaf_indices_
            leaf.right_or_count = count | 0x80000000u; // leaf flag
            // Append the triangle indices to leaf_indices_
            for (uint32_t i = start; i < end; ++i)
                leaf_indices_.push_back(indices[i]);
            nodes_.push_back(leaf);
            return static_cast<uint32_t>(nodes_.size() - 1);
        }

        // Internal node: split along longest axis
        uint32_t axis = 0;
        float extent_x = vector_math::get_x(box.max) - vector_math::get_x(box.min);
        float extent_y = vector_math::get_y(box.max) - vector_math::get_y(box.min);
        float extent_z = vector_math::get_z(box.max) - vector_math::get_z(box.min);
        if (extent_y > extent_x && extent_y > extent_z) axis = 1;
        if (extent_z > extent_x && extent_z > extent_y) axis = 2;

        // Median split based on triangle centroids
        uint32_t mid = start + count / 2;
        std::nth_element(indices + start, indices + mid, indices + end,
            [&](uint32_t a, uint32_t b) {
                float ca = get_centroid_component(triangles_[a], axis);
                float cb = get_centroid_component(triangles_[b], axis);
                return ca < cb;
            });

        BVHNode internal;
        internal.bounds = box;
        // push placeholder, children will be set later
        uint32_t my_idx = static_cast<uint32_t>(nodes_.size());
        nodes_.push_back(internal);
        uint32_t left = build_recursive(indices, start, mid, depth + 1);
        uint32_t right = build_recursive(indices, mid, end, depth + 1);
        nodes_[my_idx].left_or_first = left;
        nodes_[my_idx].right_or_count = right;
        return my_idx;
    }

    // Get centroid coordinate along axis for a triangle
    static float get_centroid_component(const PackedTriangle& tri, uint32_t axis) noexcept {
        float cx = (vector_math::get_x(tri.p0) + vector_math::get_x(tri.p1) + vector_math::get_x(tri.p2)) / 3.0f;
        float cy = (vector_math::get_y(tri.p0) + vector_math::get_y(tri.p1) + vector_math::get_y(tri.p2)) / 3.0f;
        float cz = (vector_math::get_z(tri.p0) + vector_math::get_z(tri.p1) + vector_math::get_z(tri.p2)) / 3.0f;
        return (axis == 0) ? cx : (axis == 1) ? cy : cz;
    }

    // Ray‑AABB intersection test (slabs)
    static bool ray_aabb_intersect(const geometry::Ray& ray, const geometry::AABB& box,
                                   float inv_dir_x, float inv_dir_y, float inv_dir_z,
                                   float t_max) noexcept {
        float t0 = (vector_math::get_x(box.min) - vector_math::get_x(ray.origin)) * inv_dir_x;
        float t1 = (vector_math::get_x(box.max) - vector_math::get_x(ray.origin)) * inv_dir_x;
        float tx_min = std::min(t0, t1);
        float tx_max = std::max(t0, t1);

        t0 = (vector_math::get_y(box.min) - vector_math::get_y(ray.origin)) * inv_dir_y;
        t1 = (vector_math::get_y(box.max) - vector_math::get_y(ray.origin)) * inv_dir_y;
        float ty_min = std::min(t0, t1);
        float ty_max = std::max(t0, t1);

        t0 = (vector_math::get_z(box.min) - vector_math::get_z(ray.origin)) * inv_dir_z;
        t1 = (vector_math::get_z(box.max) - vector_math::get_z(ray.origin)) * inv_dir_z;
        float tz_min = std::min(t0, t1);
        float tz_max = std::max(t0, t1);

        float t_enter = std::max({tx_min, ty_min, tz_min, ray.t_min});
        float t_exit  = std::min({tx_max, ty_max, tz_max, t_max});
        return t_enter <= t_exit;
    }

    // Ray‑triangle intersection (Möller‑Trumbore) – returns barycentric (u,v) and t
    static bool ray_triangle_intersect(const geometry::Ray& ray,
                                       DirectX::FXMVECTOR v0, DirectX::FXMVECTOR v1, DirectX::FXMVECTOR v2,
                                       float& t, float& u, float& v) noexcept {
        DirectX::XMVECTOR e1 = DirectX::XMVectorSubtract(v1, v0);
        DirectX::XMVECTOR e2 = DirectX::XMVectorSubtract(v2, v0);
        DirectX::XMVECTOR h = vector_math::cross3(ray.direction, e2);
        float a = vector_math::dot3_scalar(e1, h);
        if (std::fabs(a) < 1e-12f) return false;
        float f = 1.0f / a;
        DirectX::XMVECTOR s = DirectX::XMVectorSubtract(ray.origin, v0);
        u = f * vector_math::dot3_scalar(s, h);
        if (u < 0.0f || u > 1.0f) return false;
        DirectX::XMVECTOR q = vector_math::cross3(s, e1);
        v = f * vector_math::dot3_scalar(ray.direction, q);
        if (v < 0.0f || u + v > 1.0f) return false;
        t = f * vector_math::dot3_scalar(e2, q);
        return t >= ray.t_min;
    }
};

} // namespace mesh_raycasting
} // namespace SimulationMath

#endif // CORE_MATH_MESH_RAYCASTING_H