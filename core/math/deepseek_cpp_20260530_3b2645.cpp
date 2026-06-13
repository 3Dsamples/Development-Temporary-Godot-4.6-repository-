// File 0025 : core/math/bvh.h
// Bounding Volume Hierarchy (BVH) for ray casting, overlap queries, and frustum culling of AABB and triangle primitives.

#pragma once

#include "aabb.h"
#include "ray.h"
#include "triangle.h"
#include "sphere.h"
#include "frustum.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stack>
#include <functional>

namespace wp {

template <typename T>
class BVH {
public:
    struct alignas(32) Node {
        aabb<T>  box;               // bounding volume (AABB)
        union {
            struct {
                int32   left;       // index of left child (negative for internal? will use positive)
                int32   right;
            } inner;
            struct {
                int32   first_prim; // first primitive index in the leaf
                int32   prim_count; // number of primitives
            } leaf;
        };
        uint32  flags;              // bit0: 1 = leaf, others reserved

        bool is_leaf() const noexcept { return (flags & 1) != 0; }
    };

    // Primitives: triangles (world‑space) with AABB cache
    struct TrianglePrim {
        triangle<T> tri;
        aabb<T>     box;
    };

    std::vector<Node>         nodes;
    std::vector<TrianglePrim> primitives;     // store triangles (or can store AABBs for generic)
    int32                     root_idx = -1;

    BVH() = default;

    // ── Build from triangle list ──────────────────────────────
    void build(const std::vector<triangle<T>>& tris) {
        primitives.clear();
        primitives.reserve(tris.size());
        for (const auto& tri : tris) {
            TrianglePrim tp;
            tp.tri = tri;
            aabb<T> box;
            box.expand(tri.v0);
            box.expand(tri.v1);
            box.expand(tri.v2);
            tp.box = box;
            primitives.push_back(tp);
        }
        build_hierarchy();
    }

    // Build from generic AABB list (primitive = AABB itself)
    void build_from_aabbs(const std::vector<aabb<T>>& boxes) {
        primitives.clear();
        for (const auto& b : boxes) {
            TrianglePrim tp;
            tp.tri = triangle<T>(); // unused
            tp.box = b;
            primitives.push_back(tp);
        }
        build_hierarchy();
    }

    // ── Intersection queries ────────────────────────────────

    // Intersect ray with all triangles, return nearest hit
    bool intersect_ray(const ray<T>& r, T& t_min, triangle<T>& hit_tri, int32& hit_idx) const {
        if (root_idx < 0 || root_idx >= static_cast<int32>(nodes.size())) return false;
        bool hit = false;
        t_min = std::numeric_limits<T>::max();
        std::stack<int32, std::vector<int32>> stk;
        stk.push(root_idx);
        while (!stk.empty()) {
            int32 idx = stk.top(); stk.pop();
            const Node& node = nodes[idx];
            T tnear, tfar;
            if (!intersect_ray_aabb(r, node.box, tnear, tfar)) continue;
            if (tnear >= t_min) continue;   // farther than current best
            if (node.is_leaf()) {
                // test all primitives in leaf
                for (int32 i = node.leaf.first_prim; i < node.leaf.first_prim + node.leaf.prim_count; ++i) {
                    T t;
                    T u, v;
                    const TrianglePrim& tp = primitives[i];
                    if (intersect_ray_triangle(r, tp.tri.v0, tp.tri.v1, tp.tri.v2, t, u, v)) {
                        if (t < t_min) {
                            t_min = t;
                            hit_tri = tp.tri;
                            hit_idx = i;
                            hit = true;
                        }
                    }
                }
            } else {
                int32 left = node.inner.left;
                int32 right = node.inner.right;
                // push farther child first (order for earlier hits)
                // compute t for children? approximate with box tmin
                T t1_min, t1_max, t2_min, t2_max;
                bool hit1 = intersect_ray_aabb(r, nodes[left].box, t1_min, t1_max);
                bool hit2 = intersect_ray_aabb(r, nodes[right].box, t2_min, t2_max);
                if (hit1 && hit2) {
                    if (t1_min < t2_min) {
                        stk.push(right);
                        stk.push(left);
                    } else {
                        stk.push(left);
                        stk.push(right);
                    }
                } else if (hit1) {
                    stk.push(left);
                } else if (hit2) {
                    stk.push(right);
                }
            }
        }
        return hit;
    }

    // Intersect ray with any AABB primitive (used for AABB-only BVH)
    bool intersect_ray_primitive(const ray<T>& r, T& t_min, int32& hit_idx) const {
        if (root_idx < 0) return false;
        bool hit = false;
        t_min = std::numeric_limits<T>::max();
        std::stack<int32, std::vector<int32>> stk;
        stk.push(root_idx);
        while (!stk.empty()) {
            int32 idx = stk.top(); stk.pop();
            const Node& node = nodes[idx];
            T tnear, tfar;
            if (!intersect_ray_aabb(r, node.box, tnear, tfar)) continue;
            if (tnear >= t_min) continue;
            if (node.is_leaf()) {
                for (int32 i = node.leaf.first_prim; i < node.leaf.first_prim + node.leaf.prim_count; ++i) {
                    T t;
                    if (intersect_ray_aabb(r, primitives[i].box, t)) {
                        if (t < t_min) {
                            t_min = t;
                            hit_idx = i;
                            hit = true;
                        }
                    }
                }
            } else {
                int32 left = node.inner.left, right = node.inner.right;
                T t1, t2;
                bool h1 = intersect_ray_aabb(r, nodes[left].box, t1);
                bool h2 = intersect_ray_aabb(r, nodes[right].box, t2);
                if (h1 && h2) {
                    if (t1 < t2) { stk.push(right); stk.push(left); }
                    else { stk.push(left); stk.push(right); }
                } else if (h1) stk.push(left);
                else if (h2) stk.push(right);
            }
        }
        return hit;
    }

    // Overlap query: find all primitives whose AABB intersects a given sphere (returns indices)
    void overlap_sphere(const sphere<T>& s, std::vector<int32>& out_indices) const {
        if (root_idx < 0) return;
        std::stack<int32> stk;
        stk.push(root_idx);
        while (!stk.empty()) {
            int32 idx = stk.top(); stk.pop();
            const Node& node = nodes[idx];
            if (!intersect(node.box, s)) continue;
            if (node.is_leaf()) {
                for (int32 i = node.leaf.first_prim; i < node.leaf.first_prim + node.leaf.prim_count; ++i) {
                    if (intersect(primitives[i].box, s))
                        out_indices.push_back(i);
                }
            } else {
                stk.push(node.inner.right);
                stk.push(node.inner.left);
            }
        }
    }

    // Frustum culling: collect primitives completely or partially inside frustum
    void frustum_cull(const frustum<T>& fr, std::vector<int32>& visible) const {
        if (root_idx < 0) return;
        std::stack<int32> stk;
        stk.push(root_idx);
        while (!stk.empty()) {
            int32 idx = stk.top(); stk.pop();
            const Node& node = nodes[idx];
            if (!fr.intersects_aabb(node.box)) continue;
            if (node.is_leaf()) {
                for (int32 i = node.leaf.first_prim; i < node.leaf.first_prim + node.leaf.prim_count; ++i) {
                    if (fr.intersects_aabb(primitives[i].box))
                        visible.push_back(i);
                }
            } else {
                stk.push(node.inner.right);
                stk.push(node.inner.left);
            }
        }
    }

private:
    // ── Build median‑split BVH recursively ───────────────────
    struct BuildTask {
        int32 node_idx;
        int32 start;
        int32 end;
    };

    void build_hierarchy() {
        if (primitives.empty()) { root_idx = -1; return; }
        nodes.clear();
        // temporary array of primitive indices (we'll reorder later)
        std::vector<TrianglePrim> tmp_prims = primitives;
        nodes.reserve(tmp_prims.size() * 2);
        root_idx = 0;
        nodes.emplace_back(); // root placeholder
        // iterative build to avoid stack overflow
        std::vector<BuildTask> tasks;
        tasks.push_back({0, 0, static_cast<int32>(tmp_prims.size())});

        while (!tasks.empty()) {
            auto [node_idx, start, end] = tasks.back(); tasks.pop_back();
            if (end - start <= 4) {
                // create leaf
                Node& node = nodes[node_idx];
                node.box = aabb<T>();
                for (int32 i = start; i < end; ++i)
                    node.box.expand(tmp_prims[i].box);
                node.flags = 1; // leaf
                node.leaf.first_prim = start;
                node.leaf.prim_count = end - start;
                continue;
            }

            // compute centroid bounds
            aabb<T> centroid_box;
            for (int32 i = start; i < end; ++i)
                centroid_box.expand(tmp_prims[i].box.center());
            int axis = 0;
            vec3<T> ext = centroid_box.size();
            if (ext.y > ext.x && ext.y > ext.z) axis = 1;
            else if (ext.z > ext.x) axis = 2;

            // split along axis using median of centroids
            int32 mid = (start + end) / 2;
            std::nth_element(tmp_prims.begin() + start, tmp_prims.begin() + mid, tmp_prims.begin() + end,
                [axis](const TrianglePrim& a, const TrianglePrim& b) {
                    return a.box.center()[axis] < b.box.center()[axis];
                });

            // create left and right children
            int32 left_idx = static_cast<int32>(nodes.size());
            nodes.emplace_back();
            int32 right_idx = static_cast<int32>(nodes.size());
            nodes.emplace_back();
            Node& node = nodes[node_idx];
            node.flags = 0; // internal
            node.inner.left = left_idx;
            node.inner.right = right_idx;
            // compute AABB for current node as union of children (will be done after children are built)
            // push tasks for children in correct order (will compute AABB later)
            tasks.push_back({right_idx, mid, end});
            tasks.push_back({left_idx, start, mid});
        }

        // After build, compute AABBs for internal nodes (post‑order)
        // We can do a simple bottom‑up pass using a stack
        std::vector<int32> postorder;
        std::stack<int32> stk;
        stk.push(root_idx);
        while (!stk.empty()) {
            int32 idx = stk.top(); stk.pop();
            postorder.push_back(idx);
            const Node& node = nodes[idx];
            if (!node.is_leaf()) {
                stk.push(node.inner.right);
                stk.push(node.inner.left);
            }
        }
        // Process in reverse
        for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
            Node& node = nodes[*it];
            if (!node.is_leaf()) {
                node.box = nodes[node.inner.left].box;
                node.box.expand(nodes[node.inner.right].box);
            }
        }

        // Finally, reorder primitives according to leaves (they are already in‑place because we didn't move actual primitives, but we used tmp_prims)
        primitives = std::move(tmp_prims);
    }
};

using BVHf = BVH<float>;
using BVHd = BVH<double>;

} // namespace wp