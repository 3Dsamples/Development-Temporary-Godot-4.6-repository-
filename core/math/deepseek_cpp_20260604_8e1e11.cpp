// system name : onetbb-warp
// File 0048 : core/math/bvh.h
// Description : Bounding volume hierarchy construction (SAH, binning), ray traversal, nearest queries.

#ifndef __TBB_WARP_CORE_MATH_BVH_H
#define __TBB_WARP_CORE_MATH_BVH_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector3.h"
#include "core/math/geometry.h"
#include <vector>
#include <array>
#include <algorithm>
#include <limits>
#include <functional>
#include <memory>
#include <stack>
#include <cmath>
#include <cstdint>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// BVH node
// ============================================================

template<typename T>
struct bvh_node {
    aabb<T> bounds;
    std::uint32_t left_child;   // index in node array, or first primitive index if leaf
    std::uint32_t right_or_count; // right child index, or primitive count if leaf
    bool is_leaf() const noexcept { return right_or_count & 0x80000000; }
    std::uint32_t primitive_count() const noexcept { return right_or_count & 0x7FFFFFFF; }
};

// ============================================================
// BVH build data
// ============================================================

template<typename T>
struct bvh_primitive {
    aabb<T> bounds;
    vector3<T> centroid;
    std::uint32_t index;
};

// ============================================================
// SAH (Surface Area Heuristic) binning
// ============================================================

template<typename T>
struct sah_bin {
    aabb<T> bounds;
    std::uint32_t count = 0;
};

template<typename T>
void find_best_split(const std::vector<bvh_primitive<T>>& primitives,
                     std::uint32_t start, std::uint32_t end,
                     const aabb<T>& node_bounds,
                     std::uint32_t& split_axis, std::uint32_t& split_pos) {
    constexpr std::uint32_t num_bins = 16;
    T best_cost = std::numeric_limits<T>::max();
    split_axis = 0;
    split_pos = (start + end) / 2;

    T node_area = node_bounds.surface_area();
    if (node_area < T(1e-12)) node_area = T(1);

    for (std::uint32_t axis = 0; axis < 3; ++axis) {
        // Compute min/max of centroids along axis
        T min_c = std::numeric_limits<T>::max();
        T max_c = -std::numeric_limits<T>::max();
        for (std::uint32_t i = start; i < end; ++i) {
            min_c = std::min(min_c, primitives[i].centroid[axis]);
            max_c = std::max(max_c, primitives[i].centroid[axis]);
        }
        if (max_c - min_c < T(1e-12)) continue;

        // Fill bins
        std::array<sah_bin<T>, num_bins> bins;
        T inv_range = T(num_bins) / (max_c - min_c);
        for (std::uint32_t i = start; i < end; ++i) {
            std::uint32_t b = static_cast<std::uint32_t>((primitives[i].centroid[axis] - min_c) * inv_range);
            if (b >= num_bins) b = num_bins - 1;
            bins[b].bounds.expand(primitives[i].bounds);
            bins[b].count++;
        }

        // Sweep from left
        std::array<sah_bin<T>, num_bins - 1> left_sweep;
        sah_bin<T> left;
        for (std::uint32_t i = 0; i < num_bins - 1; ++i) {
            left.bounds.expand(bins[i].bounds);
            left.count += bins[i].count;
            left_sweep[i] = left;
        }

        // Sweep from right and evaluate cost
        sah_bin<T> right;
        for (std::uint32_t i = num_bins - 1; i > 0; --i) {
            right.bounds.expand(bins[i].bounds);
            right.count += bins[i].count;
            if (left_sweep[i - 1].count == 0 || right.count == 0) continue;
            T left_area = left_sweep[i - 1].bounds.surface_area();
            T right_area = right.bounds.surface_area();
            T cost = T(1) + (left_area * left_sweep[i - 1].count + right_area * right.count) / node_area;
            if (cost < best_cost) {
                best_cost = cost;
                split_axis = axis;
                // split position: primitives with centroid <= bin boundary
                T split_val = min_c + (static_cast<T>(i) / num_bins) * (max_c - min_c);
                // Find split index in actual array by partitioning
                auto it = std::partition(primitives.begin() + start, primitives.begin() + end,
                    [axis, split_val](const bvh_primitive<T>& p) {
                        return p.centroid[axis] <= split_val;
                    });
                split_pos = static_cast<std::uint32_t>(it - primitives.begin());
            }
        }
    }
}

// ============================================================
// Recursive BVH build with SAH
// ============================================================

template<typename T>
std::uint32_t build_bvh_recursive(std::vector<bvh_node<T>>& nodes,
                                   std::vector<bvh_primitive<T>>& primitives,
                                   std::uint32_t start, std::uint32_t end) {
    std::uint32_t node_idx = static_cast<std::uint32_t>(nodes.size());
    nodes.emplace_back();
    bvh_node<T>& node = nodes.back();

    // Compute bounds of this node
    aabb<T> node_bounds;
    for (std::uint32_t i = start; i < end; ++i)
        node_bounds.expand(primitives[i].bounds);
    node.bounds = node_bounds;

    std::uint32_t count = end - start;
    if (count <= 4) {
        // Leaf
        node.left_child = start;
        node.right_or_count = count | 0x80000000;
        return node_idx;
    }

    // Find best split
    std::uint32_t axis, split;
    find_best_split(primitives, start, end, node_bounds, axis, split);
    if (split == start || split == end) {
        // Unable to split, make leaf
        node.left_child = start;
        node.right_or_count = count | 0x80000000;
        return node_idx;
    }

    std::uint32_t left_child = build_bvh_recursive(nodes, primitives, start, split);
    std::uint32_t right_child = build_bvh_recursive(nodes, primitives, split, end);
    node.left_child = left_child;
    node.right_or_count = right_child;
    return node_idx;
}

template<typename T>
std::vector<bvh_node<T>> build_bvh(std::vector<aabb<T>>& primitive_bounds) {
    std::vector<bvh_primitive<T>> primitives(primitive_bounds.size());
    for (std::size_t i = 0; i < primitive_bounds.size(); ++i) {
        primitives[i].bounds = primitive_bounds[i];
        primitives[i].centroid = primitive_bounds[i].center();
        primitives[i].index = static_cast<std::uint32_t>(i);
    }
    std::vector<bvh_node<T>> nodes;
    if (!primitives.empty())
        build_bvh_recursive(nodes, primitives, 0, static_cast<std::uint32_t>(primitives.size()));
    return nodes;
}

// ============================================================
// Ray‑BVH intersection using slab method
// ============================================================

template<typename T>
bool ray_aabb_intersect(const ray<T>& r, const aabb<T>& box, T& t_min, T& t_max) noexcept {
    vector3<T> inv_dir(T(1)/r.direction.x, T(1)/r.direction.y, T(1)/r.direction.z);
    T t1 = (box.min.x - r.origin.x) * inv_dir.x;
    T t2 = (box.max.x - r.origin.x) * inv_dir.x;
    t_min = std::min(t1, t2);
    t_max = std::max(t1, t2);
    t1 = (box.min.y - r.origin.y) * inv_dir.y;
    t2 = (box.max.y - r.origin.y) * inv_dir.y;
    t_min = std::max(t_min, std::min(t1, t2));
    t_max = std::min(t_max, std::max(t1, t2));
    t1 = (box.min.z - r.origin.z) * inv_dir.z;
    t2 = (box.max.z - r.origin.z) * inv_dir.z;
    t_min = std::max(t_min, std::min(t1, t2));
    t_max = std::min(t_max, std::max(t1, t2));
    return t_max >= std::max(T(0), t_min);
}

// ============================================================
// BVH ray traversal (stack‑based)
// ============================================================

template<typename T, typename HitCallback>
void traverse_bvh_ray(const std::vector<bvh_node<T>>& nodes,
                      const std::vector<aabb<T>>& primitive_bounds,
                      const ray<T>& r, HitCallback&& hit_callback) {
    if (nodes.empty()) return;
    struct stack_entry { std::uint32_t node_idx; T t_min; };
    std::stack<stack_entry> st;
    T root_t_min, root_t_max;
    if (!ray_aabb_intersect(r, nodes[0].bounds, root_t_min, root_t_max)) return;
    st.push({0, root_t_min});

    while (!st.empty()) {
        auto [node_idx, t_min] = st.top(); st.pop();
        if (t_min > r.max_distance) continue;
        const auto& node = nodes[node_idx];
        if (node.is_leaf()) {
            std::uint32_t count = node.primitive_count();
            for (std::uint32_t i = 0; i < count; ++i) {
                std::uint32_t prim_idx = node.left_child + i;
                if (prim_idx >= primitive_bounds.size()) continue;
                const aabb<T>& box = primitive_bounds[prim_idx];
                T prim_t_min, prim_t_max;
                if (ray_aabb_intersect(r, box, prim_t_min, prim_t_max) && prim_t_min <= r.max_distance) {
                    hit_callback(prim_idx, prim_t_min);
                }
            }
        } else {
            std::uint32_t left = node.left_child;
            std::uint32_t right = node.right_or_count;
            T lt_min, lt_max, rt_min, rt_max;
            bool hit_left = ray_aabb_intersect(r, nodes[left].bounds, lt_min, lt_max);
            bool hit_right = ray_aabb_intersect(r, nodes[right].bounds, rt_min, rt_max);
            if (hit_left && hit_right) {
                if (lt_min < rt_min) {
                    st.push({right, rt_min});
                    st.push({left, lt_min});
                } else {
                    st.push({left, lt_min});
                    st.push({right, rt_min});
                }
            } else if (hit_left) {
                st.push({left, lt_min});
            } else if (hit_right) {
                st.push({right, rt_min});
            }
        }
    }
}

// ============================================================
// Nearest neighbour search in BVH using priority queue
// ============================================================

template<typename T>
std::uint32_t nearest_bvh(const std::vector<bvh_node<T>>& nodes,
                          const std::vector<aabb<T>>& primitive_bounds,
                          const vector3<T>& query, T& out_distance) {
    if (nodes.empty()) return std::uint32_t(-1);
    struct nn_entry { std::uint32_t node_idx; T dist; };
    auto cmp = [](const nn_entry& a, const nn_entry& b) { return a.dist > b.dist; };
    std::priority_queue<nn_entry, std::vector<nn_entry>, decltype(cmp)> pq(cmp);
    T init_dist = aabb_distance_sq(nodes[0].bounds, query);
    pq.push({0, init_dist});

    T best_dist = std::numeric_limits<T>::max();
    std::uint32_t best_prim = std::uint32_t(-1);

    while (!pq.empty()) {
        auto [node_idx, dist] = pq.top(); pq.pop();
        if (dist > best_dist) continue;
        const auto& node = nodes[node_idx];
        if (node.is_leaf()) {
            std::uint32_t count = node.primitive_count();
            for (std::uint32_t i = 0; i < count; ++i) {
                std::uint32_t prim = node.left_child + i;
                T d = aabb_distance_sq(primitive_bounds[prim], query);
                if (d < best_dist) { best_dist = d; best_prim = prim; }
            }
        } else {
            std::uint32_t left = node.left_child, right = node.right_or_count;
            T dl = aabb_distance_sq(nodes[left].bounds, query);
            T dr = aabb_distance_sq(nodes[right].bounds, query);
            if (dl < best_dist) pq.push({left, dl});
            if (dr < best_dist) pq.push({right, dr});
            if (dl < dr) {
                if (dr < best_dist) pq.push({right, dr});
                if (dl < best_dist) pq.push({left, dl});
            } else {
                if (dl < best_dist) pq.push({left, dl});
                if (dr < best_dist) pq.push({right, dr});
            }
        }
    }
    out_distance = std::sqrt(best_dist);
    return best_prim;
}

// ============================================================
// Distance squared from a point to AABB
// ============================================================

template<typename T>
T aabb_distance_sq(const aabb<T>& box, const vector3<T>& point) noexcept {
    T sq = T(0);
    for (int i = 0; i < 3; ++i) {
        T v = point[i];
        if (v < box.min[i]) sq += (box.min[i] - v) * (box.min[i] - v);
        else if (v > box.max[i]) sq += (v - box.max[i]) * (v - box.max[i]);
    }
    return sq;
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_BVH_H