//File 0033 : core/xgeometry.hpp
//Spatial search structures (KD-tree, AABB-tree), intersection tests, distance queries for 2D/3D real-time simulation and nearest neighbor search.
#ifndef XTENSOR_XGEOMETRY_HPP
#define XTENSOR_XGEOMETRY_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"
#include "xexpression.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xarray.hpp"
#include "xeval.hpp"
#include "xreducer.hpp"
#include "xsort.hpp"
#include "xlinalg.hpp"
#include "xstatistics.hpp"
#include "xrandom.hpp"

namespace xt {
namespace geometry {

    /*********************************************
     * Basic vector types for 2D/3D
     *********************************************/
    template <class T, std::size_t D>
    using vec = std::array<T, D>;

    template <class T>
    using vec2 = vec<T, 2>;
    template <class T>
    using vec3 = vec<T, 3>;

    /*********************************************
     * Geometric primitives
     *********************************************/
    template <class T>
    struct AABB {
        vec3<T> min;
        vec3<T> max;

        AABB() : min{std::numeric_limits<T>::max(), std::numeric_limits<T>::max(), std::numeric_limits<T>::max()},
                 max{-std::numeric_limits<T>::max(), -std::numeric_limits<T>::max(), -std::numeric_limits<T>::max()} {}
        AABB(const vec3<T>& a, const vec3<T>& b) : min{a}, max{b} {}

        vec3<T> center() const { return {(min[0]+max[0])/2, (min[1]+max[1])/2, (min[2]+max[2])/2}; }
        vec3<T> extent() const { return {(max[0]-min[0])/2, (max[1]-min[1])/2, (max[2]-min[2])/2}; }

        bool contains(const vec3<T>& p) const {
            return p[0]>=min[0] && p[0]<=max[0] && p[1]>=min[1] && p[1]<=max[1] && p[2]>=min[2] && p[2]<=max[2];
        }
        bool intersects(const AABB<T>& other) const {
            return (min[0] <= other.max[0] && max[0] >= other.min[0]) &&
                   (min[1] <= other.max[1] && max[1] >= other.min[1]) &&
                   (min[2] <= other.max[2] && max[2] >= other.min[2]);
        }
        AABB<T> Union(const AABB<T>& other) const {
            return { {std::min(min[0],other.min[0]), std::min(min[1],other.min[1]), std::min(min[2],other.min[2])},
                     {std::max(max[0],other.max[0]), std::max(max[1],other.max[1]), std::max(max[2],other.max[2])} };
        }
        T surfaceArea() const {
            vec3<T> d{max[0]-min[0], max[1]-min[1], max[2]-min[2]};
            return 2*(d[0]*d[1] + d[1]*d[2] + d[2]*d[0]);
        }
    };

    // Ray: origin + t * direction
    template <class T>
    struct Ray {
        vec3<T> origin;
        vec3<T> direction;
        T t_min;
        T t_max;
        Ray(const vec3<T>& o, const vec3<T>& d, T t_min=T(1e-4), T t_max=std::numeric_limits<T>::max())
            : origin(o), direction(d), t_min(t_min), t_max(t_max) {}
    };

    // Triangle mesh element
    template <class T>
    struct Triangle {
        vec3<T> v0, v1, v2;
        Triangle(const vec3<T>& a, const vec3<T>& b, const vec3<T>& c) : v0(a), v1(b), v2(c) {}
        vec3<T> centroid() const { return {(v0[0]+v1[0]+v2[0])/3, (v0[1]+v1[1]+v2[1])/3, (v0[2]+v1[2]+v2[2])/3}; }
        AABB<T> boundingBox() const {
            return { {std::min({v0[0],v1[0],v2[0]}), std::min({v0[1],v1[1],v2[1]}), std::min({v0[2],v1[2],v2[2]})},
                     {std::max({v0[0],v1[0],v2[0]}), std::max({v0[1],v1[1],v2[1]}), std::max({v0[2],v1[2],v2[2]})} };
        }
    };

    // Ray-Triangle intersection (Möller–Trumbore)
    template <class T>
    bool ray_triangle_intersect(const Ray<T>& ray, const Triangle<T>& tri, T& t, T& u, T& v) {
        vec3<T> e1 = {tri.v1[0]-tri.v0[0], tri.v1[1]-tri.v0[1], tri.v1[2]-tri.v0[2]};
        vec3<T> e2 = {tri.v2[0]-tri.v0[0], tri.v2[1]-tri.v0[1], tri.v2[2]-tri.v0[2]};
        vec3<T> h = cross(ray.direction, e2);
        T a = dot(e1, h);
        if (std::abs(a) < 1e-12) return false;
        T f = 1.0f / a;
        vec3<T> s = {ray.origin[0]-tri.v0[0], ray.origin[1]-tri.v0[1], ray.origin[2]-tri.v0[2]};
        u = f * dot(s, h);
        if (u < 0.0 || u > 1.0) return false;
        vec3<T> q = cross(s, e1);
        v = f * dot(ray.direction, q);
        if (v < 0.0 || u + v > 1.0) return false;
        t = f * dot(e2, q);
        return t >= ray.t_min && t <= ray.t_max;
    }

    // Ray-AABB intersection (slab method)
    template <class T>
    bool ray_aabb_intersect(const Ray<T>& ray, const AABB<T>& box, T& t_min, T& t_max) {
        t_min = ray.t_min; t_max = ray.t_max;
        for (int i = 0; i < 3; ++i) {
            T invD = 1.0f / ray.direction[i];
            T t0 = (box.min[i] - ray.origin[i]) * invD;
            T t1 = (box.max[i] - ray.origin[i]) * invD;
            if (invD < 0.0f) std::swap(t0, t1);
            t_min = std::max(t_min, t0);
            t_max = std::min(t_max, t1);
            if (t_max < t_min) return false;
        }
        return true;
    }

    // Dot and cross product utility for vec3
    template <class T>
    T dot(const vec3<T>& a, const vec3<T>& b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
    template <class T>
    vec3<T> cross(const vec3<T>& a, const vec3<T>& b) {
        return {a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]};
    }
    template <class T>
    T length(const vec3<T>& v) { return std::sqrt(dot(v,v)); }
    template <class T>
    vec3<T> normalize(const vec3<T>& v) { T len = length(v); return {v[0]/len, v[1]/len, v[2]/len}; }

    /*********************************************
     * KD-Tree for nearest neighbor search
     *********************************************/
    template <class T, std::size_t D = 3>
    class KDTree {
    public:
        struct Node {
            vec<T, D> point;
            std::size_t index;
            std::unique_ptr<Node> left;
            std::unique_ptr<Node> right;
            Node(const vec<T, D>& p, std::size_t idx) : point(p), index(idx) {}
        };

        KDTree() : root_(nullptr) {}

        void build(const std::vector<vec<T, D>>& points) {
            indices_.resize(points.size());
            std::iota(indices_.begin(), indices_.end(), 0);
            root_ = buildRecursive(points, indices_.begin(), indices_.end(), 0);
        }

        std::size_t nearestNeighbor(const vec<T, D>& query) const {
            std::size_t best_idx = 0;
            T best_dist = std::numeric_limits<T>::max();
            searchNearest(root_.get(), query, 0, best_idx, best_dist);
            return best_idx;
        }

        std::vector<std::size_t> knearest(const vec<T, D>& query, std::size_t k) const {
            using Pair = std::pair<T, std::size_t>;
            auto cmp = [](const Pair& a, const Pair& b) { return a.first < b.first; };
            std::priority_queue<Pair, std::vector<Pair>, decltype(cmp)> pq(cmp);
            searchKNearest(root_.get(), query, 0, k, pq);
            std::vector<std::size_t> result;
            while (!pq.empty()) { result.push_back(pq.top().second); pq.pop(); }
            std::reverse(result.begin(), result.end());
            return result;
        }

    private:
        std::unique_ptr<Node> root_;
        std::vector<std::size_t> indices_;

        template <class Iter>
        std::unique_ptr<Node> buildRecursive(const std::vector<vec<T, D>>& pts, Iter begin, Iter end, std::size_t depth) {
            if (begin == end) return nullptr;
            std::size_t axis = depth % D;
            auto mid = begin + (end - begin) / 2;
            std::nth_element(begin, mid, end, [axis](const std::size_t& a, const std::size_t& b) {
                return pts[a][axis] < pts[b][axis];
            });
            auto node = std::make_unique<Node>(pts[*mid], *mid);
            node->left = buildRecursive(pts, begin, mid, depth+1);
            node->right = buildRecursive(pts, mid+1, end, depth+1);
            return node;
        }

        void searchNearest(const Node* node, const vec<T, D>& query, std::size_t depth,
                          std::size_t& best_idx, T& best_dist) const {
            if (!node) return;
            T dist = distance(query, node->point);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = node->index;
            }
            std::size_t axis = depth % D;
            T diff = query[axis] - node->point[axis];
            const Node* first = diff < 0 ? node->left.get() : node->right.get();
            const Node* second = diff < 0 ? node->right.get() : node->left.get();
            searchNearest(first, query, depth+1, best_idx, best_dist);
            if (diff*diff < best_dist)
                searchNearest(second, query, depth+1, best_idx, best_dist);
        }

        void searchKNearest(const Node* node, const vec<T, D>& query, std::size_t depth, std::size_t k,
                           std::priority_queue<std::pair<T,std::size_t>, std::vector<std::pair<T,std::size_t>>,
                               std::less<std::pair<T,std::size_t>>>& pq) const {
            if (!node) return;
            T dist = distance(query, node->point);
            if (pq.size() < k) {
                pq.emplace(dist, node->index);
            } else if (dist < pq.top().first) {
                pq.pop();
                pq.emplace(dist, node->index);
            }
            std::size_t axis = depth % D;
            T diff = query[axis] - node->point[axis];
            const Node* first = diff < 0 ? node->left.get() : node->right.get();
            const Node* second = diff < 0 ? node->right.get() : node->left.get();
            searchKNearest(first, query, depth+1, k, pq);
            if (diff*diff < pq.top().first || pq.size() < k)
                searchKNearest(second, query, depth+1, k, pq);
        }

        static T distance(const vec<T, D>& a, const vec<T, D>& b) {
            T sum = 0;
            for (std::size_t i = 0; i < D; ++i) sum += (a[i]-b[i])*(a[i]-b[i]);
            return std::sqrt(sum);
        }
    };

    /*********************************************
     * BVH (Bounding Volume Hierarchy) for ray tracing
     *********************************************/
    template <class T>
    class BVH {
    public:
        struct Node {
            AABB<T> box;
            std::unique_ptr<Node> left;
            std::unique_ptr<Node> right;
            std::vector<std::size_t> triangle_indices; // leaf node data
            Node() = default;
        };

        BVH() : root_(nullptr) {}

        void build(const std::vector<Triangle<T>>& triangles) {
            triangles_ = &triangles;
            std::vector<std::size_t> indices(triangles.size());
            std::iota(indices.begin(), indices.end(), 0);
            root_ = buildRecursive(indices, 0);
        }

        bool intersect(const Ray<T>& ray, T& t, std::size_t& triangle_idx) const {
            t = std::numeric_limits<T>::max();
            return intersectNode(root_.get(), ray, t, triangle_idx);
        }

    private:
        const std::vector<Triangle<T>>* triangles_;
        std::unique_ptr<Node> root_;

        std::unique_ptr<Node> buildRecursive(std::vector<std::size_t>& indices, std::size_t depth) {
            if (indices.empty()) return nullptr;
            auto node = std::make_unique<Node>();
            // Compute bounding box of all triangles in this node
            AABB<T> box;
            for (auto idx : indices) {
                box = box.Union((*triangles_)[idx].boundingBox());
            }
            node->box = box;
            if (indices.size() <= 4) { // leaf
                node->triangle_indices = std::move(indices);
                return node;
            }
            // Split along longest axis
            vec3<T> extent = box.extent();
            int axis = (extent[0] > extent[1] && extent[0] > extent[2]) ? 0 :
                       (extent[1] > extent[2] ? 1 : 2);
            T mid = box.center()[axis];
            std::vector<std::size_t> left_idx, right_idx;
            for (auto idx : indices) {
                if ((*triangles_)[idx].centroid()[axis] < mid)
                    left_idx.push_back(idx);
                else
                    right_idx.push_back(idx);
            }
            if (left_idx.empty() || right_idx.empty()) {
                // fallback: split half-half
                std::size_t half = indices.size()/2;
                left_idx.assign(indices.begin(), indices.begin()+half);
                right_idx.assign(indices.begin()+half, indices.end());
            }
            indices.clear();
            node->left = buildRecursive(left_idx, depth+1);
            node->right = buildRecursive(right_idx, depth+1);
            return node;
        }

        bool intersectNode(const Node* node, const Ray<T>& ray, T& t, std::size_t& tri_idx) const {
            if (!node) return false;
            T t_min, t_max;
            if (!ray_aabb_intersect(ray, node->box, t_min, t_max)) return false;
            bool hit = false;
            if (!node->triangle_indices.empty()) {
                for (auto idx : node->triangle_indices) {
                    T tu, tv, tt;
                    if (ray_triangle_intersect(ray, (*triangles_)[idx], tt, tu, tv)) {
                        if (tt < t) { t = tt; tri_idx = idx; hit = true; }
                    }
                }
            } else {
                hit |= intersectNode(node->left.get(), ray, t, tri_idx);
                hit |= intersectNode(node->right.get(), ray, t, tri_idx);
            }
            return hit;
        }
    };

    /*********************************************
     * Distance between geometric entities
     *********************************************/
    template <class T>
    T point_to_point_distance(const vec3<T>& a, const vec3<T>& b) {
        return length(vec3<T>{a[0]-b[0], a[1]-b[1], a[2]-b[2]});
    }

    template <class T>
    T point_to_line_distance(const vec3<T>& point, const vec3<T>& line_origin, const vec3<T>& line_dir) {
        vec3<T> v = {point[0]-line_origin[0], point[1]-line_origin[1], point[2]-line_origin[2]};
        T proj = dot(v, line_dir);
        vec3<T> closest = {line_origin[0]+proj*line_dir[0], line_origin[1]+proj*line_dir[1], line_origin[2]+proj*line_dir[2]};
        return point_to_point_distance(point, closest);
    }

    template <class T>
    T point_to_plane_distance(const vec3<T>& point, const vec3<T>& plane_normal, T plane_d) {
        return std::abs(dot(point, plane_normal) + plane_d) / length(plane_normal);
    }

} // namespace geometry
} // namespace xt

#endif // XTENSOR_XGEOMETRY_HPP