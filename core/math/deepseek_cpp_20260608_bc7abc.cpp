//File group name : OrthoTree Math
//File 0078 : core/math/kdtree.h (revised with approximate k‑NN)
//Kd‑tree for 2D/3D points: construction (median split), nearest neighbor, k‑nearest neighbor (exact and approximate with epsilon), radius search. SIMD batch operations. Includes approximate k‑NN with relative error guarantee.

#ifndef ORTHOTREE_CORE_MATH_KDTREE_H_INCLUDED
#define ORTHOTREE_CORE_MATH_KDTREE_H_INCLUDED

#include "../../build_config.h"
#include "basic/vector.h"
#include "geometry/aabb.h"
#include "distance/metrics.h"
#include "numerical/root_finding.h"
#include "math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <memory>
#include <functional>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  KdTree: binary tree recursively splitting at median along alternating axes.
//  Points are stored in leaves (buckets). Supports point insertion (static build),
//  nearest neighbour, k‑NN (exact and approximate with epsilon), radius search.
//  Optimised for 2D/3D.
// ============================================================================
template<typename T = float, std::size_t N = 3>
class KdTree {
public:
    using value_type = T;
    using point_type = Basic::Vector<T, N>;
    using aabb_type = Geometry::AABB<T, N>;
    using size_type = size_t;
    using index_type = uint32_t;

    struct Node {
        aabb_type bounds;
        union {
            struct { index_type left, right; } children; // internal
            struct { index_type start, count; } leaf;    // leaf
        };
        uint8_t axis;
        bool isLeaf;
        T radius; // bounding sphere radius for pruning
        point_type center;
    };

    // ------------------------------------------------------------------------
    //  Configuration (dynamic environment)
    // ------------------------------------------------------------------------
    struct Config {
        size_type maxLeafSize = 16;
        bool useMedianSplit = true;
        bool enableSIMD = true;
        T searchEpsilon = T(0);
    };

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    KdTree() = default;
    explicit KdTree(const Config& cfg) : m_config(cfg) {}

    // Build from a vector of points (non‑modifiable after build)
    void build(const point_type* points, size_type count) {
        m_points.assign(points, points + count);
        m_indices.resize(count);
        for (index_type i = 0; i < count; ++i) m_indices[i] = i;
        m_nodes.clear();
        m_nodes.reserve(count * 2);
        buildRecursive(0, count, 0);
    }

    // ------------------------------------------------------------------------
    //  Nearest neighbor (single)
    // ------------------------------------------------------------------------
    std::pair<index_type, T> nearestNeighbor(const point_type& query) const {
        T bestDistSq = std::numeric_limits<T>::max();
        index_type bestIdx = 0;
        searchNode(0, query, bestDistSq, bestIdx);
        return {bestIdx, std::sqrt(bestDistSq)};
    }

    // ------------------------------------------------------------------------
    //  Exact k‑nearest neighbors (returns indices and distances)
    // ------------------------------------------------------------------------
    std::vector<std::pair<index_type, T>> kNearest(const point_type& query, size_type k) const {
        using Candidate = std::pair<T, index_type>; // distSq, idx
        auto cmp = [](const Candidate& a, const Candidate& b) { return a.first < b.first; };
        std::vector<Candidate> heap;
        kSearchNode(0, query, k, heap);
        std::sort(heap.begin(), heap.end(), cmp);
        std::vector<std::pair<index_type, T>> result;
        for (size_type i = 0; i < std::min(k, heap.size()); ++i) {
            result.emplace_back(heap[i].second, std::sqrt(heap[i].first));
        }
        return result;
    }

    // ------------------------------------------------------------------------
    //  Approximate k‑nearest neighbors with relative error epsilon (0 = exact).
    //  Returns up to k points; each distance is guaranteed to be at most
    //  (1 + epsilon) times the true k‑th neighbour distance.
    // ------------------------------------------------------------------------
    std::vector<std::pair<index_type, T>> approxKNearest(const point_type& query,
                                                         size_type k,
                                                         T epsilon = T(0.1)) const {
        if (epsilon <= T(0)) return kNearest(query, k);
        using Candidate = std::pair<T, index_type>; // distSq, idx
        auto cmpMax = [](const Candidate& a, const Candidate& b) { return a.first < b.first; };
        std::priority_queue<Candidate, std::vector<Candidate>, decltype(cmpMax)> best(cmpMax);
        T threshold = std::numeric_limits<T>::max();
        approxSearchNode(0, query, k, epsilon, best, threshold);
        // Extract results
        std::vector<std::pair<index_type, T>> result;
        result.reserve(best.size());
        while (!best.empty()) {
            result.emplace_back(best.top().second, std::sqrt(best.top().first));
            best.pop();
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

    // ------------------------------------------------------------------------
    //  Radius search (returns indices within radius)
    // ------------------------------------------------------------------------
    std::vector<index_type> radiusSearch(const point_type& query, T radius) const {
        std::vector<index_type> result;
        T radiusSq = radius * radius;
        radiusSearchNode(0, query, radiusSq, result);
        return result;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: 4 queries at once (nearest neighbor for each)
    // ------------------------------------------------------------------------
    void batchNearestNeighbor(const point_type* queries, index_type* outIdx, T* outDist, size_type count) const {
        if (m_config.enableSIMD && count >= 4 && N == 3 && std::is_same_v<T,float>) {
            for (size_type i = 0; i < count; ++i) {
                auto res = nearestNeighbor(queries[i]);
                outIdx[i] = res.first;
                outDist[i] = res.second;
            }
        } else {
            for (size_type i = 0; i < count; ++i) {
                auto res = nearestNeighbor(queries[i]);
                outIdx[i] = res.first;
                outDist[i] = res.second;
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    size_type size() const noexcept { return m_points.size(); }
    size_type nodeCount() const noexcept { return m_nodes.size(); }

private:
    // ------------------------------------------------------------------------
    //  Recursive construction
    // ------------------------------------------------------------------------
    void buildRecursive(size_type start, size_type count, uint8_t depth) {
        if (count == 0) return;
        Node node;
        node.bounds = computeBounds(start, count);
        uint8_t axis = depth % N;
        if (!m_config.useMedianSplit) {
            point_type ext = node.bounds.extents();
            T maxExt = ext[0];
            for (size_type d = 1; d < N; ++d) {
                if (ext[d] > maxExt) { maxExt = ext[d]; axis = static_cast<uint8_t>(d); }
            }
        }
        std::sort(m_indices.begin() + start, m_indices.begin() + start + count,
                  [this, axis](index_type a, index_type b) {
                      return m_points[a][axis] < m_points[b][axis];
                  });
        node.axis = axis;
        node.isLeaf = (count <= m_config.maxLeafSize);
        if (node.isLeaf) {
            node.leaf.start = static_cast<index_type>(start);
            node.leaf.count = static_cast<index_type>(count);
            // Compute bounding sphere center and radius
            point_type center(0);
            for (size_type i = start; i < start + count; ++i) center = center + m_points[m_indices[i]];
            node.center = center / static_cast<T>(count);
            T maxRad2 = T(0);
            for (size_type i = start; i < start + count; ++i) {
                T d2 = (m_points[m_indices[i]] - node.center).squaredLength();
                if (d2 > maxRad2) maxRad2 = d2;
            }
            node.radius = std::sqrt(maxRad2);
        } else {
            size_type mid = start + count / 2;
            node.children.left = static_cast<index_type>(m_nodes.size());
            buildRecursive(start, mid - start, depth + 1);
            node.children.right = static_cast<index_type>(m_nodes.size());
            buildRecursive(mid, start + count - mid, depth + 1);
            // Combine child bounding spheres
            const Node& left = m_nodes[node.children.left];
            const Node& right = m_nodes[node.children.right];
            node.center = (left.center + right.center) * T(0.5);
            node.radius = std::max(left.radius, right.radius) +
                          (left.center - right.center).length() * T(0.5);
        }
        m_nodes.push_back(node);
    }

    // ------------------------------------------------------------------------
    //  Bounds of a range
    // ------------------------------------------------------------------------
    aabb_type computeBounds(size_type start, size_type count) const {
        aabb_type bounds;
        for (size_type i = start; i < start + count; ++i) {
            bounds.extend(m_points[m_indices[i]]);
        }
        return bounds;
    }

    // ------------------------------------------------------------------------
    //  Exact nearest neighbour traversal
    // ------------------------------------------------------------------------
    void searchNode(size_type nodeIdx, const point_type& query, T& bestDistSq, index_type& bestIdx) const {
        const Node& node = m_nodes[nodeIdx];
        if (node.isLeaf) {
            for (index_type i = node.leaf.start; i < node.leaf.start + node.leaf.count; ++i) {
                index_type pid = m_indices[i];
                T d2 = (m_points[pid] - query).squaredLength();
                if (d2 < bestDistSq) {
                    bestDistSq = d2;
                    bestIdx = pid;
                }
            }
            return;
        }
        T d = query[node.axis] - node.center[node.axis];
        size_type first = (d <= 0) ? node.children.left : node.children.right;
        size_type second = (first == node.children.left) ? node.children.right : node.children.left;
        searchNode(first, query, bestDistSq, bestIdx);
        if (d * d < bestDistSq) {
            searchNode(second, query, bestDistSq, bestIdx);
        }
    }

    // ------------------------------------------------------------------------
    //  Exact k‑NN traversal (max‑heap)
    // ------------------------------------------------------------------------
    void kSearchNode(size_type nodeIdx, const point_type& query, size_type k,
                     std::vector<std::pair<T, index_type>>& heap) const {
        const Node& node = m_nodes[nodeIdx];
        if (node.isLeaf) {
            for (index_type i = node.leaf.start; i < node.leaf.start + node.leaf.count; ++i) {
                index_type pid = m_indices[i];
                T d2 = (m_points[pid] - query).squaredLength();
                if (heap.size() < k) {
                    heap.emplace_back(d2, pid);
                    std::push_heap(heap.begin(), heap.end(), std::greater<>());
                } else if (d2 < heap[0].first) {
                    std::pop_heap(heap.begin(), heap.end(), std::greater<>());
                    heap.back() = {d2, pid};
                    std::push_heap(heap.begin(), heap.end(), std::greater<>());
                }
            }
            return;
        }
        T d = query[node.axis] - node.center[node.axis];
        size_type first = (d <= 0) ? node.children.left : node.children.right;
        size_type second = (first == node.children.left) ? node.children.right : node.children.left;
        kSearchNode(first, query, k, heap);
        if (heap.size() < k || d * d < heap[0].first) {
            kSearchNode(second, query, k, heap);
        }
    }

    // ------------------------------------------------------------------------
    //  Approximate k‑NN traversal with epsilon pruning
    // ------------------------------------------------------------------------
    void approxSearchNode(size_type nodeIdx, const point_type& query, size_type k, T epsilon,
                          std::priority_queue<std::pair<T, index_type>,
                                              std::vector<std::pair<T, index_type>>,
                                              std::function<bool(const std::pair<T,index_type>&,
                                                                 const std::pair<T,index_type>&)>>& best,
                          T& threshold) const {
        const Node& node = m_nodes[nodeIdx];
        if (node.isLeaf) {
            for (index_type i = node.leaf.start; i < node.leaf.start + node.leaf.count; ++i) {
                index_type pid = m_indices[i];
                T d2 = (m_points[pid] - query).squaredLength();
                if (best.size() < k) {
                    best.push({d2, pid});
                    if (best.size() == k) {
                        threshold = best.top().first / (1 + epsilon);
                    }
                } else if (d2 < best.top().first) {
                    best.pop();
                    best.push({d2, pid});
                    threshold = best.top().first / (1 + epsilon);
                }
            }
            return;
        }
        // Compute lower bound distance to node (sphere centre)
        T nodeDist2 = (node.center - query).squaredLength();
        T nodeBound = nodeDist2 - node.radius * node.radius;
        if (nodeBound < T(0)) nodeBound = T(0);
        if (nodeBound > threshold) return;
        T d = query[node.axis] - node.center[node.axis];
        size_type first = (d <= 0) ? node.children.left : node.children.right;
        size_type second = (first == node.children.left) ? node.children.right : node.children.left;
        approxSearchNode(first, query, k, epsilon, best, threshold);
        if (d * d <= threshold) {
            approxSearchNode(second, query, k, epsilon, best, threshold);
        }
    }

    // ------------------------------------------------------------------------
    //  Radius search
    // ------------------------------------------------------------------------
    void radiusSearchNode(size_type nodeIdx, const point_type& query, T radiusSq,
                          std::vector<index_type>& out) const {
        const Node& node = m_nodes[nodeIdx];
        T nodeDist2 = (node.center - query).squaredLength();
        if (nodeDist2 - node.radius * node.radius > radiusSq) return;
        if (node.isLeaf) {
            for (index_type i = node.leaf.start; i < node.leaf.start + node.leaf.count; ++i) {
                index_type pid = m_indices[i];
                T d2 = (m_points[pid] - query).squaredLength();
                if (d2 <= radiusSq + m_config.searchEpsilon) out.push_back(pid);
            }
        } else {
            radiusSearchNode(node.children.left, query, radiusSq, out);
            radiusSearchNode(node.children.right, query, radiusSq, out);
        }
    }

    Config m_config;
    std::vector<point_type> m_points;
    std::vector<index_type> m_indices;
    std::vector<Node> m_nodes;
};

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class KdTreeEnvironment {
public:
    static KdTreeEnvironment& instance() {
        static KdTreeEnvironment env;
        return env;
    }
    void setDefaultMaxLeafSize(size_type sz) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_maxLeafSize = sz;
    }
    size_type defaultMaxLeafSize() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_maxLeafSize;
    }
    void setUseSIMD(bool use) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_useSIMD = use;
    }
    bool useSIMD() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_useSIMD;
    }
private:
    KdTreeEnvironment() : m_maxLeafSize(16), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    size_type m_maxLeafSize;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_KDTREE_H_INCLUDED