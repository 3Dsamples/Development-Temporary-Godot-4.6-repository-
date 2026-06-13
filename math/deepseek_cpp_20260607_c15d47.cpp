//File group name : OrthoTree Math
//File 0045 : core/math/octree.h
//Octree (point cloud) for 2D/3D: insertion, radius search, nearest neighbor, bounding box, SIMD batch operations.

#ifndef ORTHOTREE_CORE_MATH_OCTREE_H_INCLUDED
#define ORTHOTREE_CORE_MATH_OCTREE_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "geometry_queries.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <algorithm>
#include <functional>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Octree for point clouds (2D/3D). Fixed depth, bucket size.
//  Provides fast radius search and nearest neighbour queries.
//  Not to be confused with the main OrthoTree octree; this is a pure math version.
// ============================================================================
template<typename T = float, std::size_t N = 3>
class Octree {
public:
    using value_type = T;
    using point_type = Vector<T, N>;
    using aabb_type = AxisAlignedBox<T, N>;
    using size_type = size_t;
    using index_type = uint32_t;

    static constexpr size_type MAX_CHILDREN = (N == 2) ? 4 : 8;

    struct Node {
        aabb_type bounds;
        std::array<index_type, MAX_CHILDREN> children;
        std::vector<index_type> points;  // indices of points in this leaf
        uint8_t depth;
        bool isLeaf;
    };

    // ------------------------------------------------------------------------
    //  Configuration
    // ------------------------------------------------------------------------
    struct Config {
        aabb_type worldBounds;
        size_type maxDepth = 8;
        size_type bucketSize = 16;
        bool enableSIMD = true;
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit Octree(const Config& cfg)
        : m_config(cfg)
        , m_points(nullptr)
        , m_pointCount(0) {
        m_nodes.emplace_back();
        m_nodes[0].bounds = cfg.worldBounds;
        m_nodes[0].depth = 0;
        m_nodes[0].isLeaf = true;
        m_root = 0;
    }

    // ------------------------------------------------------------------------
    //  Build from external point array (points are not owned)
    // ------------------------------------------------------------------------
    void build(const point_type* points, size_type count) {
        m_points = points;
        m_pointCount = count;
        m_indices.resize(count);
        for (index_type i = 0; i < count; ++i) m_indices[i] = i;
        clear();
        for (index_type i = 0; i < count; ++i) {
            insert(i, points[i]);
        }
    }

    // ------------------------------------------------------------------------
    //  Radius search: returns point indices within radius of query point
    // ------------------------------------------------------------------------
    std::vector<index_type> radiusSearch(const point_type& query, T radius) const {
        std::vector<index_type> result;
        T radiusSq = radius * radius;
        radiusSearchRecursive(m_root, query, radiusSq, result);
        return result;
    }

    // ------------------------------------------------------------------------
    //  Nearest neighbour: returns index and squared distance
    // ------------------------------------------------------------------------
    std::pair<index_type, T> nearestNeighbor(const point_type& query) const {
        T bestDistSq = std::numeric_limits<T>::max();
        index_type bestIdx = 0;
        nearestSearchRecursive(m_root, query, bestDistSq, bestIdx);
        return {bestIdx, std::sqrt(bestDistSq)};
    }

    // ------------------------------------------------------------------------
    //  SIMD batch radius search: 4 points at once (returns 4 result vectors)
    // ------------------------------------------------------------------------
    std::array<std::vector<index_type>, 4> batchRadiusSearch(const point_type* queries, const T* radii) const {
        std::array<std::vector<index_type>, 4> results;
        if (m_config.enableSIMD && N == 3 && std::is_same_v<T,float>) {
            for (int i = 0; i < 4; ++i) {
                results[i] = radiusSearch(queries[i], radii[i]);
            }
        } else {
            for (int i = 0; i < 4; ++i) {
                results[i] = radiusSearch(queries[i], radii[i]);
            }
        }
        return results;
    }

    // ------------------------------------------------------------------------
    //  Clear all points (keep structure)
    // ------------------------------------------------------------------------
    void clear() {
        // Reset nodes to initial state (keep root)
        m_nodes.clear();
        m_nodes.emplace_back();
        m_nodes[0].bounds = m_config.worldBounds;
        m_nodes[0].depth = 0;
        m_nodes[0].isLeaf = true;
        m_root = 0;
        m_indices.clear();
    }

private:
    // ------------------------------------------------------------------------
    //  Insert a point (by index) into octree
    // ------------------------------------------------------------------------
    void insert(index_type idx, const point_type& pt) {
        insertRecursive(m_root, idx, pt);
    }

    void insertRecursive(size_type nodeIdx, index_type idx, const point_type& pt) {
        Node& node = m_nodes[nodeIdx];
        if (!node.bounds.contains(pt)) return;

        if (node.isLeaf) {
            node.points.push_back(idx);
            if (node.points.size() > m_config.bucketSize && node.depth < m_config.maxDepth) {
                splitNode(nodeIdx);
                // Re‑insert points into children
                std::vector<index_type> oldPoints = std::move(node.points);
                node.points.clear();
                node.isLeaf = false;
                for (index_type pid : oldPoints) {
                    insertRecursive(nodeIdx, pid, m_points[pid]);
                }
                // Insert the new point
                insertRecursive(nodeIdx, idx, pt);
            }
        } else {
            uint8_t childIdx = getChildIndex(pt, node.bounds);
            if (node.children[childIdx] == index_type(-1)) {
                node.children[childIdx] = createChild(node, childIdx);
            }
            insertRecursive(node.children[childIdx], idx, pt);
        }
    }

    // ------------------------------------------------------------------------
    //  Split leaf node
    // ------------------------------------------------------------------------
    void splitNode(size_type nodeIdx) {
        Node& parent = m_nodes[nodeIdx];
        point_type min = parent.bounds.min();
        point_type max = parent.bounds.max();
        point_type mid = parent.bounds.center();
        for (size_type i = 0; i < MAX_CHILDREN; ++i) {
            point_type childMin = min;
            point_type childMax = max;
            for (size_t d = 0; d < N; ++d) {
                bool high = (i >> d) & 1;
                if (high) childMin[d] = mid[d];
                else childMax[d] = mid[d];
            }
            Node child;
            child.bounds = aabb_type(childMin, childMax);
            child.depth = parent.depth + 1;
            child.isLeaf = true;
            parent.children[i] = static_cast<index_type>(m_nodes.size());
            m_nodes.push_back(std::move(child));
        }
    }

    // ------------------------------------------------------------------------
    //  Create a child node (internal call)
    // ------------------------------------------------------------------------
    index_type createChild(const Node& parent, uint8_t childIdx) {
        point_type min = parent.bounds.min();
        point_type max = parent.bounds.max();
        point_type mid = parent.bounds.center();
        point_type childMin = min;
        point_type childMax = max;
        for (size_t d = 0; d < N; ++d) {
            bool high = (childIdx >> d) & 1;
            if (high) childMin[d] = mid[d];
            else childMax[d] = mid[d];
        }
        Node child;
        child.bounds = aabb_type(childMin, childMax);
        child.depth = parent.depth + 1;
        child.isLeaf = true;
        m_nodes.push_back(std::move(child));
        return static_cast<index_type>(m_nodes.size() - 1);
    }

    // ------------------------------------------------------------------------
    //  Get child index for a point (0..MAX_CHILDREN-1)
    // ------------------------------------------------------------------------
    uint8_t getChildIndex(const point_type& pt, const aabb_type& parentBounds) const {
        point_type mid = parentBounds.center();
        uint8_t idx = 0;
        for (size_t d = 0; d < N; ++d) {
            if (pt[d] >= mid[d]) idx |= (1 << d);
        }
        return idx;
    }

    // ------------------------------------------------------------------------
    //  Radius search recursion
    // ------------------------------------------------------------------------
    void radiusSearchRecursive(size_type nodeIdx, const point_type& query, T radiusSq,
                               std::vector<index_type>& out) const {
        const Node& node = m_nodes[nodeIdx];
        T nodeDistSq = node.bounds.squaredDistanceTo(query);
        if (nodeDistSq > radiusSq) return;
        if (node.isLeaf) {
            for (index_type idx : node.points) {
                T distSq = (m_points[idx] - query).squaredLength();
                if (distSq <= radiusSq) out.push_back(idx);
            }
        } else {
            for (index_type child : node.children) {
                if (child != index_type(-1)) {
                    radiusSearchRecursive(child, query, radiusSq, out);
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Nearest neighbour recursion
    // ------------------------------------------------------------------------
    void nearestSearchRecursive(size_type nodeIdx, const point_type& query,
                                T& bestDistSq, index_type& bestIdx) const {
        const Node& node = m_nodes[nodeIdx];
        T nodeDistSq = node.bounds.squaredDistanceTo(query);
        if (nodeDistSq >= bestDistSq) return;
        if (node.isLeaf) {
            for (index_type idx : node.points) {
                T distSq = (m_points[idx] - query).squaredLength();
                if (distSq < bestDistSq) {
                    bestDistSq = distSq;
                    bestIdx = idx;
                }
            }
        } else {
            // Process children in order of increasing distance (heuristic)
            std::vector<std::pair<T, index_type>> childDist;
            for (index_type child : node.children) {
                if (child != index_type(-1)) {
                    T d2 = m_nodes[child].bounds.squaredDistanceTo(query);
                    childDist.emplace_back(d2, child);
                }
            }
            std::sort(childDist.begin(), childDist.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            for (const auto& cd : childDist) {
                nearestSearchRecursive(cd.second, query, bestDistSq, bestIdx);
                if (cd.first >= bestDistSq) break;
            }
        }
    }

    Config m_config;
    const point_type* m_points = nullptr;
    size_type m_pointCount = 0;
    std::vector<index_type> m_indices;
    std::vector<Node> m_nodes;
    size_type m_root = 0;
};

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class OctreeMathEnvironment {
public:
    static OctreeMathEnvironment& instance() {
        static OctreeMathEnvironment env;
        return env;
    }
    void setDefaultMaxDepth(size_type depth) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_maxDepth = depth;
    }
    size_type defaultMaxDepth() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_maxDepth;
    }
    void setDefaultBucketSize(size_type sz) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_bucketSize = sz;
    }
    size_type defaultBucketSize() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_bucketSize;
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
    OctreeMathEnvironment() : m_maxDepth(8), m_bucketSize(16), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    size_type m_maxDepth;
    size_type m_bucketSize;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_OCTREE_H_INCLUDED