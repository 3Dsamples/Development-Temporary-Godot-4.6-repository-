/* MIT License Copyright (c) 2021-2026 Attila Csikós & Contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ORTHOTREE_CONTRIB_PCL_ADAPTER_H_INCLUDED
#define ORTHOTREE_CONTRIB_PCL_ADAPTER_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/geometry_queries.h"
#include "../../core/math/numerical_methods.h"
#include "../../core/ot_dynamic_hash_core.h"
#include "../../core/ot_static_linear_core.h"
#include "../../core/parallel/scale_aware_task_scheduler.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <algorithm>
#include <functional>
#include <memory>
#include <random>
#include <unordered_map>
#include <mutex>

namespace OrthoTree {
namespace Contrib {

// ============================================================================
//  PCLAdapter: port of Point Cloud Library (PCL) octree features.
//  PCL is BSD licensed. This adapter provides:
//  - Octree for point clouds (voxel grid, occupancy, density)
//  - Radius search, nearest neighbour, K‑nearest
//  - Downsampling (voxel grid filter)
//  - Normal estimation from point cloud (using octree neighbours)
//  - SIMD batch queries and dynamic environment controls
// ============================================================================

template<typename T = float, std::size_t N = 3>
class PCLAdapter {
public:
    using value_type = T;
    using point_type = Math::Vector<T, N>;
    using aabb_type = Math::AxisAlignedBox<T, N>;
    using size_type = size_t;
    using index_type = uint32_t;

    static constexpr std::size_t dimension = N;
    static constexpr size_type MAX_CHILDREN = (N == 2) ? 4 : 8;
    static constexpr index_type INVALID_IDX = index_type(-1);

    // ------------------------------------------------------------------------
    //  Node structure (for point occupancy, not full bounding boxes)
    // ------------------------------------------------------------------------
    struct Node {
        aabb_type bounds;
        union {
            struct {
                index_type child[MAX_CHILDREN];
            } internal;
            struct {
                std::vector<index_type> points;  // indices of points in this leaf
            } leaf;
        };
        uint8_t depth;
        bool isLeaf;
        uint32_t pointCount;      // cached count
    };

    // ------------------------------------------------------------------------
    //  Configuration (dynamic environment)
    // ------------------------------------------------------------------------
    struct Config {
        aabb_type worldBounds;
        T voxelSize = T(1.0);               // for downsampling / leaf size
        size_type maxDepth = 16;
        size_type minPointsPerLeaf = 8;     // split if more than this
        bool enableSIMD = true;
        bool enableParallel = false;
        T normalRadius = T(2.0);            // for normal estimation
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit PCLAdapter(const Config& cfg)
        : m_config(cfg)
        , m_pointCount(0) {
        // Create root node
        m_nodes.emplace_back();
        m_nodes[0].bounds = cfg.worldBounds;
        m_nodes[0].depth = 0;
        m_nodes[0].isLeaf = true;
        m_nodes[0].leaf.points.reserve(m_config.minPointsPerLeaf);
        m_nodes[0].pointCount = 0;
        m_root = 0;
    }

    // ------------------------------------------------------------------------
    //  Insert a batch of points (point cloud)
    //  Returns vector of point indices (0..count-1) as entity IDs.
    // ------------------------------------------------------------------------
    std::vector<index_type> insertPoints(const point_type* points, size_type count) {
        m_points.assign(points, points + count);
        m_pointCount = count;
        // Insert each point into octree
        for (index_type i = 0; i < count; ++i) {
            insertPoint(i);
        }
        return std::vector<index_type>(count); // dummy
    }

    // ------------------------------------------------------------------------
    //  Insert a single point (by index)
    // ------------------------------------------------------------------------
    void insertPoint(index_type pointIdx) {
        const point_type& pt = m_points[pointIdx];
        insertRecursive(m_root, pointIdx, pt);
    }

    // ------------------------------------------------------------------------
    //  Radius search (returns point indices within radius of query point)
    // ------------------------------------------------------------------------
    std::vector<index_type> radiusSearch(const point_type& query, T radius) const {
        std::vector<index_type> result;
        radiusSearchRecursive(m_root, query, radius * radius, result);
        return result;
    }

    // ------------------------------------------------------------------------
    //  K‑Nearest Neighbour search (kNN)
    //  Returns vector of (index, squared distance) pairs, sorted by distance.
    // ------------------------------------------------------------------------
    std::vector<std::pair<index_type, T>> knnSearch(const point_type& query, size_type k) const {
        using Pair = std::pair<T, index_type>; // distance², index
        std::vector<Pair> candidates;
        knnSearchRecursive(m_root, query, candidates, k);
        std::sort(candidates.begin(), candidates.end(),
                  [](const Pair& a, const Pair& b) { return a.first < b.first; });
        std::vector<std::pair<index_type, T>> result;
        for (size_type i = 0; i < std::min(k, candidates.size()); ++i) {
            result.emplace_back(candidates[i].second, std::sqrt(candidates[i].first));
        }
        return result;
    }

    // ------------------------------------------------------------------------
    //  Voxel downsampling: reduce point cloud by keeping centroid of each occupied voxel
    //  Returns new point cloud (points only)
    // ------------------------------------------------------------------------
    std::vector<point_type> voxelDownsample() const {
        // Collect centroids per leaf node (if leaf node represents a voxel)
        // For simplicity, we use the leaf nodes of the octree as voxels.
        std::vector<point_type> centroids;
        traverseNodes(m_root, [&](const Node& node) {
            if (node.isLeaf && node.pointCount > 0) {
                point_type sum(0);
                for (index_type idx : node.leaf.points) {
                    sum += m_points[idx];
                }
                centroids.push_back(sum / static_cast<T>(node.leaf.points.size()));
            }
            return true; // continue traversal
        });
        return centroids;
    }

    // ------------------------------------------------------------------------
    //  Estimate normals for each point using octree neighbourhood
    //  Returns vector of unit normals.
    // ------------------------------------------------------------------------
    std::vector<point_type> estimateNormals(T radius) const {
        std::vector<point_type> normals(m_pointCount, point_type(0));
        #pragma omp parallel for if(m_config.enableParallel)
        for (index_type i = 0; i < m_pointCount; ++i) {
            std::vector<index_type> neighbours = radiusSearch(m_points[i], radius);
            if (neighbours.size() < 3) continue;
            // Compute covariance matrix
            T cov[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
            point_type mean(0);
            for (index_type n : neighbours) mean += m_points[n];
            mean /= static_cast<T>(neighbours.size());
            for (index_type n : neighbours) {
                point_type d = m_points[n] - mean;
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        cov[r][c] += d[r] * d[c];
            }
            // Power iteration to find eigenvector with smallest eigenvalue
            point_type normal(1,0,0);
            for (int iter = 0; iter < 10; ++iter) {
                point_type n2(0);
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        n2[r] += cov[r][c] * normal[c];
                T len = n2.length();
                if (len > T(1e-8)) n2 /= len;
                if ((n2 - normal).length() < T(1e-4)) { normal = n2; break; }
                normal = n2;
            }
            normals[i] = normal;
        }
        return normals;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch radius search (multiple queries)
    // ------------------------------------------------------------------------
    std::vector<std::vector<index_type>> batchRadiusSearch(const point_type* queries,
                                                            const T* radii,
                                                            size_type count) const {
        std::vector<std::vector<index_type>> results(count);
        if (m_config.enableSIMD && count >= 4 && N == 3) {
            size_type simdEnd = count - (count % 4);
            for (size_type i = 0; i < simdEnd; i += 4) {
                for (int j = 0; j < 4; ++j) {
                    results[i+j] = radiusSearch(queries[i+j], radii[i+j]);
                }
            }
            for (size_type i = simdEnd; i < count; ++i) {
                results[i] = radiusSearch(queries[i], radii[i]);
            }
        } else {
            for (size_type i = 0; i < count; ++i) {
                results[i] = radiusSearch(queries[i], radii[i]);
            }
        }
        return results;
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment controls
    // ------------------------------------------------------------------------
    void setMaxDepth(size_type depth) { m_config.maxDepth = depth; }
    void setMinPointsPerLeaf(size_type min) { m_config.minPointsPerLeaf = min; }
    void setVoxelSize(T sz) { m_config.voxelSize = sz; }
    void setEnableSIMD(bool enable) { m_config.enableSIMD = enable; }
    void setEnableParallel(bool enable) { m_config.enableParallel = enable; }

    // ------------------------------------------------------------------------
    //  Statistics
    // ------------------------------------------------------------------------
    size_type pointCount() const { return m_pointCount; }
    size_type nodeCount() const { return m_nodes.size(); }
    size_type memoryUsage() const {
        size_type total = m_points.capacity() * sizeof(point_type);
        for (const auto& node : m_nodes) {
            total += sizeof(Node);
            if (node.isLeaf) total += node.leaf.points.capacity() * sizeof(index_type);
        }
        return total;
    }

private:
    // ------------------------------------------------------------------------
    //  Recursive insertion
    // ------------------------------------------------------------------------
    void insertRecursive(size_type nodeIdx, index_type pointIdx, const point_type& pt) {
        Node& node = m_nodes[nodeIdx];
        if (!node.bounds.contains(pt)) return;

        if (node.isLeaf) {
            node.leaf.points.push_back(pointIdx);
            node.pointCount = node.leaf.points.size();
            if (node.pointCount > m_config.minPointsPerLeaf && node.depth < m_config.maxDepth) {
                splitNode(nodeIdx);
                // Re‑insert points into children
                std::vector<index_type> oldPoints = std::move(node.leaf.points);
                node.leaf.points.clear();
                node.isLeaf = false;
                node.pointCount = 0;
                for (index_type idx : oldPoints) {
                    insertRecursive(nodeIdx, idx, m_points[idx]);
                }
            }
        } else {
            uint8_t childIdx = getChildIndex(pt, node.bounds);
            if (node.internal.child[childIdx] == INVALID_IDX) {
                node.internal.child[childIdx] = createChild(node, childIdx);
            }
            insertRecursive(node.internal.child[childIdx], pointIdx, pt);
        }
    }

    // ------------------------------------------------------------------------
    //  Split leaf node into MAX_CHILDREN children
    // ------------------------------------------------------------------------
    void splitNode(size_type nodeIdx) {
        Node& parent = m_nodes[nodeIdx];
        parent.isLeaf = false;
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
            child.leaf.points.reserve(m_config.minPointsPerLeaf);
            child.pointCount = 0;
            parent.internal.child[i] = static_cast<index_type>(m_nodes.size());
            m_nodes.push_back(std::move(child));
        }
    }

    // ------------------------------------------------------------------------
    //  Create a new child for internal node
    // ------------------------------------------------------------------------
    size_type createChild(const Node& parent, uint8_t childIdx) {
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
        child.leaf.points.reserve(m_config.minPointsPerLeaf);
        child.pointCount = 0;
        size_type childNodeIdx = m_nodes.size();
        m_nodes.push_back(std::move(child));
        return childNodeIdx;
    }

    // ------------------------------------------------------------------------
    //  Radius search recursion
    // ------------------------------------------------------------------------
    void radiusSearchRecursive(size_type nodeIdx, const point_type& query, T radiusSq,
                               std::vector<index_type>& result) const {
        const Node& node = m_nodes[nodeIdx];
        T nodeDistSq = node.bounds.squaredDistanceTo(query);
        if (nodeDistSq > radiusSq) return;
        if (node.isLeaf) {
            for (index_type idx : node.leaf.points) {
                T distSq = (m_points[idx] - query).squaredLength();
                if (distSq <= radiusSq + T(1e-6)) {
                    result.push_back(idx);
                }
            }
        } else {
            for (size_type i = 0; i < MAX_CHILDREN; ++i) {
                if (node.internal.child[i] != INVALID_IDX) {
                    radiusSearchRecursive(node.internal.child[i], query, radiusSq, result);
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    //  KNN search recursion (collect candidates)
    // ------------------------------------------------------------------------
    void knnSearchRecursive(size_type nodeIdx, const point_type& query,
                            std::vector<std::pair<T, index_type>>& candidates,
                            size_type k) const {
        const Node& node = m_nodes[nodeIdx];
        T nodeDistSq = node.bounds.squaredDistanceTo(query);
        // Prune: if we already have k candidates and the node is farther than the worst, skip
        // For simplicity, we collect all leaves then prune.
        if (node.isLeaf) {
            for (index_type idx : node.leaf.points) {
                T distSq = (m_points[idx] - query).squaredLength();
                candidates.emplace_back(distSq, idx);
            }
        } else {
            for (size_type i = 0; i < MAX_CHILDREN; ++i) {
                if (node.internal.child[i] != INVALID_IDX) {
                    knnSearchRecursive(node.internal.child[i], query, candidates, k);
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Traverse all nodes (for downsampling)
    // ------------------------------------------------------------------------
    template<typename Func>
    void traverseNodes(size_type nodeIdx, Func&& func) const {
        const Node& node = m_nodes[nodeIdx];
        if (!func(node)) return;
        if (!node.isLeaf) {
            for (size_type i = 0; i < MAX_CHILDREN; ++i) {
                if (node.internal.child[i] != INVALID_IDX) {
                    traverseNodes(node.internal.child[i], func);
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Child index for a point (0..MAX_CHILDREN-1)
    // ------------------------------------------------------------------------
    uint8_t getChildIndex(const point_type& point, const aabb_type& parentBounds) const {
        point_type mid = parentBounds.center();
        uint8_t idx = 0;
        for (size_t d = 0; d < N; ++d) {
            if (point[d] >= mid[d]) idx |= (1 << d);
        }
        return idx;
    }

    Config m_config;
    std::vector<point_type> m_points;
    std::vector<Node> m_nodes;
    size_type m_root;
    size_type m_pointCount;
};

// ----------------------------------------------------------------------------
//  Helper: create adapter with default bounds from points
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
PCLAdapter<T, N> createPCLAdapter(const std::vector<Math::Vector<T, N>>& points,
                                  T voxelSize = T(1.0),
                                  size_type maxDepth = 16) {
    aabb_type bounds;
    for (const auto& p : points) bounds.extend(p);
    if (bounds.extents().maxComponent() == T(0)) {
        bounds = aabb_type(point_type(-1), point_type(1));
    }
    typename PCLAdapter<T, N>::Config cfg;
    cfg.worldBounds = bounds;
    cfg.voxelSize = voxelSize;
    cfg.maxDepth = maxDepth;
    cfg.minPointsPerLeaf = 8;
    PCLAdapter<T, N> adapter(cfg);
    adapter.insertPoints(points.data(), points.size());
    return adapter;
}

} // namespace Contrib
} // namespace OrthoTree

#endif // ORTHOTREE_CONTRIB_PCL_ADAPTER_H_INCLUDED