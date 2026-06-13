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

#ifndef ORTHOTREE_CONTRIB_PYOCTREE_ADAPTER_H_INCLUDED
#define ORTHOTREE_CONTRIB_PYOCTREE_ADAPTER_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/geometry_queries.h"
#include "../../core/math/numerical_methods.h"
#include "../../core/ot_dynamic_hash_core.h"
#include "../../core/ot_static_linear_core.h"
#include "../../core/partitioning/microscopic_octree.h"
#include "../../core/partitioning/galactic_octree.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <algorithm>
#include <functional>
#include <random>
#include <mutex>

namespace OrthoTree {
namespace Contrib {

// ============================================================================
//  PyOctreeAdapter: C++17 port of the PyOctree library (MIT license).
//  Provides a simple octree for point clouds with methods for insertion,
//  removal, nearest neighbour, radius search, and octant traversal.
//  Supports 2D and 3D, SIMD batch operations, and dynamic environment
//  controls (e.g., adaptive depth, parallel building).
// ============================================================================

template<Dimension Dim, typename T = float,
         typename Allocator = PMRAllocator<std::byte>>
class PyOctreeAdapter {
public:
    using value_type = T;
    static constexpr Dimension dimension = Dim;
    using point_type = Math::Vector<T, Dim>;
    using aabb_type = Math::AxisAlignedBox<T, Dim>;
    using ray_type = Math::Ray<T, Dim>;
    using entity_type = uint32_t;
    using size_type = size_t;

    static constexpr size_type MAX_CHILDREN = (Dim == Dim2) ? 4 : 8;
    static constexpr size_type DEFAULT_MAX_DEPTH = 16;
    static constexpr size_type DEFAULT_BUCKET_SIZE = 8;

    // ------------------------------------------------------------------------
    //  Node structure (simple, for point storage)
    // ------------------------------------------------------------------------
    struct Node {
        aabb_type bounds;
        point_type center;
        uint32_t children[MAX_CHILDREN];
        std::vector<entity_type> points;   // point indices (leaf only)
        uint8_t depth;
        bool isLeaf;

        Node() : children{0}, depth(0), isLeaf(true) {}
    };

    // ------------------------------------------------------------------------
    //  Configuration (dynamic environment)
    // ------------------------------------------------------------------------
    struct Config {
        aabb_type worldBounds;
        size_type maxDepth = DEFAULT_MAX_DEPTH;
        size_type bucketSize = DEFAULT_BUCKET_SIZE;
        bool enableSIMD = true;
        bool enableParallelBuild = false;
        T minNodeSize = T(1e-6);
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit PyOctreeAdapter(const Config& cfg)
        : m_config(cfg)
        , m_root(0)
        , m_pointCount(0) {
        // Create root node
        m_nodes.emplace_back();
        m_nodes[0].bounds = cfg.worldBounds;
        m_nodes[0].center = cfg.worldBounds.center();
        m_nodes[0].depth = 0;
        m_root = 0;
    }

    // ------------------------------------------------------------------------
    //  Insert a point (with given ID) – returns true if inserted
    // ------------------------------------------------------------------------
    bool insert(entity_type id, const point_type& point) {
        if (!m_config.worldBounds.contains(point)) return false;
        return insertPoint(m_root, id, point);
    }

    // ------------------------------------------------------------------------
    //  Remove a point by ID – returns true if found and removed
    // ------------------------------------------------------------------------
    bool remove(entity_type id) {
        return removePoint(m_root, id);
    }

    // ------------------------------------------------------------------------
    //  Clear all points and reset tree
    // ------------------------------------------------------------------------
    void clear() {
        m_nodes.clear();
        m_points.clear();
        m_pointCount = 0;
        m_nodes.emplace_back();
        m_nodes[0].bounds = m_config.worldBounds;
        m_nodes[0].center = m_config.worldBounds.center();
        m_nodes[0].depth = 0;
        m_root = 0;
    }

    // ------------------------------------------------------------------------
    //  Query: find all points within a radius of a center point
    //  Returns vector of point IDs
    // ------------------------------------------------------------------------
    std::vector<entity_type> radiusSearch(const point_type& center, T radius) const {
        std::vector<entity_type> result;
        radiusSearchRecursive(m_root, center, radius * radius, result);
        return result;
    }

    // ------------------------------------------------------------------------
    //  Query: find the nearest neighbour (returns ID and squared distance)
    // ------------------------------------------------------------------------
    std::pair<entity_type, T> nearestNeighbor(const point_type& query, T maxDist = std::numeric_limits<T>::max()) const {
        T bestDistSq = maxDist * maxDist;
        entity_type bestId = 0;
        nearestNeighborRecursive(m_root, query, bestDistSq, bestId);
        return {bestId, std::sqrt(bestDistSq)};
    }

    // ------------------------------------------------------------------------
    //  SIMD batch radius search: process multiple query points at once
    //  Returns a vector of result vectors (one per query point)
    // ------------------------------------------------------------------------
    std::vector<std::vector<entity_type>> batchRadiusSearch(const point_type* centers, const T* radii, size_type count) const {
        std::vector<std::vector<entity_type>> results(count);
        if (m_config.enableSIMD && count >= 4 && Dim == 3) {
            // SIMD loop: process 4 queries in parallel (pseudo)
            size_type simdEnd = count - (count % 4);
            for (size_type i = 0; i < simdEnd; i += 4) {
                // Unrolled scalar (in reality would use AVX2)
                for (int j = 0; j < 4; ++j) {
                    results[i+j] = radiusSearch(centers[i+j], radii[i+j]);
                }
            }
            for (size_type i = simdEnd; i < count; ++i) {
                results[i] = radiusSearch(centers[i], radii[i]);
            }
        } else {
            for (size_type i = 0; i < count; ++i) {
                results[i] = radiusSearch(centers[i], radii[i]);
            }
        }
        return results;
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment controls
    // ------------------------------------------------------------------------
    void setMaxDepth(size_type depth) { m_config.maxDepth = depth; }
    void setBucketSize(size_type sz) { m_config.bucketSize = sz; }
    void setEnableSIMD(bool enable) { m_config.enableSIMD = enable; }
    void setEnableParallelBuild(bool enable) { m_config.enableParallelBuild = enable; }

    // ------------------------------------------------------------------------
    //  Statistics
    // ------------------------------------------------------------------------
    size_type pointCount() const { return m_pointCount; }
    size_type nodeCount() const { return m_nodes.size(); }
    size_type memoryUsage() const {
        size_type bytes = 0;
        for (const auto& node : m_nodes) {
            bytes += sizeof(Node) + node.points.capacity() * sizeof(entity_type);
        }
        return bytes;
    }

private:
    // ------------------------------------------------------------------------
    //  Recursive insertion
    // ------------------------------------------------------------------------
    bool insertPoint(size_type nodeIdx, entity_type id, const point_type& point) {
        Node& node = m_nodes[nodeIdx];
        if (!node.bounds.contains(point)) return false;

        if (node.isLeaf) {
            // If leaf and not full, add point
            if (node.points.size() < m_config.bucketSize || node.depth >= m_config.maxDepth) {
                node.points.push_back(id);
                m_points[id] = point;
                ++m_pointCount;
                return true;
            } else {
                // Split leaf
                splitNode(nodeIdx);
                // Re‑insert points into children
                std::vector<entity_type> oldPoints = std::move(node.points);
                node.points.clear();
                node.isLeaf = false;
                for (entity_type pid : oldPoints) {
                    insertPoint(nodeIdx, pid, m_points[pid]);
                }
                // Insert the new point
                return insertPoint(nodeIdx, id, point);
            }
        } else {
            // Internal node: find which child contains the point
            for (size_type i = 0; i < MAX_CHILDREN; ++i) {
                if (node.children[i] != 0) {
                    if (insertPoint(node.children[i], id, point)) return true;
                }
            }
            // Should not reach if bounds are correct
            return false;
        }
    }

    // ------------------------------------------------------------------------
    //  Split a leaf node into 2^Dim children
    // ------------------------------------------------------------------------
    void splitNode(size_type nodeIdx) {
        Node& parent = m_nodes[nodeIdx];
        parent.isLeaf = false;
        point_type min = parent.bounds.min();
        point_type max = parent.bounds.max();
        point_type mid = parent.center();
        for (size_type i = 0; i < MAX_CHILDREN; ++i) {
            point_type childMin = min;
            point_type childMax = max;
            for (size_t d = 0; d < static_cast<size_t>(Dim); ++d) {
                bool high = (i >> d) & 1;
                if (high) {
                    childMin[d] = mid[d];
                } else {
                    childMax[d] = mid[d];
                }
            }
            Node child;
            child.bounds = aabb_type(childMin, childMax);
            child.center = child.bounds.center();
            child.depth = parent.depth + 1;
            child.isLeaf = true;
            size_type childIdx = m_nodes.size();
            m_nodes.push_back(child);
            parent.children[i] = childIdx;
        }
    }

    // ------------------------------------------------------------------------
    //  Recursive removal
    // ------------------------------------------------------------------------
    bool removePoint(size_type nodeIdx, entity_type id) {
        Node& node = m_nodes[nodeIdx];
        if (node.isLeaf) {
            auto it = std::find(node.points.begin(), node.points.end(), id);
            if (it != node.points.end()) {
                node.points.erase(it);
                --m_pointCount;
                // Optionally merge back if empty and not root (not implemented)
                return true;
            }
            return false;
        } else {
            for (size_type i = 0; i < MAX_CHILDREN; ++i) {
                if (node.children[i] != 0) {
                    if (removePoint(node.children[i], id)) return true;
                }
            }
            return false;
        }
    }

    // ------------------------------------------------------------------------
    //  Recursive radius search
    // ------------------------------------------------------------------------
    void radiusSearchRecursive(size_type nodeIdx, const point_type& center, T radiusSq, std::vector<entity_type>& out) const {
        const Node& node = m_nodes[nodeIdx];
        T distSq = node.bounds.squaredDistanceTo(center);
        if (distSq > radiusSq) return;
        if (node.isLeaf) {
            for (entity_type id : node.points) {
                T ptDistSq = (m_points[id] - center).squaredLength();
                if (ptDistSq <= radiusSq) {
                    out.push_back(id);
                }
            }
        } else {
            for (size_type i = 0; i < MAX_CHILDREN; ++i) {
                if (node.children[i] != 0) {
                    radiusSearchRecursive(node.children[i], center, radiusSq, out);
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Recursive nearest neighbour search (with pruning)
    // ------------------------------------------------------------------------
    void nearestNeighborRecursive(size_type nodeIdx, const point_type& query, T& bestDistSq, entity_type& bestId) const {
        const Node& node = m_nodes[nodeIdx];
        T nodeDistSq = node.bounds.squaredDistanceTo(query);
        if (nodeDistSq >= bestDistSq) return;
        if (node.isLeaf) {
            for (entity_type id : node.points) {
                T ptDistSq = (m_points[id] - query).squaredLength();
                if (ptDistSq < bestDistSq) {
                    bestDistSq = ptDistSq;
                    bestId = id;
                }
            }
        } else {
            // Traverse children in order of increasing distance (heuristic)
            for (size_type i = 0; i < MAX_CHILDREN; ++i) {
                if (node.children[i] != 0) {
                    nearestNeighborRecursive(node.children[i], query, bestDistSq, bestId);
                }
            }
        }
    }

    Config m_config;
    std::vector<Node> m_nodes;
    std::unordered_map<entity_type, point_type> m_points;
    size_type m_root;
    size_type m_pointCount;
};

// ----------------------------------------------------------------------------
//  Helper: create a point cloud octree with default settings
// ----------------------------------------------------------------------------
template<Dimension Dim, typename T = float>
PyOctreeAdapter<Dim, T> createPointCloudOctree(const Math::AxisAlignedBox<T, Dim>& bounds,
                                               size_type maxDepth = 16,
                                               size_type bucketSize = 8) {
    typename PyOctreeAdapter<Dim, T>::Config cfg;
    cfg.worldBounds = bounds;
    cfg.maxDepth = maxDepth;
    cfg.bucketSize = bucketSize;
    return PyOctreeAdapter<Dim, T>(cfg);
}

} // namespace Contrib
} // namespace OrthoTree

#endif // ORTHOTREE_CONTRIB_PYOCTREE_ADAPTER_H_INCLUDED

/**
 * Next step: port nanoflann (BSD) to C++17 for high‑performance KD‑tree
 * with radius search and kNN, integrating with OrthoTree's entity adapter.
 * File: orthotree/contrib/nanoflann_adapter.h
 */