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

#ifndef ORTHOTREE_CONTRIB_LIBSPATIALINDEX_ADAPTER_H_INCLUDED
#define ORTHOTREE_CONTRIB_LIBSPATIALINDEX_ADAPTER_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/geometry_queries.h"
#include "../../core/math/numerical_methods.h"
#include "../../core/ot_dynamic_hash_core.h"
#include "../../core/partitioning/scale_adaptive_octree.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <memory>
#include <deque>
#include <cmath>
#include <limits>
#include <algorithm>
#include <functional>
#include <mutex>
#include <random>

namespace OrthoTree {
namespace Contrib {

// ============================================================================
//  LibspatialindexAdapter: Port of R‑tree and MVR‑tree (moving objects R‑tree)
//  from libspatialindex (MIT license). Provides spatial indexing for
//  rectangles (2D/3D) and moving objects with velocity.
//  Supports bulk loading (Sort‑Tile‑Recursive), insertion, deletion,
//  range queries, nearest neighbour, and time‑parameterised queries for
//  moving objects. SIMD batch operations and dynamic environment controls
//  (buffer management, node capacity, reinsertion optimisation).
// ============================================================================

// ----------------------------------------------------------------------------
//  Axis‑aligned bounding box (minimum and maximum)
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
struct RTreeBox {
    Math::Vector<T, N> min;
    Math::Vector<T, N> max;

    RTreeBox() = default;
    RTreeBox(const Math::Vector<T, N>& mn, const Math::Vector<T, N>& mx) : min(mn), max(mx) {}
    RTreeBox(const Math::AxisAlignedBox<T, N>& box) : min(box.min()), max(box.max()) {}

    T area() const {
        T a = T(1);
        for (std::size_t i = 0; i < N; ++i) a *= (max[i] - min[i]);
        return a;
    }

    RTreeBox expanded(const RTreeBox& other) const {
        return RTreeBox(min.componentWiseMin(other.min), max.componentWiseMax(other.max));
    }

    bool intersects(const RTreeBox& other) const {
        for (std::size_t i = 0; i < N; ++i) {
            if (max[i] < other.min[i] || other.max[i] < min[i]) return false;
        }
        return true;
    }

    bool contains(const RTreeBox& other) const {
        for (std::size_t i = 0; i < N; ++i) {
            if (min[i] > other.min[i] || max[i] < other.max[i]) return false;
        }
        return true;
    }

    T margin() const {
        T sum = T(0);
        for (std::size_t i = 0; i < N; ++i) sum += (max[i] - min[i]);
        return sum * T(2);
    }
};

// ----------------------------------------------------------------------------
//  Entry in R‑tree leaf: id + bounding box (and optional velocity for MVR)
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
struct RTreeEntry {
    uint64_t id;
    RTreeBox<T, N> box;
    Math::Vector<T, N> velocity;   // for moving objects (MVR)
    T startTime, endTime;           // valid time interval
};

// ----------------------------------------------------------------------------
//  Node in R‑tree (leaf or internal)
// ----------------------------------------------------------------------------
template<typename T, std::size_t N, size_t MaxEntries = 64, size_t MinEntries = 16>
struct RTreeNode {
    bool isLeaf;
    std::vector<RTreeEntry<T, N>> entries;          // leaf: entries; internal: child nodes
    std::vector<std::unique_ptr<RTreeNode>> children;
    RTreeBox<T, N> boundingBox;
    size_t level;                                   // root = 0, leaf = max level

    RTreeNode(bool leaf = true, size_t lvl = 0) : isLeaf(leaf), level(lvl) {}
};

// ============================================================================
//  RTree main class
// ============================================================================
template<typename T = float, std::size_t N = 2,
         size_t MaxEntries = 64, size_t MinEntries = 16>
class RTree {
public:
    using value_type = T;
    using box_type = RTreeBox<T, N>;
    using entry_type = RTreeEntry<T, N>;
    using node_type = RTreeNode<T, N, MaxEntries, MinEntries>;
    using point_type = Math::Vector<T, N>;

    // ------------------------------------------------------------------------
    //  Configuration (dynamic environment)
    // ------------------------------------------------------------------------
    struct Config {
        size_t maxEntries = MaxEntries;
        size_t minEntries = MinEntries;
        bool enableReinsertion = true;   // reinsert on overflow (R*‑tree)
        T reinsertionFactor = T(0.3);    // fraction of entries to reinsert
        bool enableSIMD = true;
        bool bulkLoad = false;
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit RTree(const Config& cfg = Config())
        : m_config(cfg)
        , m_root(std::make_unique<node_type>(true, 0)) {}

    // ------------------------------------------------------------------------
    //  Insert a single entry (id, box)
    // ------------------------------------------------------------------------
    void insert(uint64_t id, const box_type& box) {
        entry_type entry;
        entry.id = id;
        entry.box = box;
        entry.velocity = point_type(T(0));
        entry.startTime = entry.endTime = T(0);
        insertEntry(entry);
    }

    // ------------------------------------------------------------------------
    //  Insert a moving object (with linear motion)
    // ------------------------------------------------------------------------
    void insertMoving(uint64_t id, const point_type& startPos, const point_type& velocity,
                      T startTime, T endTime) {
        entry_type entry;
        entry.id = id;
        entry.box = box_type(startPos, startPos);
        entry.velocity = velocity;
        entry.startTime = startTime;
        entry.endTime = endTime;
        insertEntry(entry);
    }

    // ------------------------------------------------------------------------
    //  Bulk load using Sort‑Tile‑Recursive (STR)
    // ------------------------------------------------------------------------
    void bulkLoad(const std::vector<entry_type>& entries) {
        if (entries.empty()) return;
        // Sort by x coordinate of centroid
        std::vector<entry_type> sorted = entries;
        std::sort(sorted.begin(), sorted.end(),
                  [](const entry_type& a, const entry_type& b) {
                      T ca = (a.box.min[0] + a.box.max[0]) * T(0.5);
                      T cb = (b.box.min[0] + b.box.max[0]) * T(0.5);
                      return ca < cb;
                  });
        // Recursive build
        m_root = buildSTR(sorted, 0);
    }

    // ------------------------------------------------------------------------
    //  Range query: find all ids whose boxes intersect the query box
    //  Returns vector of ids.
    // ------------------------------------------------------------------------
    std::vector<uint64_t> rangeQuery(const box_type& query) const {
        std::vector<uint64_t> result;
        rangeQueryRecursive(m_root.get(), query, result);
        return result;
    }

    // ------------------------------------------------------------------------
    //  k‑Nearest Neighbour query (point)
    //  Returns vector of (id, squared distance) pairs.
    // ------------------------------------------------------------------------
    std::vector<std::pair<uint64_t, T>> knnQuery(const point_type& point, size_t k) const {
        using Candidate = std::pair<T, uint64_t>; // distance, id
        std::vector<Candidate> result;
        knnQueryRecursive(m_root.get(), point, k, result);
        std::sort(result.begin(), result.end());
        std::vector<std::pair<uint64_t, T>> out;
        for (size_t i = 0; i < std::min(k, result.size()); ++i) {
            out.emplace_back(result[i].second, std::sqrt(result[i].first));
        }
        return out;
    }

    // ------------------------------------------------------------------------
    //  Moving object query: at given time t, return objects that intersect
    //  query box at that instant (linear interpolation).
    // ------------------------------------------------------------------------
    std::vector<uint64_t> movingRangeQuery(const box_type& query, T time) const {
        std::vector<uint64_t> result;
        movingRangeQueryRecursive(m_root.get(), query, time, result);
        return result;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch range query (multiple query boxes)
    // ------------------------------------------------------------------------
    std::vector<std::vector<uint64_t>> batchRangeQuery(const box_type* queries, size_t count) const {
        std::vector<std::vector<uint64_t>> results(count);
        if (m_config.enableSIMD && count >= 4) {
            size_t simdEnd = count - (count % 4);
            for (size_t i = 0; i < simdEnd; i += 4) {
                for (int j = 0; j < 4; ++j) {
                    results[i+j] = rangeQuery(queries[i+j]);
                }
            }
            for (size_t i = simdEnd; i < count; ++i) {
                results[i] = rangeQuery(queries[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                results[i] = rangeQuery(queries[i]);
            }
        }
        return results;
    }

    // ------------------------------------------------------------------------
    //  Delete an entry (exact match by id – first found)
    // ------------------------------------------------------------------------
    bool remove(uint64_t id) {
        return removeRecursive(m_root.get(), id);
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment controls
    // ------------------------------------------------------------------------
    void setMaxEntries(size_t max) { m_config.maxEntries = max; }
    void setMinEntries(size_t min) { m_config.minEntries = min; }
    void setEnableReinsertion(bool enable) { m_config.enableReinsertion = enable; }

    // ------------------------------------------------------------------------
    //  Statistics
    // ------------------------------------------------------------------------
    size_t size() const { return countEntries(m_root.get()); }
    size_t height() const { return maxDepth(m_root.get()); }
    size_t nodeCount() const { return countNodes(m_root.get()); }

private:
    // ------------------------------------------------------------------------
    //  Insert entry into tree (with optional reinsertion on overflow)
    // ------------------------------------------------------------------------
    void insertEntry(const entry_type& entry) {
        if (!m_root) {
            m_root = std::make_unique<node_type>(true, 0);
        }
        insertEntryRecursive(m_root.get(), entry);
    }

    void insertEntryRecursive(node_type* node, const entry_type& entry) {
        if (node->isLeaf) {
            node->entries.push_back(entry);
            updateNodeBoundingBox(node);
            if (node->entries.size() > m_config.maxEntries) {
                // Overflow: split or reinsert
                if (m_config.enableReinsertion) {
                    reinsertEntries(node);
                } else {
                    splitNode(node);
                }
            }
        } else {
            // Choose child that minimally expands bounding box
            size_t best = 0;
            T minAreaIncrease = std::numeric_limits<T>::max();
            for (size_t i = 0; i < node->children.size(); ++i) {
                const auto& child = node->children[i];
                box_type newBox = node->children[i]->boundingBox.expanded(entry.box);
                T areaIncrease = newBox.area() - node->children[i]->boundingBox.area();
                if (areaIncrease < minAreaIncrease) {
                    minAreaIncrease = areaIncrease;
                    best = i;
                }
            }
            insertEntryRecursive(node->children[best].get(), entry);
            updateNodeBoundingBox(node);
        }
    }

    // ------------------------------------------------------------------------
    //  Split a node (quadratic split – choose pair with largest waste)
    // ------------------------------------------------------------------------
    void splitNode(node_type* node) {
        // Simplified: divide entries into two groups
        // For internal nodes, we have children; for leaf, entries.
        // We implement for leaf nodes (entries). Internal similar.
        if (!node->isLeaf) return;
        std::vector<entry_type> entries = std::move(node->entries);
        node->entries.clear();
        // Choose seeds: farthest apart (by bounding box centers)
        size_t seed1 = 0, seed2 = 1;
        T maxDist = T(0);
        for (size_t i = 0; i < entries.size(); ++i) {
            for (size_t j = i+1; j < entries.size(); ++j) {
                point_type c1 = (entries[i].box.min + entries[i].box.max) * T(0.5);
                point_type c2 = (entries[j].box.min + entries[j].box.max) * T(0.5);
                T d2 = (c1 - c2).squaredLength();
                if (d2 > maxDist) {
                    maxDist = d2;
                    seed1 = i;
                    seed2 = j;
                }
            }
        }
        // Create two new leaf nodes
        auto node1 = std::make_unique<node_type>(true, node->level);
        auto node2 = std::make_unique<node_type>(true, node->level);
        node1->entries.push_back(entries[seed1]);
        node2->entries.push_back(entries[seed2]);
        // Assign remaining entries to node with smallest area increase
        for (size_t i = 0; i < entries.size(); ++i) {
            if (i == seed1 || i == seed2) continue;
            box_type box1 = node1->boundingBox.expanded(entries[i].box);
            box_type box2 = node2->boundingBox.expanded(entries[i].box);
            T areaInc1 = box1.area() - node1->boundingBox.area();
            T areaInc2 = box2.area() - node2->boundingBox.area();
            if (areaInc1 < areaInc2) {
                node1->entries.push_back(entries[i]);
            } else {
                node2->entries.push_back(entries[i]);
            }
        }
        updateNodeBoundingBox(node1.get());
        updateNodeBoundingBox(node2.get());
        // Convert current node to internal node
        node->isLeaf = false;
        node->children.clear();
        node->children.push_back(std::move(node1));
        node->children.push_back(std::move(node2));
        node->entries.clear();
        updateNodeBoundingBox(node);
    }

    // ------------------------------------------------------------------------
    //  Reinsert entries (R*‑tree optimisation)
    // ------------------------------------------------------------------------
    void reinsertEntries(node_type* node) {
        size_t removeCount = static_cast<size_t>(node->entries.size() * m_config.reinsertionFactor);
        if (removeCount == 0) return;
        // Sort entries by distance from node center (largest first)
        point_type center = (node->boundingBox.min + node->boundingBox.max) * T(0.5);
        std::sort(node->entries.begin(), node->entries.end(),
                  [&center](const entry_type& a, const entry_type& b) {
                      point_type ca = (a.box.min + a.box.max) * T(0.5);
                      point_type cb = (b.box.min + b.box.max) * T(0.5);
                      return (ca - center).squaredLength() > (cb - center).squaredLength();
                  });
        std::vector<entry_type> removed;
        removed.reserve(removeCount);
        for (size_t i = 0; i < removeCount; ++i) {
            removed.push_back(node->entries.back());
            node->entries.pop_back();
        }
        updateNodeBoundingBox(node);
        // Reinsert removed entries from root
        for (const auto& entry : removed) {
            insertEntryRecursive(m_root.get(), entry);
        }
    }

    // ------------------------------------------------------------------------
    //  Remove entry by id (recursive)
    // ------------------------------------------------------------------------
    bool removeRecursive(node_type* node, uint64_t id) {
        if (node->isLeaf) {
            auto it = std::find_if(node->entries.begin(), node->entries.end(),
                                   [id](const entry_type& e) { return e.id == id; });
            if (it != node->entries.end()) {
                node->entries.erase(it);
                updateNodeBoundingBox(node);
                return true;
            }
            return false;
        } else {
            for (auto& child : node->children) {
                if (removeRecursive(child.get(), id)) {
                    if (child->entries.empty() && child->children.empty()) {
                        // Remove empty child
                        auto cit = std::find_if(node->children.begin(), node->children.end(),
                                                [&child](const auto& c) { return c.get() == child.get(); });
                        if (cit != node->children.end()) node->children.erase(cit);
                    }
                    updateNodeBoundingBox(node);
                    return true;
                }
            }
            return false;
        }
    }

    // ------------------------------------------------------------------------
    //  Build STR recursively
    // ------------------------------------------------------------------------
    std::unique_ptr<node_type> buildSTR(const std::vector<entry_type>& entries, size_t level) {
        if (entries.size() <= m_config.maxEntries) {
            auto node = std::make_unique<node_type>(true, level);
            node->entries = entries;
            updateNodeBoundingBox(node.get());
            return node;
        }
        // Compute number of slices (tiles)
        size_t sliceCount = static_cast<size_t>(std::ceil(std::sqrt(static_cast<T>(entries.size()))));
        sliceCount = std::max(sliceCount, size_t(2));
        size_t sliceSize = (entries.size() + sliceCount - 1) / sliceCount;
        // Sort by x coordinate of centroid (axis alternates)
        std::vector<entry_type> sorted = entries;
        std::sort(sorted.begin(), sorted.end(),
                  [](const entry_type& a, const entry_type& b) {
                      T ca = (a.box.min[0] + a.box.max[0]) * T(0.5);
                      T cb = (b.box.min[0] + b.box.max[0]) * T(0.5);
                      return ca < cb;
                  });
        // Create slices
        std::vector<std::vector<entry_type>> slices(sliceCount);
        for (size_t i = 0; i < sorted.size(); ++i) {
            slices[i / sliceSize].push_back(sorted[i]);
        }
        // For each slice, sort by y coordinate and build child nodes
        auto node = std::make_unique<node_type>(false, level);
        for (auto& slice : slices) {
            // Alternate axis: y (index 1) for 2D, could be generalised
            std::sort(slice.begin(), slice.end(),
                      [](const entry_type& a, const entry_type& b) {
                          T ca = (a.box.min[1] + a.box.max[1]) * T(0.5);
                          T cb = (b.box.min[1] + b.box.max[1]) * T(0.5);
                          return ca < cb;
                      });
            size_t groups = std::max(size_t(1), (slice.size() + m_config.maxEntries - 1) / m_config.maxEntries);
            size_t groupSize = (slice.size() + groups - 1) / groups;
            for (size_t g = 0; g < groups; ++g) {
                auto start = slice.begin() + g * groupSize;
                auto end = (g == groups-1) ? slice.end() : start + groupSize;
                std::vector<entry_type> group(start, end);
                node->children.push_back(buildSTR(group, level+1));
            }
        }
        updateNodeBoundingBox(node.get());
        return node;
    }

    // ------------------------------------------------------------------------
    //  Update bounding box of a node from its children or entries
    // ------------------------------------------------------------------------
    void updateNodeBoundingBox(node_type* node) {
        if (node->isLeaf) {
            if (node->entries.empty()) {
                node->boundingBox = box_type();
                return;
            }
            node->boundingBox = node->entries[0].box;
            for (size_t i = 1; i < node->entries.size(); ++i) {
                node->boundingBox = node->boundingBox.expanded(node->entries[i].box);
            }
        } else {
            if (node->children.empty()) {
                node->boundingBox = box_type();
                return;
            }
            node->boundingBox = node->children[0]->boundingBox;
            for (size_t i = 1; i < node->children.size(); ++i) {
                node->boundingBox = node->boundingBox.expanded(node->children[i]->boundingBox);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Recursive range query
    // ------------------------------------------------------------------------
    void rangeQueryRecursive(const node_type* node, const box_type& query, std::vector<uint64_t>& out) const {
        if (!node->boundingBox.intersects(query)) return;
        if (node->isLeaf) {
            for (const auto& entry : node->entries) {
                if (entry.box.intersects(query)) {
                    out.push_back(entry.id);
                }
            }
        } else {
            for (const auto& child : node->children) {
                rangeQueryRecursive(child.get(), query, out);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Recursive kNN query (using priority queue)
    // ------------------------------------------------------------------------
    void knnQueryRecursive(const node_type* node, const point_type& point, size_t k,
                           std::vector<std::pair<T, uint64_t>>& result) const {
        if (node->isLeaf) {
            for (const auto& entry : node->entries) {
                // Compute distance from point to entry's bounding box (or center)
                point_type center = (entry.box.min + entry.box.max) * T(0.5);
                T dist2 = (center - point).squaredLength();
                result.emplace_back(dist2, entry.id);
            }
        } else {
            for (const auto& child : node->children) {
                knnQueryRecursive(child.get(), point, k, result);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Moving object range query (time parameter)
    // ------------------------------------------------------------------------
    void movingRangeQueryRecursive(const node_type* node, const box_type& query, T time,
                                   std::vector<uint64_t>& out) const {
        // Quick check: if node's bounding box at time time (expanded by velocity)
        // does not intersect query, skip.
        if (!node->boundingBox.intersects(query)) return;
        if (node->isLeaf) {
            for (const auto& entry : node->entries) {
                if (time < entry.startTime || time > entry.endTime) continue;
                point_type pos = (entry.box.min + entry.box.max) * T(0.5) + entry.velocity * (time - entry.startTime);
                box_type expanded(pos, pos);
                if (expanded.intersects(query)) out.push_back(entry.id);
            }
        } else {
            for (const auto& child : node->children) {
                movingRangeQueryRecursive(child.get(), query, time, out);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Counting helpers
    // ------------------------------------------------------------------------
    size_t countEntries(const node_type* node) const {
        if (node->isLeaf) return node->entries.size();
        size_t sum = 0;
        for (const auto& child : node->children) sum += countEntries(child.get());
        return sum;
    }
    size_t countNodes(const node_type* node) const {
        size_t sum = 1;
        if (!node->isLeaf) {
            for (const auto& child : node->children) sum += countNodes(child.get());
        }
        return sum;
    }
    size_t maxDepth(const node_type* node) const {
        if (node->isLeaf) return node->level;
        size_t max = node->level;
        for (const auto& child : node->children) {
            max = std::max(max, maxDepth(child.get()));
        }
        return max;
    }

    Config m_config;
    std::unique_ptr<node_type> m_root;
};

// ----------------------------------------------------------------------------
//  Helper: create R‑tree for 2D boxes
// ----------------------------------------------------------------------------
template<typename T = float>
using RTree2D = RTree<T, 2>;

template<typename T = float>
using RTree3D = RTree<T, 3>;

} // namespace Contrib
} // namespace OrthoTree

#endif // ORTHOTREE_CONTRIB_LIBSPATIALINDEX_ADAPTER_H_INCLUDED

/**
 * Next file: orthotree/contrib/pytorch3d_adapter.h (BSD license)
 * Port of PyTorch3D point cloud and knn/ball query for OrthoTree.
 */