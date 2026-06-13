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

#ifndef ORTHOTREE_CORE_PARTITIONING_SCALE_ADAPTIVE_OCTREE_H_INCLUDED
#define ORTHOTREE_CORE_PARTITIONING_SCALE_ADAPTIVE_OCTREE_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/geometry_queries.h"
#include "../../core/math/numerical_methods.h"
#include "../../core/configuration.h"
#include "../../detail/common.h"
#include "../../detail/inplace_vector.h"
#include "../../detail/memory_resource.h"
#include "../../detail/si_morton.h"
#include "../morton/morton_128bit.h"
#include "../morton/hierarchical_morton_key.h"

#include <array>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>

namespace OrthoTree {
namespace Partitioning {

// ============================================================================
//  ScaleAdaptiveOctree: octree that adapts cell size based on local density
//  and distance from viewer (LOD). Supports both 2D and 3D.
// ============================================================================
template<Dimension Dim, typename T = float,
         typename Allocator = PMRAllocator<std::byte>>
class ScaleAdaptiveOctree {
public:
    using point_type = Math::Vector<T, Dim>;
    using aabb_type = Math::AxisAlignedBox<T, Dim>;
    using morton_type = Morton::HierarchicalMortonKey;
    using size_type = std::size_t;
    using entity_type = uint32_t;  // entity ID (user manages actual data)

    static constexpr size_type MAX_CHILDREN = (Dim == Dim2) ? 4 : 8;

    // ------------------------------------------------------------------------
    //  Node structure with adaptive scale
    // ------------------------------------------------------------------------
    struct Node {
        aabb_type bounds;                     // world bounds of this node
        morton_type key;                      // hierarchical key (scale + code)
        uint32_t children[MAX_CHILDREN];      // child node indices (or INVALID)
        uint32_t firstEntity;                 // start index in global entity list
        uint32_t entityCount;                 // number of entities in this leaf
        uint8_t depth;                        // tree depth (0 = root)
        bool isLeaf : 1;
        bool hasEntities : 1;

        static constexpr uint32_t INVALID = ~0u;

        Node() : bounds(), key(), children{INVALID}, firstEntity(0),
                 entityCount(0), depth(0), isLeaf(true), hasEntities(false) {}
    };

    // ------------------------------------------------------------------------
    //  Construction and configuration
    // ------------------------------------------------------------------------
    ScaleAdaptiveOctree(const aabb_type& worldBounds,
                        uint8_t maxScale = 20,
                        size_type bucketSize = 8,
                        const Allocator& alloc = Allocator())
        : m_alloc(alloc)
        , m_worldBounds(worldBounds)
        , m_maxScale(maxScale)
        , m_bucketSize(bucketSize)
        , m_root(INVALID_NODE)
        , m_nodeCount(0)
        , m_entityCount(0)
        , m_densityMap(alloc)
        , m_hotspotTracker(alloc) {
        // Create root node at scale = 0 (largest cell)
        createRoot();
    }

    // ------------------------------------------------------------------------
    //  Insertion / update / removal
    // ------------------------------------------------------------------------
    bool insert(entity_type entity, const aabb_type& bounds) {
        point_type center = bounds.center();
        uint8_t optimalScale = computeOptimalScale(center, bounds.extents().maxComponent());
        morton_type key = pointToKey(center, optimalScale);
        NodeIndex nodeIdx = findOrCreateNode(key, optimalScale);
        Node& node = m_nodes[nodeIdx];
        if (node.entityCount >= m_bucketSize && node.isLeaf && node.depth < m_maxScale) {
            splitNode(nodeIdx);
            node = m_nodes[nodeIdx];  // refresh after split (node may have become internal)
        }
        // Insert entity into leaf (may be the same node or new leaf after split)
        // Re‑find the correct leaf after split
        NodeIndex leafIdx = findLeafNode(key);
        return insertIntoLeaf(leafIdx, entity, bounds);
    }

    bool remove(entity_type entity) {
        // For simplicity, find entity in all leaves (expensive, but acceptable for sparse)
        for (auto& node : m_nodes) {
            if (!node.isLeaf) continue;
            for (size_type i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
                if (m_entities[i] == entity) {
                    // remove and compact
                    m_entities[i] = m_entities.back();
                    m_entities.pop_back();
                    --node.entityCount;
                    --m_entityCount;
                    // If node becomes empty and not root, try to merge
                    if (node.entityCount == 0 && node.depth > 0) {
                        tryMerge(node);
                    }
                    return true;
                }
            }
        }
        return false;
    }

    bool update(entity_type entity, const aabb_type& oldBounds, const aabb_type& newBounds) {
        if (remove(entity)) {
            return insert(entity, newBounds);
        }
        return false;
    }

    void clear() {
        m_nodes.clear();
        m_entities.clear();
        m_densityMap.clear();
        m_hotspotTracker.clear();
        m_nodeCount = 0;
        m_entityCount = 0;
        createRoot();
    }

    // ------------------------------------------------------------------------
    //  Queries (range, point, ray)
    // ------------------------------------------------------------------------
    template<typename OutputIt>
    size_type queryPoint(const point_type& point, OutputIt out) const {
        size_type count = 0;
        morton_type key = pointToKey(point, 0);  // start from any scale
        NodeIndex idx = findLeafNode(key);
        if (idx != INVALID_NODE) {
            const Node& node = m_nodes[idx];
            for (size_type i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
                // caller can test precise geometry; we just return all entities in leaf
                *out++ = m_entities[i];
                ++count;
            }
        }
        return count;
    }

    template<typename OutputIt>
    size_type queryBox(const aabb_type& box, OutputIt out) const {
        size_type count = 0;
        traverseBox(box, [&](const Node& node) {
            if (node.isLeaf) {
                for (size_type i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
                    // In a real implementation, we would test entity AABB vs box.
                    // Here we simply return all entities in overlapping leaves.
                    *out++ = m_entities[i];
                    ++count;
                }
            }
        });
        return count;
    }

    // Ray cast (simplified)
    std::optional<entity_type> raycast(const Math::Ray<T, Dim>& ray, T& tOut) const {
        T closest = std::numeric_limits<T>::max();
        std::optional<entity_type> hit;
        traverseRay(ray, [&](const Node& node) -> bool {
            T tMin, tMax;
            if (!node.bounds.intersectRay(ray.origin(), ray.direction(), tMin, tMax))
                return false; // no intersection, skip node
            if (tMin > closest) return false;
            if (node.isLeaf) {
                for (size_type i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
                    // Placeholder: assume entity is a point at node center for demo
                    // In real code, test exact entity geometry.
                    // We'll just use node center as hit point.
                    point_type center = node.bounds.center();
                    Math::Ray<T, Dim> dummyRay(ray.origin(), ray.direction());
                    T t;
                    if (dummyRay.pointAt(t) == center) { // not correct; just placeholder
                        if (t < closest) {
                            closest = t;
                            hit = m_entities[i];
                        }
                    }
                }
                return true; // continue traversal? Actually leaf, no children.
            }
            return true; // continue to children
        });
        if (hit) tOut = closest;
        return hit;
    }

    // ------------------------------------------------------------------------
    //  Statistics and introspection
    // ------------------------------------------------------------------------
    size_type size() const noexcept { return m_entityCount; }
    size_type nodeCount() const noexcept { return m_nodeCount; }
    size_type memoryUsage() const noexcept {
        return m_nodes.capacity() * sizeof(Node) +
               m_entities.capacity() * sizeof(entity_type) +
               m_densityMap.size() * sizeof(typename DensityMap::value_type) +
               m_hotspotTracker.size() * sizeof(typename HotspotMap::value_type);
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control (hotspot tracking, density‑based refinement)
    // ------------------------------------------------------------------------
    void recordQuery(const point_type& point) {
        morton_type key = pointToKey(point, 0);
        m_hotspotTracker[key] = m_hotspotTracker[key] + 1;
    }

    void updateDensity(const point_type& point, T density) {
        morton_type key = pointToKey(point, 0);
        m_densityMap[key] = density;
    }

    void adaptiveRefine(T densityThreshold) {
        // Refine nodes where density > threshold
        for (auto& [key, density] : m_densityMap) {
            if (density > densityThreshold) {
                NodeIndex idx = findNodeByKey(key);
                if (idx != INVALID_NODE && m_nodes[idx].isLeaf && m_nodes[idx].depth < m_maxScale) {
                    splitNode(idx);
                }
            }
        }
    }

private:
    using NodeIndex = uint32_t;
    static constexpr NodeIndex INVALID_NODE = NodeIndex(-1);
    using NodeVector = std::vector<Node, typename Allocator::template rebind<Node>::other>;
    using EntityVector = std::vector<entity_type, typename Allocator::template rebind<entity_type>::other>;
    using DensityMap = std::unordered_map<morton_type, T,
                                          std::hash<morton_type>, std::equal_to<morton_type>,
                                          typename Allocator::template rebind<std::pair<const morton_type, T>>::other>;
    using HotspotMap = std::unordered_map<morton_type, uint32_t,
                                          std::hash<morton_type>, std::equal_to<morton_type>,
                                          typename Allocator::template rebind<std::pair<const morton_type, uint32_t>>::other>;

    void createRoot() {
        Node root;
        root.bounds = m_worldBounds;
        root.key = morton_type(0, 0);
        root.depth = 0;
        root.isLeaf = true;
        m_nodes.push_back(root);
        m_root = 0;
        m_nodeCount = 1;
    }

    morton_type pointToKey(const point_type& point, uint8_t scale) const {
        // Normalize point to [0,1] range
        point_type t = (point - m_worldBounds.min()) / m_worldBounds.extents();
        // Quantize to 2^scale cells per dimension
        uint64_t maxCoord = (scale < 64) ? (uint64_t(1) << scale) : ~uint64_t(0);
        uint64_t ix = static_cast<uint64_t>(t[0] * static_cast<T>(maxCoord - 1));
        uint64_t iy = static_cast<uint64_t>(t[1] * static_cast<T>(maxCoord - 1));
        uint64_t iz = (Dim == Dim2) ? 0 : static_cast<uint64_t>(t[2] * static_cast<T>(maxCoord - 1));
        Morton::Morton128Bit mort;
        if constexpr (Dim == Dim2) mort.encode2D(ix, iy);
        else mort.encode3D(ix, iy, iz);
        // Shift to scale
        uint128_t shifted = mort.code() >> ((Morton::HierarchicalMortonKey::MAX_SCALE - scale) * Morton::HierarchicalMortonKey::dimensionBits(Dim));
        return morton_type(scale, shifted);
    }

    uint8_t computeOptimalScale(const point_type& center, T entitySize) const {
        // Based on distance from viewer? For now, use entity size relative to world.
        T worldSize = m_worldBounds.extents().maxComponent();
        T relativeSize = entitySize / worldSize;
        if (relativeSize <= T(0)) return m_maxScale;
        int scaleEst = static_cast<int>(std::log2(T(1) / relativeSize));
        return static_cast<uint8_t>(Math::clamp(scaleEst, 0, static_cast<int>(m_maxScale)));
    }

    NodeIndex findOrCreateNode(const morton_type& key, uint8_t targetScale) {
        // Traverse from root down to target scale, creating missing nodes.
        NodeIndex current = m_root;
        for (uint8_t depth = 0; depth < targetScale; ++depth) {
            uint8_t childIdx = key.childIndexAtDepth(depth);
            Node& node = m_nodes[current];
            if (node.children[childIdx] == Node::INVALID) {
                // create child
                Node child;
                child.depth = depth + 1;
                child.isLeaf = true;
                child.bounds = computeChildBounds(node.bounds, childIdx);
                child.key = key.parentAtDepth(depth); // approximate
                child.children.fill(Node::INVALID);
                child.firstEntity = 0;
                child.entityCount = 0;
                NodeIndex newIdx = static_cast<NodeIndex>(m_nodes.size());
                m_nodes.push_back(child);
                m_nodeCount++;
                node.children[childIdx] = newIdx;
            }
            current = node.children[childIdx];
        }
        return current;
    }

    NodeIndex findLeafNode(const morton_type& key) const {
        NodeIndex current = m_root;
        while (current != INVALID_NODE) {
            const Node& node = m_nodes[current];
            if (node.isLeaf) return current;
            uint8_t childIdx = key.childIndexAtDepth(node.depth);
            current = node.children[childIdx];
        }
        return INVALID_NODE;
    }

    NodeIndex findNodeByKey(const morton_type& key) const {
        NodeIndex current = m_root;
        for (uint8_t depth = 0; depth <= key.scale(); ++depth) {
            if (current == INVALID_NODE) break;
            const Node& node = m_nodes[current];
            if (depth == key.scale()) return current;
            uint8_t childIdx = key.childIndexAtDepth(depth);
            current = node.children[childIdx];
        }
        return INVALID_NODE;
    }

    aabb_type computeChildBounds(const aabb_type& parent, uint8_t childIdx) const {
        point_type min = parent.min();
        point_type max = parent.max();
        point_type mid = parent.center();
        for (size_t d = 0; d < static_cast<size_t>(Dim); ++d) {
            bool high = (childIdx >> d) & 1;
            if (high) min[d] = mid[d];
            else max[d] = mid[d];
        }
        return aabb_type(min, max);
    }

    bool insertIntoLeaf(NodeIndex leafIdx, entity_type entity, const aabb_type&) {
        Node& leaf = m_nodes[leafIdx];
        if (leaf.entityCount == 0) {
            leaf.firstEntity = static_cast<uint32_t>(m_entities.size());
        }
        // For simplicity, we assume contiguous append; in production we might need to handle gaps.
        m_entities.push_back(entity);
        leaf.entityCount++;
        m_entityCount++;
        return true;
    }

    void splitNode(NodeIndex nodeIdx) {
        Node& node = m_nodes[nodeIdx];
        if (!node.isLeaf) return;
        // Create children
        for (uint8_t i = 0; i < MAX_CHILDREN; ++i) {
            Node child;
            child.depth = node.depth + 1;
            child.isLeaf = true;
            child.bounds = computeChildBounds(node.bounds, i);
            child.key = node.key.child(i);
            child.children.fill(Node::INVALID);
            child.firstEntity = 0;
            child.entityCount = 0;
            NodeIndex childIdx = static_cast<NodeIndex>(m_nodes.size());
            m_nodes.push_back(child);
            m_nodeCount++;
            node.children[i] = childIdx;
        }
        node.isLeaf = false;
        // Redistribute entities to children
        for (size_type i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
            entity_type ent = m_entities[i];
            // Find which child contains entity (approximate using its bounds; here we use point center)
            // In real code, we would store entity bounds. For simplicity, we use node center.
            point_type center = node.bounds.center(); // placeholder
            uint8_t childIdx = getChildIndexForPoint(node.bounds, center);
            Node& child = m_nodes[node.children[childIdx]];
            if (child.entityCount == 0) child.firstEntity = i;
            // We can't simply move because entity list is contiguous. We'll rebuild leaf later.
            // For now, we leave entities in parent and mark parent as internal; queries must descend.
        }
        // Clear entity list from parent (they are now in children)
        node.firstEntity = 0;
        node.entityCount = 0;
    }

    uint8_t getChildIndexForPoint(const aabb_type& box, const point_type& point) const {
        point_type mid = box.center();
        uint8_t idx = 0;
        for (size_t d = 0; d < static_cast<size_t>(Dim); ++d) {
            if (point[d] >= mid[d]) idx |= (1 << d);
        }
        return idx;
    }

    void tryMerge(Node& node) {
        if (node.depth == 0) return; // root
        // Check siblings: if all children are leaves and have zero entities, merge.
        // Not implemented fully.
    }

    template<typename Func>
    void traverseBox(const aabb_type& box, Func&& func) const {
        std::vector<NodeIndex> stack;
        stack.push_back(m_root);
        while (!stack.empty()) {
            NodeIndex idx = stack.back();
            stack.pop_back();
            const Node& node = m_nodes[idx];
            if (!node.bounds.overlaps(box)) continue;
            func(node);
            if (!node.isLeaf) {
                for (uint32_t child : node.children) {
                    if (child != Node::INVALID) stack.push_back(child);
                }
            }
        }
    }

    template<typename Func>
    void traverseRay(const Math::Ray<T, Dim>& ray, Func&& func) const {
        // Simple recursive traversal; for performance we would use priority queue.
        std::vector<NodeIndex> stack;
        stack.push_back(m_root);
        while (!stack.empty()) {
            NodeIndex idx = stack.back();
            stack.pop_back();
            const Node& node = m_nodes[idx];
            T tMin, tMax;
            if (!node.bounds.intersectRay(ray.origin(), ray.direction(), tMin, tMax))
                continue;
            if (!func(node)) continue;
            if (!node.isLeaf) {
                // In production, sort children by tMin for better order.
                for (uint32_t child : node.children) {
                    if (child != Node::INVALID) stack.push_back(child);
                }
            }
        }
    }

    Allocator m_alloc;
    aabb_type m_worldBounds;
    uint8_t m_maxScale;
    size_type m_bucketSize;
    NodeIndex m_root;
    NodeVector m_nodes;
    EntityVector m_entities;
    size_type m_nodeCount;
    size_type m_entityCount;
    DensityMap m_densityMap;
    HotspotMap m_hotspotTracker;
};

} // namespace Partitioning
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_PARTITIONING_SCALE_ADAPTIVE_OCTREE_H_INCLUDED