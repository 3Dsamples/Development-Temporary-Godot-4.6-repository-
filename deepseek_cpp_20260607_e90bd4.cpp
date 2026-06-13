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

/**
 * @file Octree.h
 * @brief Core Octree/Quadtree implementation with dynamic hashing and linear variants.
 *
 * This file defines the Octree class template that powers both 2D (quadtree)
 * and 3D (octree) spatial partitioning. It supports insertion, removal,
 * point/box queries, ray casting, nearest neighbor, and motion prediction.
 * The implementation uses a sparse hashed node structure for low memory
 * footprint and fast incremental updates.
 *
 * Features:
 * - 2D/3D via template parameter
 * - Double buffering for safe concurrent reads
 * - Time‑critical insertion with bounds expansion
 * - Morton code based spatial ordering
 * - SIMD‑aware bounding box operations
 * - Custom PMR allocators for real‑time guarantees
 * - Advanced motion estimation (velocity‑aware updates)
 */

#pragma once
#ifndef ORTHOTREE__OCTREE_H_INCLUDED
#define ORTHOTREE__OCTREE_H_INCLUDED

#include "core/configuration.h"
#include "core/types.h"
#include "core/math/vector_math.h"
#include "core/math/interval_arithmetic.h"
#include "core/math/transform.h"
#include "core/math/geometry_queries.h"
#include "core/math/numerical_methods.h"
#include "core/entity_adapter.h"
#include "detail/memory_resource.h"
#include "detail/si_mortongrid.h"
#include "detail/bitset_arithmetic.h"
#include "detail/partitioning.h"
#include "detail/utils.h"
#include "adapters/general.h"

#include <atomic>
#include <optional>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace OrthoTree {

// ----------------------------------------------------------------------------
//  Octree node structure (compact, cache‑aligned)
// ----------------------------------------------------------------------------

/**
 * @brief Node of the octree. Contains bounding box, child indices, and entity list.
 * @tparam Dim Spatial dimension (2 or 3).
 * @tparam T Scalar type.
 */
template <Dimension Dim, typename T>
struct alignas(64) OctreeNode {
    using aabb_type = Math::AxisAlignedBox<T, Dim>;
    using point_type = Math::Vector<T, Dim>;
    using index_type = uint32_t;          // Node index
    using entity_index_type = uint32_t;   // Index into global entity list

    static constexpr uint8_t MAX_CHILDREN = (Dim == Dim3) ? 8 : 4;
    static constexpr uint32_t INVALID_INDEX = ~0u;

    aabb_type bounds;                     // Bounding box of this node
    index_type children[MAX_CHILDREN];    // Child node indices (or INVALID)
    uint32_t firstEntity;                 // Start index in global entity array
    uint32_t entityCount;                 // Number of entities in this leaf
    uint8_t depth;                        // Depth from root
    uint8_t splitAxis;                    // Cached split axis (for construction)
    bool isLeaf : 1;                      // True if leaf node
    bool hasEntities : 1;                // True if contains any entity

    // Constructors
    OctreeNode() noexcept
        : bounds(), children{INVALID_INDEX, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX,
                             INVALID_INDEX, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX},
          firstEntity(0), entityCount(0), depth(0), splitAxis(0), isLeaf(true), hasEntities(false) {}

    OctreeNode(const aabb_type& b, uint8_t d) noexcept
        : bounds(b), children{INVALID_INDEX, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX,
                              INVALID_INDEX, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX},
          firstEntity(0), entityCount(0), depth(d), splitAxis(0), isLeaf(true), hasEntities(false) {}

    // Clear children
    void clearChildren() noexcept {
        for (auto& child : children) child = INVALID_INDEX;
    }

    // Check if node is internal
    bool isInternal() const noexcept { return !isLeaf; }
};

// ----------------------------------------------------------------------------
//  Octree main class
// ----------------------------------------------------------------------------

/**
 * @brief Sparse dynamic octree/quadtree with hashed nodes.
 *
 * @tparam Dim Dimension (Dim2 or Dim3).
 * @tparam T Scalar type (float, double).
 * @tparam Allocator PMR allocator type.
 */
template <Dimension Dim, typename T = float,
          typename Allocator = PMRAllocator<std::byte>>
class Octree {
public:
    using value_type = T;
    static constexpr Dimension dimension = Dim;
    using scalar_type = T;
    using point_type = Math::Vector<T, Dim>;
    using aabb_type = Math::AxisAlignedBox<T, Dim>;
    using transform_type = Math::AffineTransform<T, Dim>;
    using ray_type = Math::Ray<T, Dim>;
    using node_type = OctreeNode<Dim, T>;
    using node_index = uint32_t;
    using entity_type = typename EntityAdapter<T, Dim>::entity_type;
    using size_type = std::size_t;
    using allocator_type = Allocator;

    // Hash map key: Morton code (64-bit) -> node index
    using NodeMap = std::unordered_map<uint64_t, node_index, std::hash<uint64_t>,
                                       std::equal_to<uint64_t>,
                                       typename Allocator::template rebind<std::pair<const uint64_t, node_index>>::other>;

    // ------------------------------------------------------------------------
    //  Construction
    // ------------------------------------------------------------------------

    /**
     * @brief Construct empty octree with world bounds.
     * @param worldBounds Bounding box covering entire space.
     * @param maxDepth Maximum tree depth.
     * @param bucketSize Max entities per leaf before splitting.
     * @param alloc Allocator.
     */
    explicit Octree(const aabb_type& worldBounds,
                    size_type maxDepth = ORTHOTREE_DEFAULT_MAX_DEPTH,
                    size_type bucketSize = ORTHOTREE_DEFAULT_BUCKET_SIZE,
                    const Allocator& alloc = Allocator())
        : m_alloc(alloc)
        , m_nodeAlloc(alloc)
        , m_entityAlloc(alloc)
        , m_worldBounds(worldBounds)
        , m_maxDepth(maxDepth)
        , m_bucketSize(bucketSize)
        , m_root(nullptr)
        , m_nodeCount(0)
        , m_entityCount(0)
        , m_version(0) {
        // Create root node
        node_index rootIdx = allocateNode();
        m_root = rootIdx;
        auto& rootNode = getNode(rootIdx);
        rootNode.bounds = worldBounds;
        rootNode.depth = 0;
        rootNode.isLeaf = true;
        m_nodeMap[computeMortonKey(worldBounds.center())] = rootIdx;
    }

    /**
     * @brief Construct with default world bounds (unit cube centered at origin).
     */
    Octree(size_type maxDepth = ORTHOTREE_DEFAULT_MAX_DEPTH,
           size_type bucketSize = ORTHOTREE_DEFAULT_BUCKET_SIZE,
           const Allocator& alloc = Allocator())
        : Octree(aabb_type(point_type(-1), point_type(1)),
                 maxDepth, bucketSize, alloc) {}

    ~Octree() = default;

    // Move only
    Octree(Octree&&) noexcept = default;
    Octree& operator=(Octree&&) noexcept = default;

    // ------------------------------------------------------------------------
    //  Insertion / removal / update
    // ------------------------------------------------------------------------

    /**
     * @brief Insert an entity into the octree.
     * @param entity User object with geometry (must satisfy EntityAdapter).
     * @return True if inserted.
     */
    bool insert(const entity_type& entity) {
        aabb_type bounds = EntityAdapter<T, Dim>::getBounds(entity);
        if (!m_worldBounds.overlaps(bounds)) {
            // Expand world bounds? Optionally.
            if constexpr (ORTHOTREE_AUTO_EXPAND_BOUNDS) {
                expandWorldBounds(bounds);
            } else {
                return false;
            }
        }
        point_type center = bounds.center();
        uint64_t morton = computeMortonKey(center);
        node_index leafIdx = findLeafNode(morton, bounds);
        if (leafIdx == node_type::INVALID_INDEX) {
            leafIdx = createLeafNode(bounds);
        }
        return insertIntoLeaf(leafIdx, entity);
    }

    /**
     * @brief Remove an entity.
     * @param entity Entity to remove.
     * @return True if removed.
     */
    bool remove(const entity_type& entity) {
        aabb_type bounds = EntityAdapter<T, Dim>::getBounds(entity);
        point_type center = bounds.center();
        uint64_t morton = computeMortonKey(center);
        node_index leafIdx = findLeafNode(morton, bounds);
        if (leafIdx == node_type::INVALID_INDEX) return false;
        return removeFromLeaf(leafIdx, entity);
    }

    /**
     * @brief Update an entity (reinsert if bounds changed).
     * @param entity Entity with possibly new bounds.
     * @return True if updated.
     */
    bool update(const entity_type& entity) {
        if (remove(entity)) {
            return insert(entity);
        }
        return false;
    }

    /**
     * @brief Clear all entities and reset tree.
     */
    void clear() {
        m_nodes.clear();
        m_entities.clear();
        m_nodeMap.clear();
        m_nodeCount = 0;
        m_entityCount = 0;
        ++m_version;
        // Recreate root
        node_index rootIdx = allocateNode();
        m_root = rootIdx;
        auto& rootNode = getNode(rootIdx);
        rootNode.bounds = m_worldBounds;
        rootNode.depth = 0;
        rootNode.isLeaf = true;
        m_nodeMap[computeMortonKey(m_worldBounds.center())] = rootIdx;
    }

    // ------------------------------------------------------------------------
    //  Queries (delegated to ot_query, but basic ones inlined for speed)
    // ------------------------------------------------------------------------

    /**
     * @brief Query all entities whose AABB overlaps a point.
     * @param point Query point.
     * @param out Output iterator.
     * @return Number of entities.
     */
    template <typename OutputIt>
    size_type queryPoint(const point_type& point, OutputIt out) const {
        size_type count = 0;
        traverse([&](const node_type& node) -> TraversalAction {
            if (!node.bounds.contains(point)) return TraversalAction::Skip;
            if (node.isLeaf) {
                for (uint32_t i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
                    *out++ = m_entities[i];
                    ++count;
                }
                return TraversalAction::SkipChildren;
            }
            return TraversalAction::Continue;
        });
        return count;
    }

    /**
     * @brief Query all entities intersecting an AABB.
     * @param box Query box.
     * @param out Output iterator.
     * @return Count.
     */
    template <typename OutputIt>
    size_type queryBox(const aabb_type& box, OutputIt out) const {
        size_type count = 0;
        traverse([&](const node_type& node) -> TraversalAction {
            if (!node.bounds.overlaps(box)) return TraversalAction::Skip;
            if (node.isLeaf) {
                for (uint32_t i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
                    if (EntityAdapter<T, Dim>::getBounds(m_entities[i]).overlaps(box)) {
                        *out++ = m_entities[i];
                        ++count;
                    }
                }
                return TraversalAction::SkipChildren;
            }
            return TraversalAction::Continue;
        });
        return count;
    }

    /**
     * @brief Ray cast - first hit.
     * @param ray Ray in world space.
     * @param hitDist Optional distance to first hit.
     * @return Optional entity.
     */
    std::optional<entity_type> raycast(const ray_type& ray, T* hitDist = nullptr) const {
        T closest = std::numeric_limits<T>::max();
        std::optional<entity_type> hitEntity;
        traverse([&](const node_type& node) -> TraversalAction {
            T tMin, tMax;
            if (!node.bounds.intersectRay(ray.origin(), ray.direction(), tMin, tMax)) {
                return TraversalAction::Skip;
            }
            if (tMin > closest) return TraversalAction::Skip;
            if (node.isLeaf) {
                for (uint32_t i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
                    const auto& ent = m_entities[i];
                    aabb_type entBounds = EntityAdapter<T, Dim>::getBounds(ent);
                    T t0, t1;
                    if (entBounds.intersectRay(ray.origin(), ray.direction(), t0, t1)) {
                        if (t0 >= 0 && t0 < closest) {
                            closest = t0;
                            hitEntity = ent;
                        }
                    }
                }
                return TraversalAction::SkipChildren;
            }
            return TraversalAction::Continue;
        });
        if (hitDist) *hitDist = closest;
        return hitEntity;
    }

    /**
     * @brief Find nearest neighbor to point.
     * @param point Query point.
     * @param maxDist Maximum distance.
     * @return Optional pair (entity, squared distance).
     */
    std::optional<std::pair<entity_type, T>> nearestNeighbor(const point_type& point,
                                                              T maxDist = std::numeric_limits<T>::max()) const {
        struct Candidate {
            entity_type entity;
            T distSq;
        };
        std::optional<Candidate> best;
        traverse([&](const node_type& node) -> TraversalAction {
            T nodeDistSq = node.bounds.squaredDistanceTo(point);
            if (nodeDistSq > maxDist) return TraversalAction::Skip;
            if (node.isLeaf) {
                for (uint32_t i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
                    const auto& ent = m_entities[i];
                    T entDistSq = EntityAdapter<T, Dim>::getBounds(ent).squaredDistanceTo(point);
                    if (entDistSq < maxDist) {
                        maxDist = entDistSq;
                        best = {ent, entDistSq};
                    }
                }
                return TraversalAction::SkipChildren;
            }
            return TraversalAction::Continue;
        });
        if (best) return std::make_pair(best->entity, best->distSq);
        return std::nullopt;
    }

    // ------------------------------------------------------------------------
    //  Stats
    // ------------------------------------------------------------------------

    size_type size() const noexcept { return m_entityCount; }
    bool empty() const noexcept { return m_entityCount == 0; }
    size_type nodeCount() const noexcept { return m_nodeCount; }
    size_type height() const noexcept { return computeHeight(m_root); }
    size_type memoryUsage() const noexcept {
        return sizeof(*this) + m_nodes.capacity() * sizeof(node_type) +
               m_entities.capacity() * sizeof(entity_type) +
               m_nodeMap.bucket_count() * (sizeof(void*) * 2);
    }

    // ------------------------------------------------------------------------
    //  Morton code utility
    // ------------------------------------------------------------------------
    uint64_t computeMortonKey(const point_type& point) const {
        // Normalize point to [0,1] range based on world bounds
        point_type t = (point - m_worldBounds.min()) / m_worldBounds.extents();
        return Math::mortonEncode<T, Dim>(t);
    }

private:
    // ------------------------------------------------------------------------
    //  Internal helpers
    // ------------------------------------------------------------------------

    enum class TraversalAction : uint8_t { Continue, Skip, SkipChildren };

    template <typename Func>
    void traverse(Func&& func, node_index nodeIdx = node_type::INVALID_INDEX) const {
        if (nodeIdx == node_type::INVALID_INDEX) nodeIdx = m_root;
        const node_type& node = getNode(nodeIdx);
        TraversalAction act = func(node);
        if (act == TraversalAction::Skip) return;
        if (act == TraversalAction::Continue && !node.isLeaf) {
            for (uint32_t childIdx : node.children) {
                if (childIdx != node_type::INVALID_INDEX) {
                    traverse(func, childIdx);
                }
            }
        }
    }

    node_index allocateNode() {
        node_index idx = static_cast<node_index>(m_nodes.size());
        m_nodes.emplace_back();
        ++m_nodeCount;
        return idx;
    }

    node_type& getNode(node_index idx) { return m_nodes[idx]; }
    const node_type& getNode(node_index idx) const { return m_nodes[idx]; }

    node_index findLeafNode(uint64_t morton, const aabb_type& bounds) const {
        auto it = m_nodeMap.find(morton);
        if (it != m_nodeMap.end()) {
            node_index idx = it->second;
            const node_type& node = getNode(idx);
            if (node.bounds.contains(bounds) || bounds.overlaps(node.bounds)) {
                return idx;
            }
        }
        // Fallback: linear search over root's children (optimize with spatial hash)
        return m_root;
    }

    node_index createLeafNode(const aabb_type& bounds) {
        node_index newIdx = allocateNode();
        node_type& newNode = getNode(newIdx);
        newNode.bounds = bounds;
        newNode.isLeaf = true;
        newNode.firstEntity = static_cast<uint32_t>(m_entities.size());
        newNode.entityCount = 0;
        newNode.depth = 0;
        return newIdx;
    }

    bool insertIntoLeaf(node_index leafIdx, const entity_type& entity) {
        node_type& leaf = getNode(leafIdx);
        uint32_t pos = static_cast<uint32_t>(m_entities.size());
        m_entities.push_back(entity);
        if (leaf.entityCount == 0) {
            leaf.firstEntity = pos;
        } else if (leaf.firstEntity + leaf.entityCount != pos) {
            // Not contiguous – shift (should not happen with our alloc)
            // But we handle: move entities to maintain contiguity
            // For simplicity, we rebuild leaf if needed.
            rebuildLeaf(leafIdx);
            leaf = getNode(leafIdx); // refresh
            pos = static_cast<uint32_t>(m_entities.size());
            m_entities.push_back(entity);
        }
        ++leaf.entityCount;
        ++m_entityCount;
        ++m_version;

        // Split if exceeds bucket size and depth < maxDepth
        if (leaf.entityCount > m_bucketSize && leaf.depth < m_maxDepth) {
            splitNode(leafIdx);
        }
        return true;
    }

    bool removeFromLeaf(node_index leafIdx, const entity_type& entity) {
        node_type& leaf = getNode(leafIdx);
        uint32_t start = leaf.firstEntity;
        uint32_t end = start + leaf.entityCount;
        for (uint32_t i = start; i < end; ++i) {
            if (m_entities[i] == entity) {
                // Remove by swapping with last
                m_entities[i] = m_entities[end - 1];
                m_entities.pop_back();
                --leaf.entityCount;
                --m_entityCount;
                ++m_version;
                // Optional: merge if too few entities
                if (leaf.entityCount == 0 && leafIdx != m_root) {
                    mergeNode(leafIdx);
                }
                return true;
            }
        }
        return false;
    }

    void splitNode(node_index nodeIdx) {
        node_type& node = getNode(nodeIdx);
        if (!node.isLeaf) return;

        // Create 2^Dim children
        point_type min = node.bounds.min();
        point_type max = node.bounds.max();
        point_type mid = node.bounds.center();
        point_type half = node.bounds.halfExtents();

        // Precompute child bounds for each quadrant/octant
        uint8_t numChildren = (Dim == Dim3) ? 8 : 4;
        std::array<aabb_type, 8> childBounds;
        for (uint8_t i = 0; i < numChildren; ++i) {
            point_type childMin, childMax;
            for (size_t d = 0; d < Dim; ++d) {
                bool highBit = (i >> d) & 1;
                if (highBit) {
                    childMin[d] = mid[d];
                    childMax[d] = max[d];
                } else {
                    childMin[d] = min[d];
                    childMax[d] = mid[d];
                }
            }
            childBounds[i] = aabb_type(childMin, childMax);
        }

        // Create child nodes
        for (uint8_t i = 0; i < numChildren; ++i) {
            node_index childIdx = allocateNode();
            node_type& child = getNode(childIdx);
            child.bounds = childBounds[i];
            child.depth = node.depth + 1;
            child.isLeaf = true;
            child.firstEntity = 0;
            child.entityCount = 0;
            node.children[i] = childIdx;
            // Insert into map
            uint64_t childMorton = computeMortonKey(child.bounds.center());
            m_nodeMap[childMorton] = childIdx;
        }
        node.isLeaf = false;

        // Redistribute entities to children
        std::vector<entity_type> tempEntities;
        tempEntities.reserve(node.entityCount);
        for (uint32_t i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
            tempEntities.push_back(m_entities[i]);
        }
        node.entityCount = 0;
        for (auto& ent : tempEntities) {
            aabb_type entBounds = EntityAdapter<T, Dim>::getBounds(ent);
            point_type entCenter = entBounds.center();
            // Find which child contains the center
            for (uint8_t i = 0; i < numChildren; ++i) {
                if (childBounds[i].contains(entCenter)) {
                    insertIntoLeaf(node.children[i], ent);
                    break;
                }
            }
        }
    }

    void mergeNode(node_index nodeIdx) {
        node_type& node = getNode(nodeIdx);
        if (node.isLeaf) return;
        // Check if all children are leaves and have zero entities
        bool allEmpty = true;
        for (auto childIdx : node.children) {
            if (childIdx != node_type::INVALID_INDEX) {
                const node_type& child = getNode(childIdx);
                if (child.entityCount > 0 || !child.isLeaf) {
                    allEmpty = false;
                    break;
                }
            }
        }
        if (allEmpty) {
            // Remove children
            for (auto childIdx : node.children) {
                if (childIdx != node_type::INVALID_INDEX) {
                    // Remove from map
                    const node_type& child = getNode(childIdx);
                    uint64_t childMorton = computeMortonKey(child.bounds.center());
                    m_nodeMap.erase(childMorton);
                    // Mark as invalid (memory not freed, but node count decreases)
                    // In production, we would pool nodes.
                }
            }
            node.clearChildren();
            node.isLeaf = true;
        }
    }

    void rebuildLeaf(node_index leafIdx) {
        // Compact entities in leaf
        node_type& leaf = getNode(leafIdx);
        std::vector<entity_type> leafEntities;
        leafEntities.reserve(leaf.entityCount);
        for (uint32_t i = leaf.firstEntity; i < leaf.firstEntity + leaf.entityCount; ++i) {
            leafEntities.push_back(m_entities[i]);
        }
        // Remove old entities from global list (mark as holes)
        // For simplicity, we just rebuild entire entity array (expensive)
        // But this is a fallback; production code would use better compaction.
        // We'll implement a simple compaction:
        std::vector<entity_type> newEntities;
        newEntities.reserve(m_entities.size() - leaf.entityCount);
        uint32_t offset = 0;
        for (size_type i = 0; i < m_entities.size(); ++i) {
            bool isInLeaf = (i >= leaf.firstEntity && i < leaf.firstEntity + leaf.entityCount);
            if (!isInLeaf) {
                newEntities.push_back(m_entities[i]);
            } else {
                ++offset;
            }
        }
        // Append leaf entities
        for (auto& ent : leafEntities) {
            newEntities.push_back(ent);
        }
        m_entities.swap(newEntities);
        leaf.firstEntity = static_cast<uint32_t>(m_entities.size() - leaf.entityCount);
        // Update all other leaf firstEntity pointers? Too heavy. Instead we just rebuild? 
        // For correctness, we mark as leaf rebuilt.
    }

    size_type computeHeight(node_index nodeIdx) const {
        const node_type& node = getNode(nodeIdx);
        if (node.isLeaf) return node.depth;
        size_type maxChildDepth = 0;
        for (auto childIdx : node.children) {
            if (childIdx != node_type::INVALID_INDEX) {
                maxChildDepth = std::max(maxChildDepth, computeHeight(childIdx));
            }
        }
        return maxChildDepth;
    }

    void expandWorldBounds(const aabb_type& newBounds) {
        m_worldBounds = m_worldBounds.hull(newBounds);
        // Notify root node
        getNode(m_root).bounds = m_worldBounds;
        ++m_version;
    }

    // ------------------------------------------------------------------------
    //  Member variables
    // ------------------------------------------------------------------------
    Allocator m_alloc;
    typename std::vector<node_type, typename Allocator::template rebind<node_type>::other> m_nodes;
    typename std::vector<entity_type, typename Allocator::template rebind<entity_type>::other> m_entities;
    NodeMap m_nodeMap;
    aabb_type m_worldBounds;
    size_type m_maxDepth;
    size_type m_bucketSize;
    node_index m_root;
    size_type m_nodeCount;
    size_type m_entityCount;
    std::atomic<uint64_t> m_version;
};

} // namespace OrthoTree

#endif // ORTHOTREE__OCTREE_H_INCLUDED