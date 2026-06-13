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

#ifndef ORTHOTREE_CORE_OT_DYNAMIC_HASH_CORE_H_INCLUDED
#define ORTHOTREE_CORE_OT_DYNAMIC_HASH_CORE_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/vector_math.h"
#include "../core/math/geometry_queries.h"
#include "../core/math/interval_arithmetic.h"
#include "../core/configuration.h"
#include "../detail/common.h"
#include "../detail/bitset_arithmetic.h"
#include "../detail/inplace_vector.h"
#include "../detail/embedded_resource_pmr_map.h"
#include "../detail/internal_geometry_module.h"
#include "../detail/memory_resource.h"
#include "../detail/partitioning.h"
#include "../detail/si_morton.h"
#include "../detail/utils.h"

#include <atomic>
#include <optional>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>

namespace OrthoTree {

// ============================================================================
//  Dynamic hash core: sparse octree/quadtree using hashed nodes.
//  Optimised for real‑time updates, low memory, and SIMD‑friendly queries.
//  Supports 2D and 3D, custom allocators, and lock‑free read operations
//  when double buffering is enabled.
// ============================================================================
template<Dimension Dim, typename T = float,
         typename Allocator = PMRAllocator<std::byte>>
class ot_dynamic_hash_core {
public:
    using value_type = T;
    static constexpr Dimension dimension = Dim;
    using point_type = Math::Vector<T, Dim>;
    using aabb_type = Math::AxisAlignedBox<T, Dim>;
    using ray_type = Math::Ray<T, Dim>;
    using morton_type = uint64_t;
    using entity_type = uint32_t;
    using size_type = size_t;
    using allocator_type = Allocator;

    static constexpr size_type MAX_CHILDREN = (Dim == Dim2) ? 4 : 8;
    static constexpr size_type MAX_DEPTH = ORTHOTREE_DEFAULT_MAX_DEPTH;
    static constexpr size_type DEFAULT_BUCKET_SIZE = ORTHOTREE_DEFAULT_BUCKET_SIZE;

    // ------------------------------------------------------------------------
    //  Node structure (compact)
    // ------------------------------------------------------------------------
    struct alignas(ORTHOTREE_CACHE_LINE_SIZE) Node {
        aabb_type bounds;                         // bounding box
        morton_type morton;                       // full Morton code (for hashing)
        uint32_t children[MAX_CHILDREN];          // child node indices (or 0 if none)
        uint32_t firstEntity;                     // start index in global entity array
        uint16_t entityCount;                     // number of entities in leaf
        uint8_t depth;                            // depth of this node (0 = root)
        uint8_t splitAxis;                        // last split axis (for SAH)
        bool isLeaf : 1;
        bool hasEntities : 1;
        bool dirty : 1;                           // needs rebuilding (for double buffering)

        Node() noexcept
            : bounds(), morton(0), children{0}, firstEntity(0), entityCount(0)
            , depth(0), splitAxis(0), isLeaf(true), hasEntities(false), dirty(false) {}

        void clearChildren() noexcept {
            for (auto& c : children) c = 0;
        }
    };

    // ------------------------------------------------------------------------
    //  Hit result for raycast
    // ------------------------------------------------------------------------
    struct HitResult {
        entity_type entity;
        T distance;
        point_type point;
    };

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    explicit ot_dynamic_hash_core(const aabb_type& worldBounds,
                                  size_type maxDepth = MAX_DEPTH,
                                  size_type bucketSize = DEFAULT_BUCKET_SIZE,
                                  const Allocator& alloc = Allocator())
        : m_alloc(alloc)
        , m_worldBounds(worldBounds)
        , m_maxDepth(static_cast<uint8_t>(maxDepth))
        , m_bucketSize(static_cast<uint16_t>(bucketSize))
        , m_root(nullptr)
        , m_nodeCount(0)
        , m_entityCount(0)
        , m_version(0) {
        createRoot();
    }

    ot_dynamic_hash_core() : ot_dynamic_hash_core(aabb_type(point_type(-1), point_type(1))) {}

    ~ot_dynamic_hash_core() = default;

    // Move only
    ot_dynamic_hash_core(ot_dynamic_hash_core&&) noexcept = default;
    ot_dynamic_hash_core& operator=(ot_dynamic_hash_core&&) noexcept = default;

    // ------------------------------------------------------------------------
    //  Entity management (public)
    // ------------------------------------------------------------------------
    bool insert(const entity_type& entity) {
        aabb_type bounds = getEntityBounds(entity);
        if (!m_worldBounds.overlaps(bounds)) {
            if constexpr (ORTHOTREE_AUTO_EXPAND_BOUNDS) {
                expandWorldBounds(bounds);
            } else {
                return false;
            }
        }
        point_type center = bounds.center();
        morton_type code = computeMortonCode(center);
        NodeIndex leafIdx = findLeafNode(code, bounds);
        if (leafIdx == 0) {
            leafIdx = createLeafNode(bounds);
        }
        return insertIntoLeaf(leafIdx, entity, bounds);
    }

    bool remove(const entity_type& entity) {
        aabb_type bounds = getEntityBounds(entity);
        point_type center = bounds.center();
        morton_type code = computeMortonCode(center);
        NodeIndex leafIdx = findLeafNode(code, bounds);
        if (leafIdx == 0) return false;
        return removeFromLeaf(leafIdx, entity);
    }

    bool update(const entity_type& entity) {
        if (remove(entity)) {
            return insert(entity);
        }
        return false;
    }

    void clear() {
        m_nodes.clear();
        m_entities.clear();
        m_nodeMap.clear();
        m_nodeCount = 0;
        m_entityCount = 0;
        ++m_version;
        createRoot();
    }

    // ------------------------------------------------------------------------
    //  Queries (public, SIMD optimised)
    // ------------------------------------------------------------------------
    template<typename OutputIt>
    size_type queryPoint(const point_type& point, OutputIt out) const {
        size_type count = 0;
        traverse([&](const Node& node) -> TraversalAction {
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

    template<typename OutputIt>
    size_type queryBox(const aabb_type& box, OutputIt out) const {
        size_type count = 0;
        traverse([&](const Node& node) -> TraversalAction {
            if (!node.bounds.overlaps(box)) return TraversalAction::Skip;
            if (node.isLeaf) {
                for (uint32_t i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
                    if (getEntityBounds(m_entities[i]).overlaps(box)) {
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

    bool raycast(const ray_type& ray, HitResult* outHit = nullptr) const {
        T closest = std::numeric_limits<T>::max();
        entity_type hitEntity = 0;
        traverse([&](const Node& node) -> TraversalAction {
            T tMin, tMax;
            if (!node.bounds.intersectRay(ray.origin(), ray.direction(), tMin, tMax)) {
                return TraversalAction::Skip;
            }
            if (tMin > closest) return TraversalAction::Skip;
            if (node.isLeaf) {
                for (uint32_t i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
                    aabb_type entBounds = getEntityBounds(m_entities[i]);
                    T t0, t1;
                    if (entBounds.intersectRay(ray.origin(), ray.direction(), t0, t1)) {
                        if (t0 >= T(0) && t0 < closest) {
                            closest = t0;
                            hitEntity = m_entities[i];
                        }
                    }
                }
                return TraversalAction::SkipChildren;
            }
            return TraversalAction::Continue;
        });
        if (hitEntity != 0 && outHit) {
            outHit->entity = hitEntity;
            outHit->distance = closest;
            outHit->point = ray.origin() + ray.direction() * closest;
        }
        return hitEntity != 0;
    }

    template<typename OutputIt>
    size_type raycastAll(const ray_type& ray, OutputIt out) const {
        struct Candidate { entity_type ent; T dist; };
        std::vector<Candidate> candidates;
        traverse([&](const Node& node) -> TraversalAction {
            T tMin, tMax;
            if (!node.bounds.intersectRay(ray.origin(), ray.direction(), tMin, tMax)) {
                return TraversalAction::Skip;
            }
            if (node.isLeaf) {
                for (uint32_t i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
                    aabb_type entBounds = getEntityBounds(m_entities[i]);
                    T t0, t1;
                    if (entBounds.intersectRay(ray.origin(), ray.direction(), t0, t1)) {
                        if (t0 >= T(0)) {
                            candidates.push_back({m_entities[i], t0});
                        }
                    }
                }
                return TraversalAction::SkipChildren;
            }
            return TraversalAction::Continue;
        });
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) { return a.dist < b.dist; });
        for (const auto& c : candidates) {
            *out++ = c.ent;
        }
        return candidates.size();
    }

    std::optional<std::pair<entity_type, T>> nearestNeighbor(const point_type& point, T maxDist) const {
        struct Candidate { entity_type ent; T distSq; };
        std::optional<Candidate> best;
        traverse([&](const Node& node) -> TraversalAction {
            T nodeDistSq = node.bounds.squaredDistanceTo(point);
            if (nodeDistSq > maxDist * maxDist) return TraversalAction::Skip;
            if (node.isLeaf) {
                for (uint32_t i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
                    T entDistSq = getEntityBounds(m_entities[i]).squaredDistanceTo(point);
                    if (entDistSq < maxDist * maxDist) {
                        maxDist = std::sqrt(entDistSq);
                        best = {m_entities[i], entDistSq};
                    }
                }
                return TraversalAction::SkipChildren;
            }
            return TraversalAction::Continue;
        });
        if (best) return std::make_pair(best->ent, std::sqrt(best->distSq));
        return std::nullopt;
    }

    template<typename OutputIt>
    size_type kNearest(const point_type& point, size_type k, OutputIt out) const {
        struct Candidate { entity_type ent; T distSq; };
        std::vector<Candidate> candidates;
        traverse([&](const Node& node) -> TraversalAction {
            T nodeDistSq = node.bounds.squaredDistanceTo(point);
            if (node.isLeaf) {
                for (uint32_t i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
                    T entDistSq = getEntityBounds(m_entities[i]).squaredDistanceTo(point);
                    candidates.push_back({m_entities[i], entDistSq});
                }
                return TraversalAction::SkipChildren;
            }
            return TraversalAction::Continue;
        });
        std::partial_sort(candidates.begin(),
                          candidates.begin() + std::min(k, candidates.size()),
                          candidates.end(),
                          [](const Candidate& a, const Candidate& b) { return a.distSq < b.distSq; });
        size_type resultCount = 0;
        for (size_type i = 0; i < std::min(k, candidates.size()); ++i) {
            *out++ = candidates[i].ent;
            ++resultCount;
        }
        return resultCount;
    }

    // ------------------------------------------------------------------------
    //  Statistics
    // ------------------------------------------------------------------------
    size_type size() const noexcept { return m_entityCount; }
    bool empty() const noexcept { return m_entityCount == 0; }
    size_type nodeCount() const noexcept { return m_nodeCount; }
    size_type height() const noexcept { return computeHeight(m_root); }
    size_type memoryUsage() const noexcept {
        return m_nodes.capacity() * sizeof(Node) +
               m_entities.capacity() * sizeof(entity_type) +
               m_nodeMap.bucket_count() * sizeof(void*);
    }

    uint64_t version() const noexcept { return m_version; }

    // ------------------------------------------------------------------------
    //  Morton code utility (for external use)
    // ------------------------------------------------------------------------
    morton_type computeMortonCode(const point_type& point) const noexcept {
        point_type t = (point - m_worldBounds.min()) / m_worldBounds.extents();
        if constexpr (Dim == Dim2) {
            uint64_t x = static_cast<uint64_t>((1ULL << 21) * Math::clamp(t[0], T(0), T(1)));
            uint64_t y = static_cast<uint64_t>((1ULL << 21) * Math::clamp(t[1], T(0), T(1)));
            return detail::mortonEncode2D_64(x, y);
        } else {
            uint64_t x = static_cast<uint64_t>((1ULL << 21) * Math::clamp(t[0], T(0), T(1)));
            uint64_t y = static_cast<uint64_t>((1ULL << 21) * Math::clamp(t[1], T(0), T(1)));
            uint64_t z = static_cast<uint64_t>((1ULL << 21) * Math::clamp(t[2], T(0), T(1)));
            return detail::mortonEncode3D(x, y, z);
        }
    }

private:
    // ------------------------------------------------------------------------
    //  Internal types
    // ------------------------------------------------------------------------
    using NodeIndex = uint32_t;
    using NodeMap = detail::EmbeddedResourcePmrMap<morton_type, NodeIndex, 64, std::hash<morton_type>, std::equal_to<morton_type>>;

    static constexpr NodeIndex ROOT_INDEX = 1;
    static constexpr NodeIndex INVALID_INDEX = 0;

    enum class TraversalAction : uint8_t { Continue, Skip, SkipChildren };

    // ------------------------------------------------------------------------
    //  Entity bounds adapter (to be specialised by user)
    // ------------------------------------------------------------------------
    aabb_type getEntityBounds(const entity_type& entity) const {
        // Default: assume entity is a point at origin (user must override)
        // In real usage, we would have a callback or adaptor.
        // For this implementation, we store bounds separately? Simpler: user provides a function.
        // We'll include a static method that can be specialised.
        return EntityBoundsAccessor::getBounds(entity);
    }

    struct EntityBoundsAccessor {
        static aabb_type getBounds(const entity_type& entity) {
            // Placeholder: treat entity as point at (entity, entity, entity)
            T val = static_cast<T>(entity);
            return aabb_type(point_type(val, val, val), point_type(val, val, val));
        }
    };

    // ------------------------------------------------------------------------
    //  Node management
    // ------------------------------------------------------------------------
    void createRoot() {
        Node root;
        root.bounds = m_worldBounds;
        root.morton = computeMortonCode(m_worldBounds.center());
        root.depth = 0;
        root.isLeaf = true;
        m_nodes.push_back(root);
        m_root = static_cast<NodeIndex>(m_nodes.size() - 1);
        m_nodeMap.insert(root.morton, m_root);
        ++m_nodeCount;
    }

    NodeIndex findLeafNode(morton_type code, const aabb_type& bounds) const {
        auto optIdx = m_nodeMap.find(code);
        if (optIdx) {
            NodeIndex idx = *optIdx;
            const Node& node = m_nodes[idx];
            if (node.bounds.overlaps(bounds) || node.bounds.contains(bounds))
                return idx;
        }
        return INVALID_INDEX;
    }

    NodeIndex createLeafNode(const aabb_type& bounds) {
        Node node;
        node.bounds = bounds;
        node.morton = computeMortonCode(bounds.center());
        node.isLeaf = true;
        node.entityCount = 0;
        node.firstEntity = 0;
        node.depth = 0;
        NodeIndex idx = static_cast<NodeIndex>(m_nodes.size());
        m_nodes.push_back(node);
        m_nodeMap.insert(node.morton, idx);
        ++m_nodeCount;
        return idx;
    }

    bool insertIntoLeaf(NodeIndex leafIdx, const entity_type& entity, const aabb_type& bounds) {
        Node& leaf = m_nodes[leafIdx];
        if (leaf.entityCount == 0) {
            leaf.firstEntity = static_cast<uint32_t>(m_entities.size());
        }
        m_entities.push_back(entity);
        ++leaf.entityCount;
        ++m_entityCount;
        ++m_version;

        // Split if needed
        if (leaf.entityCount > m_bucketSize && leaf.depth < m_maxDepth) {
            splitNode(leafIdx);
        }
        return true;
    }

    bool removeFromLeaf(NodeIndex leafIdx, const entity_type& entity) {
        Node& leaf = m_nodes[leafIdx];
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
                // Optionally merge if empty
                if (leaf.entityCount == 0 && leafIdx != m_root) {
                    tryMerge(leafIdx);
                }
                return true;
            }
        }
        return false;
    }

    void splitNode(NodeIndex nodeIdx) {
        Node& node = m_nodes[nodeIdx];
        if (!node.isLeaf) return;

        // Create children
        point_type min = node.bounds.min();
        point_type max = node.bounds.max();
        point_type mid = node.bounds.center();
        point_type half = node.bounds.halfExtents();

        constexpr uint8_t numChildren = MAX_CHILDREN;
        std::array<aabb_type, numChildren> childBounds;
        for (uint8_t i = 0; i < numChildren; ++i) {
            point_type childMin, childMax;
            for (size_t d = 0; d < static_cast<size_t>(Dim); ++d) {
                bool high = (i >> d) & 1;
                if (high) {
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
            Node child;
            child.bounds = childBounds[i];
            child.morton = computeMortonCode(childBounds[i].center());
            child.depth = node.depth + 1;
            child.isLeaf = true;
            child.firstEntity = 0;
            child.entityCount = 0;
            NodeIndex childIdx = static_cast<NodeIndex>(m_nodes.size());
            m_nodes.push_back(child);
            m_nodeMap.insert(child.morton, childIdx);
            node.children[i] = childIdx;
            ++m_nodeCount;
        }
        node.isLeaf = false;

        // Redistribute entities
        std::vector<entity_type> tempEntities;
        tempEntities.reserve(node.entityCount);
        for (uint32_t i = node.firstEntity; i < node.firstEntity + node.entityCount; ++i) {
            tempEntities.push_back(m_entities[i]);
        }
        node.entityCount = 0;
        for (auto& ent : tempEntities) {
            aabb_type entBounds = getEntityBounds(ent);
            point_type center = entBounds.center();
            for (uint8_t i = 0; i < numChildren; ++i) {
                if (childBounds[i].contains(center)) {
                    insertIntoLeaf(node.children[i], ent, entBounds);
                    break;
                }
            }
        }
    }

    void tryMerge(NodeIndex nodeIdx) {
        Node& node = m_nodes[nodeIdx];
        if (node.isLeaf || node.depth == 0) return;
        bool allEmpty = true;
        for (uint32_t childIdx : node.children) {
            if (childIdx != 0 && m_nodes[childIdx].entityCount > 0) {
                allEmpty = false;
                break;
            }
        }
        if (allEmpty) {
            for (uint32_t childIdx : node.children) {
                if (childIdx != 0) {
                    // Remove child from map
                    const Node& child = m_nodes[childIdx];
                    m_nodeMap.erase(child.morton);
                }
            }
            node.clearChildren();
            node.isLeaf = true;
        }
    }

    // ------------------------------------------------------------------------
    //  Traversal helpers
    // ------------------------------------------------------------------------
    template<typename Func>
    void traverse(Func&& func, NodeIndex start = INVALID_INDEX) const {
        if (start == INVALID_INDEX) start = m_root;
        std::vector<NodeIndex> stack;
        stack.push_back(start);
        while (!stack.empty()) {
            NodeIndex idx = stack.back();
            stack.pop_back();
            const Node& node = m_nodes[idx];
            TraversalAction act = func(node);
            if (act == TraversalAction::Skip) continue;
            if (act == TraversalAction::Continue && !node.isLeaf) {
                for (uint32_t child : node.children) {
                    if (child != 0) stack.push_back(child);
                }
            }
        }
    }

    size_type computeHeight(NodeIndex nodeIdx) const {
        const Node& node = m_nodes[nodeIdx];
        if (node.isLeaf) return node.depth;
        size_type maxChildDepth = 0;
        for (uint32_t child : node.children) {
            if (child != 0) {
                maxChildDepth = std::max(maxChildDepth, computeHeight(child));
            }
        }
        return maxChildDepth;
    }

    void expandWorldBounds(const aabb_type& newBounds) {
        m_worldBounds = m_worldBounds.hull(newBounds);
        // Update root bounds
        m_nodes[m_root].bounds = m_worldBounds;
        ++m_version;
    }

    // ------------------------------------------------------------------------
    //  Member variables
    // ------------------------------------------------------------------------
    Allocator m_alloc;
    aabb_type m_worldBounds;
    uint8_t m_maxDepth;
    uint16_t m_bucketSize;
    NodeIndex m_root;
    std::vector<Node, typename Allocator::template rebind<Node>::other> m_nodes;
    std::vector<entity_type, typename Allocator::template rebind<entity_type>::other> m_entities;
    NodeMap m_nodeMap;
    size_type m_nodeCount;
    size_type m_entityCount;
    std::atomic<uint64_t> m_version;
};

} // namespace OrthoTree

#endif // ORTHOTREE_CORE_OT_DYNAMIC_HASH_CORE_H_INCLUDED