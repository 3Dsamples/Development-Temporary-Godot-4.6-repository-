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

#ifndef ORTHOTREE_CORE_OT_STATIC_LINEAR_CORE_H_INCLUDED
#define ORTHOTREE_CORE_OT_STATIC_LINEAR_CORE_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/vector_math.h"
#include "../core/math/geometry_queries.h"
#include "../core/math/interval_arithmetic.h"
#include "../core/configuration.h"
#include "../detail/common.h"
#include "../detail/bitset_arithmetic.h"
#include "../detail/inplace_vector.h"
#include "../detail/internal_geometry_module.h"
#include "../detail/memory_resource.h"
#include "../detail/partitioning.h"
#include "../detail/si_morton.h"
#include "../detail/utils.h"

#include <algorithm>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <limits>
#include <cmath>
#include <atomic>
#include <type_traits>

namespace OrthoTree {

// ============================================================================
//  Static linear BVH core (LBVH) – Morton‑coded linear bounding volume hierarchy.
//  Optimised for static geometry, extremely fast traversal, low memory.
//  Supports 2D and 3D, SIMD‑aware node layout, and cache‑friendly queries.
// ============================================================================
template<Dimension Dim, typename T = float,
         typename Allocator = PMRAllocator<std::byte>>
class ot_static_linear_core {
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

    // ------------------------------------------------------------------------
    //  Linear BVH node (16 bytes on 64‑bit, cache‑line friendly)
    // ------------------------------------------------------------------------
    struct alignas(16) LinearNode {
        aabb_type bounds;      // 2 * Dim * sizeof(T) – 24 bytes for 3D float
        uint32_t leftChild;    // index of left child or first primitive (if leaf)
        uint32_t rightChild;   // index of right child or primitive count (leaf)
        uint8_t axis;          // split axis (0,1,2)
        bool isLeaf : 1;
        uint8_t : 7;           // padding
        uint16_t primitiveCount; // number of primitives in leaf (if leaf)

        LinearNode() noexcept : bounds(), leftChild(0), rightChild(0), axis(0), isLeaf(true), primitiveCount(0) {}
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
    //  Constructor – builds BVH from a list of entity bounds.
    //  For static geometry, we build once and then only query.
    // ------------------------------------------------------------------------
    template<typename InputIt>
    ot_static_linear_core(InputIt first, InputIt last,
                          const Allocator& alloc = Allocator())
        : m_alloc(alloc)
        , m_nodes(alloc)
        , m_primitiveIndices(alloc) {
        build(first, last);
    }

    // Empty constructor (no primitives)
    ot_static_linear_core() : ot_static_linear_core(nullptr, nullptr) {}

    // ------------------------------------------------------------------------
    //  Rebuild from new data (destroys old structure)
    // ------------------------------------------------------------------------
    template<typename InputIt>
    void rebuild(InputIt first, InputIt last) {
        clear();
        build(first, last);
    }

    // Clear all data
    void clear() {
        m_nodes.clear();
        m_primitiveIndices.clear();
        m_entityBounds.clear();
    }

    // ------------------------------------------------------------------------
    //  Queries (const only – BVH is immutable after build)
    // ------------------------------------------------------------------------
    template<typename OutputIt>
    size_type queryPoint(const point_type& point, OutputIt out) const {
        size_type count = 0;
        traverse([&](const LinearNode& node) -> TraversalAction {
            if (!node.bounds.contains(point)) return TraversalAction::Skip;
            if (node.isLeaf) {
                for (uint32_t i = node.leftChild; i < node.leftChild + node.primitiveCount; ++i) {
                    *out++ = m_primitiveIndices[i];
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
        traverse([&](const LinearNode& node) -> TraversalAction {
            if (!node.bounds.overlaps(box)) return TraversalAction::Skip;
            if (node.isLeaf) {
                for (uint32_t i = node.leftChild; i < node.leftChild + node.primitiveCount; ++i) {
                    entity_type ent = m_primitiveIndices[i];
                    if (getEntityBounds(ent).overlaps(box)) {
                        *out++ = ent;
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
        traverse([&](const LinearNode& node) -> TraversalAction {
            T tMin, tMax;
            if (!node.bounds.intersectRay(ray.origin(), ray.direction(), tMin, tMax)) {
                return TraversalAction::Skip;
            }
            if (tMin > closest) return TraversalAction::Skip;
            if (node.isLeaf) {
                for (uint32_t i = node.leftChild; i < node.leftChild + node.primitiveCount; ++i) {
                    entity_type ent = m_primitiveIndices[i];
                    aabb_type entBounds = getEntityBounds(ent);
                    T t0, t1;
                    if (entBounds.intersectRay(ray.origin(), ray.direction(), t0, t1)) {
                        if (t0 >= T(0) && t0 < closest) {
                            closest = t0;
                            hitEntity = ent;
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
        traverse([&](const LinearNode& node) -> TraversalAction {
            T tMin, tMax;
            if (!node.bounds.intersectRay(ray.origin(), ray.direction(), tMin, tMax)) {
                return TraversalAction::Skip;
            }
            if (node.isLeaf) {
                for (uint32_t i = node.leftChild; i < node.leftChild + node.primitiveCount; ++i) {
                    entity_type ent = m_primitiveIndices[i];
                    aabb_type entBounds = getEntityBounds(ent);
                    T t0, t1;
                    if (entBounds.intersectRay(ray.origin(), ray.direction(), t0, t1)) {
                        if (t0 >= T(0)) {
                            candidates.push_back({ent, t0});
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
        traverse([&](const LinearNode& node) -> TraversalAction {
            T nodeDistSq = node.bounds.squaredDistanceTo(point);
            if (nodeDistSq > maxDist * maxDist) return TraversalAction::Skip;
            if (node.isLeaf) {
                for (uint32_t i = node.leftChild; i < node.leftChild + node.primitiveCount; ++i) {
                    entity_type ent = m_primitiveIndices[i];
                    T entDistSq = getEntityBounds(ent).squaredDistanceTo(point);
                    if (entDistSq < maxDist * maxDist) {
                        maxDist = std::sqrt(entDistSq);
                        best = {ent, entDistSq};
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
        traverse([&](const LinearNode& node) -> TraversalAction {
            if (node.isLeaf) {
                for (uint32_t i = node.leftChild; i < node.leftChild + node.primitiveCount; ++i) {
                    entity_type ent = m_primitiveIndices[i];
                    T entDistSq = getEntityBounds(ent).squaredDistanceTo(point);
                    candidates.push_back({ent, entDistSq});
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
    size_type size() const noexcept { return m_primitiveIndices.size(); }
    bool empty() const noexcept { return m_primitiveIndices.empty(); }
    size_type nodeCount() const noexcept { return m_nodes.size(); }
    size_type memoryUsage() const noexcept {
        return m_nodes.capacity() * sizeof(LinearNode) +
               m_primitiveIndices.capacity() * sizeof(entity_type) +
               m_entityBounds.capacity() * sizeof(aabb_type);
    }

    // ------------------------------------------------------------------------
    //  Access to internal nodes (for debugging)
    // ------------------------------------------------------------------------
    const std::vector<LinearNode>& nodes() const noexcept { return m_nodes; }
    const std::vector<entity_type>& primitiveIndices() const noexcept { return m_primitiveIndices; }

private:
    // ------------------------------------------------------------------------
    //  Entity bounds adapter (to be specialised by user)
    // ------------------------------------------------------------------------
    aabb_type getEntityBounds(const entity_type& entity) const {
        // Default: assume entity is point at (entity,entity,entity)
        T val = static_cast<T>(entity);
        return aabb_type(point_type(val, val, val), point_type(val, val, val));
    }

    // ------------------------------------------------------------------------
    //  Build helpers
    // ------------------------------------------------------------------------
    template<typename InputIt>
    void build(InputIt first, InputIt last) {
        // Collect primitive bounds and compute Morton codes
        size_type numPrimitives = std::distance(first, last);
        if (numPrimitives == 0) return;

        m_primitiveIndices.reserve(numPrimitives);
        m_entityBounds.reserve(numPrimitives);
        std::vector<morton_type> mortonCodes;
        mortonCodes.reserve(numPrimitives);

        // Compute global bounds of all primitives
        aabb_type globalBounds;
        size_t idx = 0;
        for (auto it = first; it != last; ++it, ++idx) {
            entity_type ent = static_cast<entity_type>(idx);
            aabb_type bounds = getEntityBounds(ent);
            m_entityBounds.push_back(bounds);
            m_primitiveIndices.push_back(ent);
            globalBounds.extend(bounds);
            // Compute centroid
            point_type center = bounds.center();
            // Normalise to [0,1] range
            point_type t = (center - globalBounds.min()) / globalBounds.extents();
            morton_type code = 0;
            if constexpr (Dim == Dim2) {
                uint64_t x = static_cast<uint64_t>(t[0] * static_cast<T>((1ULL << 21) - 1));
                uint64_t y = static_cast<uint64_t>(t[1] * static_cast<T>((1ULL << 21) - 1));
                code = detail::mortonEncode2D_64(x, y);
            } else {
                uint64_t x = static_cast<uint64_t>(t[0] * static_cast<T>((1ULL << 21) - 1));
                uint64_t y = static_cast<uint64_t>(t[1] * static_cast<T>((1ULL << 21) - 1));
                uint64_t z = static_cast<uint64_t>(t[2] * static_cast<T>((1ULL << 21) - 1));
                code = detail::mortonEncode3D(x, y, z);
            }
            mortonCodes.push_back(code);
        }

        // Sort primitives by Morton code
        std::vector<size_type> indices(numPrimitives);
        for (size_type i = 0; i < numPrimitives; ++i) indices[i] = i;
        std::sort(indices.begin(), indices.end(),
                  [&](size_type a, size_type b) { return mortonCodes[a] < mortonCodes[b]; });

        // Reorder primitive indices and bounds
        std::vector<entity_type> sortedEntities;
        std::vector<aabb_type> sortedBounds;
        sortedEntities.reserve(numPrimitives);
        sortedBounds.reserve(numPrimitives);
        for (size_type i : indices) {
            sortedEntities.push_back(m_primitiveIndices[i]);
            sortedBounds.push_back(m_entityBounds[i]);
        }
        m_primitiveIndices.swap(sortedEntities);
        m_entityBounds.swap(sortedBounds);

        // Build LBVH recursively
        m_nodes.clear();
        buildRecursive(0, numPrimitives, globalBounds);
    }

    NodeIndex buildRecursive(size_type start, size_type count, const aabb_type& nodeBounds) {
        NodeIndex nodeIdx = static_cast<NodeIndex>(m_nodes.size());
        m_nodes.emplace_back();
        LinearNode& node = m_nodes.back();
        node.bounds = nodeBounds;

        if (count <= 2) { // leaf threshold (can be tuned)
            node.isLeaf = true;
            node.leftChild = static_cast<uint32_t>(start);
            node.primitiveCount = static_cast<uint16_t>(count);
            return nodeIdx;
        }

        // Find best split axis based on largest extent
        point_type extents = nodeBounds.extents();
        uint8_t axis = 0;
        T maxExt = extents[0];
        for (uint8_t i = 1; i < static_cast<uint8_t>(Dim); ++i) {
            if (extents[i] > maxExt) {
                maxExt = extents[i];
                axis = i;
            }
        }
        node.axis = axis;

        // Find split position using median of centroids along axis
        T splitCoord = findMedianSplit(start, count, axis);
        size_type splitIndex = partitionPrimitives(start, count, axis, splitCoord);

        if (splitIndex == start || splitIndex == start + count) {
            // Fallback: split into two equal parts
            splitIndex = start + count / 2;
        }

        aabb_type leftBounds = computeBounds(start, splitIndex);
        aabb_type rightBounds = computeBounds(splitIndex, start + count);

        NodeIndex leftChild = buildRecursive(start, splitIndex - start, leftBounds);
        NodeIndex rightChild = buildRecursive(splitIndex, start + count - splitIndex, rightBounds);

        node.isLeaf = false;
        node.leftChild = leftChild;
        node.rightChild = rightChild;
        node.primitiveCount = 0;
        return nodeIdx;
    }

    T findMedianSplit(size_type start, size_type count, uint8_t axis) const {
        // Collect centroids
        std::vector<T> centroids;
        centroids.reserve(count);
        for (size_type i = start; i < start + count; ++i) {
            centroids.push_back(m_entityBounds[i].center()[axis]);
        }
        std::nth_element(centroids.begin(), centroids.begin() + count / 2, centroids.end());
        return centroids[count / 2];
    }

    size_type partitionPrimitives(size_type start, size_type count, uint8_t axis, T splitCoord) {
        size_type left = start;
        size_type right = start + count - 1;
        while (left <= right) {
            while (left <= right && m_entityBounds[left].center()[axis] < splitCoord) ++left;
            while (left <= right && m_entityBounds[right].center()[axis] >= splitCoord) --right;
            if (left < right) {
                std::swap(m_primitiveIndices[left], m_primitiveIndices[right]);
                std::swap(m_entityBounds[left], m_entityBounds[right]);
                ++left;
                --right;
            }
        }
        return left;
    }

    aabb_type computeBounds(size_type start, size_type end) const {
        aabb_type bounds;
        for (size_type i = start; i < end; ++i) {
            bounds.extend(m_entityBounds[i]);
        }
        return bounds;
    }

    // ------------------------------------------------------------------------
    //  Traversal helper (iterative, stack‑based)
    // ------------------------------------------------------------------------
    enum class TraversalAction : uint8_t { Continue, Skip, SkipChildren };

    template<typename Func>
    void traverse(Func&& func) const {
        if (m_nodes.empty()) return;
        std::vector<NodeIndex> stack;
        stack.push_back(0);
        while (!stack.empty()) {
            NodeIndex idx = stack.back();
            stack.pop_back();
            const LinearNode& node = m_nodes[idx];
            TraversalAction act = func(node);
            if (act == TraversalAction::Skip) continue;
            if (act == TraversalAction::Continue && !node.isLeaf) {
                stack.push_back(node.rightChild);
                stack.push_back(node.leftChild);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Member variables
    // ------------------------------------------------------------------------
    Allocator m_alloc;
    std::vector<LinearNode, typename Allocator::template rebind<LinearNode>::other> m_nodes;
    std::vector<entity_type, typename Allocator::template rebind<entity_type>::other> m_primitiveIndices;
    std::vector<aabb_type, typename Allocator::template rebind<aabb_type>::other> m_entityBounds;
};

} // namespace OrthoTree

#endif // ORTHOTREE_CORE_OT_STATIC_LINEAR_CORE_H_INCLUDED