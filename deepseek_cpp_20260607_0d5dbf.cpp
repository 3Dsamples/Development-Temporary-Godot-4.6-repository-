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
 * @file bvh.h
 * @brief Unified BVH (Bounding Volume Hierarchy) interface for 2D/3D spatial queries.
 *
 * This file provides a high-level wrapper around both static linear BVH and
 * dynamic octree implementations. It abstracts the underlying core and exposes
 * a consistent API for ray casting, frustum culling, nearest neighbor search,
 * and range queries. Designed for real-time rendering, physics simulation,
 * and collision detection.
 *
 * Key features:
 * - Compile‑time selection of static (LBVH) or dynamic (octree) backend
 * - Automatic dimension and scalar type deduction
 * - PMR allocator support for low‑latency workloads
 * - Advanced math integration (quaternions, transforms, intervals)
 * - Thread‑safe read‑only queries (when underlying core supports it)
 */

#pragma once
#ifndef ORTHOTREE__BVH_H_INCLUDED
#define ORTHOTREE__BVH_H_INCLUDED

#include "core/configuration.h"
#include "core/types.h"
#include "core/math/vector_math.h"
#include "core/math/geometry_queries.h"
#include "core/math/transform.h"
#include "core/math/numerical_methods.h"
#include "core/entity_adapter.h"
#include "core/ot_static_linear_core.h"
#include "core/ot_dynamic_hash_core.h"
#include "core/ot_query.h"
#include "detail/memory_resource.h"
#include "adapters/general.h"

#include <type_traits>
#include <optional>
#include <vector>
#include <functional>
#include <memory>

namespace OrthoTree {

// ----------------------------------------------------------------------------
//  BVH policy tags for compile‑time backend selection
// ----------------------------------------------------------------------------

/**
 * @brief Tag to select the static linear BVH backend (fast, read‑only).
 */
struct StaticPolicy {};

/**
 * @brief Tag to select the dynamic octree backend (mutable, sparse).
 */
struct DynamicPolicy {};

// ----------------------------------------------------------------------------
//  BVH class template – main user interface
// ----------------------------------------------------------------------------

/**
 * @brief Bounding Volume Hierarchy for generic spatial indexing.
 *
 * @tparam Dim Dimension (2 or 3).
 * @tparam T Scalar type (float/double).
 * @tparam Policy Compile‑time backend selector (StaticPolicy or DynamicPolicy).
 * @tparam Allocator Allocator type (default PMR).
 */
template <Dimension Dim, typename T = float,
          typename Policy = DynamicPolicy,
          typename Allocator = PMRAllocator<std::byte>>
class BVH {
    // Select the appropriate core implementation based on Policy
    using CoreType = std::conditional_t<
        std::is_same_v<Policy, StaticPolicy>,
        ot_static_linear_core<Dim, T, Allocator>,
        ot_dynamic_hash_core<Dim, T, Allocator>
    >;

public:
    using value_type      = T;
    using scalar_type     = T;
    static constexpr Dimension dimension = Dim;
    using point_type      = Math::Vector<T, Dim>;
    using aabb_type       = Math::AxisAlignedBox<T, Dim>;
    using transform_type  = Math::AffineTransform<T, Dim>;
    using query_engine    = ot_query<Dim, T, CoreType>;
    using size_type       = std::size_t;
    using index_type      = typename CoreType::index_type;
    using entity_type     = typename CoreType::entity_type;
    using entity_adapter  = typename CoreType::entity_adapter;
    using allocator_type  = Allocator;

    // ------------------------------------------------------------------------
    //  Construction / destruction
    // ------------------------------------------------------------------------

    /**
     * @brief Construct an empty BVH.
     * @param alloc Allocator instance (if PMR).
     */
    explicit BVH(const Allocator& alloc = Allocator())
        : m_core(alloc), m_query(m_core) {}

    /**
     * @brief Construct BVH from a range of entities.
     * @tparam Iter Input iterator type.
     * @param first Start of range.
     * @param last End of range.
     * @param alloc Allocator.
     */
    template <typename Iter>
    BVH(Iter first, Iter last, const Allocator& alloc = Allocator())
        : m_core(alloc), m_query(m_core) {
        insert(first, last);
    }

    /**
     * @brief Construct BVH with initial bounding box (for dynamic octree).
     * @param worldBounds Global bounding box.
     * @param maxDepth Maximum tree depth.
     * @param bucketSize Max elements per leaf.
     * @param alloc Allocator.
     */
    BVH(const aabb_type& worldBounds, size_type maxDepth = 8,
        size_type bucketSize = 8, const Allocator& alloc = Allocator())
        : m_core(worldBounds, maxDepth, bucketSize, alloc), m_query(m_core) {}

    // Disable copying (heavy), but allow moving
    BVH(const BVH&) = delete;
    BVH& operator=(const BVH&) = delete;
    BVH(BVH&&) noexcept = default;
    BVH& operator=(BVH&&) noexcept = default;

    // ------------------------------------------------------------------------
    //  Insertion / removal / update
    // ------------------------------------------------------------------------

    /**
     * @brief Insert a single entity.
     * @param entity User‑defined object.
     * @return True if inserted, false if already present.
     */
    bool insert(const entity_type& entity) {
        return m_core.insert(entity);
    }

    /**
     * @brief Insert a range of entities.
     * @tparam Iter Input iterator.
     * @param first Start.
     * @param last End.
     */
    template <typename Iter>
    void insert(Iter first, Iter last) {
        for (auto it = first; it != last; ++it) {
            m_core.insert(*it);
        }
    }

    /**
     * @brief Remove an entity.
     * @param entity Entity to remove.
     * @return True if removed.
     */
    bool remove(const entity_type& entity) {
        return m_core.remove(entity);
    }

    /**
     * @brief Update an entity's position or bounds (if changed).
     * @param entity Entity with updated geometry.
     * @return True if bounds changed and tree updated.
     */
    bool update(const entity_type& entity) {
        return m_core.update(entity);
    }

    /**
     * @brief Clear all entities.
     */
    void clear() {
        m_core.clear();
    }

    // ------------------------------------------------------------------------
    //  Geometry queries (delegated to ot_query)
    // ------------------------------------------------------------------------

    /**
     * @brief Query all entities whose AABB overlaps a given point.
     * @param point Query point.
     * @param out Output iterator (e.g., back_inserter).
     * @return Number of entities found.
     */
    template <typename OutputIt>
    size_type queryPoint(const point_type& point, OutputIt out) const {
        return m_query.queryPoint(point, out);
    }

    /**
     * @brief Query all entities intersecting a given AABB.
     * @param box Query box.
     * @param out Output iterator.
     * @return Count.
     */
    template <typename OutputIt>
    size_type queryBox(const aabb_type& box, OutputIt out) const {
        return m_query.queryBox(box, out);
    }

    /**
     * @brief Ray cast – find first intersection.
     * @param ray Ray in world space.
     * @param outHit Optional reference to store hit details.
     * @return True if any entity hit.
     */
    bool raycast(const Math::Ray<T, Dim>& ray,
                 typename query_engine::HitResult* outHit = nullptr) const {
        return m_query.raycast(ray, outHit);
    }

    /**
     * @brief Ray cast – collect all intersections along the ray.
     * @param ray Ray.
     * @param out Output iterator for hit results.
     * @return Number of hits.
     */
    template <typename OutputIt>
    size_type raycastAll(const Math::Ray<T, Dim>& ray, OutputIt out) const {
        return m_query.raycastAll(ray, out);
    }

    /**
     * @brief Find nearest neighbor to a point.
     * @param point Query point.
     * @param maxDist Maximum search distance (optional).
     * @return Optional pair (entity, squared distance).
     */
    std::optional<std::pair<entity_type, T>> nearestNeighbor(
        const point_type& point, T maxDist = std::numeric_limits<T>::max()) const {
        return m_query.nearestNeighbor(point, maxDist);
    }

    /**
     * @brief Find k nearest neighbors.
     * @param point Query point.
     * @param k Number of neighbors.
     * @param out Output iterator for (entity, squared distance) pairs.
     * @return Number of found neighbors (≤ k).
     */
    template <typename OutputIt>
    size_type kNearest(const point_type& point, size_type k, OutputIt out) const {
        return m_query.kNearest(point, k, out);
    }

    /**
     * @brief Frustum culling (3D only).
     * @param frustumPlanes Array of 6 planes (left, right, bottom, top, near, far).
     * @param out Output iterator.
     * @return Count of visible entities.
     */
    template <typename OutputIt>
    size_type cullFrustum(const std::array<Math::Plane<T, 3>, 6>& frustumPlanes,
                          OutputIt out) const {
        static_assert(Dim == 3, "Frustum culling only in 3D");
        return m_query.cullFrustum(frustumPlanes, out);
    }

    // ------------------------------------------------------------------------
    //  Transformed queries (object‑space to world‑space)
    // ------------------------------------------------------------------------

    /**
     * @brief Query point after applying inverse transform.
     * Useful for querying against a moving BVH.
     * @param worldPoint Point in world coordinates.
     * @param transform Transform from world to local.
     * @param out Output iterator.
     */
    template <typename OutputIt>
    size_type queryPointTransformed(const point_type& worldPoint,
                                    const transform_type& transform,
                                    OutputIt out) const {
        point_type localPoint = transform.inverse().transform(worldPoint);
        return queryPoint(localPoint, out);
    }

    // ------------------------------------------------------------------------
    //  Statistics and introspection
    // ------------------------------------------------------------------------

    /**
     * @brief Number of entities stored.
     */
    size_type size() const noexcept {
        return m_core.size();
    }

    /**
     * @brief Check if empty.
     */
    bool empty() const noexcept {
        return m_core.empty();
    }

    /**
     * @brief Total number of nodes in the tree.
     */
    size_type nodeCount() const noexcept {
        return m_core.nodeCount();
    }

    /**
     * @brief Height of the tree (max depth).
     */
    size_type height() const noexcept {
        return m_core.height();
    }

    /**
     * @brief Memory usage in bytes.
     */
    size_type memoryUsage() const noexcept {
        return m_core.memoryUsage();
    }

    /**
     * @brief Get the underlying core (for advanced use).
     */
    const CoreType& core() const noexcept { return m_core; }
    CoreType& core() noexcept { return m_core; }

    /**
     * @brief Force a rebuild (useful after many updates in static mode).
     */
    void rebuild() {
        if constexpr (std::is_same_v<Policy, StaticPolicy>) {
            m_core.rebuild();
        }
    }

    /**
     * @brief Optimize tree layout (e.g., reorder nodes for cache efficiency).
     */
    void optimize() {
        m_core.optimize();
    }

    // ------------------------------------------------------------------------
    //  Advanced: custom spatial hashing / morton code overrides
    // ------------------------------------------------------------------------

    /**
     * @brief Compute Morton code for a point (dimension‑aware).
     * @param point Input point.
     * @return 64‑bit Morton code.
     */
    uint64_t mortonCode(const point_type& point) const {
        return m_core.mortonCode(point);
    }

private:
    CoreType m_core;
    query_engine m_query;
};

// ----------------------------------------------------------------------------
//  Convenience type aliases for common use cases
// ----------------------------------------------------------------------------

// 2D quadtree (dynamic)
using Quadtree = BVH<Dim2, float, DynamicPolicy>;

// 3D octree (dynamic)
using Octree = BVH<Dim3, float, DynamicPolicy>;

// Static 3D BVH
using StaticBVH = BVH<Dim3, float, StaticPolicy>;

// Double‑precision 3D octree for simulation
using OctreeD = BVH<Dim3, double, DynamicPolicy>;

// 2D static BVH
using StaticQuadtree = BVH<Dim2, float, StaticPolicy>;

} // namespace OrthoTree

#endif // ORTHOTREE__BVH_H_INCLUDED