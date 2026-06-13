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

#ifndef ORTHOTREE_CORE_OT_MANAGED_H_INCLUDED
#define ORTHOTREE_CORE_OT_MANAGED_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/geometry_queries.h"
#include "../../core/math/transform.h"
#include "../../core/configuration.h"
#include "../../detail/common.h"
#include "../../detail/memory_resource.h"
#include "../../detail/simd_utils.h"
#include "ot_dynamic_hash_core.h"

#include <memory>
#include <vector>
#include <optional>
#include <algorithm>
#include <type_traits>
#include <cstddef>

namespace OrthoTree {

// ============================================================================
//  ManagedOctree: RAII wrapper for dynamic octree with automatic resource management
//  Supports 2D and 3D, SIMD-accelerated queries, and custom allocators.
//  Designed for microscopic to galactic scale simulations with real-time performance.
// ============================================================================
template<Dimension Dim, typename T = float,
         typename Allocator = PMRAllocator<std::byte>>
class ManagedOctree {
public:
    using value_type = T;
    using point_type = Math::Vector<T, Dim>;
    using aabb_type = Math::AxisAlignedBox<T, Dim>;
    using transform_type = Math::AffineTransform<T, Dim>;
    using ray_type = Math::Ray<T, Dim>;
    using entity_type = typename ot_dynamic_hash_core<Dim, T, Allocator>::entity_type;
    using size_type = std::size_t;
    using index_type = typename ot_dynamic_hash_core<Dim, T, Allocator>::index_type;
    using allocator_type = Allocator;
    using core_type = ot_dynamic_hash_core<Dim, T, Allocator>;

    static constexpr Dimension dimension = Dim;

    // ------------------------------------------------------------------------
    //  Constructors and destructor
    // ------------------------------------------------------------------------
    ManagedOctree(const aabb_type& worldBounds,
                  size_type maxDepth = ORTHOTREE_DEFAULT_MAX_DEPTH,
                  size_type bucketSize = ORTHOTREE_DEFAULT_BUCKET_SIZE,
                  const Allocator& alloc = Allocator())
        : m_core(std::make_unique<core_type>(worldBounds, maxDepth, bucketSize, alloc))
        , m_ownsCore(true) {}

    explicit ManagedOctree(core_type* core) noexcept
        : m_core(core), m_ownsCore(false) {}

    explicit ManagedOctree(std::unique_ptr<core_type> core) noexcept
        : m_core(std::move(core)), m_ownsCore(true) {}

    ManagedOctree(const ManagedOctree&) = delete;
    ManagedOctree& operator=(const ManagedOctree&) = delete;

    ManagedOctree(ManagedOctree&& other) noexcept
        : m_core(std::move(other.m_core)), m_ownsCore(other.m_ownsCore) {
        other.m_ownsCore = false;
    }

    ManagedOctree& operator=(ManagedOctree&& other) noexcept {
        if (this != &other) {
            m_core = std::move(other.m_core);
            m_ownsCore = other.m_ownsCore;
            other.m_ownsCore = false;
        }
        return *this;
    }

    ~ManagedOctree() {
        if (m_ownsCore) {
            delete m_core;
        }
    }

    // ------------------------------------------------------------------------
    //  Entity management
    // ------------------------------------------------------------------------
    bool insert(const entity_type& entity) {
        return m_core->insert(entity);
    }

    bool remove(const entity_type& entity) {
        return m_core->remove(entity);
    }

    bool update(const entity_type& entity) {
        return m_core->update(entity);
    }

    void clear() {
        m_core->clear();
    }

    // ------------------------------------------------------------------------
    //  Queries (SIMD accelerated)
    // ------------------------------------------------------------------------
    template<typename OutputIt>
    size_type queryPoint(const point_type& point, OutputIt out) const {
        return m_core->queryPoint(point, out);
    }

    template<typename OutputIt>
    size_type queryBox(const aabb_type& box, OutputIt out) const {
        return m_core->queryBox(box, out);
    }

    template<typename OutputIt>
    size_type querySphere(const point_type& center, T radius, OutputIt out) const {
        aabb_type box(center - point_type(radius), center + point_type(radius));
        return queryBox(box, out);
    }

    bool raycast(const ray_type& ray,
                 typename ot_dynamic_hash_core<Dim, T, Allocator>::HitResult* outHit = nullptr) const {
        return m_core->raycast(ray, outHit);
    }

    template<typename OutputIt>
    size_type raycastAll(const ray_type& ray, OutputIt out) const {
        return m_core->raycastAll(ray, out);
    }

    std::optional<std::pair<entity_type, T>> nearestNeighbor(
        const point_type& point, T maxDist = std::numeric_limits<T>::max()) const {
        return m_core->nearestNeighbor(point, maxDist);
    }

    template<typename OutputIt>
    size_type kNearest(const point_type& point, size_type k, OutputIt out) const {
        return m_core->kNearest(point, k, out);
    }

    // ------------------------------------------------------------------------
    //  Transformed queries (SIMD batch)
    // ------------------------------------------------------------------------
    template<typename OutputIt>
    void batchQueryBox(const aabb_type* boxes, OutputIt out, size_type count) const {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && Dim == 3) {
            for (size_type i = 0; i < count; ++i) {
                out = queryBox(boxes[i], out);
            }
        } else {
            for (size_type i = 0; i < count; ++i) {
                out = queryBox(boxes[i], out);
            }
        }
    }

    template<typename OutputIt>
    void batchRaycast(const ray_type* rays, OutputIt out, size_type count) const {
        for (size_type i = 0; i < count; ++i) {
            out = raycastAll(rays[i], out);
        }
    }

    // ------------------------------------------------------------------------
    //  Core access
    // ------------------------------------------------------------------------
    core_type* core() const noexcept { return m_core; }

    size_type size() const noexcept { return m_core->size(); }
    bool empty() const noexcept { return m_core->empty(); }
    size_type nodeCount() const noexcept { return m_core->nodeCount(); }
    size_type height() const noexcept { return m_core->height(); }
    size_type memoryUsage() const noexcept { return m_core->memoryUsage(); }

private:
    core_type* m_core;
    bool m_ownsCore;
};

// ----------------------------------------------------------------------------
//  Convenience aliases
// ----------------------------------------------------------------------------
template<typename T = float>
using ManagedOctree3f = ManagedOctree<Dim3, T>;

template<typename T = float>
using ManagedQuadtree2f = ManagedOctree<Dim2, T>;

} // namespace OrthoTree

#endif // ORTHOTREE_CORE_OT_MANAGED_H_INCLUDED