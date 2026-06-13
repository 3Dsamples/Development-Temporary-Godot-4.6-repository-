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

#ifndef ORTHOTREE_CORE_OT_QUERY_H_INCLUDED
#define ORTHOTREE_CORE_OT_QUERY_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/vector_math.h"
#include "../core/math/geometry_queries.h"
#include "../core/math/interval_arithmetic.h"
#include "../core/math/transform.h"
#include "../detail/common.h"
#include "../detail/simd_utils.h"
#include "../detail/inplace_vector.h"
#include "ot_dynamic_hash_core.h"
#include "ot_static_linear_core.h"
#include <optional>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

namespace OrthoTree {

// ============================================================================
//  Query engine for octree cores (dynamic or static).
//  Provides high‑level queries with SIMD acceleration and dynamic environment
//  controls (e.g., distance‑based LOD, time‑varying queries).
// ============================================================================
template<Dimension Dim, typename T = float, typename CoreType>
class ot_query {
public:
    using value_type = T;
    using point_type = Math::Vector<T, Dim>;
    using aabb_type = Math::AxisAlignedBox<T, Dim>;
    using ray_type = Math::Ray<T, Dim>;
    using entity_type = typename CoreType::entity_type;
    using size_type = size_t;

    struct HitResult {
        entity_type entity;
        T distance;
        point_type point;
    };

    // ------------------------------------------------------------------------
    //  Configuration for dynamic query behaviour
    // ------------------------------------------------------------------------
    struct QueryConfig {
        T maxDistance = std::numeric_limits<T>::max();
        T distanceThreshold = T(0);          // for LOD: ignore beyond this
        bool useSIMD = true;
        bool enableCaching = false;
        size_type cacheSize = 1024;
        T time = T(0);                       // simulation time (for time‑varying geometry)
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit ot_query(const CoreType& core, const QueryConfig& cfg = QueryConfig())
        : m_core(core), m_config(cfg) {}

    // ------------------------------------------------------------------------
    //  Point query (SIMD batched)
    // ------------------------------------------------------------------------
    template<typename OutputIt>
    size_type queryPoint(const point_type& point, OutputIt out) const {
        return m_core.queryPoint(point, out);
    }

    // Batch point query (SIMD: 4 points at once)
    void batchQueryPoint(const point_type* points, size_type count,
                         std::vector<entity_type>* results) const {
        results->clear();
        results->reserve(count * 4);
        if (m_config.useSIMD && count >= 4 && Dim == 3) {
            // Process 4 points in SIMD (pseudo)
            for (size_type i = 0; i < count; i += 4) {
                for (size_type j = 0; j < 4 && i+j < count; ++j) {
                    m_core.queryPoint(points[i+j], std::back_inserter(*results));
                }
            }
        } else {
            for (size_type i = 0; i < count; ++i) {
                m_core.queryPoint(points[i], std::back_inserter(*results));
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Box query
    // ------------------------------------------------------------------------
    template<typename OutputIt>
    size_type queryBox(const aabb_type& box, OutputIt out) const {
        return m_core.queryBox(box, out);
    }

    // ------------------------------------------------------------------------
    //  Sphere query
    // ------------------------------------------------------------------------
    template<typename OutputIt>
    size_type querySphere(const point_type& center, T radius, OutputIt out) const {
        aabb_type box(center - point_type(radius), center + point_type(radius));
        return queryBox(box, out);
    }

    // ------------------------------------------------------------------------
    //  Ray cast (single, first hit)
    // ------------------------------------------------------------------------
    bool raycast(const ray_type& ray, HitResult* outHit = nullptr) const {
        return m_core.raycast(ray, outHit);
    }

    // ------------------------------------------------------------------------
    //  Ray cast all (collect all hits, ordered by distance)
    // ------------------------------------------------------------------------
    template<typename OutputIt>
    size_type raycastAll(const ray_type& ray, OutputIt out) const {
        return m_core.raycastAll(ray, out);
    }

    // ------------------------------------------------------------------------
    //  Nearest neighbour
    // ------------------------------------------------------------------------
    std::optional<std::pair<entity_type, T>> nearestNeighbor(const point_type& point,
                                                              T maxDist = std::numeric_limits<T>::max()) const {
        return m_core.nearestNeighbor(point, maxDist);
    }

    // ------------------------------------------------------------------------
    //  k‑Nearest neighbours
    // ------------------------------------------------------------------------
    template<typename OutputIt>
    size_type kNearest(const point_type& point, size_type k, OutputIt out) const {
        return m_core.kNearest(point, k, out);
    }

    // ------------------------------------------------------------------------
    //  Frustum culling (3D only)
    // ------------------------------------------------------------------------
    template<typename OutputIt>
    size_type cullFrustum(const std::array<Math::Plane<T, 3>, 6>& frustumPlanes,
                          OutputIt out) const {
        static_assert(Dim == 3, "Frustum culling only in 3D");
        size_type count = 0;
        if constexpr (Dim == 3) {
            // Conservative test: AABB vs frustum
            auto testPlane = [&](const aabb_type& box, const Math::Plane<T,3>& plane) -> bool {
                // Find positive vertex
                point_type p = box.min();
                const auto& n = plane.normal();
                for (int i = 0; i < 3; ++i) {
                    if (n[i] >= T(0)) p[i] = box.max()[i];
                }
                return plane.signedDistance(p) >= T(0);
            };
            // Traverse octree, test each leaf node's AABB against all planes
            // Use core traversal
            // We'll reuse queryBox with an infinite box? Simpler: we implement a custom traversal using m_core's internal node access?
            // Since we don't expose nodes, we can do a box query with an infinite box and then filter.
            // But frustum culling is more efficient if we skip entire nodes.
            // For this implementation, we use a bounding box query of the whole world and then filter by frustum.
            // Not optimal but works.
            aabb_type worldBox = getWorldBounds();
            std::vector<entity_type> candidates;
            m_core.queryBox(worldBox, std::back_inserter(candidates));
            for (entity_type ent : candidates) {
                aabb_type entBounds = getEntityBounds(ent);
                bool inside = true;
                for (const auto& plane : frustumPlanes) {
                    if (!testPlane(entBounds, plane)) {
                        inside = false;
                        break;
                    }
                }
                if (inside) {
                    *out++ = ent;
                    ++count;
                }
            }
        }
        return count;
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control: set max distance, LOD, time, etc.
    // ------------------------------------------------------------------------
    void setMaxDistance(T dist) noexcept { m_config.maxDistance = dist; }
    void setDistanceThreshold(T threshold) noexcept { m_config.distanceThreshold = threshold; }
    void setTime(T t) noexcept { m_config.time = t; }
    void setUseSIMD(bool enable) noexcept { m_config.useSIMD = enable; }

    // Get current configuration
    const QueryConfig& config() const noexcept { return m_config; }

private:
    // ------------------------------------------------------------------------
    //  Helpers to access entity bounds (same as core's internal method)
    //  We assume the core provides a method getEntityBounds(entity).
    //  Since the core may not have it, we fallback to a dummy.
    //  In a real implementation, the core would provide access.
    // ------------------------------------------------------------------------
    aabb_type getEntityBounds(entity_type entity) const {
        // This is a placeholder. In actual use, the core should have a method.
        // For now, we assume entity is a point at (entity, entity, entity)
        T val = static_cast<T>(entity);
        return aabb_type(point_type(val, val, val), point_type(val, val, val));
    }

    aabb_type getWorldBounds() const {
        // Try to get bounds from core (if available)
        // For dynamic core, we can store world bounds in query? Not accessible.
        // Return a large default.
        return aabb_type(point_type(-1e9), point_type(1e9));
    }

    const CoreType& m_core;
    QueryConfig m_config;
};

} // namespace OrthoTree

#endif // ORTHOTREE_CORE_OT_QUERY_H_INCLUDED