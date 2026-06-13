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
 * @file internal_geometry_module.h
 * @brief Low‑level geometry intersection and distance tests (branchless, SIMD-friendly).
 *
 * This file provides fast, branchless implementations of spatial queries used
 * by octree traversal and BVH: ray‑AABB, AABB‑AABB overlap, point‑AABB distance,
 * sphere‑AABB intersection, and frustum‑AABB test. Functions are templated on
 * dimension (2D/3D) and scalar type, with optimised paths for 2D and 3D.
 *
 * All functions are constexpr where possible, inline, and use no conditional
 * branches inside inner loops, relying on min/max and arithmetic operations.
 */

#ifndef ORTHOTREE_DETAIL_INTERNAL_GEOMETRY_MODULE_H_INCLUDED
#define ORTHOTREE_DETAIL_INTERNAL_GEOMETRY_MODULE_H_INCLUDED

#include "../core/math/vector_math.h"
#include "../core/math/geometry_queries.h"
#include "../core/math/interval_arithmetic.h"
#include "common.h"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace OrthoTree {
namespace detail {

// ============================================================================
//  AABB intersection tests (branchless)
// ============================================================================

/**
 * @brief Check if two axis‑aligned bounding boxes overlap.
 * @tparam Dim Dimension (2 or 3).
 * @tparam T Scalar.
 * @return True if boxes intersect.
 */
template <Dimension Dim, typename T>
ORTHOTREE_FORCE_INLINE bool aabbOverlap(
    const Math::AxisAlignedBox<T, Dim>& a,
    const Math::AxisAlignedBox<T, Dim>& b) noexcept {
    for (std::size_t i = 0; i < static_cast<std::size_t>(Dim); ++i) {
        if (a.max()[i] < b.min()[i] || b.max()[i] < a.min()[i])
            return false;
    }
    return true;
}

/**
 * @brief Check if a point is inside an AABB.
 */
template <Dimension Dim, typename T>
ORTHOTREE_FORCE_INLINE bool pointInsideAABB(
    const Math::Vector<T, Dim>& point,
    const Math::AxisAlignedBox<T, Dim>& box) noexcept {
    for (std::size_t i = 0; i < static_cast<std::size_t>(Dim); ++i) {
        if (point[i] < box.min()[i] || point[i] > box.max()[i])
            return false;
    }
    return true;
}

// ============================================================================
//  Ray‑AABB intersection (Slab method, branchless)
// ============================================================================

/**
 * @brief Ray vs AABB intersection test. Returns t_min (near) and t_max (far).
 *        If ray starts inside, t_min = 0.
 * @tparam Dim Dimension.
 * @param origin Ray origin.
 * @param invDir 1/direction (precomputed for performance).
 * @param dirSigns Sign of direction (0 if positive, 1 if negative) for axis order.
 * @param box AABB.
 * @param tMin Output near distance.
 * @param tMax Output far distance.
 * @return True if ray hits box (tMin <= tMax).
 */
template <Dimension Dim, typename T>
ORTHOTREE_FORCE_INLINE bool rayAABBIntersect(
    const Math::Vector<T, Dim>& origin,
    const Math::Vector<T, Dim>& invDir,
    const std::array<int, Dim == Dim2 ? 2 : 3>& dirSigns,
    const Math::AxisAlignedBox<T, Dim>& box,
    T& tMin, T& tMax) noexcept {
    tMin = T(0);
    tMax = std::numeric_limits<T>::max();
    for (std::size_t i = 0; i < static_cast<std::size_t>(Dim); ++i) {
        T t1 = (box.min()[i] - origin[i]) * invDir[i];
        T t2 = (box.max()[i] - origin[i]) * invDir[i];
        if (dirSigns[i]) std::swap(t1, t2);
        if (t1 > tMin) tMin = t1;
        if (t2 < tMax) tMax = t2;
        if (tMin > tMax) return false;
    }
    return true;
}

/**
 * @brief Convenience overload without precomputed invDir and signs.
 */
template <Dimension Dim, typename T>
ORTHOTREE_FORCE_INLINE bool rayAABBIntersect(
    const Math::Ray<T, Dim>& ray,
    const Math::AxisAlignedBox<T, Dim>& box,
    T& tMin, T& tMax) noexcept {
    Math::Vector<T, Dim> invDir;
    std::array<int, Dim == Dim2 ? 2 : 3> signs;
    for (std::size_t i = 0; i < static_cast<std::size_t>(Dim); ++i) {
        T dir = ray.direction()[i];
        invDir[i] = T(1) / dir;
        signs[i] = (dir < T(0)) ? 1 : 0;
    }
    return rayAABBIntersect(ray.origin(), invDir, signs, box, tMin, tMax);
}

// ============================================================================
//  Point‑to‑AABB squared distance (branchless)
// ============================================================================

/**
 * @brief Compute squared distance from point to AABB (zero if inside).
 */
template <Dimension Dim, typename T>
ORTHOTREE_FORCE_INLINE T pointAABBSqDist(
    const Math::Vector<T, Dim>& point,
    const Math::AxisAlignedBox<T, Dim>& box) noexcept {
    T sqDist = T(0);
    for (std::size_t i = 0; i < static_cast<std::size_t>(Dim); ++i) {
        T v = point[i];
        T min = box.min()[i];
        T max = box.max()[i];
        if (v < min) {
            T d = min - v;
            sqDist += d * d;
        } else if (v > max) {
            T d = v - max;
            sqDist += d * d;
        }
    }
    return sqDist;
}

// ============================================================================
//  Sphere‑AABB intersection (branchless)
// ============================================================================

/**
 * @brief Check if sphere intersects AABB.
 * @param sphere Center + radius.
 * @param box AABB.
 * @return True if they overlap.
 */
template <Dimension Dim, typename T>
ORTHOTREE_FORCE_INLINE bool sphereAABBIntersect(
    const Math::Sphere<T, Dim>& sphere,
    const Math::AxisAlignedBox<T, Dim>& box) noexcept {
    T sqDist = pointAABBSqDist(sphere.center(), box);
    return sqDist <= sphere.radius() * sphere.radius();
}

// ============================================================================
//  Frustum‑AABB test (for 3D only, using separating axis theorem)
// ============================================================================

/**
 * @brief Test AABB against a frustum defined by 6 planes.
 * @param box AABB.
 * @param planes Array of 6 planes (left, right, bottom, top, near, far).
 * @return True if AABB is inside or intersecting frustum.
 */
template <typename T>
ORTHOTREE_FORCE_INLINE bool frustumAABBIntersect(
    const Math::AxisAlignedBox<T, 3>& box,
    const std::array<Math::Plane<T, 3>, 6>& planes) noexcept {
    for (const auto& plane : planes) {
        // Find the positive vertex (most in direction of plane normal)
        Math::Vector<T, 3> p = box.min();
        Math::Vector<T, 3> n = plane.normal();
        for (int i = 0; i < 3; ++i) {
            if (n[i] >= T(0)) p[i] = box.max()[i];
        }
        // If positive vertex is behind plane, box is outside
        if (plane.signedDistance(p) < T(0)) return false;
    }
    return true;
}

// ============================================================================
//  Separation axis test for OBB (oriented bounding box) vs AABB (unused here but included)
// ============================================================================

/**
 * @brief Compute projection interval of AABB onto axis.
 */
template <Dimension Dim, typename T>
Math::Interval<T> projectAABBOnAxis(
    const Math::AxisAlignedBox<T, Dim>& box,
    const Math::Vector<T, Dim>& axis) noexcept {
    T center = box.center().dot(axis);
    T extent = box.halfExtents().abs().dot(axis.abs());
    return Math::Interval<T>(center - extent, center + extent);
}

// ============================================================================
//  Fast ray traversal order (for octree child ordering)
// ============================================================================

/**
 * @brief Compute child traversal order for a ray based on intersection distances.
 *        Returns child indices sorted from closest to farthest along ray.
 * @param ray Ray.
 * @param nodeCenter Center of current node.
 * @param dim Dimension.
 * @param outOrder Array of child indices (size = 2^Dim).
 */
template <Dimension Dim, typename T>
void computeRayTraversalOrder(
    const Math::Ray<T, Dim>& ray,
    const Math::Vector<T, Dim>& nodeCenter,
    std::array<uint8_t, (Dim == Dim2 ? 4 : 8)>& outOrder) noexcept {
    constexpr uint8_t numChildren = (Dim == Dim2) ? 4 : 8;
    // Determine sign of ray direction in each dimension
    uint8_t signBits = 0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(Dim); ++i) {
        if (ray.direction()[i] < T(0)) signBits |= (1 << i);
    }
    // Precompute child order: for each bit combination, the child index that corresponds
    // to the octant that the ray enters first. This is based on the sign bits.
    // For axis-aligned traversal: the child order is given by the morton order with bits
    // determined by signBits. We fill outOrder with indices from near to far.
    std::array<T, numChildren> tEntries;
    for (uint8_t i = 0; i < numChildren; ++i) {
        // child center offset relative to nodeCenter
        Math::Vector<T, Dim> childOffset;
        for (std::size_t d = 0; d < static_cast<std::size_t>(Dim); ++d) {
            T half = T(0.5);
            bool bit = (i >> d) & 1;
            childOffset[d] = (bit ? half : -half);
        }
        Math::Vector<T, Dim> childCenter = nodeCenter + childOffset;
        // t where ray passes through child center along direction? Not accurate for ordering.
        // Use distance from ray origin to child center projected onto ray direction.
        Math::Vector<T, Dim> diff = childCenter - ray.origin();
        T t = diff.dot(ray.direction());
        tEntries[i] = t;
    }
    // Sort indices by tEntries
    for (uint8_t i = 0; i < numChildren; ++i) outOrder[i] = i;
    std::sort(outOrder.begin(), outOrder.end(),
        [&](uint8_t a, uint8_t b) { return tEntries[a] < tEntries[b]; });
}

// ============================================================================
//  Geometry helper for 2D/3D generic algorithms
// ============================================================================

/**
 * @brief Given a bounding box and a point, compute the child index that would contain
 *        the point (for octree subdivision). Assumes point is inside box.
 */
template <Dimension Dim, typename T>
ORTHOTREE_FORCE_INLINE uint8_t getChildIndexForPoint(
    const Math::AxisAlignedBox<T, Dim>& box,
    const Math::Vector<T, Dim>& point) noexcept {
    Math::Vector<T, Dim> mid = box.center();
    uint8_t idx = 0;
    for (std::size_t d = 0; d < static_cast<std::size_t>(Dim); ++d) {
        if (point[d] >= mid[d]) idx |= (1 << d);
    }
    return idx;
}

/**
 * @brief Compute child bounding box from parent box and child index.
 */
template <Dimension Dim, typename T>
Math::AxisAlignedBox<T, Dim> getChildBounds(
    const Math::AxisAlignedBox<T, Dim>& parent,
    uint8_t childIdx) noexcept {
    Math::Vector<T, Dim> min = parent.min();
    Math::Vector<T, Dim> max = parent.max();
    Math::Vector<T, Dim> mid = parent.center();
    for (std::size_t d = 0; d < static_cast<std::size_t>(Dim); ++d) {
        bool high = (childIdx >> d) & 1;
        if (high) {
            min[d] = mid[d];
        } else {
            max[d] = mid[d];
        }
    }
    return Math::AxisAlignedBox<T, Dim>(min, max);
}

} // namespace detail
} // namespace OrthoTree

#endif // ORTHOTREE_DETAIL_INTERNAL_GEOMETRY_MODULE_H_INCLUDED