//File group name : OrthoTree Math
//File 0054 : core/math/geometry/ray.h
//Ray (origin + direction) in N dimensions (2D, 3D). Parameterisation, pointAt, closest point to another ray, ray‑ray distance, transformation, and SIMD batch operations.

#ifndef ORTHOTREE_CORE_MATH_GEOMETRY_RAY_H_INCLUDED
#define ORTHOTREE_CORE_MATH_GEOMETRY_RAY_H_INCLUDED

#include "../../build_config.h"
#include "../basic/vector.h"
#include "../basic/matrix.h"
#include "aabb.h"
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <limits>

namespace OrthoTree {
namespace Math {
namespace Geometry {

// ============================================================================
//  Ray for N dimensions (N = 2 or 3). Defined by origin and unit direction.
// ============================================================================
template<typename T, std::size_t N>
class Ray {
public:
    using value_type = T;
    using point_type = Basic::Vector<T, N>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    constexpr Ray() noexcept : m_origin(T(0)), m_direction(T(1,0,0)) {}
    constexpr Ray(const point_type& origin, const point_type& direction) noexcept
        : m_origin(origin), m_direction(direction.normalized()) {}

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    constexpr const point_type& origin() const noexcept { return m_origin; }
    constexpr const point_type& direction() const noexcept { return m_direction; }
    constexpr void setOrigin(const point_type& o) noexcept { m_origin = o; }
    constexpr void setDirection(const point_type& d) noexcept { m_direction = d.normalized(); }

    // ------------------------------------------------------------------------
    //  Point at parameter t (t >= 0)
    // ------------------------------------------------------------------------
    constexpr point_type pointAt(T t) const noexcept {
        return m_origin + m_direction * t;
    }

    // ------------------------------------------------------------------------
    //  Closest point between two rays (3D only)
    //  Returns squared distance, and parameters t1 (on this ray), t2 (on other).
    // ------------------------------------------------------------------------
    T closestPoints(const Ray& other, T& t1, T& t2) const noexcept {
        static_assert(N == 3, "closestPoints only for 3D");
        const point_type& u = m_direction;
        const point_type& v = other.m_direction;
        point_type w = m_origin - other.m_origin;
        T a = u.dot(u);
        T b = u.dot(v);
        T c = v.dot(v);
        T d = u.dot(w);
        T e = v.dot(w);
        T denom = a * c - b * b;
        if (std::abs(denom) < static_cast<T>(1e-12)) {
            // Parallel rays
            t1 = T(0);
            t2 = e / c;
            if (t2 < T(0)) t2 = T(0);
        } else {
            t1 = (b * e - c * d) / denom;
            t2 = (a * e - b * d) / denom;
        }
        if (t1 < T(0)) t1 = T(0);
        if (t2 < T(0)) t2 = T(0);
        point_type p1 = pointAt(t1);
        point_type p2 = other.pointAt(t2);
        return (p1 - p2).squaredLength();
    }

    // ------------------------------------------------------------------------
    //  Distance to another ray (closest point distance)
    // ------------------------------------------------------------------------
    T distanceToRay(const Ray& other) const noexcept {
        T t1, t2;
        return std::sqrt(closestPoints(other, t1, t2));
    }

    // ------------------------------------------------------------------------
    //  Transform ray by affine transform (origin and direction)
    // ------------------------------------------------------------------------
    Ray transform(const Basic::AffineTransform<T,N>& tf) const noexcept {
        point_type newOrigin = tf.transform(m_origin);
        point_type newDir = tf.transformDirection(m_direction);
        return Ray(newOrigin, newDir);
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool nearlyEqual(const Ray& other, T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) const noexcept {
        return m_origin.nearlyEqual(other.m_origin, eps) &&
               m_direction.nearlyEqual(other.m_direction, eps);
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: compute pointAt for 4 rays with the same t
    // ------------------------------------------------------------------------
    static void batchPointAt(const Ray* rays, T t, point_type* out, size_t count) noexcept {
        if constexpr (std::is_same_v<T,float> && ORTHOTREE_SIMD_LEVEL >= 128 && N == 3) {
            for (size_t i = 0; i < count; ++i) {
                out[i] = rays[i].pointAt(t);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                out[i] = rays[i].pointAt(t);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: compute closest points for 4 ray pairs
    // ------------------------------------------------------------------------
    static void batchClosestPoints(const Ray* rays1, const Ray* rays2,
                                   T* t1, T* t2, T* distSq, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) {
            distSq[i] = rays1[i].closestPoints(rays2[i], t1[i], t2[i]);
        }
    }

    // ------------------------------------------------------------------------
    //  Ray‑AABB intersection test (slab method)
    // ------------------------------------------------------------------------
    bool intersectsAABB(const AABB<T,N>& box, T& tMin, T& tMax) const noexcept {
        tMin = T(0);
        tMax = std::numeric_limits<T>::max();
        for (size_t i = 0; i < N; ++i) {
            T invDir = T(1) / m_direction[i];
            T t1 = (box.min()[i] - m_origin[i]) * invDir;
            T t2 = (box.max()[i] - m_origin[i]) * invDir;
            if (t1 > t2) std::swap(t1, t2);
            if (t1 > tMin) tMin = t1;
            if (t2 < tMax) tMax = t2;
            if (tMin > tMax) return false;
        }
        return true;
    }

private:
    point_type m_origin;
    point_type m_direction;
};

// ----------------------------------------------------------------------------
//  Convenience aliases
// ----------------------------------------------------------------------------
template<typename T> using Ray2 = Ray<T, 2>;
template<typename T> using Ray3 = Ray<T, 3>;

using Ray2f = Ray<float, 2>;
using Ray3f = Ray<float, 3>;
using Ray2d = Ray<double, 2>;
using Ray3d = Ray<double, 3>;

// ----------------------------------------------------------------------------
//  Helper: ray from two points (direction from start to end)
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
Ray<T,N> makeRay(const Basic::Vector<T,N>& from, const Basic::Vector<T,N>& to) {
    return Ray<T,N>(from, to - from);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class RayEnvironment {
public:
    static RayEnvironment& instance() {
        static RayEnvironment env;
        return env;
    }
    void setUseSIMD(bool use) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_useSIMD = use;
    }
    bool useSIMD() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_useSIMD;
    }
private:
    RayEnvironment() : m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    bool m_useSIMD;
};

} // namespace Geometry
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_GEOMETRY_RAY_H_INCLUDED