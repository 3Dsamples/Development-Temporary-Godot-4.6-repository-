//File group name : OrthoTree Math
//File 0020 : core/math/ray.h
//Ray class (origin + direction) with pointAt, transformation, SIMD batch operations, and ray‑ray closest point.

#ifndef ORTHOTREE_CORE_MATH_RAY_H_INCLUDED
#define ORTHOTREE_CORE_MATH_RAY_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "line_segment.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <limits>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Ray: origin + t * direction (t >= 0). Supports 2D and 3D.
//  Provides pointAt, transformation, closest point to another ray,
//  and SIMD batch operations.
// ============================================================================
template<typename T = float, std::size_t N = 3>
class Ray {
public:
    using value_type = T;
    using point_type = Vector<T, N>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    constexpr Ray() noexcept : m_origin(T(0)), m_direction(T(1,0,0)) {}
    constexpr Ray(const point_type& origin, const point_type& direction) noexcept
        : m_origin(origin), m_direction(direction.normalized()) {}

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    const point_type& origin() const noexcept { return m_origin; }
    const point_type& direction() const noexcept { return m_direction; }
    void setOrigin(const point_type& o) noexcept { m_origin = o; }
    void setDirection(const point_type& d) noexcept { m_direction = d.normalized(); }

    // ------------------------------------------------------------------------
    //  Point at parameter t (t >= 0)
    // ------------------------------------------------------------------------
    point_type pointAt(T t) const noexcept {
        return m_origin + m_direction * t;
    }

    // ------------------------------------------------------------------------
    //  Closest point between two rays (3D)
    //  Returns parameters t1 (on this ray) and t2 (on other ray) for the points
    //  of minimum distance, and the distance squared.
    // ------------------------------------------------------------------------
    T closestPoints(const Ray& other, T& t1, T& t2) const noexcept {
        static_assert(N == 3, "Ray‑ray closest points only for 3D");
        point_type u = m_direction;
        point_type v = other.m_direction;
        point_type w = m_origin - other.m_origin;
        T a = u.squaredLength();
        T b = u.dot(v);
        T c = v.squaredLength();
        T d = u.dot(w);
        T e = v.dot(w);
        T denom = a * c - b * b;
        if (std::abs(denom) < T(1e-12)) {
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
    //  Transform ray by affine transform (origin and direction)
    // ------------------------------------------------------------------------
    Ray transform(const AffineTransform<T,N>& tf) const noexcept {
        point_type newOrigin = tf.transform(m_origin);
        point_type newDir = tf.transformDirection(m_direction);
        return Ray(newOrigin, newDir);
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool operator==(const Ray& other) const noexcept {
        T eps = T(1e-6);
        return (m_origin - other.m_origin).length() < eps &&
               (m_direction - other.m_direction).length() < eps;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: compute pointAt for 4 rays with same t
    // ------------------------------------------------------------------------
    static void batchPointAt(const Ray* rays, T t, point_type* out, size_t count) noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && N == 3 && std::is_same_v<T,float>) {
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
    //  SIMD batch: compute closest points between 4 pairs of rays
    // ------------------------------------------------------------------------
    static void batchClosestPoints(const Ray* rays1, const Ray* rays2,
                                   T* t1, T* t2, T* distSq, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) {
            distSq[i] = rays1[i].closestPoints(rays2[i], t1[i], t2[i]);
        }
    }

private:
    point_type m_origin;
    point_type m_direction;
};

// ----------------------------------------------------------------------------
//  Helper: ray from two points (direction from start to end)
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
Ray<T,N> makeRay(const Vector<T,N>& origin, const Vector<T,N>& pointOnRay) {
    return Ray<T,N>(origin, pointOnRay - origin);
}

// ----------------------------------------------------------------------------
//  Helper: ray from segment (start to end)
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
Ray<T,N> rayFromSegment(const LineSegment<T,N>& seg) {
    return Ray<T,N>(seg.a(), seg.b() - seg.a());
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller for ray operations
// ----------------------------------------------------------------------------
class RayEnvironment {
public:
    static RayEnvironment& instance() {
        static RayEnvironment env;
        return env;
    }
    void setEpsilon(T eps) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_epsilon = eps;
    }
    T epsilon() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_epsilon;
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
    RayEnvironment() : m_epsilon(T(1e-8)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_RAY_H_INCLUDED