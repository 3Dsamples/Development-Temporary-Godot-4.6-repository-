//File group name : OrthoTree Math
//File 0016 : core/math/plane.h
//Plane representation (normal + distance), point classification, projection, intersection with ray/line, and SIMD batch operations.

#ifndef ORTHOTREE_CORE_MATH_PLANE_H_INCLUDED
#define ORTHOTREE_CORE_MATH_PLANE_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "ray_intersection.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <limits>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Plane class: defined by normal (unit vector) and distance from origin.
//  Supports signed distance, projection, point classification, and intersection
//  with ray, line, and other planes.
// ============================================================================

template<typename T = float>
class Plane {
public:
    using value_type = T;
    using point_type = Vector<T, 3>;
    using ray_type = Ray<T, 3>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    constexpr Plane() noexcept : m_normal(point_type(0,0,1)), m_d(T(0)) {}
    constexpr Plane(const point_type& normal, T d) noexcept
        : m_normal(normal.normalized()), m_d(d) {}
    constexpr Plane(const point_type& normal, const point_type& point) noexcept
        : m_normal(normal.normalized()), m_d(-m_normal.dot(point)) {}
    Plane(const point_type& p1, const point_type& p2, const point_type& p3) noexcept {
        m_normal = cross(p2 - p1, p3 - p1).normalized();
        m_d = -m_normal.dot(p1);
    }

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    const point_type& normal() const noexcept { return m_normal; }
    T d() const noexcept { return m_d; }
    void setNormal(const point_type& n) noexcept { m_normal = n.normalized(); }
    void setD(T d) noexcept { m_d = d; }

    // ------------------------------------------------------------------------
    //  Distance / classification
    // ------------------------------------------------------------------------
    T signedDistance(const point_type& point) const noexcept {
        return m_normal.dot(point) + m_d;
    }
    T distance(const point_type& point) const noexcept {
        return std::abs(signedDistance(point));
    }
    int classify(const point_type& point, T eps = T(1e-8)) const noexcept {
        T dist = signedDistance(point);
        if (dist > eps) return 1;   // front
        if (dist < -eps) return -1; // back
        return 0;                   // on
    }

    // ------------------------------------------------------------------------
    //  Project point onto plane
    // ------------------------------------------------------------------------
    point_type project(const point_type& point) const noexcept {
        return point - m_normal * signedDistance(point);
    }

    // ------------------------------------------------------------------------
    //  Intersection with ray
    // ------------------------------------------------------------------------
    bool intersectRay(const ray_type& ray, T& t) const noexcept {
        T denom = m_normal.dot(ray.direction());
        if (std::abs(denom) < T(1e-8)) return false;
        t = -(m_normal.dot(ray.origin()) + m_d) / denom;
        return t >= T(0);
    }

    // ------------------------------------------------------------------------
    //  Intersection of two planes (returns line direction and point)
    //  Returns false if planes are parallel.
    // ------------------------------------------------------------------------
    bool intersectPlanes(const Plane& other, point_type& lineDir, point_type& linePoint) const noexcept {
        lineDir = cross(m_normal, other.m_normal);
        T lenSq = lineDir.squaredLength();
        if (lenSq < T(1e-12)) return false;
        lineDir = lineDir / std::sqrt(lenSq);
        // Solve for point on both planes
        Matrix<T, 2, 2> A;
        A(0,0) = m_normal[0]; A(0,1) = m_normal[1];
        A(1,0) = other.m_normal[0]; A(1,1) = other.m_normal[1];
        Vector<T,2> b(-m_d, -other.m_d);
        Vector<T,2> xy = A.inverse() * b; // simplified: 2x2 inverse
        linePoint = point_type(xy[0], xy[1], T(0));
        return true;
    }

    // ------------------------------------------------------------------------
    //  Transform plane by affine transform (normal is transformed by inverse transpose)
    // ------------------------------------------------------------------------
    Plane transform(const AffineTransform<T,3>& tf) const noexcept {
        point_type newNormal = tf.transformNormal(m_normal);
        T newD = m_d - newNormal.dot(tf.translation());
        return Plane(newNormal, newD);
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: compute signed distances for 4 points
    // ------------------------------------------------------------------------
    void batchSignedDistance(const point_type* points, T* out, size_t count) const noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && std::is_same_v<T,float>) {
            // In real SIMD, we would use _mm256_load_ps, _mm256_mul_ps, etc.
            // For clarity, we unroll scalar.
            for (size_t i = 0; i < count; ++i) {
                out[i] = signedDistance(points[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                out[i] = signedDistance(points[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool operator==(const Plane& other) const noexcept {
        T eps = T(1e-6);
        return (m_normal - other.m_normal).length() < eps && std::abs(m_d - other.m_d) < eps;
    }
    bool operator!=(const Plane& other) const noexcept { return !(*this == other); }

private:
    point_type m_normal;
    T m_d;
};

// ----------------------------------------------------------------------------
//  Helper: create plane from normal and point
// ----------------------------------------------------------------------------
template<typename T>
Plane<T> makePlaneFromNormalAndPoint(const Vector<T,3>& normal, const Vector<T,3>& point) {
    return Plane<T>(normal, point);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller for plane operations
// ----------------------------------------------------------------------------
class PlaneEnvironment {
public:
    static PlaneEnvironment& instance() {
        static PlaneEnvironment env;
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
private:
    PlaneEnvironment() : m_epsilon(T(1e-8)) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_PLANE_H_INCLUDED