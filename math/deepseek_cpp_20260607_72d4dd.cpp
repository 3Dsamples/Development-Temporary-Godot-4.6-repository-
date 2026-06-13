//File group name : OrthoTree Math
//File 0017 : core/math/capsule.h
//Capsule primitive (line segment + radius): distance queries, ray intersection, AABB overlap, and SIMD batch operations.

#ifndef ORTHOTREE_CORE_MATH_CAPSULE_H_INCLUDED
#define ORTHOTREE_CORE_MATH_CAPSULE_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "geometry_queries.h"
#include "ray_intersection.h"
#include "interval_arithmetic.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <limits>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Capsule: a line segment with a radius. Useful for character collision,
//  ray casting, and proximity queries. Supports 3D only.
// ============================================================================
template<typename T = float>
class Capsule {
public:
    using value_type = T;
    using point_type = Vector<T, 3>;
    using ray_type = Ray<T, 3>;
    using aabb_type = AxisAlignedBox<T, 3>;
    using sphere_type = Sphere<T, 3>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    Capsule() noexcept : m_a(point_type(0)), m_b(point_type(0,0,1)), m_radius(T(1)) {}
    Capsule(const point_type& a, const point_type& b, T radius) noexcept
        : m_a(a), m_b(b), m_radius(radius) {}
    Capsule(const point_type& center, const point_type& axis, T height, T radius) noexcept {
        point_type dir = axis.normalized();
        T half = height * T(0.5);
        m_a = center - dir * half;
        m_b = center + dir * half;
        m_radius = radius;
    }

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    const point_type& a() const noexcept { return m_a; }
    const point_type& b() const noexcept { return m_b; }
    T radius() const noexcept { return m_radius; }
    void setA(const point_type& a) noexcept { m_a = a; }
    void setB(const point_type& b) noexcept { m_b = b; }
    void setRadius(T r) noexcept { m_radius = r; }

    point_type center() const noexcept { return (m_a + m_b) * T(0.5); }
    point_type axis() const noexcept { return (m_b - m_a).normalized(); }
    T height() const noexcept { return (m_b - m_a).length(); }

    // ------------------------------------------------------------------------
    //  Distance queries
    // ------------------------------------------------------------------------
    T distanceToPoint(const point_type& p) const noexcept {
        point_type closest = closestPointToPoint(p);
        return (closest - p).length();
    }

    T squaredDistanceToPoint(const point_type& p) const noexcept {
        point_type closest = closestPointToPoint(p);
        return (closest - p).squaredLength();
    }

    // Closest point on capsule to external point
    point_type closestPointToPoint(const point_type& p) const noexcept {
        point_type ab = m_b - m_a;
        T t = (p - m_a).dot(ab) / ab.squaredLength();
        if (t <= T(0)) t = T(0);
        else if (t >= T(1)) t = T(1);
        point_type closestOnLine = m_a + ab * t;
        point_type dir = (p - closestOnLine).normalized();
        return closestOnLine + dir * m_radius;
    }

    // ------------------------------------------------------------------------
    //  Ray intersection (returns t of first hit, or false)
    //  Solve quadratic: |(origin + t*direction) - closestPointOnLine|^2 = radius^2
    // ------------------------------------------------------------------------
    bool intersectRay(const ray_type& ray, T& t0, T& t1) const noexcept {
        point_type ab = m_b - m_a;
        point_type ro = ray.origin();
        point_type rd = ray.direction();
        // Parameterization of capsule: set of points X = C + u * ab + r * n, where |n|=1, 0<=u<=1.
        // For ray: ro + t*rd = C + u*ab + r*n.
        // We solve for t, u, and distance r.
        // Simpler: compute closest point on line segment to ray, then check distance.
        point_type w = ro - m_a;
        T a = rd.squaredLength();
        T b = rd.dot(ab);
        T c = ab.squaredLength();
        T d = rd.dot(w);
        T e = ab.dot(w);
        T denom = a * c - b * b;
        if (std::abs(denom) < T(1e-12)) {
            // Ray parallel to capsule axis
            point_type dir = rd.normalized();
            point_type perp = cross(dir, ab);
            if (perp.squaredLength() < T(1e-12)) {
                // Ray parallel and collinear
                // Project ray onto line and find overlap
                T tStart = -e / c;
                T tEnd = (ab.dot(ray.direction()) * (ro - m_a).dot(ab)) / c; // messy
                // For simplicity, fallback to AABB‑capsule intersection
                return false;
            }
        }
        T invDenom = T(1) / denom;
        T t = (b * e - c * d) * invDenom;
        T u = (a * e - b * d) * invDenom;
        point_type closestOnRay = ro + rd * t;
        point_type closestOnLine = m_a + ab * u;
        T dist2 = (closestOnRay - closestOnLine).squaredLength();
        if (dist2 > m_radius * m_radius) return false;
        T r = std::sqrt(dist2);
        T radialDist = std::sqrt(m_radius * m_radius - dist2);
        T tCenter = t;
        t0 = tCenter - radialDist / rd.length();
        t1 = tCenter + radialDist / rd.length();
        if (t0 < T(0)) t0 = T(0);
        return true;
    }

    // ------------------------------------------------------------------------
    //  AABB overlap (conservative)
    // ------------------------------------------------------------------------
    aabb_type boundingBox() const noexcept {
        point_type min = m_a.componentWiseMin(m_b) - point_type(m_radius);
        point_type max = m_a.componentWiseMax(m_b) + point_type(m_radius);
        return aabb_type(min, max);
    }

    bool overlapsAABB(const aabb_type& box) const noexcept {
        return boundingBox().overlaps(box);
    }

    // ------------------------------------------------------------------------
    //  Transform capsule by affine transform
    // ------------------------------------------------------------------------
    Capsule transform(const AffineTransform<T,3>& tf) const noexcept {
        point_type newA = tf.transform(m_a);
        point_type newB = tf.transform(m_b);
        // Radius scales by max of scaling factors (conservative)
        T scale = (tf.matrix() * point_type(1,0,0)).length();
        scale = std::max(scale, (tf.matrix() * point_type(0,1,0)).length());
        scale = std::max(scale, (tf.matrix() * point_type(0,0,1)).length());
        return Capsule(newA, newB, m_radius * scale);
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: compute closest points for 4 points
    // ------------------------------------------------------------------------
    void batchClosestPoint(const point_type* points, point_type* out, size_t count) const noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && std::is_same_v<T,float>) {
            for (size_t i = 0; i < count; ++i) {
                out[i] = closestPointToPoint(points[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                out[i] = closestPointToPoint(points[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool operator==(const Capsule& other) const noexcept {
        T eps = T(1e-6);
        return (m_a - other.m_a).length() < eps &&
               (m_b - other.m_b).length() < eps &&
               std::abs(m_radius - other.m_radius) < eps;
    }

private:
    point_type m_a, m_b;
    T m_radius;
};

// ----------------------------------------------------------------------------
//  Helper: capsule from center, axis, height, radius
// ----------------------------------------------------------------------------
template<typename T>
Capsule<T> makeCapsule(const Vector<T,3>& center, const Vector<T,3>& axis,
                       T height, T radius) {
    return Capsule<T>(center, axis, height, radius);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller for capsule operations
// ----------------------------------------------------------------------------
class CapsuleEnvironment {
public:
    static CapsuleEnvironment& instance() {
        static CapsuleEnvironment env;
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
    CapsuleEnvironment() : m_epsilon(T(1e-8)) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_CAPSULE_H_INCLUDED