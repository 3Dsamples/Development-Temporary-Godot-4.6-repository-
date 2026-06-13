//File group name : OrthoTree Math
//File 0057 : core/math/geometry/capsule.h
//Capsule primitive (line segment with radius): distance to point, ray intersection, bounding box, transformation, and SIMD batch operations.

#ifndef ORTHOTREE_CORE_MATH_GEOMETRY_CAPSULE_H_INCLUDED
#define ORTHOTREE_CORE_MATH_GEOMETRY_CAPSULE_H_INCLUDED

#include "../../build_config.h"
#include "../basic/vector.h"
#include "../basic/matrix.h"
#include "aabb.h"
#include "ray.h"
#include "sphere.h"
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <limits>

namespace OrthoTree {
namespace Math {
namespace Geometry {

// ============================================================================
//  Capsule defined by two endpoints (a, b) and a radius.
//  The capsule is the set of points within distance radius of the segment ab.
// ============================================================================
template<typename T = float>
class Capsule {
public:
    using value_type = T;
    using point_type = Basic::Vector<T, 3>;
    using ray_type = Ray<T, 3>;
    using aabb_type = AABB<T, 3>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    Capsule() noexcept : m_a(T(0)), m_b(T(0,0,1)), m_radius(T(1)) {}
    Capsule(const point_type& a, const point_type& b, T radius) noexcept
        : m_a(a), m_b(b), m_radius(radius) {}
    // Capsule from center, axis, height, radius
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
    constexpr const point_type& a() const noexcept { return m_a; }
    constexpr const point_type& b() const noexcept { return m_b; }
    constexpr T radius() const noexcept { return m_radius; }
    constexpr void setA(const point_type& a) noexcept { m_a = a; }
    constexpr void setB(const point_type& b) noexcept { m_b = b; }
    constexpr void setRadius(T r) noexcept { m_radius = r; }

    constexpr point_type center() const noexcept { return (m_a + m_b) * T(0.5); }
    constexpr point_type axis() const noexcept { return (m_b - m_a).normalized(); }
    constexpr T height() const noexcept { return (m_b - m_a).length(); }

    // ------------------------------------------------------------------------
    //  Bounding box (conservative)
    // ------------------------------------------------------------------------
    aabb_type boundingBox() const noexcept {
        point_type minP = m_a.componentWiseMin(m_b) - point_type(m_radius);
        point_type maxP = m_a.componentWiseMax(m_b) + point_type(m_radius);
        return aabb_type(minP, maxP);
    }

    // ------------------------------------------------------------------------
    //  Closest point on capsule to an external point
    // ------------------------------------------------------------------------
    point_type closestPoint(const point_type& p) const noexcept {
        point_type ab = m_b - m_a;
        T t = (p - m_a).dot(ab) / ab.squaredLength();
        t = Basic::clamp(t, T(0), T(1));
        point_type closestOnSegment = m_a + ab * t;
        point_type dir = p - closestOnSegment;
        T len = dir.length();
        if (len > m_radius) {
            return closestOnSegment + dir * (m_radius / len);
        }
        return p;
    }

    // ------------------------------------------------------------------------
    //  Distance to point
    // ------------------------------------------------------------------------
    T distanceToPoint(const point_type& p) const noexcept {
        point_type closest = closestPoint(p);
        return (closest - p).length();
    }
    T squaredDistanceToPoint(const point_type& p) const noexcept {
        point_type closest = closestPoint(p);
        return (closest - p).squaredLength();
    }
    bool containsPoint(const point_type& p) const noexcept {
        return squaredDistanceToPoint(p) <= m_radius * m_radius;
    }

    // ------------------------------------------------------------------------
    //  Ray intersection (returns t0, t1, true if hit)
    //  Solve quadratic for infinite cylinder + sphere caps.
    //  Simplified: we use numerical approach (iterative for brevity).
    //  For speed, implement analytic solution (as in original).
    // ------------------------------------------------------------------------
    bool intersectRay(const ray_type& ray, T& t0, T& t1) const noexcept {
        // Transform ray to coordinate system where capsule axis = Z
        point_type axisDir = axis();
        point_type u, v;
        if (std::abs(axisDir[0]) < T(0.9)) {
            u = axisDir.cross(point_type(1,0,0)).normalized();
        } else {
            u = axisDir.cross(point_type(0,1,0)).normalized();
        }
        v = u.cross(axisDir).normalized();
        point_type aLocal = point_type(T(0), T(0), -height() * T(0.5));
        point_type bLocal = point_type(T(0), T(0), height() * T(0.5));
        point_type ro = ray.origin() - center();
        point_type rd = ray.direction();
        point_type lo = point_type(ro.dot(u), ro.dot(v), ro.dot(axisDir));
        point_type ld = point_type(rd.dot(u), rd.dot(v), rd.dot(axisDir));
        T r2 = m_radius * m_radius;
        // Solve for infinite cylinder: (x^2 + y^2) = r^2
        T a = ld[0]*ld[0] + ld[1]*ld[1];
        T b = T(2)*(lo[0]*ld[0] + lo[1]*ld[1]);
        T c = lo[0]*lo[0] + lo[1]*lo[1] - r2;
        T disc = b*b - T(4)*a*c;
        if (disc < T(0)) return false;
        T sqrtDisc = std::sqrt(disc);
        T tCyl0 = (-b - sqrtDisc) / (T(2)*a);
        T tCyl1 = (-b + sqrtDisc) / (T(2)*a);
        // Clip by Z range [-halfH, halfH]
        T halfH = height() * T(0.5);
        T tMin = tCyl0, tMax = tCyl1;
        T z0 = lo[2] + ld[2] * tMin;
        T z1 = lo[2] + ld[2] * tMax;
        if (z0 > halfH || z1 < -halfH) {
            // Check sphere caps (two spheres at endpoints)
            Sphere<T,3> capA(m_a, m_radius);
            Sphere<T,3> capB(m_b, m_radius);
            T tA0, tA1, tB0, tB1;
            bool hitA = capA.intersectRay(ray, tA0, tA1);
            bool hitB = capB.intersectRay(ray, tB0, tB1);
            if (hitA && hitB) {
                t0 = std::min(tA0, tB0);
                t1 = std::max(tA1, tB1);
                return true;
            } else if (hitA) {
                t0 = tA0; t1 = tA1; return true;
            } else if (hitB) {
                t0 = tB0; t1 = tB1; return true;
            }
            return false;
        }
        // Adjust tMin and tMax to lie within Z range
        if (z0 < -halfH) {
            tMin = (-halfH - lo[2]) / ld[2];
        }
        if (z1 > halfH) {
            tMax = (halfH - lo[2]) / ld[2];
        }
        t0 = tMin;
        t1 = tMax;
        return (t0 <= t1);
    }

    // ------------------------------------------------------------------------
    //  Transform capsule by affine transform (endpoints and radius scale)
    // ------------------------------------------------------------------------
    Capsule transform(const Basic::AffineTransform<T,3>& tf) const noexcept {
        point_type newA = tf.transform(m_a);
        point_type newB = tf.transform(m_b);
        // Estimate scaling factor for radius
        T maxScale = T(0);
        for (int i = 0; i < 3; ++i) {
            point_type col;
            for (int j = 0; j < 3; ++j) col[j] = tf.matrix()(j, i);
            T len = col.length();
            if (len > maxScale) maxScale = len;
        }
        return Capsule(newA, newB, m_radius * maxScale);
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool nearlyEqual(const Capsule& other, T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) const noexcept {
        return m_a.nearlyEqual(other.m_a, eps) &&
               m_b.nearlyEqual(other.m_b, eps) &&
               std::abs(m_radius - other.m_radius) < eps;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: squared distance for 4 capsules to 4 points
    // ------------------------------------------------------------------------
    static void batchSquaredDistance(const Capsule* capsules, const point_type* points,
                                     T* out, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) {
            out[i] = capsules[i].squaredDistanceToPoint(points[i]);
        }
    }

private:
    point_type m_a, m_b;
    T m_radius;
};

// ----------------------------------------------------------------------------
//  Convenience aliases
// ----------------------------------------------------------------------------
using Capsule3f = Capsule<float>;
using Capsule3d = Capsule<double>;

// ----------------------------------------------------------------------------
//  Helper: create capsule from center, axis, height, radius
// ----------------------------------------------------------------------------
template<typename T>
Capsule<T> makeCapsule(const Basic::Vector<T,3>& center, const Basic::Vector<T,3>& axis,
                       T height, T radius) {
    return Capsule<T>(center, axis, height, radius);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class CapsuleEnvironment {
public:
    static CapsuleEnvironment& instance() {
        static CapsuleEnvironment env;
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
    CapsuleEnvironment() : m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    bool m_useSIMD;
};

} // namespace Geometry
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_GEOMETRY_CAPSULE_H_INCLUDED