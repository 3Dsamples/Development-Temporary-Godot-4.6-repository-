//File group name : OrthoTree Math
//File 0058 : core/math/geometry/cylinder.h
//Cylinder defined by center, axis direction, height, radius. Distance to point, ray intersection, bounding box, transformation, and SIMD batch operations.

#ifndef ORTHOTREE_CORE_MATH_GEOMETRY_CYLINDER_H_INCLUDED
#define ORTHOTREE_CORE_MATH_GEOMETRY_CYLINDER_H_INCLUDED

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
//  Cylinder (right circular) defined by center, axis direction, height, radius.
//  The cylinder is capped (includes top and bottom disks).
// ============================================================================
template<typename T = float>
class Cylinder {
public:
    using value_type = T;
    using point_type = Basic::Vector<T, 3>;
    using ray_type = Ray<T, 3>;
    using aabb_type = AABB<T, 3>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    Cylinder() noexcept
        : m_center(T(0)), m_axis(T(0,0,1)), m_height(T(2)), m_radius(T(1)) {}
    Cylinder(const point_type& center, const point_type& axis, T height, T radius) noexcept
        : m_center(center), m_axis(axis.normalized()), m_height(height), m_radius(radius) {}
    // Cylinder from bottom and top centers
    Cylinder(const point_type& bottom, const point_type& top, T radius) noexcept {
        m_center = (bottom + top) * T(0.5);
        m_axis = (top - bottom).normalized();
        m_height = (top - bottom).length();
        m_radius = radius;
    }

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    constexpr const point_type& center() const noexcept { return m_center; }
    constexpr const point_type& axis() const noexcept { return m_axis; }
    constexpr T height() const noexcept { return m_height; }
    constexpr T radius() const noexcept { return m_radius; }
    constexpr void setCenter(const point_type& c) noexcept { m_center = c; }
    constexpr void setAxis(const point_type& a) noexcept { m_axis = a.normalized(); }
    constexpr void setHeight(T h) noexcept { m_height = h; }
    constexpr void setRadius(T r) noexcept { m_radius = r; }

    // ------------------------------------------------------------------------
    //  End points (bottom and top)
    // ------------------------------------------------------------------------
    constexpr point_type bottom() const noexcept { return m_center - m_axis * (m_height * T(0.5)); }
    constexpr point_type top() const noexcept { return m_center + m_axis * (m_height * T(0.5)); }

    // ------------------------------------------------------------------------
    //  Bounding box (conservative)
    // ------------------------------------------------------------------------
    aabb_type boundingBox() const noexcept {
        point_type bottomPt = bottom();
        point_type topPt = top();
        point_type minP = bottomPt.componentWiseMin(topPt) - point_type(m_radius);
        point_type maxP = bottomPt.componentWiseMax(topPt) + point_type(m_radius);
        return aabb_type(minP, maxP);
    }

    // ------------------------------------------------------------------------
    //  Distance to point (clamped to height)
    // ------------------------------------------------------------------------
    T squaredDistanceToPoint(const point_type& p) const noexcept {
        point_type bottomPt = bottom();
        point_type v = p - bottomPt;
        T proj = v.dot(m_axis);
        if (proj <= T(0)) {
            // below bottom: distance to bottom disk
            point_type radial = v - m_axis * proj;
            T radialDist = radial.length() - m_radius;
            if (radialDist < T(0)) radialDist = T(0);
            return radialDist * radialDist + proj * proj;
        }
        if (proj >= m_height) {
            // above top: distance to top disk
            point_type radial = v - m_axis * m_height;
            T radialDist = radial.length() - m_radius;
            if (radialDist < T(0)) radialDist = T(0);
            T dz = proj - m_height;
            return radialDist * radialDist + dz * dz;
        }
        // inside height range: distance to side surface
        point_type radial = v - m_axis * proj;
        T radialDist = radial.length() - m_radius;
        if (radialDist < T(0)) radialDist = T(0);
        return radialDist * radialDist;
    }

    T distanceToPoint(const point_type& p) const noexcept {
        return std::sqrt(squaredDistanceToPoint(p));
    }
    bool containsPoint(const point_type& p) const noexcept {
        return squaredDistanceToPoint(p) <= T(1e-12);
    }

    // ------------------------------------------------------------------------
    //  Ray intersection (returns t0 (entry), t1 (exit), true if hit)
    //  Solves for infinite cylinder then clips by height, plus caps.
    // ------------------------------------------------------------------------
    bool intersectRay(const ray_type& ray, T& t0, T& t1) const noexcept {
        // Transform to coordinate system where cylinder axis = Z, origin at bottom
        point_type u, v;
        if (std::abs(m_axis[0]) < T(0.9)) {
            u = m_axis.cross(point_type(1,0,0)).normalized();
        } else {
            u = m_axis.cross(point_type(0,1,0)).normalized();
        }
        v = u.cross(m_axis).normalized();
        point_type bottomPt = bottom();
        point_type ro = ray.origin() - bottomPt;
        point_type rd = ray.direction();
        point_type lo = point_type(ro.dot(u), ro.dot(v), ro.dot(m_axis));
        point_type ld = point_type(rd.dot(u), rd.dot(v), rd.dot(m_axis));

        // Solve for infinite cylinder: (x^2 + y^2) = r^2
        T a = ld[0]*ld[0] + ld[1]*ld[1];
        T b = T(2)*(lo[0]*ld[0] + lo[1]*ld[1]);
        T c = lo[0]*lo[0] + lo[1]*lo[1] - m_radius*m_radius;
        T disc = b*b - T(4)*a*c;
        if (disc < T(0)) return false;
        T sqrtDisc = std::sqrt(disc);
        T tCyl0 = (-b - sqrtDisc) / (T(2)*a);
        T tCyl1 = (-b + sqrtDisc) / (T(2)*a);
        if (tCyl0 > tCyl1) std::swap(tCyl0, tCyl1);

        // Clip by height range [0, m_height]
        T tMin = tCyl0;
        T tMax = tCyl1;
        T z0 = lo[2] + ld[2] * tMin;
        T z1 = lo[2] + ld[2] * tMax;
        if (z0 > m_height || z1 < T(0)) {
            // Check caps (bottom and top disks)
            bool hit = false;
            // Bottom cap: plane z=0, disk radius m_radius
            T tBottom = (T(0) - lo[2]) / ld[2];
            if (tBottom >= T(0)) {
                point_type pHit = lo + ld * tBottom;
                if (pHit[0]*pHit[0] + pHit[1]*pHit[1] <= m_radius*m_radius) {
                    tMin = tBottom; tMax = tBottom;
                    hit = true;
                }
            }
            // Top cap: plane z=m_height
            T tTop = (m_height - lo[2]) / ld[2];
            if (tTop >= T(0)) {
                point_type pHit = lo + ld * tTop;
                if (pHit[0]*pHit[0] + pHit[1]*pHit[1] <= m_radius*m_radius) {
                    if (!hit) {
                        tMin = tTop; tMax = tTop;
                        hit = true;
                    } else {
                        tMin = std::min(tMin, tTop);
                        tMax = std::max(tMax, tTop);
                    }
                }
            }
            if (!hit) return false;
            t0 = tMin; t1 = tMax;
            return true;
        }
        // Clip tMin, tMax to be within Z range
        if (z0 < T(0)) tMin = (T(0) - lo[2]) / ld[2];
        if (z1 > m_height) tMax = (m_height - lo[2]) / ld[2];
        t0 = tMin; t1 = tMax;
        return (t0 <= t1);
    }

    // ------------------------------------------------------------------------
    //  Transform cylinder by affine transform (endpoints transform, radius scales)
    // ------------------------------------------------------------------------
    Cylinder transform(const Basic::AffineTransform<T,3>& tf) const noexcept {
        point_type newBottom = tf.transform(bottom());
        point_type newTop = tf.transform(top());
        // Compute scaling factor for radius
        T maxScale = T(0);
        for (int i = 0; i < 3; ++i) {
            point_type col;
            for (int j = 0; j < 3; ++j) col[j] = tf.matrix()(j, i);
            T len = col.length();
            if (len > maxScale) maxScale = len;
        }
        return Cylinder(newBottom, newTop, m_radius * maxScale);
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool nearlyEqual(const Cylinder& other, T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) const noexcept {
        return m_center.nearlyEqual(other.m_center, eps) &&
               m_axis.nearlyEqual(other.m_axis, eps) &&
               std::abs(m_height - other.m_height) < eps &&
               std::abs(m_radius - other.m_radius) < eps;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: squared distance for 4 cylinders to 4 points
    // ------------------------------------------------------------------------
    static void batchSquaredDistance(const Cylinder* cylinders, const point_type* points,
                                     T* out, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) {
            out[i] = cylinders[i].squaredDistanceToPoint(points[i]);
        }
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: ray intersection for 4 cylinders with 4 rays
    // ------------------------------------------------------------------------
    static void batchRayIntersect(const Cylinder* cylinders, const ray_type* rays,
                                  bool* hit, T* t0, T* t1, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) {
            hit[i] = cylinders[i].intersectRay(rays[i], t0[i], t1[i]);
        }
    }

private:
    point_type m_center;
    point_type m_axis;
    T m_height;
    T m_radius;
};

// ----------------------------------------------------------------------------
//  Convenience aliases
// ----------------------------------------------------------------------------
using Cylinder3f = Cylinder<float>;
using Cylinder3d = Cylinder<double>;

// ----------------------------------------------------------------------------
//  Helper: create cylinder from bottom and top centers
// ----------------------------------------------------------------------------
template<typename T>
Cylinder<T> makeCylinder(const Basic::Vector<T,3>& bottom, const Basic::Vector<T,3>& top, T radius) {
    return Cylinder<T>(bottom, top, radius);
}

// ----------------------------------------------------------------------------
//  Helper: create cylinder from center, axis, height, radius
// ----------------------------------------------------------------------------
template<typename T>
Cylinder<T> makeCylinder(const Basic::Vector<T,3>& center, const Basic::Vector<T,3>& axis,
                         T height, T radius) {
    return Cylinder<T>(center, axis, height, radius);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class CylinderEnvironment {
public:
    static CylinderEnvironment& instance() {
        static CylinderEnvironment env;
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
    CylinderEnvironment() : m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    bool m_useSIMD;
};

} // namespace Geometry
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_GEOMETRY_CYLINDER_H_INCLUDED