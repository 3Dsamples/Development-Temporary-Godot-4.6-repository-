//File group name : OrthoTree Math
//File 0024 : core/math/cylinder.h
//Cylinder primitive: center, axis direction, height, radius. Supports point distance, ray intersection, bounding box, transformation, and SIMD batch operations.

#ifndef ORTHOTREE_CORE_MATH_CYLINDER_H_INCLUDED
#define ORTHOTREE_CORE_MATH_CYLINDER_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "ray_intersection.h"
#include "line_segment.h"
#include "geometry_queries.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <limits>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Cylinder: defined by center point, axis direction (unit), height, and radius.
//  Cylinder is capped (end caps included). Provides distance to point,
//  ray intersection (quadratic solution), bounding box, and transformation.
//  Supports 3D only.
// ============================================================================
template<typename T = float>
class Cylinder {
public:
    using value_type = T;
    using point_type = Vector<T, 3>;
    using ray_type = Ray<T, 3>;
    using aabb_type = AxisAlignedBox<T, 3>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    Cylinder() noexcept
        : m_center(T(0)), m_axis(T(0,0,1)), m_height(T(2)), m_radius(T(1)) {}
    Cylinder(const point_type& center, const point_type& axis, T height, T radius) noexcept
        : m_center(center), m_axis(axis.normalized()), m_height(height), m_radius(radius) {}
    // Cylinder from two end points (bottom, top) and radius
    Cylinder(const point_type& bottom, const point_type& top, T radius) noexcept {
        m_center = (bottom + top) * T(0.5);
        m_axis = (top - bottom).normalized();
        m_height = (top - bottom).length();
        m_radius = radius;
    }

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    const point_type& center() const noexcept { return m_center; }
    const point_type& axis() const noexcept { return m_axis; }
    T height() const noexcept { return m_height; }
    T radius() const noexcept { return m_radius; }
    void setCenter(const point_type& c) noexcept { m_center = c; }
    void setAxis(const point_type& a) noexcept { m_axis = a.normalized(); }
    void setHeight(T h) noexcept { m_height = h; }
    void setRadius(T r) noexcept { m_radius = r; }

    // ------------------------------------------------------------------------
    //  End points (bottom and top)
    // ------------------------------------------------------------------------
    point_type bottom() const noexcept { return m_center - m_axis * (m_height * T(0.5)); }
    point_type top() const noexcept { return m_center + m_axis * (m_height * T(0.5)); }

    // ------------------------------------------------------------------------
    //  Distance to point (clamped to height)
    // ------------------------------------------------------------------------
    T squaredDistanceToPoint(const point_type& p) const noexcept {
        point_type bottomPt = bottom();
        point_type topPt = top();
        point_type v = p - bottomPt;
        T proj = v.dot(m_axis);
        if (proj <= T(0)) return (bottomPt - p).squaredLength();
        if (proj >= m_height) return (topPt - p).squaredLength();
        point_type closestAxis = bottomPt + m_axis * proj;
        T radialDist = (p - closestAxis).length() - m_radius;
        if (radialDist < T(0)) return T(0);
        return radialDist * radialDist;
    }

    T distanceToPoint(const point_type& p) const noexcept {
        return std::sqrt(squaredDistanceToPoint(p));
    }

    // ------------------------------------------------------------------------
    //  Ray intersection (returns t0 (entry), t1 (exit), false if no hit)
    //  Solve for intersection with infinite cylinder and then clip by caps.
    // ------------------------------------------------------------------------
    bool intersectRay(const ray_type& ray, T& t0, T& t1) const noexcept {
        point_type ro = ray.origin();
        point_type rd = ray.direction();
        point_type bottomPt = bottom();
        point_type axis = m_axis;
        T height = m_height;
        T r = m_radius;

        // Transform to coordinate system where cylinder is upright (axis = +Z)
        // Build orthonormal basis (u, v, axis)
        point_type u, v;
        if (std::abs(axis[0]) < T(0.9)) {
            u = normalize(cross(axis, point_type(1,0,0)));
        } else {
            u = normalize(cross(axis, point_type(0,1,0)));
        }
        v = cross(u, axis);
        // Change of basis matrix
        auto toLocal = [&](const point_type& p) -> point_type {
            return point_type(dot(p - bottomPt, u), dot(p - bottomPt, v), dot(p - bottomPt, axis));
        };
        point_type lo = toLocal(ro);
        point_type ld = point_type(dot(rd, u), dot(rd, v), dot(rd, axis));

        // Solve (x^2 + y^2 = r^2) intersection with infinite cylinder
        T a = ld[0]*ld[0] + ld[1]*ld[1];
        T b = T(2) * (lo[0]*ld[0] + lo[1]*ld[1]);
        T c = lo[0]*lo[0] + lo[1]*lo[1] - r*r;
        T disc = b*b - T(4)*a*c;
        if (disc < T(0)) return false;
        T sqrtDisc = std::sqrt(disc);
        T tCyl0 = (-b - sqrtDisc) / (T(2)*a);
        T tCyl1 = (-b + sqrtDisc) / (T(2)*a);
        if (tCyl0 > tCyl1) std::swap(tCyl0, tCyl1);
        // Clip by height (z in [0, height])
        T tNear = tCyl0;
        T tFar = tCyl1;
        T z0 = lo[2] + ld[2] * tNear;
        T z1 = lo[2] + ld[2] * tFar;
        if (z0 > height || z1 < T(0)) {
            // Check cap disks (top and bottom)
            // Bottom cap: plane z=0, circle radius r
            T tBottom = (T(0) - lo[2]) / ld[2];
            if (tBottom >= T(0)) {
                point_type pHit = lo + ld * tBottom;
                if (pHit[0]*pHit[0] + pHit[1]*pHit[1] <= r*r) {
                    tNear = tBottom;
                    // For exit, we need to find far end (could be top cap)
                    // Find intersection with top cap
                    T tTop = (height - lo[2]) / ld[2];
                    if (tTop >= tBottom) {
                        tFar = tTop;
                    } else {
                        tFar = tBottom;
                    }
                    // Also check if infinite cylinder part lies within height
                    // We'll simplify: compute both caps and take min/max.
                }
            }
            // Top cap: plane z=height
            T tTop = (height - lo[2]) / ld[2];
            if (tTop >= T(0)) {
                point_type pHit = lo + ld * tTop;
                if (pHit[0]*pHit[0] + pHit[1]*pHit[1] <= r*r) {
                    if (tNear > tTop) tNear = tTop;
                    if (tFar < tTop) tFar = tTop;
                }
            }
            if (tNear > tFar) return false;
        }
        t0 = tNear;
        t1 = tFar;
        return true;
    }

    // ------------------------------------------------------------------------
    //  Bounding box (conservative)
    // ------------------------------------------------------------------------
    aabb_type boundingBox() const noexcept {
        point_type ext = m_axis * (m_height * T(0.5));
        point_type center = m_center;
        point_type minP = center - ext;
        point_type maxP = center + ext;
        // Add radius margin
        point_type rad(m_radius);
        minP = minP - rad;
        maxP = maxP + rad;
        return aabb_type(minP, maxP);
    }

    // ------------------------------------------------------------------------
    //  Transform cylinder by affine transform (center, axis direction, radius scales)
    //  Note: height and axis may change length; we recompute height from transformed endpoints.
    //  Radius scales by max singular value.
    // ------------------------------------------------------------------------
    Cylinder transform(const AffineTransform<T,3>& tf) const noexcept {
        point_type newBottom = tf.transform(bottom());
        point_type newTop = tf.transform(top());
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
    //  SIMD batch: intersect 4 rays with 4 cylinders (pairs)
    // ------------------------------------------------------------------------
    static void batchRayIntersect(const Cylinder* cylinders, const ray_type* rays,
                                  bool* hit, T* t0, T* t1, size_t count) noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && std::is_same_v<T,float>) {
            for (size_t i = 0; i < count; ++i) {
                hit[i] = cylinders[i].intersectRay(rays[i], t0[i], t1[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                hit[i] = cylinders[i].intersectRay(rays[i], t0[i], t1[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool operator==(const Cylinder& other) const noexcept {
        T eps = T(1e-6);
        return (m_center - other.m_center).length() < eps &&
               (m_axis - other.m_axis).length() < eps &&
               std::abs(m_height - other.m_height) < eps &&
               std::abs(m_radius - other.m_radius) < eps;
    }

private:
    point_type m_center;
    point_type m_axis;
    T m_height;
    T m_radius;
};

// ----------------------------------------------------------------------------
//  Helper: create cylinder from bottom and top centers and radius
// ----------------------------------------------------------------------------
template<typename T>
Cylinder<T> makeCylinder(const Vector<T,3>& bottom, const Vector<T,3>& top, T radius) {
    return Cylinder<T>(bottom, top, radius);
}

// ----------------------------------------------------------------------------
//  Helper: create cylinder from center, axis, height, radius
// ----------------------------------------------------------------------------
template<typename T>
Cylinder<T> makeCylinder(const Vector<T,3>& center, const Vector<T,3>& axis, T height, T radius) {
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
    CylinderEnvironment() : m_epsilon(T(1e-8)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_CYLINDER_H_INCLUDED