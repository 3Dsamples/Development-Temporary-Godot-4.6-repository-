//File group name : OrthoTree Math
//File 0025 : core/math/cone.h
//Cone (and truncated cone) primitive: center, axis direction, height, base radius, top radius (optional). Distance to point, ray intersection, bounding box, transformation, and SIMD batch operations.

#ifndef ORTHOTREE_CORE_MATH_CONE_H_INCLUDED
#define ORTHOTREE_CORE_MATH_CONE_H_INCLUDED

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
#include <algorithm>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Cone (or truncated cone, i.e., frustum of a cone) defined by:
//  - center: midpoint between bottom and top centers.
//  - axis: unit direction from bottom to top.
//  - height: distance between bottom and top planes.
//  - radiusBottom: radius at bottom (larger end)
//  - radiusTop: radius at top (smaller end). If equal to 0, it's a standard cone.
//  For a standard cone (radiusTop = 0), the apex is at the top.
//  The bottom is at center - axis * (height/2), top at center + axis * (height/2).
//  Provides distance to point, ray intersection, bounding box, transformation.
// ============================================================================
template<typename T = float>
class Cone {
public:
    using value_type = T;
    using point_type = Vector<T, 3>;
    using ray_type = Ray<T, 3>;
    using aabb_type = AxisAlignedBox<T, 3>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    Cone() noexcept
        : m_center(T(0)), m_axis(T(0,0,1)), m_height(T(2)),
          m_radiusBottom(T(1)), m_radiusTop(T(0)) {}
    Cone(const point_type& center, const point_type& axis, T height,
         T radiusBottom, T radiusTop = T(0)) noexcept
        : m_center(center), m_axis(axis.normalized()), m_height(height),
          m_radiusBottom(radiusBottom), m_radiusTop(radiusTop) {}
    // Standard cone from apex, axis direction, height, base radius
    static Cone fromApex(const point_type& apex, const point_type& axis,
                         T height, T radiusBottom) noexcept {
        point_type center = apex + axis * (height * T(0.5));
        return Cone(center, axis, height, radiusBottom, T(0));
    }
    // Truncated cone from bottom center, top center, bottom radius, top radius
    static Cone fromEnds(const point_type& bottom, const point_type& top,
                         T radiusBottom, T radiusTop) noexcept {
        point_type center = (bottom + top) * T(0.5);
        point_type axis = (top - bottom).normalized();
        T height = (top - bottom).length();
        return Cone(center, axis, height, radiusBottom, radiusTop);
    }

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    const point_type& center() const noexcept { return m_center; }
    const point_type& axis() const noexcept { return m_axis; }
    T height() const noexcept { return m_height; }
    T radiusBottom() const noexcept { return m_radiusBottom; }
    T radiusTop() const noexcept { return m_radiusTop; }
    void setCenter(const point_type& c) noexcept { m_center = c; }
    void setAxis(const point_type& a) noexcept { m_axis = a.normalized(); }
    void setHeight(T h) noexcept { m_height = h; }
    void setRadii(T bottom, T top) noexcept { m_radiusBottom = bottom; m_radiusTop = top; }

    // ------------------------------------------------------------------------
    //  End points
    // ------------------------------------------------------------------------
    point_type bottom() const noexcept { return m_center - m_axis * (m_height * T(0.5)); }
    point_type top() const noexcept { return m_center + m_axis * (m_height * T(0.5)); }

    // ------------------------------------------------------------------------
    //  Distance to point (clamped to height, with radial interpolation)
    // ------------------------------------------------------------------------
    T squaredDistanceToPoint(const point_type& p) const noexcept {
        point_type bottomPt = bottom();
        point_type v = p - bottomPt;
        T proj = v.dot(m_axis);
        if (proj <= T(0)) {
            // below bottom: distance to bottom disk (circle)
            point_type radial = v - m_axis * proj;
            T radialDist = radial.length() - m_radiusBottom;
            if (radialDist < T(0)) radialDist = T(0);
            return radialDist * radialDist + proj * proj;
        }
        if (proj >= m_height) {
            // above top: distance to top disk
            point_type radial = v - m_axis * m_height;
            T radialDist = radial.length() - m_radiusTop;
            if (radialDist < T(0)) radialDist = T(0);
            T dz = proj - m_height;
            return radialDist * radialDist + dz * dz;
        }
        // inside height range: interpolate radius linearly
        T t = proj / m_height;
        T r_interp = m_radiusBottom * (T(1) - t) + m_radiusTop * t;
        point_type radial = v - m_axis * proj;
        T radialDist = radial.length() - r_interp;
        if (radialDist < T(0)) radialDist = T(0);
        return radialDist * radialDist;
    }

    T distanceToPoint(const point_type& p) const noexcept {
        return std::sqrt(squaredDistanceToPoint(p));
    }

    // ------------------------------------------------------------------------
    //  Ray intersection (returns t0, t1, false if no hit)
    //  Solve for intersection with infinite cone (or truncated cone)
    //  and clip by height.
    // ------------------------------------------------------------------------
    bool intersectRay(const ray_type& ray, T& t0, T& t1) const noexcept {
        // Implementation: transform to coordinate system where cone axis = Z,
        // origin at bottom center.
        point_type bottomPt = bottom();
        point_type u, v;
        if (std::abs(m_axis[0]) < T(0.9)) {
            u = normalize(cross(m_axis, point_type(1,0,0)));
        } else {
            u = normalize(cross(m_axis, point_type(0,1,0)));
        }
        v = cross(u, m_axis);
        auto toLocal = [&](const point_type& p) -> point_type {
            point_type d = p - bottomPt;
            return point_type(dot(d, u), dot(d, v), dot(d, m_axis));
        };
        point_type lo = toLocal(ray.origin());
        point_type ld = point_type(dot(ray.direction(), u), dot(ray.direction(), v), dot(ray.direction(), m_axis));

        T r_b = m_radiusBottom;
        T r_t = m_radiusTop;
        T h = m_height;
        T slope = (r_t - r_b) / h;  // linear radius variation
        T intercept = r_b;

        // For a point at height z (0..h), radius = intercept + slope * z
        // The cone surface equation: x^2 + y^2 = (intercept + slope * z)^2
        // This is a quadratic in z: (x^2 + y^2) - (intercept^2 + 2*intercept*slope*z + slope^2*z^2) = 0
        // Let's solve for intersection of ray (lo + ld * t) with this surface.
        // Substitute: x = lo.x + ld.x * t, y = lo.y + ld.y * t, z = lo.z + ld.z * t.
        // Then we have a quadratic in t: A t^2 + B t + C = 0.
        T a = ld[0]*ld[0] + ld[1]*ld[1] - slope*slope * ld[2]*ld[2];
        T b = T(2)*(lo[0]*ld[0] + lo[1]*ld[1]) - T(2)*slope*ld[2]*(intercept + slope*lo[2]);
        T c = lo[0]*lo[0] + lo[1]*lo[1] - (intercept + slope*lo[2])*(intercept + slope*lo[2]);
        T disc = b*b - T(4)*a*c;
        if (disc < T(0)) return false;
        T sqrtDisc = std::sqrt(disc);
        T tCone0 = (-b - sqrtDisc) / (T(2)*a);
        T tCone1 = (-b + sqrtDisc) / (T(2)*a);
        if (tCone0 > tCone1) std::swap(tCone0, tCone1);
        // Clip by height range [0, h]
        T z0 = lo[2] + ld[2] * tCone0;
        T z1 = lo[2] + ld[2] * tCone1;
        // Also consider intersection with bottom and top disks
        t0 = tCone0;
        t1 = tCone1;
        // Simple: if both z outside range, check caps
        bool hit = false;
        // Bottom cap: z=0, disk radius r_b
        T tBottom = (T(0) - lo[2]) / ld[2];
        if (tBottom >= T(0)) {
            point_type pHit = lo + ld * tBottom;
            if (pHit[0]*pHit[0] + pHit[1]*pHit[1] <= r_b*r_b) {
                if (!hit || tBottom < t0) t0 = tBottom;
                if (!hit || tBottom > t1) t1 = tBottom;
                hit = true;
            }
        }
        // Top cap: z=h, disk radius r_t
        T tTop = (h - lo[2]) / ld[2];
        if (tTop >= T(0)) {
            point_type pHit = lo + ld * tTop;
            if (pHit[0]*pHit[0] + pHit[1]*pHit[1] <= r_t*r_t) {
                if (!hit || tTop < t0) t0 = tTop;
                if (!hit || tTop > t1) t1 = tTop;
                hit = true;
            }
        }
        // If no hit from caps, check if cone intersection segments are within height
        if (!hit && z0 <= h && z1 >= T(0)) {
            if (z0 < T(0)) t0 = (-lo[2]) / ld[2];
            else t0 = tCone0;
            if (z1 > h) t1 = (h - lo[2]) / ld[2];
            else t1 = tCone1;
            if (t0 <= t1) hit = true;
        }
        return hit;
    }

    // ------------------------------------------------------------------------
    //  Bounding box (conservative)
    // ------------------------------------------------------------------------
    aabb_type boundingBox() const noexcept {
        point_type bottomPt = bottom();
        point_type topPt = top();
        T maxRad = std::max(m_radiusBottom, m_radiusTop);
        point_type minP = bottomPt.componentWiseMin(topPt) - point_type(maxRad);
        point_type maxP = bottomPt.componentWiseMax(topPt) + point_type(maxRad);
        return aabb_type(minP, maxP);
    }

    // ------------------------------------------------------------------------
    //  Transform cone by affine transform (approximate: scale radii by max scale)
    //  Height and axis recomputed from transformed bottom and top.
    // ------------------------------------------------------------------------
    Cone transform(const AffineTransform<T,3>& tf) const noexcept {
        point_type newBottom = tf.transform(bottom());
        point_type newTop = tf.transform(top());
        // Compute max scale factor from matrix
        T maxScale = T(0);
        for (int i = 0; i < 3; ++i) {
            point_type col;
            for (int j = 0; j < 3; ++j) col[j] = tf.matrix()(j, i);
            T len = col.length();
            if (len > maxScale) maxScale = len;
        }
        return Cone::fromEnds(newBottom, newTop, m_radiusBottom * maxScale, m_radiusTop * maxScale);
    }

    // ------------------------------------------------------------------------
    //  SIMD batch ray intersection (4 cones, 4 rays)
    // ------------------------------------------------------------------------
    static void batchRayIntersect(const Cone* cones, const ray_type* rays,
                                  bool* hit, T* t0, T* t1, size_t count) noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && std::is_same_v<T,float>) {
            for (size_t i = 0; i < count; ++i) {
                hit[i] = cones[i].intersectRay(rays[i], t0[i], t1[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                hit[i] = cones[i].intersectRay(rays[i], t0[i], t1[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool operator==(const Cone& other) const noexcept {
        T eps = T(1e-6);
        return (m_center - other.m_center).length() < eps &&
               (m_axis - other.m_axis).length() < eps &&
               std::abs(m_height - other.m_height) < eps &&
               std::abs(m_radiusBottom - other.m_radiusBottom) < eps &&
               std::abs(m_radiusTop - other.m_radiusTop) < eps;
    }

private:
    point_type m_center;
    point_type m_axis;
    T m_height;
    T m_radiusBottom;
    T m_radiusTop;
};

// ----------------------------------------------------------------------------
//  Helper: create standard cone (apex at top)
// ----------------------------------------------------------------------------
template<typename T>
Cone<T> makeCone(const Vector<T,3>& apex, const Vector<T,3>& axis, T height, T baseRadius) {
    return Cone<T>::fromApex(apex, axis, height, baseRadius);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class ConeEnvironment {
public:
    static ConeEnvironment& instance() {
        static ConeEnvironment env;
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
    ConeEnvironment() : m_epsilon(T(1e-8)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_CONE_H_INCLUDED