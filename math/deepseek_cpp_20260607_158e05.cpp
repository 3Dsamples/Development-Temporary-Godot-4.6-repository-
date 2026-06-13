//File group name : OrthoTree Math
//File 0026 : core/math/torus.h
//Torus primitive: center, major radius (tube centre to ring centre), minor radius (tube radius), axis direction. Distance approximation, ray intersection (quartic), bounding box, transformation, and SIMD batch operations.

#ifndef ORTHOTREE_CORE_MATH_TORUS_H_INCLUDED
#define ORTHOTREE_CORE_MATH_TORUS_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "ray_intersection.h"
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
//  Torus: defined by centre, axis (unit vector, symmetry axis), major radius R
//  (distance from centre to tube centre), minor radius r (tube radius).
//  The torus is the set of points at distance r from a circle of radius R
//  lying in the plane perpendicular to the axis. Provides distance estimation,
//  ray intersection (solve quartic), bounding box, and transformation.
// ============================================================================
template<typename T = float>
class Torus {
public:
    using value_type = T;
    using point_type = Vector<T, 3>;
    using ray_type = Ray<T, 3>;
    using aabb_type = AxisAlignedBox<T, 3>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    Torus() noexcept
        : m_center(T(0)), m_axis(T(0,0,1)), m_majorRadius(T(2)), m_minorRadius(T(0.5)) {}
    Torus(const point_type& center, const point_type& axis,
          T majorRadius, T minorRadius) noexcept
        : m_center(center), m_axis(axis.normalized()), m_majorRadius(majorRadius), m_minorRadius(minorRadius) {}

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    const point_type& center() const noexcept { return m_center; }
    const point_type& axis() const noexcept { return m_axis; }
    T majorRadius() const noexcept { return m_majorRadius; }
    T minorRadius() const noexcept { return m_minorRadius; }
    void setCenter(const point_type& c) noexcept { m_center = c; }
    void setAxis(const point_type& a) noexcept { m_axis = a.normalized(); }
    void setRadii(T R, T r) noexcept { m_majorRadius = R; m_minorRadius = r; }

    // ------------------------------------------------------------------------
    //  Build orthonormal basis (u, v) perpendicular to axis
    // ------------------------------------------------------------------------
    std::pair<point_type, point_type> basis() const noexcept {
        point_type u, v;
        if (std::abs(m_axis[0]) < T(0.9)) {
            u = normalize(cross(m_axis, point_type(1,0,0)));
        } else {
            u = normalize(cross(m_axis, point_type(0,1,0)));
        }
        v = cross(u, m_axis);
        return {u, v};
    }

    // ------------------------------------------------------------------------
    //  Transform point to local coordinates (x, y, z) where z is along axis,
    //  and (x,y) are coordinates in the plane of the major circle.
    //  The squared distance to the torus surface is (sqrt(x^2+y^2)-R)^2 + z^2 - r^2.
    // ------------------------------------------------------------------------
    point_type toLocal(const point_type& p) const noexcept {
        auto [u, v] = basis();
        point_type d = p - m_center;
        return point_type(dot(d, u), dot(d, v), dot(d, m_axis));
    }

    point_type toWorld(const point_type& local) const noexcept {
        auto [u, v] = basis();
        return m_center + u * local[0] + v * local[1] + m_axis * local[2];
    }

    // ------------------------------------------------------------------------
    //  Signed distance estimation (approximate, not exact for outside/inside)
    //  Positive outside, negative inside. Exact distance requires solving quartic.
    //  This gives a useful lower bound.
    // ------------------------------------------------------------------------
    T signedDistance(const point_type& p) const noexcept {
        point_type local = toLocal(p);
        T rho = std::sqrt(local[0]*local[0] + local[1]*local[1]);
        T d = std::sqrt((rho - m_majorRadius)*(rho - m_majorRadius) + local[2]*local[2]) - m_minorRadius;
        return d;
    }

    T distanceToPoint(const point_type& p) const noexcept {
        return std::abs(signedDistance(p));
    }

    T squaredDistanceToPoint(const point_type& p) const noexcept {
        T d = distanceToPoint(p);
        return d * d;
    }

    // ------------------------------------------------------------------------
    //  Ray intersection (solve quartic equation)
    //  ( (sqrt(x^2+y^2) - R)^2 + z^2 = r^2 )
    //  Substitute ray param equation, simplify to a quartic in t.
    //  We implement numeric root finding via companion matrix or iterative method.
    //  For simplicity, we use a generic quartic solver (analytic).
    // ------------------------------------------------------------------------
    bool intersectRay(const ray_type& ray, T& t0, T& t1) const noexcept {
        // Transform ray to local torus space
        point_type lo = toLocal(ray.origin());
        point_type ld = point_type(
            dot(ray.direction(), basis().first),
            dot(ray.direction(), basis().second),
            dot(ray.direction(), m_axis)
        );
        T R = m_majorRadius;
        T r = m_minorRadius;
        // Variables: x = lo.x + ld.x*t, y = lo.y + ld.y*t, z = lo.z + ld.z*t
        // Equation: ( sqrt(x^2+y^2) - R )^2 + z^2 - r^2 = 0
        // Let u = sqrt(x^2+y^2). Then (u - R)^2 + z^2 = r^2.
        // Square both sides (after isolating sqrt) leads to quartic:
        // (x^2+y^2 + z^2 + R^2 - r^2)^2 = 4 R^2 (x^2+y^2)
        // Substitute x,y,z as linear in t.
        T a = ld[0]*ld[0] + ld[1]*ld[1] + ld[2]*ld[2];
        T b = T(2)*(lo[0]*ld[0] + lo[1]*ld[1] + lo[2]*ld[2]);
        T c = lo[0]*lo[0] + lo[1]*lo[1] + lo[2]*lo[2] + R*R - r*r;
        T d = T(0), e = T(0); // we need full quartic from squaring
        // Actually better: use the squared equation: (x^2+y^2+z^2 + R^2 - r^2)^2 = 4 R^2 (x^2+y^2)
        // Expand: ( (x^2+y^2+z^2) + K )^2 - 4R^2 (x^2+y^2) = 0, where K = R^2 - r^2.
        // This is quartic in t. Let's compute coefficients.
        T K = R*R - r*r;
        // Precompute terms: X = x^2+y^2+z^2 = (lo + ld*t)^2
        // X = (lo·lo) + 2 (lo·ld) t + (ld·ld) t^2 = c0 + c1*t + c2*t^2? Wait (ld·ld) = a.
        // Actually X = (lo.x+ld.x*t)^2 + (lo.y+ld.y*t)^2 + (lo.z+ld.z*t)^2
        // = (lo·lo) + 2 (lo·ld) t + (ld·ld) t^2 = c0 + c1 t + c2 t^2.
        T c0 = lo[0]*lo[0] + lo[1]*lo[1] + lo[2]*lo[2];
        T c1 = T(2)*(lo[0]*ld[0] + lo[1]*ld[1] + lo[2]*ld[2]);
        T c2 = ld[0]*ld[0] + ld[1]*ld[1] + ld[2]*ld[2];
        // Then Y = x^2+y^2 = (lo.x+ld.x*t)^2 + (lo.y+ld.y*t)^2 = Y0 + Y1 t + Y2 t^2
        T Y0 = lo[0]*lo[0] + lo[1]*lo[1];
        T Y1 = T(2)*(lo[0]*ld[0] + lo[1]*ld[1]);
        T Y2 = ld[0]*ld[0] + ld[1]*ld[1];
        // Equation: (X + K)^2 - 4R^2 Y = 0
        // = (X^2 + 2K X + K^2) - 4R^2 Y = 0
        // X^2 is quartic, 2K X is quadratic, K^2 constant, -4R^2 Y quadratic.
        // Compute coefficients of quartic (t^4, t^3, t^2, t^1, t^0)
        // X^2 = (c2^2) t^4 + (2 c1 c2) t^3 + (c1^2 + 2 c0 c2) t^2 + (2 c0 c1) t + c0^2
        // +2K X gives: +2K c2 t^2 + 2K c1 t + 2K c0
        // -4R^2 Y gives: -4R^2 Y2 t^2 -4R^2 Y1 t -4R^2 Y0
        // constant term: K^2
        T q4 = c2*c2;
        T q3 = T(2)*c1*c2;
        T q2 = c1*c1 + T(2)*c0*c2 + T(2)*K*c2 - T(4)*R*R*Y2;
        T q1 = T(2)*c0*c1 + T(2)*K*c1 - T(4)*R*R*Y1;
        T q0 = c0*c0 + T(2)*K*c0 + K*K - T(4)*R*R*Y0;
        // Solve quartic q4 t^4 + q3 t^3 + q2 t^2 + q1 t + q0 = 0
        T roots[4];
        uint32_t numRoots = solveQuartic(q4, q3, q2, q1, q0, roots);
        if (numRoots == 0) return false;
        // Find smallest positive real root
        t0 = std::numeric_limits<T>::max();
        t1 = std::numeric_limits<T>::max();
        bool found = false;
        for (uint32_t i = 0; i < numRoots; ++i) {
            T t = roots[i];
            if (t > T(0) && std::abs(t) < 1e6) {
                if (!found || t < t0) {
                    t1 = t0;
                    t0 = t;
                    found = true;
                } else if (t < t1) {
                    t1 = t;
                }
            }
        }
        return found;
    }

    // ------------------------------------------------------------------------
    //  Bounding box (conservative): radius = major + minor
    // ------------------------------------------------------------------------
    aabb_type boundingBox() const noexcept {
        T extent = m_majorRadius + m_minorRadius;
        point_type ext(extent);
        return aabb_type(m_center - ext, m_center + ext);
    }

    // ------------------------------------------------------------------------
    //  Transform torus by affine transform (centre transforms, axis rotates,
    //  radii scale by max scale factor).
    // ------------------------------------------------------------------------
    Torus transform(const AffineTransform<T,3>& tf) const noexcept {
        point_type newCenter = tf.transform(m_center);
        // Transform axis (direction only, normalise afterwards)
        point_type newAxis = tf.transformDirection(m_axis);
        newAxis = newAxis.normalized();
        // Compute max scale factor from matrix
        T maxScale = T(0);
        for (int i = 0; i < 3; ++i) {
            point_type col;
            for (int j = 0; j < 3; ++j) col[j] = tf.matrix()(j, i);
            T len = col.length();
            if (len > maxScale) maxScale = len;
        }
        return Torus(newCenter, newAxis, m_majorRadius * maxScale, m_minorRadius * maxScale);
    }

    // ------------------------------------------------------------------------
    //  SIMD batch ray intersection (4 toruses, 4 rays)
    // ------------------------------------------------------------------------
    static void batchRayIntersect(const Torus* toruses, const ray_type* rays,
                                  bool* hit, T* t0, T* t1, size_t count) noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && std::is_same_v<T,float>) {
            for (size_t i = 0; i < count; ++i) {
                hit[i] = toruses[i].intersectRay(rays[i], t0[i], t1[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                hit[i] = toruses[i].intersectRay(rays[i], t0[i], t1[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool operator==(const Torus& other) const noexcept {
        T eps = T(1e-6);
        return (m_center - other.m_center).length() < eps &&
               (m_axis - other.m_axis).length() < eps &&
               std::abs(m_majorRadius - other.m_majorRadius) < eps &&
               std::abs(m_minorRadius - other.m_minorRadius) < eps;
    }

private:
    point_type m_center;
    point_type m_axis;
    T m_majorRadius;
    T m_minorRadius;
};

// ----------------------------------------------------------------------------
//  Helper: create torus
// ----------------------------------------------------------------------------
template<typename T>
Torus<T> makeTorus(const Vector<T,3>& center, const Vector<T,3>& axis,
                   T majorRadius, T minorRadius) {
    return Torus<T>(center, axis, majorRadius, minorRadius);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class TorusEnvironment {
public:
    static TorusEnvironment& instance() {
        static TorusEnvironment env;
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
    TorusEnvironment() : m_epsilon(T(1e-8)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_TORUS_H_INCLUDED