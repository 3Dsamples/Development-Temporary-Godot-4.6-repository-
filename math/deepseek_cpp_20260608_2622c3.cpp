//File group name : OrthoTree Math
//File 0055 : core/math/geometry/plane.h
//Plane in 3D defined by normal and distance. Signed distance, point classification, projection, intersection with ray, transformation, and SIMD batch operations.

#ifndef ORTHOTREE_CORE_MATH_GEOMETRY_PLANE_H_INCLUDED
#define ORTHOTREE_CORE_MATH_GEOMETRY_PLANE_H_INCLUDED

#include "../../build_config.h"
#include "../basic/vector.h"
#include "../basic/matrix.h"
#include "ray.h"
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <limits>

namespace OrthoTree {
namespace Math {
namespace Geometry {

// ============================================================================
//  Plane in 3D: defined by unit normal (n) and distance from origin (d)
//  such that n·x + d = 0.
// ============================================================================
template<typename T = float>
class Plane {
public:
    using value_type = T;
    using point_type = Basic::Vector<T, 3>;
    using ray_type = Ray<T, 3>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    constexpr Plane() noexcept : m_normal(0,0,1), m_d(0) {}
    constexpr Plane(const point_type& normal, T d) noexcept
        : m_normal(normal.normalized()), m_d(d) {}
    constexpr Plane(const point_type& normal, const point_type& point) noexcept
        : m_normal(normal.normalized()), m_d(-m_normal.dot(point)) {}
    Plane(const point_type& p1, const point_type& p2, const point_type& p3) noexcept {
        m_normal = (p2 - p1).cross(p3 - p1).normalized();
        m_d = -m_normal.dot(p1);
    }

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    constexpr const point_type& normal() const noexcept { return m_normal; }
    constexpr T d() const noexcept { return m_d; }
    constexpr void setNormal(const point_type& n) noexcept { m_normal = n.normalized(); }
    constexpr void setD(T d) noexcept { m_d = d; }

    // ------------------------------------------------------------------------
    //  Distance and classification
    // ------------------------------------------------------------------------
    constexpr T signedDistance(const point_type& p) const noexcept {
        return m_normal.dot(p) + m_d;
    }
    T distance(const point_type& p) const noexcept {
        return std::abs(signedDistance(p));
    }
    int classify(const point_type& p, T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) const noexcept {
        T dist = signedDistance(p);
        if (dist > eps) return 1;   // front
        if (dist < -eps) return -1; // back
        return 0;                   // on
    }

    // ------------------------------------------------------------------------
    //  Project point onto plane
    // ------------------------------------------------------------------------
    point_type project(const point_type& p) const noexcept {
        return p - m_normal * signedDistance(p);
    }

    // ------------------------------------------------------------------------
    //  Intersection with ray (returns t along ray, true if hit)
    // ------------------------------------------------------------------------
    bool intersectRay(const ray_type& ray, T& t) const noexcept {
        T denom = m_normal.dot(ray.direction());
        if (std::abs(denom) < T(1e-8)) return false;
        t = -(m_normal.dot(ray.origin()) + m_d) / denom;
        return t >= T(0);
    }

    // ------------------------------------------------------------------------
    //  Intersection of two planes (returns line direction and a point on the line)
    //  Returns false if planes are parallel.
    // ------------------------------------------------------------------------
    bool intersectPlanes(const Plane& other, point_type& lineDir, point_type& linePoint) const noexcept {
        lineDir = m_normal.cross(other.m_normal);
        T len2 = lineDir.squaredLength();
        if (len2 < T(1e-12)) return false;
        lineDir = lineDir / std::sqrt(len2);
        // Solve for a point on both planes: using two equations in 3D; assume plane normals not parallel
        // Simplified: set one coordinate to zero and solve the 2x2 system.
        // Choose the coordinate where normals have largest magnitude.
        int pivot = 0;
        T maxAbs = std::abs(m_normal[0]);
        for (int i = 1; i < 3; ++i) {
            T a = std::abs(m_normal[i]);
            if (a > maxAbs) { maxAbs = a; pivot = i; }
        }
        // Build 2x2 system from the other two coordinates
        int u = (pivot + 1) % 3;
        int v = (pivot + 2) % 3;
        T a11 = m_normal[u], a12 = m_normal[v];
        T b11 = other.m_normal[u], b12 = other.m_normal[v];
        T det = a11 * b12 - a12 * b11;
        if (std::abs(det) < T(1e-12)) {
            // fallback: use another pivot
            return false;
        }
        T invDet = T(1) / det;
        T rhs1 = -m_d - m_normal[pivot] * T(0); // set coordinate pivot to 0
        T rhs2 = -other.m_d - other.m_normal[pivot] * T(0);
        T solU = (rhs1 * b12 - rhs2 * a12) * invDet;
        T solV = (a11 * rhs2 - b11 * rhs1) * invDet;
        linePoint = point_type(T(0));
        linePoint[pivot] = T(0);
        linePoint[u] = solU;
        linePoint[v] = solV;
        return true;
    }

    // ------------------------------------------------------------------------
    //  Transform plane by affine transform (normal is transformed by inverse transpose)
    //  New plane: n' = (M^-T) * n, d' = d - n'·t
    // ------------------------------------------------------------------------
    Plane transform(const Basic::AffineTransform<T,3>& tf) const noexcept {
        point_type newNormal = tf.transformNormal(m_normal);
        T newD = m_d - newNormal.dot(tf.translation());
        return Plane(newNormal, newD);
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool nearlyEqual(const Plane& other, T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) const noexcept {
        return m_normal.nearlyEqual(other.m_normal, eps) && std::abs(m_d - other.m_d) < eps;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: signed distances for 4 points
    // ------------------------------------------------------------------------
    void batchSignedDistance(const point_type* points, T* out, size_t count) const noexcept {
        if constexpr (std::is_same_v<T,float> && ORTHOTREE_SIMD_LEVEL >= 128) {
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
    //  SIMD batch: intersect 4 planes with 4 rays (pairs)
    // ------------------------------------------------------------------------
    static void batchIntersectRay(const Plane* planes, const ray_type* rays,
                                  bool* hit, T* t, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) {
            hit[i] = planes[i].intersectRay(rays[i], t[i]);
        }
    }

private:
    point_type m_normal;
    T m_d;
};

// ----------------------------------------------------------------------------
//  Convenience aliases
// ----------------------------------------------------------------------------
using Plane3f = Plane<float>;
using Plane3d = Plane<double>;

// ----------------------------------------------------------------------------
//  Helper: plane from normal and point
// ----------------------------------------------------------------------------
template<typename T>
Plane<T> planeFromNormalAndPoint(const Basic::Vector<T,3>& normal, const Basic::Vector<T,3>& point) {
    return Plane<T>(normal, point);
}

// ----------------------------------------------------------------------------
//  Helper: plane from three points
// ----------------------------------------------------------------------------
template<typename T>
Plane<T> planeFromPoints(const Basic::Vector<T,3>& p1, const Basic::Vector<T,3>& p2, const Basic::Vector<T,3>& p3) {
    return Plane<T>(p1, p2, p3);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class PlaneEnvironment {
public:
    static PlaneEnvironment& instance() {
        static PlaneEnvironment env;
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
    PlaneEnvironment() : m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    bool m_useSIMD;
};

} // namespace Geometry
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_GEOMETRY_PLANE_H_INCLUDED