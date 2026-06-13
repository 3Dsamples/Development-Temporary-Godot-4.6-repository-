//File group name : OrthoTree Math
//File 0070 : core/math/geometry/triangle.h
//Triangle primitive (3 points) in 3D. Area, centroid, normal, barycentric coordinates, ray intersection (Möller–Trumbore), closest point, point containment, and SIMD batch intersection.

#ifndef ORTHOTREE_CORE_MATH_GEOMETRY_TRIANGLE_H_INCLUDED
#define ORTHOTREE_CORE_MATH_GEOMETRY_TRIANGLE_H_INCLUDED

#include "../../build_config.h"
#include "../basic/vector.h"
#include "../basic/matrix.h"
#include "ray.h"
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
//  Triangle in 3D: vertices v0, v1, v2.
// ============================================================================
template<typename T = float>
class Triangle {
public:
    using value_type = T;
    using point_type = Basic::Vector<T, 3>;
    using ray_type = Ray<T, 3>;
    using aabb_type = AABB<T, 3>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    Triangle() noexcept : m_v0(T(0)), m_v1(T(1,0,0)), m_v2(T(0,1,0)) {}
    Triangle(const point_type& v0, const point_type& v1, const point_type& v2) noexcept
        : m_v0(v0), m_v1(v1), m_v2(v2) {}

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    constexpr const point_type& v0() const noexcept { return m_v0; }
    constexpr const point_type& v1() const noexcept { return m_v1; }
    constexpr const point_type& v2() const noexcept { return m_v2; }
    constexpr void setV0(const point_type& v) noexcept { m_v0 = v; }
    constexpr void setV1(const point_type& v) noexcept { m_v1 = v; }
    constexpr void setV2(const point_type& v) noexcept { m_v2 = v; }

    // ------------------------------------------------------------------------
    //  Geometric properties
    // ------------------------------------------------------------------------
    point_type normal() const noexcept {
        point_type e1 = m_v1 - m_v0;
        point_type e2 = m_v2 - m_v0;
        point_type n = e1.cross(e2);
        T len = n.length();
        if (len > T(0)) n = n / len;
        return n;
    }
    T area() const noexcept {
        point_type e1 = m_v1 - m_v0;
        point_type e2 = m_v2 - m_v0;
        return e1.cross(e2).length() * T(0.5);
    }
    point_type centroid() const noexcept {
        return (m_v0 + m_v1 + m_v2) / T(3);
    }
    aabb_type boundingBox() const noexcept {
        return aabb_type(m_v0.componentWiseMin(m_v1).componentWiseMin(m_v2),
                         m_v0.componentWiseMax(m_v1).componentWiseMax(m_v2));
    }

    // ------------------------------------------------------------------------
    //  Barycentric coordinates for a point on the triangle’s plane
    //  Returns true if inside (including edges), and outputs u, v.
    // ------------------------------------------------------------------------
    bool barycentric(const point_type& p, T& u, T& v, T eps = T(1e-8)) const noexcept {
        point_type e1 = m_v1 - m_v0;
        point_type e2 = m_v2 - m_v0;
        point_type n = e1.cross(e2);
        T det = n.squaredLength();
        if (det < eps) return false;
        point_type d = p - m_v0;
        T dot_n = d.dot(n);
        if (std::abs(dot_n) > eps) return false; // not in plane
        // Solve 2x2 linear system for (u, v)
        T a = e1.dot(e1);
        T b = e1.dot(e2);
        T c = e2.dot(e2);
        T dd = e1.dot(d);
        T ee = e2.dot(d);
        T invDenom = T(1) / (a*c - b*b);
        u = (c*dd - b*ee) * invDenom;
        v = (a*ee - b*dd) * invDenom;
        return (u >= -eps && v >= -eps && u + v <= T(1)+eps);
    }

    // ------------------------------------------------------------------------
    //  Ray intersection (Möller–Trumbore)
    //  Returns true if hit, t (along ray), u, v.
    // ------------------------------------------------------------------------
    bool intersectRay(const ray_type& ray, T& t, T& u, T& v, T eps = T(1e-8)) const noexcept {
        point_type e1 = m_v1 - m_v0;
        point_type e2 = m_v2 - m_v0;
        point_type pvec = ray.direction().cross(e2);
        T det = e1.dot(pvec);
        if (std::abs(det) < eps) return false;
        T invDet = T(1) / det;
        point_type tvec = ray.origin() - m_v0;
        u = tvec.dot(pvec) * invDet;
        if (u < T(0) || u > T(1)) return false;
        point_type qvec = tvec.cross(e1);
        v = ray.direction().dot(qvec) * invDet;
        if (v < T(0) || u + v > T(1)) return false;
        t = e2.dot(qvec) * invDet;
        return (t >= T(0));
    }

    // ------------------------------------------------------------------------
    //  Closest point on triangle to an external point.
    //  Returns closest point and optionally squared distance.
    // ------------------------------------------------------------------------
    point_type closestPoint(const point_type& p, T* outDistSq = nullptr) const noexcept {
        // Check if projection lies inside triangle via barycentric
        T u, v;
        if (barycentric(p, u, v, T(1e-6))) {
            point_type closest = m_v0 + (m_v1 - m_v0) * u + (m_v2 - m_v0) * v;
            if (outDistSq) *outDistSq = (closest - p).squaredLength();
            return closest;
        }
        // Clamp to edges and vertices
        point_type edges[3][2] = {{m_v0, m_v1}, {m_v1, m_v2}, {m_v2, m_v0}};
        point_type bestPoint = m_v0;
        T bestDistSq = (m_v0 - p).squaredLength();
        for (int i = 0; i < 3; ++i) {
            const point_type& a = edges[i][0];
            const point_type& b = edges[i][1];
            point_type ab = b - a;
            T t_ = (p - a).dot(ab) / ab.squaredLength();
            t_ = Basic::clamp(t_, T(0), T(1));
            point_type closest = a + ab * t_;
            T d2 = (closest - p).squaredLength();
            if (d2 < bestDistSq) {
                bestDistSq = d2;
                bestPoint = closest;
            }
        }
        if (outDistSq) *outDistSq = bestDistSq;
        return bestPoint;
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool nearlyEqual(const Triangle& other, T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) const noexcept {
        return m_v0.nearlyEqual(other.m_v0, eps) &&
               m_v1.nearlyEqual(other.m_v1, eps) &&
               m_v2.nearlyEqual(other.m_v2, eps);
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: intersect 4 rays with 4 triangles (pairwise)
    // ------------------------------------------------------------------------
    static void batchRayIntersect(const Triangle* triangles, const ray_type* rays,
                                  bool* hit, T* t, T* u, T* v, size_t count) noexcept {
        if constexpr (std::is_same_v<T,float> && ORTHOTREE_SIMD_LEVEL >= 128) {
            for (size_t i = 0; i < count; ++i) {
                hit[i] = triangles[i].intersectRay(rays[i], t[i], u[i], v[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                hit[i] = triangles[i].intersectRay(rays[i], t[i], u[i], v[i]);
            }
        }
    }

private:
    point_type m_v0, m_v1, m_v2;
};

// ----------------------------------------------------------------------------
//  Convenience aliases
// ----------------------------------------------------------------------------
using Triangle3f = Triangle<float>;
using Triangle3d = Triangle<double>;

// ----------------------------------------------------------------------------
//  Helper: create triangle from three points
// ----------------------------------------------------------------------------
template<typename T>
Triangle<T> makeTriangle(const Basic::Vector<T,3>& v0, const Basic::Vector<T,3>& v1, const Basic::Vector<T,3>& v2) {
    return Triangle<T>(v0, v1, v2);
}

// ----------------------------------------------------------------------------
//  Helper: compute triangle normal (non‑unit)
// ----------------------------------------------------------------------------
template<typename T>
Basic::Vector<T,3> triangleNormal(const Triangle<T>& tri) {
    return tri.normal();
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class TriangleEnvironment {
public:
    static TriangleEnvironment& instance() {
        static TriangleEnvironment env;
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
    TriangleEnvironment() : m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    bool m_useSIMD;
};

} // namespace Geometry
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_GEOMETRY_TRIANGLE_H_INCLUDED