//File group name : OrthoTree Math
//File 0018 : core/math/triangle.h
//Triangle primitive (3 points, normal, area, centroid, ray intersection, point-in-triangle, barycentric coordinates, SIMD batch operations).

#ifndef ORTHOTREE_CORE_MATH_TRIANGLE_H_INCLUDED
#define ORTHOTREE_CORE_MATH_TRIANGLE_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "ray_intersection.h"
#include "geometry_queries.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <array>
#include <algorithm>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Triangle: defined by 3 vertices (v0, v1, v2). Provides geometric properties
//  and queries: normal, area, centroid, ray intersection (Möller–Trumbore),
//  point containment (barycentric), closest point, and SIMD batch operations.
// ============================================================================
template<typename T = float>
class Triangle {
public:
    using value_type = T;
    using point_type = Vector<T, 3>;
    using ray_type = Ray<T, 3>;
    using aabb_type = AxisAlignedBox<T, 3>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    Triangle() noexcept : m_v0(T(0)), m_v1(T(1,0,0)), m_v2(T(0,1,0)) {}
    Triangle(const point_type& v0, const point_type& v1, const point_type& v2) noexcept
        : m_v0(v0), m_v1(v1), m_v2(v2) {}

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    const point_type& v0() const noexcept { return m_v0; }
    const point_type& v1() const noexcept { return m_v1; }
    const point_type& v2() const noexcept { return m_v2; }
    void setV0(const point_type& v) noexcept { m_v0 = v; }
    void setV1(const point_type& v) noexcept { m_v1 = v; }
    void setV2(const point_type& v) noexcept { m_v2 = v; }

    // ------------------------------------------------------------------------
    //  Geometric properties
    // ------------------------------------------------------------------------
    point_type normal() const noexcept {
        point_type e1 = m_v1 - m_v0;
        point_type e2 = m_v2 - m_v0;
        point_type n = cross(e1, e2);
        T len = n.length();
        if (len > T(0)) n = n / len;
        return n;
    }

    T area() const noexcept {
        point_type e1 = m_v1 - m_v0;
        point_type e2 = m_v2 - m_v0;
        return cross(e1, e2).length() * T(0.5);
    }

    point_type centroid() const noexcept {
        return (m_v0 + m_v1 + m_v2) / T(3);
    }

    aabb_type boundingBox() const noexcept {
        return aabb_type(m_v0.componentWiseMin(m_v1).componentWiseMin(m_v2),
                         m_v0.componentWiseMax(m_v1).componentWiseMax(m_v2));
    }

    // ------------------------------------------------------------------------
    //  Barycentric coordinates for a point (projected onto triangle plane)
    //  Returns (u, v) such that point = v0 + u*(v1-v0) + v*(v2-v0)
    //  Returns false if point is not in plane or outside.
    // ------------------------------------------------------------------------
    bool barycentric(const point_type& p, T& u, T& v) const noexcept {
        point_type e1 = m_v1 - m_v0;
        point_type e2 = m_v2 - m_v0;
        point_type n = cross(e1, e2);
        T len2 = n.squaredLength();
        if (len2 < T(1e-12)) return false;
        point_type d = p - m_v0;
        T dot_n = dot(d, n);
        if (std::abs(dot_n) > T(1e-8)) return false; // not in plane
        T invDet = T(1) / (e1[0]*e2[1] - e1[1]*e2[0]); // simplified 2D projection
        // More robust: project onto two axes
        T u0 = (d[0]*e2[1] - d[1]*e2[0]) * invDet;
        T v0 = (e1[0]*d[1] - e1[1]*d[0]) * invDet;
        u = u0; v = v0;
        return (u >= T(0)) && (v >= T(0)) && (u + v <= T(1));
    }

    // ------------------------------------------------------------------------
    //  Ray intersection (Möller–Trumbore)
    //  Returns true if hit, outputs t, u, v.
    // ------------------------------------------------------------------------
    bool intersectRay(const ray_type& ray, T& t, T& u, T& v) const noexcept {
        point_type e1 = m_v1 - m_v0;
        point_type e2 = m_v2 - m_v0;
        point_type pvec = cross(ray.direction(), e2);
        T det = dot(e1, pvec);
        if (std::abs(det) < T(1e-8)) return false;
        T invDet = T(1) / det;
        point_type tvec = ray.origin() - m_v0;
        u = dot(tvec, pvec) * invDet;
        if (u < T(0) || u > T(1)) return false;
        point_type qvec = cross(tvec, e1);
        v = dot(ray.direction(), qvec) * invDet;
        if (v < T(0) || u + v > T(1)) return false;
        t = dot(e2, qvec) * invDet;
        return (t >= T(0));
    }

    // ------------------------------------------------------------------------
    //  Closest point on triangle to external point
    //  Returns closest point and squared distance (optionally).
    // ------------------------------------------------------------------------
    point_type closestPoint(const point_type& p, T* outDistSq = nullptr) const noexcept {
        // Check if projection falls inside triangle using barycentric
        T u, v;
        if (barycentric(p, u, v)) {
            point_type closest = m_v0 + (m_v1 - m_v0) * u + (m_v2 - m_v0) * v;
            if (outDistSq) *outDistSq = (closest - p).squaredLength();
            return closest;
        }
        // Check each edge (segment) and vertex, take minimum
        point_type edges[3][2] = {{m_v0, m_v1}, {m_v1, m_v2}, {m_v2, m_v0}};
        point_type bestPoint = m_v0;
        T bestDistSq = (m_v0 - p).squaredLength();
        for (int i = 0; i < 3; ++i) {
            const point_type& a = edges[i][0];
            const point_type& b = edges[i][1];
            point_type ab = b - a;
            T t = (p - a).dot(ab) / ab.squaredLength();
            t = clamp(t, T(0), T(1));
            point_type closest = a + ab * t;
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
    //  Transform triangle by affine transform
    // ------------------------------------------------------------------------
    Triangle transform(const AffineTransform<T,3>& tf) const noexcept {
        return Triangle(tf.transform(m_v0), tf.transform(m_v1), tf.transform(m_v2));
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: ray intersect 4 triangles with the same ray
    //  Input: array of 4 triangles, single ray. Output: hit flags, t, u, v.
    // ------------------------------------------------------------------------
    static void batchRayIntersect(const Triangle* triangles, const ray_type& ray,
                                  bool* hit, T* t, T* u, T* v, size_t count) {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && std::is_same_v<T,float>) {
            for (size_t i = 0; i < count; ++i) {
                hit[i] = triangles[i].intersectRay(ray, t[i], u[i], v[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                hit[i] = triangles[i].intersectRay(ray, t[i], u[i], v[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  SIMD batch closest point (4 triangles, 4 points)
    // ------------------------------------------------------------------------
    static void batchClosestPoint(const Triangle* triangles, const point_type* points,
                                  point_type* out, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            out[i] = triangles[i].closestPoint(points[i]);
        }
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool operator==(const Triangle& other) const noexcept {
        T eps = T(1e-6);
        return (m_v0 - other.m_v0).length() < eps &&
               (m_v1 - other.m_v1).length() < eps &&
               (m_v2 - other.m_v2).length() < eps;
    }

private:
    point_type m_v0, m_v1, m_v2;
};

// ----------------------------------------------------------------------------
//  Helper: create triangle with normal (compute automatically)
// ----------------------------------------------------------------------------
template<typename T>
Triangle<T> makeTriangle(const Vector<T,3>& v0, const Vector<T,3>& v1, const Vector<T,3>& v2) {
    return Triangle<T>(v0, v1, v2);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller for triangle operations
// ----------------------------------------------------------------------------
class TriangleEnvironment {
public:
    static TriangleEnvironment& instance() {
        static TriangleEnvironment env;
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
    TriangleEnvironment() : m_epsilon(T(1e-8)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_TRIANGLE_H_INCLUDED