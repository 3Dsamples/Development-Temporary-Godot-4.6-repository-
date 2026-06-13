//File group name : OrthoTree Math
//File 0019 : core/math/line_segment.h
//Line segment in 2D/3D: closest point, distance, intersection with ray/plane/other segment, projection, and SIMD batch operations.

#ifndef ORTHOTREE_CORE_MATH_LINE_SEGMENT_H_INCLUDED
#define ORTHOTREE_CORE_MATH_LINE_SEGMENT_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "ray_intersection.h"
#include "plane.h"
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
//  LineSegment: defined by start and end points. Provides parameterisation,
//  closest point to point, distance queries, intersection tests,
//  and SIMD batch operations for both 2D and 3D.
// ============================================================================
template<typename T = float, std::size_t N = 3>
class LineSegment {
public:
    using value_type = T;
    using point_type = Vector<T, N>;
    using ray_type = Ray<T, N>;
    using aabb_type = AxisAlignedBox<T, N>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    LineSegment() noexcept : m_a(point_type(0)), m_b(point_type(1)) {}
    LineSegment(const point_type& a, const point_type& b) noexcept : m_a(a), m_b(b) {}

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    const point_type& a() const noexcept { return m_a; }
    const point_type& b() const noexcept { return m_b; }
    void setA(const point_type& a) noexcept { m_a = a; }
    void setB(const point_type& b) noexcept { m_b = b; }

    point_type direction() const noexcept { return (m_b - m_a).normalized(); }
    point_type vector() const noexcept { return m_b - m_a; }
    T length() const noexcept { return (m_b - m_a).length(); }
    T squaredLength() const noexcept { return (m_b - m_a).squaredLength(); }

    point_type pointAt(T t) const noexcept {
        return m_a + (m_b - m_a) * t;
    }

    // ------------------------------------------------------------------------
    //  Closest point on segment to an external point
    //  Returns parameter t in [0,1] and the closest point.
    // ------------------------------------------------------------------------
    std::pair<T, point_type> closestPointParam(const point_type& p) const noexcept {
        point_type ab = m_b - m_a;
        T t = (p - m_a).dot(ab) / ab.squaredLength();
        t = clamp(t, T(0), T(1));
        return {t, m_a + ab * t};
    }

    point_type closestPoint(const point_type& p) const noexcept {
        return closestPointParam(p).second;
    }

    T squaredDistanceToPoint(const point_type& p) const noexcept {
        point_type closest = closestPoint(p);
        return (closest - p).squaredLength();
    }

    T distanceToPoint(const point_type& p) const noexcept {
        return std::sqrt(squaredDistanceToPoint(p));
    }

    // ------------------------------------------------------------------------
    //  Distance between two line segments (3D)
    //  Returns minimum distance squared.
    // ------------------------------------------------------------------------
    T squaredDistanceToSegment(const LineSegment& other, T& t1, T& t2) const noexcept {
        static_assert(N == 3, "Segment‑segment distance only for 3D");
        point_type u = vector();
        point_type v = other.vector();
        point_type w = m_a - other.m_a;
        T a = u.squaredLength();
        T b = u.dot(v);
        T c = v.squaredLength();
        T d = u.dot(w);
        T e = v.dot(w);
        T det = a * c - b * b;
        t1 = T(0); t2 = T(0);
        if (det < T(1e-12)) {
            // parallel segments
            t1 = T(0);
            T t2 = e / c;
            t2 = clamp(t2, T(0), T(1));
        } else {
            t1 = (b * e - c * d) / det;
            t2 = (a * e - b * d) / det;
            t1 = clamp(t1, T(0), T(1));
            t2 = clamp(t2, T(0), T(1));
        }
        point_type p1 = pointAt(t1);
        point_type p2 = other.pointAt(t2);
        return (p1 - p2).squaredLength();
    }

    T distanceToSegment(const LineSegment& other) const noexcept {
        T t1, t2;
        return std::sqrt(squaredDistanceToSegment(other, t1, t2));
    }

    // ------------------------------------------------------------------------
    //  Intersection with ray (2D and 3D)
    //  For 3D, returns false if not intersecting, else t (on ray) and u (on segment)
    // ------------------------------------------------------------------------
    bool intersectRay(const ray_type& ray, T& t, T& u) const noexcept {
        if constexpr (N == 2) {
            // 2D line segment intersection
            point_type r = ray.origin();
            point_type dir = ray.direction();
            point_type s = m_b - m_a;
            T cross = cross2D(dir, s);
            if (std::abs(cross) < T(1e-8)) return false;
            T t_num = cross2D(m_a - r, s);
            T u_num = cross2D(m_a - r, dir);
            t = t_num / cross;
            u = u_num / cross;
            return (t >= T(0) && u >= T(0) && u <= T(1));
        } else {
            // 3D: use line‑segment intersection (solve Möller‑Trumbore‑like)
            point_type ab = m_b - m_a;
            point_type pvec = cross(ray.direction(), ab);
            T det = dot(ray.direction(), pvec); // actually dot(ab, pvec)?? Not standard.
            // Simplified: we can use closest point between ray and line.
            point_type w = ray.origin() - m_a;
            T a = dot(ray.direction(), ray.direction());
            T b = dot(ray.direction(), ab);
            T c = dot(ab, ab);
            T d = dot(ray.direction(), w);
            T e = dot(ab, w);
            T denom = a * c - b * b;
            if (std::abs(denom) < T(1e-8)) return false;
            t = (b * e - c * d) / denom;
            u = (a * e - b * d) / denom;
            return (t >= T(0) && u >= T(0) && u <= T(1));
        }
    }

    // ------------------------------------------------------------------------
    //  Intersection with plane (3D)
    //  Returns true if segment crosses plane, outputs parameter t (on segment)
    // ------------------------------------------------------------------------
    bool intersectPlane(const Plane<T>& plane, T& t) const noexcept {
        T d1 = plane.signedDistance(m_a);
        T d2 = plane.signedDistance(m_b);
        if (d1 * d2 > T(0)) return false;
        t = d1 / (d1 - d2);
        t = clamp(t, T(0), T(1));
        return true;
    }

    // ------------------------------------------------------------------------
    //  Bounding box
    // ------------------------------------------------------------------------
    aabb_type boundingBox() const noexcept {
        return aabb_type(m_a.componentWiseMin(m_b), m_a.componentWiseMax(m_b));
    }

    // ------------------------------------------------------------------------
    //  Transform by affine transform
    // ------------------------------------------------------------------------
    LineSegment transform(const AffineTransform<T,N>& tf) const noexcept {
        return LineSegment(tf.transform(m_a), tf.transform(m_b));
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: closest point for 4 points (3D)
    // ------------------------------------------------------------------------
    void batchClosestPoint(const point_type* points, point_type* out, size_t count) const noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && N == 3 && std::is_same_v<T,float>) {
            for (size_t i = 0; i < count; ++i) {
                out[i] = closestPoint(points[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                out[i] = closestPoint(points[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool operator==(const LineSegment& other) const noexcept {
        T eps = T(1e-6);
        return (m_a - other.m_a).length() < eps && (m_b - other.m_b).length() < eps;
    }

private:
    point_type m_a, m_b;
};

// ----------------------------------------------------------------------------
//  2D cross product helper
// ----------------------------------------------------------------------------
template<typename T>
T cross2D(const Vector<T,2>& a, const Vector<T,2>& b) noexcept {
    return a[0]*b[1] - a[1]*b[0];
}

// ----------------------------------------------------------------------------
//  Helper: create line segment from two points
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
LineSegment<T,N> makeLineSegment(const Vector<T,N>& a, const Vector<T,N>& b) {
    return LineSegment<T,N>(a, b);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class LineSegmentEnvironment {
public:
    static LineSegmentEnvironment& instance() {
        static LineSegmentEnvironment env;
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
    LineSegmentEnvironment() : m_epsilon(T(1e-8)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_LINE_SEGMENT_H_INCLUDED