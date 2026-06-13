//File group name : OrthoTree Math
//File 0071 : core/math/geometry/line_segment.h
//Line segment in 2D/3D: defined by start and end points. Closest point, squared distance, ray intersection, bounding box, transformation, and SIMD batch operations.

#ifndef ORTHOTREE_CORE_MATH_GEOMETRY_LINE_SEGMENT_H_INCLUDED
#define ORTHOTREE_CORE_MATH_GEOMETRY_LINE_SEGMENT_H_INCLUDED

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
#include <algorithm>

namespace OrthoTree {
namespace Math {
namespace Geometry {

// ============================================================================
//  LineSegment for 2D or 3D (N = 2 or 3) with endpoints a and b.
// ============================================================================
template<typename T, std::size_t N>
class LineSegment {
public:
    using value_type = T;
    using point_type = Basic::Vector<T, N>;
    using ray_type = Ray<T, N>;
    using aabb_type = AABB<T, N>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    constexpr LineSegment() noexcept : m_a(T(0)), m_b(T(1)) {}
    constexpr LineSegment(const point_type& a, const point_type& b) noexcept : m_a(a), m_b(b) {}

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    constexpr const point_type& a() const noexcept { return m_a; }
    constexpr const point_type& b() const noexcept { return m_b; }
    constexpr void setA(const point_type& a) noexcept { m_a = a; }
    constexpr void setB(const point_type& b) noexcept { m_b = b; }

    constexpr point_type direction() const noexcept { return (m_b - m_a).normalized(); }
    constexpr point_type vector() const noexcept { return m_b - m_a; }
    constexpr T length() const noexcept { return (m_b - m_a).length(); }
    constexpr T squaredLength() const noexcept { return (m_b - m_a).squaredLength(); }

    // ------------------------------------------------------------------------
    //  Parameterisation: point at t ∈ [0,1]
    // ------------------------------------------------------------------------
    constexpr point_type pointAt(T t) const noexcept {
        return m_a + (m_b - m_a) * t;
    }

    // ------------------------------------------------------------------------
    //  Closest point on segment to external point. Returns parameter t and point.
    // ------------------------------------------------------------------------
    std::pair<T, point_type> closestPointParam(const point_type& p) const noexcept {
        point_type ab = m_b - m_a;
        T t = (p - m_a).dot(ab) / ab.squaredLength();
        t = Basic::clamp(t, T(0), T(1));
        return {t, m_a + ab * t};
    }
    point_type closestPoint(const point_type& p) const noexcept {
        return closestPointParam(p).second;
    }
    T squaredDistanceToPoint(const point_type& p) const noexcept {
        point_type cp = closestPoint(p);
        return (cp - p).squaredLength();
    }
    T distanceToPoint(const point_type& p) const noexcept {
        return std::sqrt(squaredDistanceToPoint(p));
    }

    // ------------------------------------------------------------------------
    //  Distance between two line segments (3D). Returns squared distance and
    //  parameters t1 (on this) and t2 (on other).
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
        T denom = a * c - b * b;
        if (std::abs(denom) < T(1e-12)) {
            // Parallel segments
            t1 = T(0);
            t2 = e / c;
            t2 = Basic::clamp(t2, T(0), T(1));
        } else {
            t1 = (b * e - c * d) / denom;
            t2 = (a * e - b * d) / denom;
            t1 = Basic::clamp(t1, T(0), T(1));
            t2 = Basic::clamp(t2, T(0), T(1));
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
    //  Ray intersection (2D or 3D). Returns true if hit, t (on ray), u (on segment).
    //  2D: solve linear equations. 3D: solve closest point and check.
    // ------------------------------------------------------------------------
    bool intersectRay(const ray_type& ray, T& t, T& u) const noexcept {
        if constexpr (N == 2) {
            // 2D segment‑ray intersection using cross product
            point_type r = ray.origin();
            point_type dir = ray.direction();
            point_type s = m_b - m_a;
            T cross = dir[0]*s[1] - dir[1]*s[0];
            if (std::abs(cross) < T(1e-12)) return false;
            T t_num = ( (m_a[0] - r[0])*s[1] - (m_a[1] - r[1])*s[0] ) / cross;
            T u_num = ( (m_a[0] - r[0])*dir[1] - (m_a[1] - r[1])*dir[0] ) / cross;
            if (t_num >= T(0) && u_num >= T(0) && u_num <= T(1)) {
                t = t_num;
                u = u_num;
                return true;
            }
            return false;
        } else {
            // 3D: use closest point between ray and line segment
            point_type ro = ray.origin();
            point_type rd = ray.direction();
            point_type ab = m_b - m_a;
            // Solve for t (ray) and u (segment)
            T a = rd.squaredLength();
            T b = rd.dot(ab);
            T c = ab.squaredLength();
            T d = rd.dot(m_a - ro);
            T e = ab.dot(m_a - ro);
            T denom = a * c - b * b;
            if (std::abs(denom) < T(1e-12)) return false;
            t = (b * e - c * d) / denom;
            u = (a * e - b * d) / denom;
            // Clamp u to [0,1], and then adjust t accordingly
            if (u < T(0)) {
                u = T(0);
                t = -(ro - m_a).dot(rd) / a;
            } else if (u > T(1)) {
                u = T(1);
                t = -(ro - m_b).dot(rd) / a;
            }
            if (t >= T(0) && u >= T(0) && u <= T(1)) {
                // Verify that the point on the line segment is actually the closest
                point_type hitSeg = pointAt(u);
                point_type hitRay = ray.pointAt(t);
                if ((hitSeg - hitRay).squaredLength() < T(1e-8)) return true;
            }
            return false;
        }
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
    LineSegment transform(const Basic::AffineTransform<T,N>& tf) const noexcept {
        return LineSegment(tf.transformPoint(m_a), tf.transformPoint(m_b));
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool nearlyEqual(const LineSegment& other, T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) const noexcept {
        return m_a.nearlyEqual(other.m_a, eps) && m_b.nearlyEqual(other.m_b, eps);
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: squared distance to point for 4 segments
    // ------------------------------------------------------------------------
    static void batchSquaredDistanceToPoint(const LineSegment* segs, const point_type* points,
                                            T* out, size_t count) noexcept {
        if constexpr (std::is_same_v<T,float> && ORTHOTREE_SIMD_LEVEL >= 128 && N == 3) {
            for (size_t i = 0; i < count; ++i) {
                out[i] = segs[i].squaredDistanceToPoint(points[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                out[i] = segs[i].squaredDistanceToPoint(points[i]);
            }
        }
    }

private:
    point_type m_a, m_b;
};

// ----------------------------------------------------------------------------
//  Convenience aliases
// ----------------------------------------------------------------------------
template<typename T> using LineSegment2 = LineSegment<T, 2>;
template<typename T> using LineSegment3 = LineSegment<T, 3>;

using LineSegment2f = LineSegment<float, 2>;
using LineSegment2d = LineSegment<double, 2>;
using LineSegment3f = LineSegment<float, 3>;
using LineSegment3d = LineSegment<double, 3>;

// ----------------------------------------------------------------------------
//  Helper: create segment from two points
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
LineSegment<T,N> makeLineSegment(const Basic::Vector<T,N>& a, const Basic::Vector<T,N>& b) {
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
    void setUseSIMD(bool use) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_useSIMD = use;
    }
    bool useSIMD() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_useSIMD;
    }
private:
    LineSegmentEnvironment() : m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    bool m_useSIMD;
};

} // namespace Geometry
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_GEOMETRY_LINE_SEGMENT_H_INCLUDED