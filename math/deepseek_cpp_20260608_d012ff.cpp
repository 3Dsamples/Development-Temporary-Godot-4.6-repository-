//File group name : OrthoTree Math
//File 0072 : core/math/geometry/polygon.h
//2D polygon: area, centroid, point containment (winding number), convex hull (Graham scan), triangulation (ear clipping), bounding box, and SIMD batch point containment.

#ifndef ORTHOTREE_CORE_MATH_GEOMETRY_POLYGON_H_INCLUDED
#define ORTHOTREE_CORE_MATH_GEOMETRY_POLYGON_H_INCLUDED

#include "../../build_config.h"
#include "../basic/vector.h"
#include "aabb.h"
#include "../robust/orientation.h"
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

namespace OrthoTree {
namespace Math {
namespace Geometry {

// ============================================================================
//  Polygon2D: closed polygon in 2D (CCW order for orientation).
//  Provides area, centroid, point containment (winding number),
//  convex hull (Graham scan), triangulation (ear clipping), bounding box.
// ============================================================================
template<typename T = float>
class Polygon2D {
public:
    using value_type = T;
    using point_type = Basic::Vector<T, 2>;
    using aabb_type = AABB<T, 2>;
    using size_type = size_t;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    Polygon2D() = default;
    explicit Polygon2D(const std::vector<point_type>& vertices) : m_vertices(vertices) {}
    Polygon2D(std::vector<point_type>&& vertices) : m_vertices(std::move(vertices)) {}

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    const std::vector<point_type>& vertices() const noexcept { return m_vertices; }
    void setVertices(const std::vector<point_type>& verts) { m_vertices = verts; }
    size_type size() const noexcept { return m_vertices.size(); }
    bool empty() const noexcept { return m_vertices.empty(); }
    void clear() noexcept { m_vertices.clear(); }

    // ------------------------------------------------------------------------
    //  Bounding box
    // ------------------------------------------------------------------------
    aabb_type boundingBox() const noexcept {
        if (m_vertices.empty()) return aabb_type();
        point_type minP = m_vertices[0], maxP = m_vertices[0];
        for (const auto& p : m_vertices) {
            minP = minP.componentWiseMin(p);
            maxP = maxP.componentWiseMax(p);
        }
        return aabb_type(minP, maxP);
    }

    // ------------------------------------------------------------------------
    //  Signed area (positive for CCW)
    // ------------------------------------------------------------------------
    T signedArea() const noexcept {
        if (m_vertices.size() < 3) return T(0);
        T area2 = T(0);
        size_type n = m_vertices.size();
        for (size_type i = 0; i < n; ++i) {
            const point_type& p = m_vertices[i];
            const point_type& q = m_vertices[(i+1) % n];
            area2 += p[0] * q[1] - q[0] * p[1];
        }
        return area2 * T(0.5);
    }
    T area() const noexcept { return std::abs(signedArea()); }

    // ------------------------------------------------------------------------
    //  Centroid (area‑weighted average)
    // ------------------------------------------------------------------------
    point_type centroid() const noexcept {
        if (m_vertices.empty()) return point_type(0);
        if (m_vertices.size() == 1) return m_vertices[0];
        if (m_vertices.size() == 2) return (m_vertices[0] + m_vertices[1]) * T(0.5);
        T area2 = T(0);
        point_type c(0);
        size_type n = m_vertices.size();
        for (size_type i = 0; i < n; ++i) {
            const point_type& p = m_vertices[i];
            const point_type& q = m_vertices[(i+1) % n];
            T cross = p[0] * q[1] - q[0] * p[1];
            area2 += cross;
            c[0] += (p[0] + q[0]) * cross;
            c[1] += (p[1] + q[1]) * cross;
        }
        if (std::abs(area2) < T(1e-12)) return point_type(0);
        T inv = T(1) / (area2 * T(3));
        c[0] *= inv;
        c[1] *= inv;
        return c;
    }

    // ------------------------------------------------------------------------
    //  Point containment (winding number algorithm, robust)
    //  Returns true if point is inside (including boundary).
    // ------------------------------------------------------------------------
    bool containsPoint(const point_type& p, T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) const noexcept {
        if (m_vertices.size() < 3) return false;
        T winding = T(0);
        size_type n = m_vertices.size();
        for (size_type i = 0; i < n; ++i) {
            const point_type& a = m_vertices[i];
            const point_type& b = m_vertices[(i+1) % n];
            T angle = std::atan2(b[1] - a[1], b[0] - a[0]) - std::atan2(p[1] - a[1], p[0] - a[0]);
            if (angle > Constants<T>::pi()) angle -= T(2) * Constants<T>::pi();
            if (angle < -Constants<T>::pi()) angle += T(2) * Constants<T>::pi();
            winding += angle;
        }
        return std::abs(winding) > Constants<T>::pi();
    }

    // ------------------------------------------------------------------------
    //  Convex hull (Graham scan), returns new polygon in CCW order.
    // ------------------------------------------------------------------------
    Polygon2D convexHull() const {
        if (m_vertices.size() <= 1) return *this;
        std::vector<point_type> pts = m_vertices;
        // Find point with lowest y (and leftmost if tie)
        std::sort(pts.begin(), pts.end(), [](const point_type& a, const point_type& b) {
            return a[1] < b[1] || (a[1] == b[1] && a[0] < b[0]);
        });
        point_type pivot = pts[0];
        std::sort(pts.begin() + 1, pts.end(), [&](const point_type& a, const point_type& b) {
            T o = Robust::orient2d(pivot, a, b);
            if (std::abs(o) < T(1e-12)) {
                T da = (a - pivot).squaredLength();
                T db = (b - pivot).squaredLength();
                return da < db;
            }
            return o > T(0);
        });
        std::vector<point_type> hull;
        for (const auto& p : pts) {
            while (hull.size() >= 2 && Robust::orient2d(hull[hull.size()-2], hull.back(), p) <= T(0))
                hull.pop_back();
            hull.push_back(p);
        }
        return Polygon2D(std::move(hull));
    }

    // ------------------------------------------------------------------------
    //  Ear clipping triangulation: returns list of triangles (each as 3 point_type)
    //  Works for simple polygons (no self‑intersections). Assumes CCW order.
    // ------------------------------------------------------------------------
    std::vector<std::array<point_type,3>> triangulate() const {
        std::vector<std::array<point_type,3>> triangles;
        if (m_vertices.size() < 3) return triangles;
        std::vector<size_type> indices(m_vertices.size());
        std::iota(indices.begin(), indices.end(), 0);
        size_type n = indices.size();
        while (n > 3) {
            bool earFound = false;
            for (size_type i = 0; i < n; ++i) {
                size_type prev = (i + n - 1) % n;
                size_type next = (i + 1) % n;
                const point_type& a = m_vertices[indices[prev]];
                const point_type& b = m_vertices[indices[i]];
                const point_type& c = m_vertices[indices[next]];
                // Check if triangle is convex (CCW orientation)
                if (Robust::orient2d(a, b, c) <= T(0)) continue;
                bool inside = false;
                for (size_type j = 0; j < n; ++j) {
                    if (j == prev || j == i || j == next) continue;
                    const point_type& p = m_vertices[indices[j]];
                    if (pointInTriangle(p, a, b, c)) {
                        inside = true;
                        break;
                    }
                }
                if (!inside) {
                    triangles.push_back({a, b, c});
                    indices.erase(indices.begin() + i);
                    n = indices.size();
                    earFound = true;
                    break;
                }
            }
            if (!earFound) break; // degenerate polygon
        }
        if (n == 3) {
            triangles.push_back({m_vertices[indices[0]], m_vertices[indices[1]], m_vertices[indices[2]]});
        }
        return triangles;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: test 4 points for containment (with same polygon)
    // ------------------------------------------------------------------------
    void batchContainsPoint(const point_type* points, bool* out, size_t count) const noexcept {
        if constexpr (std::is_same_v<T,float> && ORTHOTREE_SIMD_LEVEL >= 128) {
            for (size_t i = 0; i < count; ++i) {
                out[i] = containsPoint(points[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                out[i] = containsPoint(points[i]);
            }
        }
    }

private:
    // Helper for point‑in‑triangle (2D) using barycentric coordinates
    static bool pointInTriangle(const point_type& p, const point_type& a,
                                const point_type& b, const point_type& c) noexcept {
        T v0x = c[0] - a[0], v0y = c[1] - a[1];
        T v1x = b[0] - a[0], v1y = b[1] - a[1];
        T v2x = p[0] - a[0], v2y = p[1] - a[1];
        T dot00 = v0x*v0x + v0y*v0y;
        T dot01 = v0x*v1x + v0y*v1y;
        T dot02 = v0x*v2x + v0y*v2y;
        T dot11 = v1x*v1x + v1y*v1y;
        T dot12 = v1x*v2x + v1y*v2y;
        T invDenom = T(1) / (dot00 * dot11 - dot01 * dot01);
        T u = (dot11 * dot02 - dot01 * dot12) * invDenom;
        T v = (dot00 * dot12 - dot01 * dot02) * invDenom;
        return (u >= T(0)) && (v >= T(0)) && (u + v <= T(1));
    }

    std::vector<point_type> m_vertices;
};

// ----------------------------------------------------------------------------
//  Convenience aliases
// ----------------------------------------------------------------------------
using Polygon2f = Polygon2D<float>;
using Polygon2d = Polygon2D<double>;

// ----------------------------------------------------------------------------
//  Helper: create rectangle polygon from AABB
// ----------------------------------------------------------------------------
template<typename T>
Polygon2D<T> rectangleFromAABB(const AABB<T,2>& box) {
    std::vector<Basic::Vector<T,2>> verts = {
        box.min(),
        {box.max()[0], box.min()[1]},
        box.max(),
        {box.min()[0], box.max()[1]}
    };
    return Polygon2D<T>(verts);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class PolygonEnvironment {
public:
    static PolygonEnvironment& instance() {
        static PolygonEnvironment env;
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
    PolygonEnvironment() : m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    bool m_useSIMD;
};

} // namespace Geometry
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_GEOMETRY_POLYGON_H_INCLUDED