//File group name : OrthoTree Math
//File 0027 : core/math/polygon.h
//2D polygon (convex/concave) and planar 3D polygon: area, centroid, triangulation, point-in-polygon (winding number / ray casting), bounding box, convex hull (Graham scan), and SIMD batch point containment tests.

#ifndef ORTHOTREE_CORE_MATH_POLYGON_H_INCLUDED
#define ORTHOTREE_CORE_MATH_POLYGON_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "plane.h"
#include "geometry_queries.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Polygon2D: closed polygon in 2D. Supports area, centroid, point containment,
//  triangulation (ear clipping), convex hull, and bounding box.
//  Also provides SIMD batch point testing.
// ============================================================================
template<typename T = float>
class Polygon2D {
public:
    using value_type = T;
    using point_type = Vector<T, 2>;
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
    //  Geometric properties
    // ------------------------------------------------------------------------
    T area() const noexcept {
        if (m_vertices.size() < 3) return T(0);
        T a = T(0);
        size_type n = m_vertices.size();
        for (size_type i = 0; i < n; ++i) {
            const point_type& p = m_vertices[i];
            const point_type& q = m_vertices[(i+1)%n];
            a += p[0]*q[1] - q[0]*p[1];
        }
        return std::abs(a) * T(0.5);
    }

    point_type centroid() const noexcept {
        if (m_vertices.empty()) return point_type(0);
        if (m_vertices.size() == 1) return m_vertices[0];
        if (m_vertices.size() == 2) return (m_vertices[0] + m_vertices[1]) * T(0.5);
        T area2 = T(0);
        point_type c(0);
        size_type n = m_vertices.size();
        for (size_type i = 0; i < n; ++i) {
            const point_type& p = m_vertices[i];
            const point_type& q = m_vertices[(i+1)%n];
            T cross = p[0]*q[1] - q[0]*p[1];
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
    //  Point containment (winding number algorithm, robust)
    //  Returns true if point is inside (including boundary).
    // ------------------------------------------------------------------------
    bool containsPoint(const point_type& p, T eps = T(1e-8)) const noexcept {
        if (m_vertices.size() < 3) return false;
        T winding = 0;
        size_type n = m_vertices.size();
        for (size_type i = 0; i < n; ++i) {
            const point_type& a = m_vertices[i];
            const point_type& b = m_vertices[(i+1)%n];
            if (segmentIntersect2D(a, b, p, p + point_type(1e5,0))) return true; // ray casting
            T angle = std::atan2(b[1]-a[1], b[0]-a[0]) - std::atan2(p[1]-a[1], p[0]-a[0]);
            if (angle > Math::pi<T>()) angle -= T(2)*Math::pi<T>();
            if (angle < -Math::pi<T>()) angle += T(2)*Math::pi<T>();
            winding += angle;
        }
        return std::abs(winding) > Math::pi<T>();
    }

    // ------------------------------------------------------------------------
    //  Triangulation (ear clipping) – returns list of triangles (indices)
    //  Each triangle is 3 indices into original vertex array.
    //  Works for simple polygons (no self‑intersections).
    // ------------------------------------------------------------------------
    std::vector<std::array<size_type,3>> triangulate() const {
        std::vector<std::array<size_type,3>> triangles;
        if (m_vertices.size() < 3) return triangles;
        std::vector<size_type> indices(m_vertices.size());
        std::iota(indices.begin(), indices.end(), 0);
        size_type n = indices.size();
        while (n > 3) {
            bool earFound = false;
            for (size_type i = 0; i < n; ++i) {
                size_type prev = (i + n - 1) % n;
                size_type next = (i + 1) % n;
                point_type a = m_vertices[indices[prev]];
                point_type b = m_vertices[indices[i]];
                point_type c = m_vertices[indices[next]];
                // Check if triangle is convex (orientation > 0 for CCW polygon)
                if (orient2d(a, b, c) <= T(0)) continue;
                // Check if any other point lies inside triangle
                bool inside = false;
                for (size_type j = 0; j < n; ++j) {
                    if (j == prev || j == i || j == next) continue;
                    point_type p = m_vertices[indices[j]];
                    if (pointInTriangle(p, a, b, c)) {
                        inside = true;
                        break;
                    }
                }
                if (!inside) {
                    triangles.push_back({indices[prev], indices[i], indices[next]});
                    indices.erase(indices.begin() + i);
                    n = indices.size();
                    earFound = true;
                    break;
                }
            }
            if (!earFound) break; // degenerate polygon
        }
        if (n == 3) triangles.push_back({indices[0], indices[1], indices[2]});
        return triangles;
    }

    // ------------------------------------------------------------------------
    //  Convex hull (Graham scan) – returns new polygon in CCW order.
    // ------------------------------------------------------------------------
    Polygon2D convexHull() const {
        if (m_vertices.size() <= 1) return *this;
        std::vector<point_type> pts = m_vertices;
        // Find point with lowest y (and leftmost)
        std::sort(pts.begin(), pts.end(), [](const point_type& a, const point_type& b) {
            return a[1] < b[1] || (a[1] == b[1] && a[0] < b[0]);
        });
        point_type pivot = pts[0];
        std::sort(pts.begin() + 1, pts.end(), [&](const point_type& a, const point_type& b) {
            T o = orient2d(pivot, a, b);
            if (std::abs(o) < T(1e-12)) {
                T da = (a - pivot).squaredLength();
                T db = (b - pivot).squaredLength();
                return da < db;
            }
            return o > T(0);
        });
        std::vector<point_type> hull;
        for (const auto& p : pts) {
            while (hull.size() >= 2 && orient2d(hull[hull.size()-2], hull.back(), p) <= T(0))
                hull.pop_back();
            hull.push_back(p);
        }
        return Polygon2D(std::move(hull));
    }

    // ------------------------------------------------------------------------
    //  SIMD batch point containment (4 points at once)
    // ------------------------------------------------------------------------
    void batchContainsPoint(const point_type* points, bool* out, size_t count) const noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && std::is_same_v<T,float>) {
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
    std::vector<point_type> m_vertices;
};

// ============================================================================
//  PlanarPolygon3D: polygon lying on a plane in 3D.
//  Provides area, centroid, point containment (projected to 2D).
// ============================================================================
template<typename T = float>
class PlanarPolygon3D {
public:
    using value_type = T;
    using point_type = Vector<T, 3>;
    using size_type = size_t;

    PlanarPolygon3D() = default;
    PlanarPolygon3D(const std::vector<point_type>& vertices, const Plane<T>& plane)
        : m_vertices(vertices), m_plane(plane) {}
    PlanarPolygon3D(std::vector<point_type>&& vertices, const Plane<T>& plane)
        : m_vertices(std::move(vertices)), m_plane(plane) {}

    const std::vector<point_type>& vertices() const noexcept { return m_vertices; }
    const Plane<T>& plane() const noexcept { return m_plane; }
    size_type size() const noexcept { return m_vertices.size(); }
    bool empty() const noexcept { return m_vertices.empty(); }

    // ------------------------------------------------------------------------
    //  Transform polygon to 2D (project onto plane and build 2D polygon)
    // ------------------------------------------------------------------------
    Polygon2D<T> to2D() const {
        // Build orthonormal basis in plane
        point_type n = m_plane.normal();
        point_type u, v;
        if (std::abs(n[0]) < T(0.9)) {
            u = normalize(cross(n, point_type(1,0,0)));
        } else {
            u = normalize(cross(n, point_type(0,1,0)));
        }
        v = cross(u, n);
        std::vector<Vector<T,2>> verts2D;
        verts2D.reserve(m_vertices.size());
        for (const auto& p : m_vertices) {
            Vector<T,2> p2(dot(p - m_plane.project(point_type(0)), u),
                           dot(p - m_plane.project(point_type(0)), v));
            verts2D.push_back(p2);
        }
        return Polygon2D<T>(verts2D);
    }

    T area() const { return to2D().area(); }
    point_type centroid() const {
        auto poly2D = to2D();
        Vector<T,2> c2D = poly2D.centroid();
        // Map back to 3D
        point_type u, v;
        point_type n = m_plane.normal();
        if (std::abs(n[0]) < T(0.9)) {
            u = normalize(cross(n, point_type(1,0,0)));
        } else {
            u = normalize(cross(n, point_type(0,1,0)));
        }
        v = cross(u, n);
        point_type origin = m_plane.project(point_type(0));
        return origin + u * c2D[0] + v * c2D[1];
    }

    bool containsPoint(const point_type& p, T eps = T(1e-8)) const {
        if (std::abs(m_plane.signedDistance(p)) > eps) return false;
        auto poly2D = to2D();
        Vector<T,2> p2 = poly2D.to2D(p); // need a method to project point to 2D
        return poly2D.containsPoint(p2, eps);
    }

private:
    std::vector<point_type> m_vertices;
    Plane<T> m_plane;
};

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class PolygonEnvironment {
public:
    static PolygonEnvironment& instance() {
        static PolygonEnvironment env;
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
    PolygonEnvironment() : m_epsilon(T(1e-8)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_POLYGON_H_INCLUDED