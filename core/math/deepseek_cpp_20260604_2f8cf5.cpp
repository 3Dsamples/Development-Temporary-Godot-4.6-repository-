// system name : onetbb-warp
// File 0024 : core/math/computational_geometry.h
// Description : Convex hull, Delaunay triangulation, Voronoi, point‑in‑polygon, polygon area.

#ifndef __TBB_WARP_CORE_MATH_COMPUTATIONAL_GEOMETRY_H
#define __TBB_WARP_CORE_MATH_COMPUTATIONAL_GEOMETRY_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/geometry.h"
#include "core/math/quickhull3d.h"   // Full QuickHull 3D implementation
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <functional>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Orientation test (2D) – signed area of triangle (p, q, r)
// ============================================================

template<typename T>
constexpr T orient2d(const vector2<T>& p, const vector2<T>& q, const vector2<T>& r) noexcept {
    return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);
}

// ============================================================
// Orientation test (3D) – signed volume of tetrahedron (a,b,c,d)
// ============================================================

template<typename T>
constexpr T orient3d(const vector3<T>& a, const vector3<T>& b, const vector3<T>& c, const vector3<T>& d) noexcept {
    return dot(cross(b - a, c - a), d - a);
}

// ============================================================
// In‑circle test (2D) – positive if d inside circumcircle of a,b,c
// ============================================================

template<typename T>
T incircle(const vector2<T>& a, const vector2<T>& b, const vector2<T>& c, const vector2<T>& d) noexcept {
    T adx = a.x - d.x, ady = a.y - d.y;
    T bdx = b.x - d.x, bdy = b.y - d.y;
    T cdx = c.x - d.x, cdy = c.y - d.y;
    T ab = adx * bdy - ady * bdx;
    T bc = bdx * cdy - bdy * cdx;
    T ca = cdx * ady - cdy * adx;
    T a2 = adx*adx + ady*ady;
    T b2 = bdx*bdx + bdy*bdy;
    T c2 = cdx*cdx + cdy*cdy;
    return a2 * bc + b2 * ca + c2 * ab;
}

// ============================================================
// 2D Convex hull – Graham scan
// ============================================================

template<typename T>
std::vector<vector2<T>> convex_hull_2d(std::vector<vector2<T>> points) {
    if (points.size() <= 2) return points;
    std::sort(points.begin(), points.end(), [](const vector2<T>& a, const vector2<T>& b) {
        return a.y < b.y || (a.y == b.y && a.x < b.x);
    });
    vector2<T> pivot = points[0];
    std::sort(points.begin() + 1, points.end(), [&](const vector2<T>& a, const vector2<T>& b) {
        T o = orient2d(pivot, a, b);
        if (o == 0) return distance_sq(pivot, a) < distance_sq(pivot, b);
        return o > 0;
    });
    std::vector<vector2<T>> hull;
    hull.push_back(points[0]);
    hull.push_back(points[1]);
    for (std::size_t i = 2; i < points.size(); ++i) {
        while (hull.size() >= 2 && orient2d(hull[hull.size()-2], hull.back(), points[i]) <= 0)
            hull.pop_back();
        hull.push_back(points[i]);
    }
    return hull;
}

// ============================================================
// 3D Convex hull – Full QuickHull (delegates to quickhull3d.h)
// ============================================================

template<typename T>
struct convex_hull_3d_face {
    std::array<int,3> vertices;
    vector3<T> normal;
    T offset;
    std::vector<int> outside_points;
};

template<typename T>
std::vector<convex_hull_3d_face<T>> convex_hull_3d(const std::vector<vector3<T>>& points) {
    // Call full QuickHull algorithm
    auto quick_faces = quickhull3d(points);
    std::vector<convex_hull_3d_face<T>> result;
    result.reserve(quick_faces.size());
    for (const auto& qf : quick_faces) {
        convex_hull_3d_face<T> f;
        f.vertices = qf.vertices;
        f.normal   = qf.normal;
        f.offset   = qf.offset;
        f.outside_points = qf.outside_points;
        result.push_back(std::move(f));
    }
    return result;
}

// ============================================================
// Point in polygon (ray casting)
// ============================================================

template<typename T>
bool point_in_polygon(const std::vector<vector2<T>>& polygon, const vector2<T>& point) {
    int n = (int)polygon.size();
    if (n < 3) return false;
    bool inside = false;
    for (int i = 0, j = n-1; i < n; j = i++) {
        const auto& pi = polygon[i];
        const auto& pj = polygon[j];
        if (((pi.y > point.y) != (pj.y > point.y)) &&
            (point.x < (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x))
            inside = !inside;
    }
    return inside;
}

// ============================================================
// Polygon signed area (2D) and centroid
// ============================================================

template<typename T>
T polygon_area_signed(const std::vector<vector2<T>>& polygon) {
    T area = 0;
    int n = (int)polygon.size();
    if (n < 3) return 0;
    for (int i = 0; i < n; ++i) {
        const auto& a = polygon[i];
        const auto& b = polygon[(i+1)%n];
        area += a.x * b.y - b.x * a.y;
    }
    return area * 0.5;
}

template<typename T>
vector2<T> polygon_centroid(const std::vector<vector2<T>>& polygon) {
    int n = (int)polygon.size();
    vector2<T> c(0,0);
    if (n < 3) return c;
    T area = 0;
    for (int i = 0; i < n; ++i) {
        const auto& a = polygon[i];
        const auto& b = polygon[(i+1)%n];
        T cross = a.x * b.y - b.x * a.y;
        area += cross;
        c.x += (a.x + b.x) * cross;
        c.y += (a.y + b.y) * cross;
    }
    T inv = 1.0 / (3.0 * area);
    c.x *= inv; c.y *= inv;
    return c;
}

// ============================================================
// Ear clipping triangulation of simple polygon (2D)
// ============================================================

template<typename T>
std::vector<std::array<int,3>> triangulate_ear_clipping(const std::vector<vector2<T>>& polygon) {
    std::vector<std::array<int,3>> triangles;
    int n = (int)polygon.size();
    if (n < 3) return triangles;
    std::vector<int> indices(n);
    for (int i=0; i<n; ++i) indices[i] = i;
    int remaining = n;
    while (remaining > 2) {
        bool ear_found = false;
        for (int i = 0; i < remaining; ++i) {
            int prev = (i-1+remaining)%remaining;
            int next = (i+1)%remaining;
            int pi = indices[prev], ci = indices[i], ni = indices[next];
            const auto& a = polygon[pi], &b = polygon[ci], &c = polygon[ni];
            if (orient2d(a, b, c) <= 0) continue; // convex ear candidate
            bool is_ear = true;
            for (int j = 0; j < remaining; ++j) {
                if (j == prev || j == i || j == next) continue;
                if (point_in_triangle(polygon[indices[j]], a, b, c)) {
                    is_ear = false; break;
                }
            }
            if (is_ear) {
                triangles.push_back({pi, ci, ni});
                indices.erase(indices.begin() + i);
                remaining--;
                ear_found = true;
                break;
            }
        }
        if (!ear_found) break; // safeguard
    }
    return triangles;
}

template<typename T>
bool point_in_triangle(const vector2<T>& p, const vector2<T>& a, const vector2<T>& b, const vector2<T>& c) {
    T o1 = orient2d(a, b, p);
    T o2 = orient2d(b, c, p);
    T o3 = orient2d(c, a, p);
    bool neg = (o1 < 0) || (o2 < 0) || (o3 < 0);
    bool pos = (o1 > 0) || (o2 > 0) || (o3 > 0);
    return !(neg && pos);
}

// ============================================================
// Delaunay triangulation (2D) – incremental Bowyer‑Watson
// ============================================================

template<typename T>
struct delaunay_triangle {
    std::array<int,3> v;
    std::array<int,3> neighbors; // adjacent triangle indices, -1 for boundary
};

template<typename T>
std::vector<delaunay_triangle<T>> delaunay_triangulate(std::vector<vector2<T>> points) {
    std::vector<delaunay_triangle<T>> tris;
    int n = (int)points.size();
    if (n < 3) return tris;
    // Compute bounding box and super‑triangle
    T min_x=points[0].x, max_x=points[0].x, min_y=points[0].y, max_y=points[0].y;
    for (const auto& p : points) {
        min_x = min(min_x, p.x); max_x = max(max_x, p.x);
        min_y = min(min_y, p.y); max_y = max(max_y, p.y);
    }
    T dx = max_x - min_x, dy = max_y - min_y;
    T dmax = max(dx, dy) * 10.0;
    vector2<T> super[3] = {
        {min_x - dmax, min_y - dmax},
        {min_x + dx/2.0 + dmax, min_y - dmax},
        {min_x - dmax, min_y + dy + dmax}
    };
    points.push_back(super[0]);
    points.push_back(super[1]);
    points.push_back(super[2]);
    int super_idx[3] = {n, n+1, n+2};
    // Start with super‑triangle
    tris.push_back({{n, n+1, n+2}, {-1,-1,-1}});
    // Insert points one by one
    for (int pi = 0; pi < n; ++pi) {
        const auto& p = points[pi];
        std::vector<int> bad_tris;
        for (int ti = 0; ti < (int)tris.size(); ++ti) {
            const auto& t = tris[ti];
            const auto& a = points[t.v[0]], &b = points[t.v[1]], &c = points[t.v[2]];
            if (incircle(a, b, c, p) > 0) bad_tris.push_back(ti);
        }
        // Find boundary edges
        struct edge { int a,b; };
        std::vector<edge> boundary;
        for (int ti : bad_tris) {
            const auto& t = tris[ti];
            auto test_edge = [&](int i, int j) {
                edge e{t.v[i], t.v[j]};
                int count = 0;
                for (int tj : bad_tris) {
                    const auto& t2 = tris[tj];
                    for (int k=0; k<3; ++k) {
                        if ((t2.v[k]==e.a && t2.v[(k+1)%3]==e.b) ||
                            (t2.v[k]==e.b && t2.v[(k+1)%3]==e.a)) ++count;
                    }
                }
                if (count == 1) boundary.push_back(e);
            };
            test_edge(0,1);
            test_edge(1,2);
            test_edge(2,0);
        }
        // Remove bad triangles
        std::sort(bad_tris.begin(), bad_tris.end(), std::greater<int>());
        for (int ti : bad_tris) {
            tris.erase(tris.begin() + ti);
            // update indices (already sorted descending)
        }
        // Re‑triangulate hole
        for (const auto& e : boundary) {
            tris.push_back({{e.a, e.b, pi}, {-1,-1,-1}});
        }
    }
    // Remove triangles that use super‑triangle vertices
    std::vector<delaunay_triangle<T>> result;
    for (const auto& t : tris) {
        if (t.v[0] >= n || t.v[1] >= n || t.v[2] >= n) continue;
        result.push_back(t);
    }
    return result;
}

// ============================================================
// Voronoi diagram (from Delaunay)
// ============================================================

template<typename T>
std::vector<std::vector<vector2<T>>> voronoi_from_delaunay(const std::vector<vector2<T>>& points,
                                                           const std::vector<delaunay_triangle<T>>& tris) {
    int n = (int)points.size();
    std::vector<std::vector<vector2<T>>> voronoi_cells(n);
    struct tri_data { vector2<T> center; };
    std::vector<tri_data> centers(tris.size());
    for (int ti=0; ti<(int)tris.size(); ++ti) {
        const auto& a = points[tris[ti].v[0]], &b = points[tris[ti].v[1]], &c = points[tris[ti].v[2]];
        T d = 2.0 * (a.x*(b.y-c.y) + b.x*(c.y-a.y) + c.x*(a.y-b.y));
        if (std::abs(d) < 1e-12) { centers[ti].center = (a+b+c)/3.0; continue; }
        T ux = ((a.x*a.x+a.y*a.y)*(b.y-c.y)+(b.x*b.x+b.y*b.y)*(c.y-a.y)+(c.x*c.x+c.y*c.y)*(a.y-b.y))/d;
        T uy = ((a.x*a.x+a.y*a.y)*(c.x-b.x)+(b.x*b.x+b.y*b.y)*(a.x-c.x)+(c.x*c.x+c.y*c.y)*(b.x-a.x))/d;
        centers[ti].center = {ux, uy};
    }
    for (int ti=0; ti<(int)tris.size(); ++ti) {
        for (int k=0; k<3; ++k) {
            int site = tris[ti].v[k];
            voronoi_cells[site].push_back(centers[ti].center);
        }
    }
    for (int i=0; i<n; ++i) {
        auto& cell = voronoi_cells[i];
        if (cell.empty()) continue;
        const auto& site = points[i];
        std::sort(cell.begin(), cell.end(), [&](const vector2<T>& a, const vector2<T>& b) {
            return std::atan2(a.y-site.y, a.x-site.x) < std::atan2(b.y-site.y, b.x-site.x);
        });
    }
    return voronoi_cells;
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_COMPUTATIONAL_GEOMETRY_H