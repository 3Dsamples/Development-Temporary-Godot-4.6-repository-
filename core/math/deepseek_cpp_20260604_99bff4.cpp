// system name : onetbb-warp
// File 0025 : core/math/quickhull3d.h
// Description : Full QuickHull algorithm for 3D convex hull computation.

#ifndef __TBB_WARP_CORE_MATH_QUICKHULL3D_H
#define __TBB_WARP_CORE_MATH_QUICKHULL3D_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector3.h"
#include "core/math/matrix3.h"
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <limits>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <stack>
#include <cstdint>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Data structures for QuickHull 3D
// ============================================================

template<typename T>
struct quickhull_face {
    std::array<int,3> vertices;       // indices into point array
    vector3<T> normal;                // outward unit normal
    T offset;                         // plane distance from origin
    std::vector<int> outside_points;  // points outside this face
    bool deleted = false;
};

template<typename T>
struct quickhull_edge {
    int a, b;
    bool operator==(const quickhull_edge& o) const {
        return (a == o.a && b == o.b) || (a == o.b && b == o.a);
    }
};

template<typename T>
struct quickhull_edge_hash {
    std::size_t operator()(const quickhull_edge<T>& e) const {
        return std::hash<int>()(e.a) ^ (std::hash<int>()(e.b) << 1);
    }
};

// ============================================================
// Helper: signed distance from point to face plane
// ============================================================

template<typename T>
T signed_face_distance(const vector3<T>& point, const vector3<T>& normal, T offset) {
    return dot(normal, point) + offset;
}

// ============================================================
// Helper: determine if point is visible from face
// ============================================================

template<typename T>
bool is_point_outside(const vector3<T>& point, const vector3<T>& normal, T offset, T epsilon = T(1e-9)) {
    return signed_face_distance(point, normal, offset) > epsilon;
}

// ============================================================
// Helper: compute face normal from three points (outward)
// ============================================================

template<typename T>
vector3<T> compute_face_normal(const vector3<T>& a, const vector3<T>& b, const vector3<T>& c,
                               const vector3<T>& interior_point) {
    vector3<T> v0 = b - a;
    vector3<T> v1 = c - a;
    vector3<T> n = cross(v0, v1);
    T len = length(n);
    if (len < T(1e-12)) return vector3<T>(T(0),T(1),T(0));
    n = n / len;
    // Make sure normal points outward (away from interior point)
    if (dot(n, a - interior_point) < T(0)) n = -n;
    return n;
}

// ============================================================
// Helper: find extreme point along direction
// ============================================================

template<typename T>
int extreme_point_index(const std::vector<vector3<T>>& points,
                        const vector3<T>& direction) {
    int best = 0;
    T best_dot = dot(points[0], direction);
    for (std::size_t i = 1; i < points.size(); ++i) {
        T d = dot(points[i], direction);
        if (d > best_dot) { best_dot = d; best = static_cast<int>(i); }
    }
    return best;
}

// ============================================================
// Helper: compute interior point (centroid of initial points)
// ============================================================

template<typename T>
vector3<T> approximate_interior(const std::vector<vector3<T>>& points) {
    vector3<T> sum(0,0,0);
    for (const auto& p : points) sum = sum + p;
    return sum / T(points.size());
}

// ============================================================
// Main QuickHull 3D algorithm
// ============================================================

template<typename T>
std::vector<quickhull_face<T>> quickhull3d(const std::vector<vector3<T>>& points) {
    std::vector<quickhull_face<T>> result;
    if (points.size() < 4) return result;

    // 1. Build initial tetrahedron from 6 extreme points
    vector3<T> axes[3] = {
        vector3<T>(1,0,0), vector3<T>(0,1,0), vector3<T>(0,0,1)
    };
    int extrema[6];
    for (int i = 0; i < 3; ++i) {
        extrema[2*i]   = extreme_point_index(points, axes[i]);
        extrema[2*i+1] = extreme_point_index(points, -axes[i]);
    }

    // Select the pair with largest distance to form first edge
    T max_dist2 = 0;
    int p0 = extrema[0], p1 = extrema[1];
    for (int i = 0; i < 6; ++i) {
        for (int j = i+1; j < 6; ++j) {
            T d2 = length_sq(points[extrema[i]] - points[extrema[j]]);
            if (d2 > max_dist2) {
                max_dist2 = d2;
                p0 = extrema[i];
                p1 = extrema[j];
            }
        }
    }

    // Find the farthest point from the line p0-p1 to form a triangle
    vector3<T> line_dir = points[p1] - points[p0];
    T line_len = length(line_dir);
    if (line_len < T(1e-12)) return result;
    line_dir = line_dir / line_len;
    T max_tri_dist = 0;
    int p2 = 0;
    for (int i = 0; i < 6; ++i) {
        if (i == p0 || i == p1) continue;
        int idx = extrema[i];
        vector3<T> v = points[idx] - points[p0];
        vector3<T> proj = line_dir * dot(v, line_dir);
        T dist = length(v - proj);
        if (dist > max_tri_dist) {
            max_tri_dist = dist;
            p2 = idx;
        }
    }

    // Find the farthest point from triangle plane to complete tetrahedron
    vector3<T> normal_012 = compute_face_normal(points[p0], points[p1], points[p2],
                                                  approximate_interior(points));
    T max_vol_dist = 0;
    int p3 = 0;
    for (int i = 0; i < 6; ++i) {
        if (i == p0 || i == p1 || i == p2) continue;
        int idx = extrema[i];
        T dist = std::abs(signed_face_distance(points[idx], normal_012, -dot(normal_012, points[p0])));
        if (dist > max_vol_dist) {
            max_vol_dist = dist;
            p3 = idx;
        }
    }

    // Check if tetrahedron is degenerate
    if (max_vol_dist < T(1e-12)) {
        // Search among all points for a non‑coplanar fourth point
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            if (i == p0 || i == p1 || i == p2) continue;
            T dist = std::abs(signed_face_distance(points[i], normal_012, -dot(normal_012, points[p0])));
            if (dist > T(1e-12)) {
                p3 = i;
                max_vol_dist = dist;
                break;
            }
        }
        if (max_vol_dist < T(1e-12)) return result; // all points coplanar
    }

    // Interior point for orientation
    vector3<T> interior = (points[p0] + points[p1] + points[p2] + points[p3]) / T(4);

    // Initial four faces (ensure outward orientation)
    auto make_face = [&](int v0, int v1, int v2) -> quickhull_face<T> {
        quickhull_face<T> face;
        face.vertices = {v0, v1, v2};
        face.normal = compute_face_normal(points[v0], points[v1], points[v2], interior);
        face.offset = -dot(face.normal, points[v0]);
        return face;
    };

    std::vector<quickhull_face<T>> hull = {
        make_face(p0, p1, p2),
        make_face(p0, p1, p3),
        make_face(p0, p2, p3),
        make_face(p1, p2, p3)
    };

    // Check orientation: ensure no other initial vertex is outside each face
    for (auto& face : hull) {
        // Find which vertex is not part of this face
        int inside_v = -1;
        for (int v : {p0, p1, p2, p3}) {
            if (v != face.vertices[0] && v != face.vertices[1] && v != face.vertices[2]) {
                inside_v = v;
                break;
            }
        }
        if (inside_v >= 0 && signed_face_distance(points[inside_v], face.normal, face.offset) > 0) {
            // Flip orientation
            face.normal = -face.normal;
            face.offset = -face.offset;
            std::swap(face.vertices[1], face.vertices[2]);
        }
    }

    // 2. Assign each point to the outside set of faces it lies outside of.
    //    Points inside all faces are discarded.
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        if (i == p0 || i == p1 || i == p2 || i == p3) continue;
        bool is_outside_any = false;
        for (auto& face : hull) {
            if (is_point_outside(points[i], face.normal, face.offset)) {
                face.outside_points.push_back(i);
                is_outside_any = true;
            }
        }
        if (!is_outside_any) {
            // point is inside hull, discard
        }
    }

    // 3. Iteratively expand hull
    while (true) {
        // Find face with the farthest outside point (largest positive distance)
        quickhull_face<T>* face_with_farthest = nullptr;
        int farthest_point_idx = -1;
        T max_dist = T(0);

        for (auto& face : hull) {
            if (face.deleted) continue;
            for (int pt_idx : face.outside_points) {
                T dist = signed_face_distance(points[pt_idx], face.normal, face.offset);
                if (dist > max_dist) {
                    max_dist = dist;
                    face_with_farthest = &face;
                    farthest_point_idx = pt_idx;
                }
            }
        }

        if (face_with_farthest == nullptr) break; // hull complete

        // 4. Find the horizon: set of edges that separate visible faces from invisible faces
        const vector3<T>& eye_point = points[farthest_point_idx];
        std::vector<int> visible_faces;
        std::unordered_set<quickhull_edge<T>, quickhull_edge_hash<T>> horizon_edges;
        std::stack<int> face_stack;

        // Mark the starting face as visible and push it
        face_with_farthest->deleted = true;
        visible_faces.push_back(static_cast<int>(&(*face_with_farthest) - hull.data()));
        // We'll need to map face pointer to index; use vector index
        // Better: store face index in hull array
        // Since we are using vector<quickhull_face>, face_with_farthest points to an element.
        // We'll use a loop to find the index.
        int start_face_idx = static_cast<int>(face_with_farthest - hull.data());

        std::vector<int> face_stack_indices;
        face_stack_indices.push_back(start_face_idx);
        std::vector<int> visited_visible;
        std::unordered_set<int> visible_set;
        visible_set.insert(start_face_idx);

        while (!face_stack_indices.empty()) {
            int fi = face_stack_indices.back();
            face_stack_indices.pop_back();
            visited_visible.push_back(fi);
            auto& face = hull[fi];
            // Iterate over edges
            for (int e = 0; e < 3; ++e) {
                int v0 = face.vertices[e];
                int v1 = face.vertices[(e+1)%3];
                quickhull_edge<T> edge{v0, v1};

                // Find adjacent face sharing this edge (not the same face)
                int adj_face_idx = -1;
                for (int j = 0; j < static_cast<int>(hull.size()); ++j) {
                    if (j == fi || hull[j].deleted) continue;
                    const auto& adj = hull[j];
                    bool shares_edge = false;
                    for (int k = 0; k < 3; ++k) {
                        if ((adj.vertices[k] == v0 && adj.vertices[(k+1)%3] == v1) ||
                            (adj.vertices[k] == v1 && adj.vertices[(k+1)%3] == v0)) {
                            shares_edge = true;
                            break;
                        }
                    }
                    if (shares_edge) {
                        adj_face_idx = j;
                        break;
                    }
                }

                if (adj_face_idx == -1) {
                    // No adjacent face, edge is on hull boundary? (should not happen for closed hull)
                    // Treat as horizon edge
                    horizon_edges.insert(edge);
                } else {
                    auto& adj_face = hull[adj_face_idx];
                    if (is_point_outside(eye_point, adj_face.normal, adj_face.offset)) {
                        // Adjacent face is also visible
                        if (visible_set.find(adj_face_idx) == visible_set.end()) {
                            visible_set.insert(adj_face_idx);
                            face_stack_indices.push_back(adj_face_idx);
                        }
                    } else {
                        // Adjacent face is invisible, this edge is part of horizon
                        horizon_edges.insert(edge);
                    }
                }
            }
        }

        // 5. Remove visible faces and assign their outside points to a temporary list
        std::vector<int> all_outside_points;
        for (int fi : visited_visible) {
            auto& face = hull[fi];
            all_outside_points.insert(all_outside_points.end(),
                                      face.outside_points.begin(),
                                      face.outside_points.end());
            face.outside_points.clear();
            face.deleted = true;
        }

        // 6. Create new faces from horizon edges to the eye point
        std::vector<int> new_face_indices;
        for (const auto& edge : horizon_edges) {
            quickhull_face<T> new_face;
            new_face.vertices = {edge.a, edge.b, farthest_point_idx};
            new_face.normal = compute_face_normal(points[edge.a], points[edge.b],
                                                  points[farthest_point_idx], interior);
            new_face.offset = -dot(new_face.normal, points[edge.a]);
            // Ensure the normal points outward (away from interior)
            if (dot(new_face.normal, points[edge.a] - interior) < T(0)) {
                new_face.normal = -new_face.normal;
                new_face.offset = -new_face.offset;
                std::swap(new_face.vertices[1], new_face.vertices[2]);
            }
            hull.push_back(new_face);
            new_face_indices.push_back(static_cast<int>(hull.size()) - 1);
        }

        // 7. Redistribute outside points to new faces
        for (int pt_idx : all_outside_points) {
            const auto& pt = points[pt_idx];
            bool assigned = false;
            for (int fi : new_face_indices) {
                auto& face = hull[fi];
                if (is_point_outside(pt, face.normal, face.offset)) {
                    face.outside_points.push_back(pt_idx);
                    assigned = true;
                }
            }
            if (!assigned) {
                // Point is now inside the hull, discard
            }
        }

        // Optional: merge coplanar faces (could be added for efficiency)
    }

    // 8. Collect non‑deleted faces
    for (auto& face : hull) {
        if (!face.deleted) {
            result.push_back(face);
        }
    }

    return result;
}

// ============================================================
// Convenience wrapper returning only vertex indices
// ============================================================

template<typename T>
std::vector<std::array<int,3>> quickhull3d_indices(const std::vector<vector3<T>>& points) {
    auto hull = quickhull3d(points);
    std::vector<std::array<int,3>> tris;
    for (const auto& f : hull) {
        tris.push_back(f.vertices);
    }
    return tris;
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_QUICKHULL3D_H