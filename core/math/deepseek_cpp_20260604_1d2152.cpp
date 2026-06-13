// system name : onetbb-warp
// File 0027 : core/math/mesh_operations.h
// Description : Mesh processing: normals, area, volume, simplification, subdivision, smoothing.

#ifndef __TBB_WARP_CORE_MATH_MESH_OPERATIONS_H
#define __TBB_WARP_CORE_MATH_MESH_OPERATIONS_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector3.h"
#include "core/math/matrix3.h"
#include "core/math/geometry.h"
#include "core/math/computational_geometry.h"
#include "core/math/catmull_clark.h"   // Full Catmull‑Clark subdivision
#include <cmath>
#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <queue>
#include <utility>
#include <limits>
#include <cstdint>
#include <functional>
#include <tuple>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Face normal (assuming CCW winding)
// ============================================================

template<typename T>
vector3<T> face_normal(const vector3<T>& v0, const vector3<T>& v1, const vector3<T>& v2) noexcept {
    vector3<T> edge1 = v1 - v0;
    vector3<T> edge2 = v2 - v0;
    return normalize(cross(edge1, edge2));
}

// ============================================================
// Compute normals for a mesh (indexed triangles)
// ============================================================

template<typename T>
void compute_vertex_normals(const std::vector<vector3<T>>& vertices,
                            const std::vector<std::array<int,3>>& faces,
                            std::vector<vector3<T>>& out_normals) {
    out_normals.assign(vertices.size(), vector3<T>(T(0)));
    for (const auto& f : faces) {
        vector3<T> fn = face_normal(vertices[f[0]], vertices[f[1]], vertices[f[2]]);
        T area = T(0.5) * length(cross(vertices[f[1]]-vertices[f[0]], vertices[f[2]]-vertices[f[0]]));
        out_normals[f[0]] = out_normals[f[0]] + fn * area;
        out_normals[f[1]] = out_normals[f[1]] + fn * area;
        out_normals[f[2]] = out_normals[f[2]] + fn * area;
    }
    for (auto& n : out_normals) {
        T len = length(n);
        if (len > T(1e-12)) n = n / len;
        else n = vector3<T>(T(0), T(1), T(0));
    }
}

// ============================================================
// Mesh surface area (sum of triangle areas)
// ============================================================

template<typename T>
T mesh_surface_area(const std::vector<vector3<T>>& vertices,
                    const std::vector<std::array<int,3>>& faces) {
    T total = T(0);
    for (const auto& f : faces) {
        total += triangle_area(vertices[f[0]], vertices[f[1]], vertices[f[2]]);
    }
    return total;
}

// ============================================================
// Mesh volume (signed, for closed manifold)
// ============================================================

template<typename T>
T mesh_signed_volume(const std::vector<vector3<T>>& vertices,
                     const std::vector<std::array<int,3>>& faces) {
    T vol = T(0);
    for (const auto& f : faces) {
        const auto& a = vertices[f[0]];
        const auto& b = vertices[f[1]];
        const auto& c = vertices[f[2]];
        vol += dot(cross(a, b), c);
    }
    return vol / T(6);
}

// ============================================================
// Quadric error metric for edge collapse (Garland & Heckbert)
// ============================================================

template<typename T>
struct quadric_matrix {
    T a00, a01, a02, a03;
    T a11, a12, a13;
    T a22, a23;
    T a33;

    quadric_matrix() : a00(0), a01(0), a02(0), a03(0),
                       a11(0), a12(0), a13(0),
                       a22(0), a23(0), a33(0) {}

    quadric_matrix(T v00,T v01,T v02,T v03,
                   T v11,T v12,T v13,
                   T v22,T v23,T v33)
        : a00(v00),a01(v01),a02(v02),a03(v03),
          a11(v11),a12(v12),a13(v13),
          a22(v22),a23(v23),a33(v33) {}

    quadric_matrix operator+(const quadric_matrix& o) const {
        return quadric_matrix(a00+o.a00,a01+o.a01,a02+o.a02,a03+o.a03,
                               a11+o.a11,a12+o.a12,a13+o.a13,
                               a22+o.a22,a23+o.a23,a33+o.a33);
    }

    quadric_matrix& operator+=(const quadric_matrix& o) {
        a00+=o.a00;a01+=o.a01;a02+=o.a02;a03+=o.a03;
        a11+=o.a11;a12+=o.a12;a13+=o.a13;
        a22+=o.a22;a23+=o.a23;a33+=o.a33;
        return *this;
    }

    T evaluate(const vector3<T>& v) const {
        return a00*v.x*v.x + T(2)*a01*v.x*v.y + T(2)*a02*v.x*v.z + T(2)*a03*v.x
               + a11*v.y*v.y + T(2)*a12*v.y*v.z + T(2)*a13*v.y
               + a22*v.z*v.z + T(2)*a23*v.z + a33;
    }

    bool solve(vector3<T>& optimal) const {
        T det = a00*(a11*a22 - a12*a12) - a01*(a01*a22 - a12*a02) + a02*(a01*a12 - a11*a02);
        if (std::abs(det) < T(1e-12)) return false;
        T inv = T(1) / det;
        optimal.x = -(a03*(a11*a22 - a12*a12) - a01*(a13*a22 - a23*a12) + a02*(a13*a12 - a11*a23)) * inv;
        optimal.y = -(a00*(a13*a22 - a23*a12) - a03*(a01*a22 - a12*a02) + a02*(a01*a23 - a13*a02)) * inv;
        optimal.z = -(a00*(a11*a23 - a13*a12) - a01*(a01*a23 - a13*a02) + a03*(a01*a12 - a11*a02)) * inv;
        return true;
    }
};

template<typename T>
quadric_matrix<T> compute_face_quadric(const vector3<T>& normal, T offset) {
    T a = normal.x, b = normal.y, c = normal.z, d = offset;
    return quadric_matrix<T>(a*a, a*b, a*c, a*d,
                               b*b, b*c, b*d,
                               c*c, c*d,
                               d*d);
}

// ============================================================
// Edge collapse data
// ============================================================

template<typename T>
struct edge_collapse {
    int v0, v1;
    vector3<T> optimal_position;
    T cost;
    bool operator>(const edge_collapse& o) const { return cost > o.cost; }
};

template<typename T>
std::vector<vector3<T>> simplify_mesh_quadric(
    std::vector<vector3<T>> vertices,
    std::vector<std::array<int,3>>& faces,
    int target_face_count)
{
    int n = static_cast<int>(vertices.size());
    int f = static_cast<int>(faces.size());
    if (f <= target_face_count || n < 3) return vertices;

    std::vector<quadric_matrix<T>> quadrics(n);
    for (const auto& face : faces) {
        vector3<T> normal = face_normal(vertices[face[0]], vertices[face[1]], vertices[face[2]]);
        T offset = -dot(normal, vertices[face[0]]);
        quadric_matrix<T> Q = compute_face_quadric(normal, offset);
        quadrics[face[0]] += Q;
        quadrics[face[1]] += Q;
        quadrics[face[2]] += Q;
    }

    std::unordered_map<std::uint64_t, edge_collapse<T>> edge_map;
    auto edge_key = [](int a, int b) -> std::uint64_t {
        if (a > b) std::swap(a, b);
        return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
    };

    for (const auto& face : faces) {
        for (int e = 0; e < 3; ++e) {
            int v0 = face[e], v1 = face[(e+1)%3];
            std::uint64_t key = edge_key(v0, v1);
            if (edge_map.find(key) != edge_map.end()) continue;
            quadric_matrix<T> Q = quadrics[v0] + quadrics[v1];
            vector3<T> optimal = (vertices[v0] + vertices[v1]) * T(0.5);
            Q.solve(optimal);
            T cost = Q.evaluate(optimal);
            edge_map[key] = {v0, v1, optimal, cost};
        }
    }

    std::priority_queue<edge_collapse<T>, std::vector<edge_collapse<T>>, std::greater<edge_collapse<T>>> pq;
    for (const auto& p : edge_map) pq.push(p.second);

    std::vector<bool> vertex_deleted(n, false);
    std::vector<int> vertex_remap(n);
    for (int i = 0; i < n; ++i) vertex_remap[i] = i;

    int current_faces = f;
    while (current_faces > target_face_count && !pq.empty()) {
        edge_collapse<T> ec = pq.top();
        pq.pop();
        if (vertex_deleted[ec.v0] || vertex_deleted[ec.v1]) continue;

        vertex_deleted[ec.v1] = true;
        vertices[ec.v0] = ec.optimal_position;
        vertex_remap[ec.v1] = ec.v0;

        quadrics[ec.v0] = quadrics[ec.v0] + quadrics[ec.v1];

        std::vector<std::array<int,3>> new_faces;
        for (const auto& face : faces) {
            int vv0 = vertex_remap[face[0]];
            int vv1 = vertex_remap[face[1]];
            int vv2 = vertex_remap[face[2]];
            if (vv0 == vv1 || vv1 == vv2 || vv0 == vv2) continue;
            bool duplicate = false;
            for (const auto& nf : new_faces) {
                int n0 = vertex_remap[nf[0]], n1 = vertex_remap[nf[1]], n2 = vertex_remap[nf[2]];
                if ((vv0==n0 && vv1==n1 && vv2==n2) ||
                    (vv0==n1 && vv1==n2 && vv2==n0) ||
                    (vv0==n2 && vv1==n0 && vv2==n1)) {
                    duplicate = true; break;
                }
            }
            if (!duplicate) new_faces.push_back({{vv0, vv1, vv2}});
        }
        faces.swap(new_faces);
        current_faces = static_cast<int>(faces.size());

        for (const auto& face : faces) {
            for (int e = 0; e < 3; ++e) {
                int v0 = face[e], v1 = face[(e+1)%3];
                std::uint64_t key = edge_key(v0, v1);
                if (edge_map.find(key) != edge_map.end()) continue;
                quadric_matrix<T> Q = quadrics[v0] + quadrics[v1];
                vector3<T> optimal = (vertices[v0] + vertices[v1]) * T(0.5);
                Q.solve(optimal);
                T cost = Q.evaluate(optimal);
                edge_map[key] = {v0, v1, optimal, cost};
                pq.push({v0, v1, optimal, cost});
            }
        }
    }

    std::vector<vector3<T>> new_vertices;
    std::vector<int> compact_map(n, -1);
    for (int i = 0; i < n; ++i) {
        if (!vertex_deleted[i]) {
            compact_map[i] = static_cast<int>(new_vertices.size());
            new_vertices.push_back(vertices[i]);
        }
    }
    for (auto& face : faces) {
        for (int i = 0; i < 3; ++i) face[i] = compact_map[vertex_remap[face[i]]];
    }
    return new_vertices;
}

// ============================================================
// Catmull‑Clark subdivision (via catmull_clark.h)
// ============================================================

template<typename T>
void catmull_clark_subdivide(std::vector<vector3<T>>& vertices,
                              std::vector<std::array<int,3>>& faces) {
    // Convert triangle faces to polygon representation
    std::vector<std::vector<int>> poly_faces;
    poly_faces.reserve(faces.size());
    for (const auto& f : faces) {
        poly_faces.push_back({f[0], f[1], f[2]});
    }

    // Call full Catmull‑Clark implementation
    auto result = catmull_clark_subdivide(vertices, poly_faces);

    // Convert result back to triangle faces (triangulate quads)
    vertices = std::move(result.vertices);
    faces.clear();
    for (const auto& f : result.faces) {
        int k = static_cast<int>(f.size());
        if (k == 3) {
            faces.push_back({{f[0], f[1], f[2]}});
        } else if (k == 4) {
            // Split quad into two triangles using the shortest diagonal
            T d02 = length_sq(vertices[f[0]] - vertices[f[2]]);
            T d13 = length_sq(vertices[f[1]] - vertices[f[3]]);
            if (d02 <= d13) {
                faces.push_back({{f[0], f[1], f[2]}});
                faces.push_back({{f[0], f[2], f[3]}});
            } else {
                faces.push_back({{f[0], f[1], f[3]}});
                faces.push_back({{f[1], f[2], f[3]}});
            }
        } else {
            // Fan triangulation for n‑gons (n > 4)
            for (int i = 1; i < k - 1; ++i) {
                faces.push_back({{f[0], f[i], f[i+1]}});
            }
        }
    }
}

// ============================================================
// Laplacian smoothing
// ============================================================

template<typename T>
void laplacian_smooth(std::vector<vector3<T>>& vertices,
                      const std::vector<std::array<int,3>>& faces,
                      int iterations = 5, T lambda = T(0.5)) {
    int n = static_cast<int>(vertices.size());
    std::vector<std::vector<int>> neighbors(n);
    for (const auto& f : faces) {
        for (int i = 0; i < 3; ++i) {
            int v = f[i], v1 = f[(i+1)%3], v2 = f[(i+2)%3];
            if (std::find(neighbors[v].begin(), neighbors[v].end(), v1) == neighbors[v].end())
                neighbors[v].push_back(v1);
            if (std::find(neighbors[v].begin(), neighbors[v].end(), v2) == neighbors[v].end())
                neighbors[v].push_back(v2);
        }
    }

    std::vector<vector3<T>> temp(n);
    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = 0; i < n; ++i) {
            if (neighbors[i].empty()) { temp[i] = vertices[i]; continue; }
            vector3<T> centroid(T(0));
            for (int nb : neighbors[i]) centroid = centroid + vertices[nb];
            centroid = centroid / T(neighbors[i].size());
            temp[i] = vertices[i] + (centroid - vertices[i]) * lambda;
        }
        vertices.swap(temp);
    }
}

// ============================================================
// Bounding box of mesh
// ============================================================

template<typename T>
aabb<T> mesh_bounding_box(const std::vector<vector3<T>>& vertices) {
    if (vertices.empty()) return aabb<T>();
    aabb<T> box(vertices[0], vertices[0]);
    for (const auto& v : vertices) box.expand(v);
    return box;
}

// ============================================================
// Mesh centroid
// ============================================================

template<typename T>
vector3<T> mesh_centroid(const std::vector<vector3<T>>& vertices) {
    if (vertices.empty()) return vector3<T>(T(0));
    vector3<T> sum(T(0));
    for (const auto& v : vertices) sum = sum + v;
    return sum / T(vertices.size());
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_MESH_OPERATIONS_H