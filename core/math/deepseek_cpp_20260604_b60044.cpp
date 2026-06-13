// system name : onetbb-warp
// File 0028 : core/math/catmull_clark.h
// Description : Full Catmull‑Clark subdivision for arbitrary polygon meshes.

#ifndef __TBB_WARP_CORE_MATH_CATMULL_CLARK_H
#define __TBB_WARP_CORE_MATH_CATMULL_CLARK_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector3.h"
#include <vector>
#include <array>
#include <unordered_map>
#include <algorithm>
#include <cstdint>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Catmull‑Clark subdivision for polygonal meshes
// ============================================================

template<typename T>
struct catmull_clark_result {
    std::vector<vector3<T>>                vertices;
    std::vector<std::vector<int>>          faces;         // each face is a list of vertex indices (quad or general)
};

template<typename T>
catmull_clark_result<T> catmull_clark_subdivide(
    const std::vector<vector3<T>>& vertices,
    const std::vector<std::vector<int>>& faces)   // each face: list of vertex indices (any degree)
{
    using index_t = int;
    const index_t nV = static_cast<index_t>(vertices.size());
    const index_t nF = static_cast<index_t>(faces.size());

    // 1. Face points: centroid of each face
    std::vector<vector3<T>> face_points(nF);
    for (index_t fi = 0; fi < nF; ++fi) {
        vector3<T> sum(0,0,0);
        for (index_t vi : faces[fi]) sum = sum + vertices[vi];
        face_points[fi] = sum / T(faces[fi].size());
    }

    // 2. Edge points
    // Build edge map: for each edge (unordered pair), store its adjacent face indices and its midpoint.
    struct EdgeInfo {
        index_t face_a = -1, face_b = -1;   // up to 2 faces
        vector3<T> midpoint;
        index_t new_vertex_index = -1;
    };
    std::unordered_map<std::uint64_t, EdgeInfo> edge_map;
    auto edge_hash = [](index_t a, index_t b) -> std::uint64_t {
        if (a > b) std::swap(a, b);
        return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
    };

    for (index_t fi = 0; fi < nF; ++fi) {
        const auto& f = faces[fi];
        index_t k = static_cast<index_t>(f.size());
        for (index_t i = 0; i < k; ++i) {
            index_t v0 = f[i];
            index_t v1 = f[(i+1)%k];
            std::uint64_t key = edge_hash(v0, v1);
            auto& info = edge_map[key];
            if (info.face_a == -1) info.face_a = fi;
            else if (info.face_b == -1) info.face_b = fi;
            // Midpoint is average of endpoints
            info.midpoint = (vertices[v0] + vertices[v1]) * T(0.5);
        }
    }

    // Compute edge point: average of the two endpoints and the two adjacent face points.
    for (auto& kv : edge_map) {
        auto& info = kv.second;
        if (info.face_a != -1 && info.face_b != -1) {
            info.midpoint = (info.midpoint * T(2) + face_points[info.face_a] + face_points[info.face_b]) * T(0.25);
        } else {
            // Boundary edge: just use midpoint of endpoints
            // info.midpoint remains the midpoint of endpoints.
        }
    }

    // 3. New vertex positions
    // For each original vertex, new position = (F + 2R + (n-3)*P) / n
    // where n = valence, F = average of adjacent face points, R = average of midpoints of incident edges,
    // P = original vertex position.
    std::vector<vector3<T>> new_vertices(nV);
    // Gather face points and edge midpoints incident to each vertex.
    std::vector<std::vector<index_t>> vertex_face_points(nV);
    std::vector<std::vector<vector3<T>>> vertex_edge_midpoints(nV);
    for (index_t fi = 0; fi < nF; ++fi) {
        const auto& f = faces[fi];
        index_t k = static_cast<index_t>(f.size());
        for (index_t i = 0; i < k; ++i) {
            index_t v = f[i];
            vertex_face_points[v].push_back(fi);
            index_t v0 = f[i], v1 = f[(i+1)%k];
            vector3<T> mid = (vertices[v0] + vertices[v1]) * T(0.5);
            vertex_edge_midpoints[v].push_back(mid);
        }
    }

    for (index_t vi = 0; vi < nV; ++vi) {
        index_t n = static_cast<index_t>(vertex_face_points[vi].size());
        if (n == 0) { new_vertices[vi] = vertices[vi]; continue; }
        vector3<T> F_avg(0,0,0);
        for (index_t fi : vertex_face_points[vi]) F_avg = F_avg + face_points[fi];
        F_avg = F_avg / T(n);
        vector3<T> R_avg(0,0,0);
        for (const auto& mid : vertex_edge_midpoints[vi]) R_avg = R_avg + mid;
        R_avg = R_avg / T(n);
        vector3<T> P = vertices[vi];
        new_vertices[vi] = (F_avg + R_avg * T(2) + P * T(n - 3)) / T(n);
    }

    // 4. Assemble new mesh
    // Assign indices to face points and edge points as new vertices.
    // New vertex list starts with the updated original vertices, then face points, then edge points.
    std::vector<vector3<T>> out_vertices = new_vertices;
    std::vector<std::vector<int>> out_faces;

    // Face point indices
    std::vector<index_t> fp_idx(nF);
    for (index_t fi = 0; fi < nF; ++fi) {
        fp_idx[fi] = static_cast<index_t>(out_vertices.size());
        out_vertices.push_back(face_points[fi]);
    }

    // Edge point indices
    std::unordered_map<std::uint64_t, index_t> edge_idx;
    for (auto& kv : edge_map) {
        edge_idx[kv.first] = static_cast<index_t>(out_vertices.size());
        out_vertices.push_back(kv.second.midpoint);
    }

    // For each original face, create k quadrilaterals.
    for (index_t fi = 0; fi < nF; ++fi) {
        const auto& f = faces[fi];
        index_t k = static_cast<index_t>(f.size());
        index_t face_center = fp_idx[fi];
        for (index_t i = 0; i < k; ++i) {
            index_t v0 = f[i];
            index_t v1 = f[(i+1)%k];
            std::uint64_t e0_key = edge_hash(v0, v1);
            std::uint64_t e1_key = edge_hash(v1, f[(i+2)%k]);  // edge (v1, next vertex)
            // Actually we need the edge point of the edge v_i - v_{i+1}
            // The quad is: new vertex of v1, edge point (v1, v_{i+1}), face point, edge point (v_{i-1}, v1)
            // More precisely: for edge i (v_i, v_{i+1}), quad formed by:
            //   new vertex of v_i
            //   edge point e_i
            //   face point
            //   edge point e_{i-1}
            index_t ev_i   = v0;
            index_t ev_ip1 = v1;
            std::uint64_t key_i   = edge_hash(ev_i, ev_ip1);
            std::uint64_t key_im1 = edge_hash(ev_i, f[(i-1+k)%k]);
            std::vector<int> quad = {
                ev_i,                           // new vertex position of v0
                edge_idx[key_i],                // edge point of e_i
                face_center,                    // face point
                edge_idx[key_im1]               // edge point of e_{i-1}
            };
            out_faces.push_back(std::move(quad));
        }
    }

    return { std::move(out_vertices), std::move(out_faces) };
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_CATMULL_CLARK_H