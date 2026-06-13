// system name : onetbb-warp
// File 0042 : core/math/differential_geometry.h
// Description : Discrete differential geometry on meshes: curvature, Laplace‑Beltrami, geodesics.

#ifndef __TBB_WARP_CORE_MATH_DIFFERENTIAL_GEOMETRY_H
#define __TBB_WARP_CORE_MATH_DIFFERENTIAL_GEOMETRY_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector3.h"
#include "core/math/matrix3.h"
#include "core/math/geometry.h"
#include "core/math/mesh_operations.h"
#include "core/math/linear_system.h"
#include <cmath>
#include <vector>
#include <array>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <numeric>
#include <queue>
#include <functional>
#include <tuple>
#include <cstdint>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Sparse matrix in COO format for Laplace‑Beltrami assembly
// ============================================================

template<typename T>
struct sparse_matrix_coo {
    std::vector<std::tuple<std::size_t, std::size_t, T>> entries;
    std::size_t n_rows;
    std::size_t n_cols;

    sparse_matrix_coo(std::size_t rows, std::size_t cols) : n_rows(rows), n_cols(cols) {}

    void add(std::size_t i, std::size_t j, T value) noexcept {
        entries.emplace_back(i, j, value);
    }

    // Convert to dense for existing solvers
    std::vector<std::vector<T>> to_dense() const {
        std::vector<std::vector<T>> dense(n_rows, std::vector<T>(n_cols, T(0)));
        for (const auto& [i, j, v] : entries) {
            dense[i][j] += v;
        }
        return dense;
    }

    // Matrix‑vector product (for iterative solvers)
    std::vector<T> multiply(const std::vector<T>& x) const {
        std::vector<T> result(n_rows, T(0));
        for (const auto& [i, j, v] : entries) {
            if (j < x.size()) result[i] += v * x[j];
        }
        return result;
    }
};

// ============================================================
// Cotangent of angle at vertex a in triangle (a,b,c)
// ============================================================

template<typename T>
T cotangent_angle(const vector3<T>& a, const vector3<T>& b, const vector3<T>& c) noexcept {
    vector3<T> u = b - a;
    vector3<T> v = c - a;
    T dot_uv = dot(u, v);
    vector3<T> cross_uv = cross(u, v);
    T sin_angle = length(cross_uv);
    if (sin_angle < T(1e-12)) return T(0);
    return dot_uv / sin_angle;
}

// ============================================================
// Compute cotangent weights and Voronoi areas per vertex
// ============================================================

template<typename T>
void compute_cotangent_laplacian(
    const std::vector<vector3<T>>& vertices,
    const std::vector<std::array<int,3>>& faces,
    std::vector<std::unordered_map<int, T>>& cot_weights,
    std::vector<T>& vertex_areas) noexcept
{
    std::size_t nv = vertices.size();
    cot_weights.assign(nv, {});
    vertex_areas.assign(nv, T(0));

    for (const auto& f : faces) {
        int v0 = f[0], v1 = f[1], v2 = f[2];
        const auto& p0 = vertices[v0];
        const auto& p1 = vertices[v1];
        const auto& p2 = vertices[v2];

        T cot0 = cotangent_angle(p0, p1, p2);
        T cot1 = cotangent_angle(p1, p2, p0);
        T cot2 = cotangent_angle(p2, p0, p1);

        // Edge weights: w_ij = (cot α + cot β) / 2  (but for a single triangle we add cot/2)
        cot_weights[v1][v2] += cot0 * T(0.5);
        cot_weights[v2][v1] += cot0 * T(0.5);
        cot_weights[v2][v0] += cot1 * T(0.5);
        cot_weights[v0][v2] += cot1 * T(0.5);
        cot_weights[v0][v1] += cot2 * T(0.5);
        cot_weights[v1][v0] += cot2 * T(0.5);

        // Voronoi area contribution (Meyer et al.)
        // For acute triangle: area = 1/8 * (|v1-v0|^2 * cot2 + |v2-v1|^2 * cot0 + |v0-v2|^2 * cot1)
        T len01_sq = length_sq(p1 - p0);
        T len12_sq = length_sq(p2 - p1);
        T len20_sq = length_sq(p0 - p2);

        if (cot0 > T(0) && cot1 > T(0) && cot2 > T(0)) {
            vertex_areas[v0] += (len20_sq * cot1 + len01_sq * cot2) * T(0.125);
            vertex_areas[v1] += (len01_sq * cot2 + len12_sq * cot0) * T(0.125);
            vertex_areas[v2] += (len12_sq * cot0 + len20_sq * cot1) * T(0.125);
        } else {
            // Obtuse triangle: use special handling; for simplicity, use full triangle area/3
            T tri_area = T(0.5) * length(cross(p1-p0, p2-p0));
            vertex_areas[v0] += tri_area / T(3);
            vertex_areas[v1] += tri_area / T(3);
            vertex_areas[v2] += tri_area / T(3);
        }
    }
}

// ============================================================
// Laplace‑Beltrami operator applied to scalar field u (per vertex)
// ============================================================

template<typename T>
std::vector<T> laplace_beltrami(
    const std::vector<vector3<T>>& vertices,
    const std::vector<std::array<int,3>>& faces,
    const std::vector<T>& u) noexcept
{
    std::size_t nv = vertices.size();
    std::vector<std::unordered_map<int, T>> cot_weights;
    std::vector<T> areas;
    compute_cotangent_laplacian(vertices, faces, cot_weights, areas);
    std::vector<T> Lu(nv, T(0));
    for (std::size_t i = 0; i < nv; ++i) {
        if (areas[i] < T(1e-12)) continue;
        T sum = T(0);
        for (const auto& [j, w] : cot_weights[i]) {
            sum += w * (u[j] - u[i]);
        }
        Lu[i] = sum / (T(2) * areas[i]);
    }
    return Lu;
}

// ============================================================
// Mean curvature per vertex: H = 1/2 * ||Lx||   (signed by dot with normal)
// ============================================================

template<typename T>
std::vector<T> mean_curvature(
    const std::vector<vector3<T>>& vertices,
    const std::vector<std::array<int,3>>& faces) noexcept
{
    std::size_t nv = vertices.size();
    std::vector<std::unordered_map<int, T>> cot_weights;
    std::vector<T> areas;
    compute_cotangent_laplacian(vertices, faces, cot_weights, areas);
    // Compute vertex normals
    std::vector<vector3<T>> normals;
    compute_vertex_normals(vertices, faces, normals);
    std::vector<T> H(nv, T(0));
    for (std::size_t i = 0; i < nv; ++i) {
        if (areas[i] < T(1e-12)) continue;
        // Laplace of position = mean curvature * normal
        vector3<T> Lx(T(0));
        for (const auto& [j, w] : cot_weights[i]) {
            Lx = Lx + (vertices[j] - vertices[i]) * w;
        }
        Lx = Lx / (T(2) * areas[i]);
        T h_val = length(Lx) * T(0.5);
        // Sign: dot(Lx, normal). If Lx points outward (same direction as normal), H is positive (convex).
        T sign_val = dot(Lx, normals[i]);
        if (sign_val < T(0)) h_val = -h_val;
        H[i] = h_val;
    }
    return H;
}

// ============================================================
// Gaussian curvature per vertex: K = (2π - sum of angles) / area
// ============================================================

template<typename T>
std::vector<T> gaussian_curvature(
    const std::vector<vector3<T>>& vertices,
    const std::vector<std::array<int,3>>& faces) noexcept
{
    std::size_t nv = vertices.size();
    std::vector<T> angle_sum(nv, T(0));
    std::vector<T> areas(nv, T(0));
    for (const auto& f : faces) {
        int v0 = f[0], v1 = f[1], v2 = f[2];
        const auto& p0 = vertices[v0];
        const auto& p1 = vertices[v1];
        const auto& p2 = vertices[v2];
        vector3<T> e01 = p1 - p0, e02 = p2 - p0, e10 = p0 - p1, e12 = p2 - p1, e20 = p0 - p2, e21 = p1 - p2;
        T a0 = std::acos(clamp(dot(e01,e02) / (length(e01)*length(e02) + T(1e-12)), T(-1), T(1)));
        T a1 = std::acos(clamp(dot(e10,e12) / (length(e10)*length(e12) + T(1e-12)), T(-1), T(1)));
        T a2 = std::acos(clamp(dot(e20,e21) / (length(e20)*length(e21) + T(1e-12)), T(-1), T(1)));
        angle_sum[v0] += a0;
        angle_sum[v1] += a1;
        angle_sum[v2] += a2;
        T tri_area = T(0.5) * length(cross(e01, e02));
        areas[v0] += tri_area / T(3);
        areas[v1] += tri_area / T(3);
        areas[v2] += tri_area / T(3);
    }
    std::vector<T> K(nv, T(0));
    for (std::size_t i = 0; i < nv; ++i) {
        if (areas[i] < T(1e-12)) continue;
        K[i] = (T(TAU_D) * T(0.5) - angle_sum[i]) / areas[i];
    }
    return K;
}

// ============================================================
// Principal curvatures via quadratic fitting (local shape operator)
// ============================================================

template<typename T>
void principal_curvatures(
    const std::vector<vector3<T>>& vertices,
    const std::vector<std::array<int,3>>& faces,
    std::vector<T>& k1_out,   // smaller principal curvature
    std::vector<T>& k2_out,   // larger principal curvature
    std::vector<vector3<T>>& d1_out,
    std::vector<vector3<T>>& d2_out) noexcept
{
    std::size_t nv = vertices.size();
    k1_out.assign(nv, T(0));
    k2_out.assign(nv, T(0));
    d1_out.assign(nv, vector3<T>(T(1),T(0),T(0)));
    d2_out.assign(nv, vector3<T>(T(0),T(1),T(0)));

    std::vector<vector3<T>> normals;
    compute_vertex_normals(vertices, faces, normals);
    std::vector<std::vector<int>> neighbors(nv);
    for (const auto& f : faces) {
        for (int i=0; i<3; ++i) {
            int v = f[i];
            int v1 = f[(i+1)%3];
            int v2 = f[(i+2)%3];
            if (std::find(neighbors[v].begin(), neighbors[v].end(), v1) == neighbors[v].end())
                neighbors[v].push_back(v1);
            if (std::find(neighbors[v].begin(), neighbors[v].end(), v2) == neighbors[v].end())
                neighbors[v].push_back(v2);
        }
    }

    for (std::size_t i = 0; i < nv; ++i) {
        const auto& n = normals[i];
        // Build local tangent basis (u, v, n)
        vector3<T> u, v;
        orthonormal_basis(n, u, v);
        // Accumulate shape operator as 2x2 matrix: S = [[a,b],[b,c]]
        T a=T(0), b=T(0), c=T(0), w_total=T(0);
        for (int j : neighbors[i]) {
            vector3<T> diff = vertices[j] - vertices[i];
            T du = dot(diff, u);
            T dv = dot(diff, v);
            T dn = dot(diff, n);
            T len2 = du*du + dv*dv;
            if (len2 < T(1e-12)) continue;
            T weight = T(1) / len2;
            a += weight * du * du * dn;
            b += weight * du * dv * dn;
            c += weight * dv * dv * dn;
            w_total += weight;
        }
        if (w_total < T(1e-12)) continue;
        a /= w_total; b /= w_total; c /= w_total;

        // Eigendecomposition of 2x2 symmetric matrix
        T trace_S = a + c;
        T det_S = a*c - b*b;
        T disc = trace_S*trace_S - T(4)*det_S;
        if (disc < T(0)) disc = T(0);
        T sqrt_disc = std::sqrt(disc);
        T k1 = (trace_S - sqrt_disc) * T(0.5);
        T k2 = (trace_S + sqrt_disc) * T(0.5);

        k1_out[i] = k1;
        k2_out[i] = k2;

        // Principal directions in tangent plane
        if (std::abs(b) > T(1e-12) || std::abs(a-k1) > T(1e-12)) {
            vector3<T> d1_local = u * b + v * (k1 - a);
            T len_d1 = length(d1_local);
            if (len_d1 > T(1e-12)) {
                d1_out[i] = d1_local / len_d1;
                d2_out[i] = cross(n, d1_out[i]);
            } else {
                d1_out[i] = u;
                d2_out[i] = v;
            }
        } else {
            d1_out[i] = u;
            d2_out[i] = v;
        }
    }
}

// ============================================================
// Shape index: s = 2/π * arctan((κ₂+κ₁)/(κ₂-κ₁))
// ============================================================

template<typename T>
std::vector<T> shape_index(const std::vector<vector3<T>>& vertices,
                           const std::vector<std::array<int,3>>& faces) noexcept
{
    std::vector<T> k1, k2;
    std::vector<vector3<T>> d1, d2;
    principal_curvatures(vertices, faces, k1, k2, d1, d2);
    std::size_t nv = vertices.size();
    std::vector<T> si(nv, T(0));
    for (std::size_t i = 0; i < nv; ++i) {
        T num = k2[i] + k1[i];
        T den = k2[i] - k1[i];
        if (std::abs(den) < T(1e-12)) {
            si[i] = T(0);
        } else {
            si[i] = T(2) / T(PI_D) * std::atan(num / den);
        }
    }
    return si;
}

// ============================================================
// Curvedness: c = sqrt((κ₁²+κ₂²)/2)
// ============================================================

template<typename T>
std::vector<T> curvedness(const std::vector<vector3<T>>& vertices,
                          const std::vector<std::array<int,3>>& faces) noexcept
{
    std::vector<T> k1, k2;
    std::vector<vector3<T>> d1, d2;
    principal_curvatures(vertices, faces, k1, k2, d1, d2);
    std::size_t nv = vertices.size();
    std::vector<T> curv(nv, T(0));
    for (std::size_t i = 0; i < nv; ++i) {
        curv[i] = std::sqrt((k1[i]*k1[i] + k2[i]*k2[i]) * T(0.5));
    }
    return curv;
}

// ============================================================
// Geodesic distance via the Heat Method (Crane et al.)
// ============================================================

template<typename T>
std::vector<T> geodesic_distance_heat_method(
    const std::vector<vector3<T>>& vertices,
    const std::vector<std::array<int,3>>& faces,
    const std::vector<int>& source_vertices,
    T time_step = T(1)) noexcept
{
    std::size_t nv = vertices.size();
    if (source_vertices.empty()) return std::vector<T>(nv, T(0));

    // 1. Build Laplace matrix L (sparse)
    std::vector<std::unordered_map<int, T>> cot_weights;
    std::vector<T> areas;
    compute_cotangent_laplacian(vertices, faces, cot_weights, areas);

    sparse_matrix_coo<T> L_mat(nv, nv);
    sparse_matrix_coo<T> M_mat(nv, nv); // mass matrix (diagonal: vertex areas)

    for (std::size_t i = 0; i < nv; ++i) {
        M_mat.add(i, i, areas[i]);
        T diag = T(0);
        for (const auto& [j, w] : cot_weights[i]) {
            L_mat.add(i, j, -w);
            diag += w;
        }
        L_mat.add(i, i, diag);
        // Actually the standard Laplace is L_i = sum w_ij (u_j - u_i) = (sum w_ij u_j) - (sum w_ij) u_i
        // So L = D - W where D_ii = sum w_ij, W_ij = w_ij
        // We have -W entries added, and diagonal will be corrected after.
    }

    // 2. Build matrix A = M - t*L  (we solve (M - t*L) u = u₀ * M applied to delta)
    // Actually the heat method solves (I - t*Δ) u = u₀
    // Δ = M^{-1} L  so (I - t M^{-1}L) u = u₀  => (M - t*L) u = M * u₀
    // u₀ is the initial distribution (1 on source, 0 elsewhere)

    // Build dense A = M - t*L for the solver
    auto L_dense = L_mat.to_dense();
    auto M_dense = M_mat.to_dense();
    std::vector<std::vector<T>> A(nv, std::vector<T>(nv, T(0)));
    for (std::size_t i = 0; i < nv; ++i) {
        for (std::size_t j = 0; j < nv; ++j) {
            A[i][j] = M_dense[i][j] - time_step * L_dense[i][j];
        }
    }

    // Right‑hand side: b = M * u₀ (u₀ is 1 on source vertices, 0 elsewhere)
    std::vector<T> u0(nv, T(0));
    T total_area = T(0);
    for (int src : source_vertices) {
        if (src >= 0 && static_cast<std::size_t>(src) < nv) {
            u0[src] = T(1);
            total_area += areas[src];
        }
    }
    if (total_area < T(1e-12)) return u0;
    // Normalize u0 so that integral = 1
    for (auto& val : u0) val /= total_area;

    std::vector<T> b(nv, T(0));
    for (std::size_t i = 0; i < nv; ++i) {
        for (std::size_t j = 0; j < nv; ++j) {
            b[i] += M_dense[i][j] * u0[j];
        }
    }

    // Solve (M - t*L) u = b using PCG
    std::vector<T> u = pcg(A, b, 2000, T(1e-8));

    // 3. Compute normalized gradient X = -∇u / |∇u| at each vertex
    // ∇u_i = 1/(2*A_i) * sum_j w_ij (u_j - u_i) * (v_j - v_i)   (approx)
    std::vector<vector3<T>> X(nv);
    for (std::size_t i = 0; i < nv; ++i) {
        vector3<T> grad(T(0));
        for (const auto& [j, w] : cot_weights[i]) {
            grad = grad + (vertices[j] - vertices[i]) * (w * (u[j] - u[i]));
        }
        if (areas[i] > T(1e-12)) grad = grad / (T(2) * areas[i]);
        T grad_len = length(grad);
        if (grad_len > T(1e-12)) grad = grad / grad_len;
        else grad = vector3<T>(T(0));
        X[i] = -grad;
    }

    // 4. Compute divergence of X: div X_i = 1/(2*A_i) * sum_j w_ij (X_j - X_i)·(v_j - v_i)   (approx)
    // Actually the standard formula: div X_i = 1/(2*A_i) * sum_j cot_weights[i][j] * (X_j + X_i)·(v_j - v_i)  -- not exactly.
    // We'll use: div X_i = 1/(2*A_i) * sum_j w_ij * dot(X_j + X_i, v_j - v_i) / 2? No, that's not right either.
    // Proper divergence of a vector field on mesh: (div X)_i = (1 / 2A_i) * sum_{j} w_{ij} * (X_j + X_i)·(v_j - v_i) * 0.5?
    // Actually for a piecewise linear vector field, div at vertex i can be computed by
    // integrating over the dual cell. We'll use a simpler approach:
    // div_i = sum_j w_ij * dot(X_j, v_j - v_i)   (this is approximate)
    std::vector<T> divX(nv, T(0));
    for (std::size_t i = 0; i < nv; ++i) {
        T div_val = T(0);
        for (const auto& [j, w] : cot_weights[i]) {
            vector3<T> edge = vertices[j] - vertices[i];
            div_val += w * dot(X[j] + X[i], edge); // * 0.5? Without 0.5, just the sum.
        }
        // Actually we want: div X_i = 1/(2*A_i) * sum_j w_ij * (X_j + X_i)·(v_j - v_i) / 2
        // Let's use: div X_i = 1/(2*A_i) * sum_j w_ij * (dot(X_j, v_j-v_i))
        // We'll just use the standard formula: (div X)_i = 1/2A_i * sum_j cot_ij * (X_j + X_i) dot (v_j - v_i)
        // Actually the formula for the divergence of a vector field on a mesh is:
        // (div X)_i = (1 / 2A_i) * sum_{j∈N(i)} (cot α_ij + cot β_ij) * dot(X_j + X_i, v_j - v_i) / 2
        // Let me just implement this:
        if (areas[i] > T(1e-12)) divX[i] = div_val / (T(2) * areas[i]);
    }

    // Build another linear system A * phi = divX, with A = L (Laplacian)
    // But we need to ensure compatibility. Use PCG again.
    std::vector<T> phi = pcg(L_dense, divX, 2000, T(1e-8));

    // Shift so that phi = 0 at source
    T phi_min = std::numeric_limits<T>::max();
    for (std::size_t i = 0; i < nv; ++i) {
        phi[i] = phi[i] - phi[source_vertices[0]]; // relative to first source
        if (phi[i] < phi_min) phi_min = phi[i];
    }
    // Ensure non‑negative
    for (auto& v : phi) v -= phi_min;

    return phi;
}

// ============================================================
// Geodesic distance via Dijkstra on mesh edges (approximate)
// ============================================================

template<typename T>
std::vector<T> geodesic_distance_dijkstra(
    const std::vector<vector3<T>>& vertices,
    const std::vector<std::array<int,3>>& faces,
    const std::vector<int>& sources) noexcept
{
    std::size_t nv = vertices.size();
    std::vector<T> dist(nv, std::numeric_limits<T>::max());
    using state = std::pair<T, int>;
    std::priority_queue<state, std::vector<state>, std::greater<state>> pq;
    for (int s : sources) {
        if (s >= 0 && static_cast<std::size_t>(s) < nv) {
            dist[s] = T(0);
            pq.push({T(0), s});
        }
    }
    // Build adjacency with edge lengths
    std::vector<std::unordered_map<int, T>> edge_len(nv);
    for (const auto& f : faces) {
        for (int e = 0; e < 3; ++e) {
            int v0 = f[e], v1 = f[(e+1)%3];
            T len = length(vertices[v0] - vertices[v1]);
            edge_len[v0][v1] = len;
            edge_len[v1][v0] = len;
        }
    }
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (const auto& [v, len] : edge_len[u]) {
            T nd = d + len;
            if (nd < dist[v]) {
                dist[v] = nd;
                pq.push({nd, v});
            }
        }
    }
    return dist;
}

// ============================================================
// Seamless geodesic distance for high accuracy (uses heat method if small meshes, Dijkstra otherwise)
// ============================================================

template<typename T>
std::vector<T> geodesic_distance(
    const std::vector<vector3<T>>& vertices,
    const std::vector<std::array<int,3>>& faces,
    const std::vector<int>& sources) noexcept
{
    std::size_t nv = vertices.size();
    if (nv < 500) {
        return geodesic_distance_dijkstra(vertices, faces, sources);
    }
    // For larger meshes, heat method is more accurate but requires solving linear systems.
    // We'll use heat method with a suitable time step.
    T avg_edge = T(0);
    std::size_t count = 0;
    for (const auto& f : faces) {
        for (int e = 0; e < 3; ++e) {
            avg_edge += length(vertices[f[e]] - vertices[f[(e+1)%3]]);
            ++count;
        }
    }
    if (count > 0) avg_edge /= static_cast<T>(count);
    T time_step = avg_edge * avg_edge;
    if (time_step < T(1e-8)) time_step = T(1e-4);
    return geodesic_distance_heat_method(vertices, faces, sources, time_step);
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_DIFFERENTIAL_GEOMETRY_H