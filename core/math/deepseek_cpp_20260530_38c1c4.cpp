// File 0033 : core/math/geometry_fitting.h
// Geometric fitting: plane, line, sphere from point sets using SVD and algebraic least squares.

#pragma once

#include "vec3.h"
#include "plane.h"
#include "sphere.h"
#include "mat3.h"
#include "svd3.h"
#include "constants.h"
#include <vector>
#include <cmath>

namespace wp {

// ── Fit plane to 3D points using orthogonal regression ─────────────────
// Returns a plane (unit normal, d = -normal·center) that minimizes the sum of squared distances.
template <typename T>
plane<T> fit_plane_least_squares(const std::vector<vec3<T>>& points) {
    if (points.empty()) return plane<T>();
    // Compute centroid
    vec3<T> centroid(T(0));
    for (const auto& p : points) centroid = centroid + p;
    centroid = centroid / T(points.size());

    // Build covariance matrix of the centered points
    mat3<T> cov(T(0));
    for (const auto& p : points) {
        vec3<T> d = p - centroid;
        cov(0,0) += d.x * d.x;
        cov(0,1) += d.x * d.y;
        cov(0,2) += d.x * d.z;
        cov(1,1) += d.y * d.y;
        cov(1,2) += d.y * d.z;
        cov(2,2) += d.z * d.z;
    }
    // Fill symmetric part
    cov(1,0) = cov(0,1);
    cov(2,0) = cov(0,2);
    cov(2,1) = cov(1,2);

    // Compute eigenvalues/eigenvectors. The normal is the eigenvector corresponding to the smallest eigenvalue.
    mat3<T> V;
    vec3<T> lambda;
    symmetric_eigen(cov, V, lambda);

    // Find index of smallest eigenvalue
    int idx = 0;
    if (lambda.y < lambda.x) idx = 1;
    if (lambda.z < lambda[idx]) idx = 2;

    vec3<T> normal = V.col(idx);
    // Ensure normal points in a consistent direction? Not needed.
    return plane<T>(normal, -dot(normal, centroid));
}

// ── Fit 3D line to points using PCA ────────────────────────────────────
// Returns a pair (point_on_line, direction). The direction is the principal component with largest variance.
template <typename T>
std::pair<vec3<T>, vec3<T>> fit_line_least_squares(const std::vector<vec3<T>>& points) {
    if (points.empty()) return {vec3<T>(T(0)), vec3<T>(T(0),T(0),T(1))};

    vec3<T> centroid(T(0));
    for (const auto& p : points) centroid = centroid + p;
    centroid = centroid / T(points.size());

    mat3<T> cov(T(0));
    for (const auto& p : points) {
        vec3<T> d = p - centroid;
        cov(0,0) += d.x * d.x;
        cov(0,1) += d.x * d.y;
        cov(0,2) += d.x * d.z;
        cov(1,1) += d.y * d.y;
        cov(1,2) += d.y * d.z;
        cov(2,2) += d.z * d.z;
    }
    cov(1,0) = cov(0,1);
    cov(2,0) = cov(0,2);
    cov(2,1) = cov(1,2);

    mat3<T> V;
    vec3<T> lambda;
    symmetric_eigen(cov, V, lambda);

    int idx = 0;
    if (lambda.y > lambda.x) idx = 1;
    if (lambda.z > lambda[idx]) idx = 2;

    vec3<T> direction = normalize(V.col(idx));
    return {centroid, direction};
}

// ── Fit sphere to points using algebraic linear method ─────────────────
// Solves min Σ ( (p_i - c)^2 - r^2 )²  by linearizing: 2c·p_i + (r² - |c|²) = |p_i|².
// Form system A * [c_x, c_y, c_z, k]^T = b where k = r² - |c|².
// Returns sphere (center, radius). If radius negative, return degenerate.
template <typename T>
sphere<T> fit_sphere_least_squares(const std::vector<vec3<T>>& points) {
    if (points.size() < 4) return sphere<T>(vec3<T>(T(0)), T(0));

    const size_t n = points.size();
    // Build normal equations matrix (4x4 symmetric) and rhs vector (4).
    // We solve A^T A x = A^T b.
    // A[i] = [2*px, 2*py, 2*pz, 1], b[i] = |p_i|².
    // The system is small, we'll accumulate directly.
    T sum_x=0, sum_y=0, sum_z=0, sum_x2=0, sum_y2=0, sum_z2=0;
    T sum_xy=0, sum_xz=0, sum_yz=0;
    T sum_x3=0, sum_y3=0, sum_z3=0;
    T sum_x2y=0, sum_x2z=0, sum_xy2=0, sum_y2z=0, sum_xz2=0, sum_yz2=0;
    T sum_x4=0, sum_y4=0, sum_z4=0;
    T sum_x2y2=0, sum_x2z2=0, sum_y2z2=0;
    T sum_b=0, sum_xb=0, sum_yb=0, sum_zb=0;

    for (const auto& p : points) {
        T x=p.x, y=p.y, z=p.z;
        T b_val = x*x + y*y + z*z;
        T x2=x*x, y2=y*y, z2=z*z;
        sum_x += x; sum_y += y; sum_z += z;
        sum_x2 += x2; sum_y2 += y2; sum_z2 += z2;
        sum_xy += x*y; sum_xz += x*z; sum_yz += y*z;
        sum_x3 += x2*x; sum_y3 += y2*y; sum_z3 += z2*z;
        sum_x2y += x2*y; sum_x2z += x2*z;
        sum_xy2 += x*y2; sum_y2z += y2*z;
        sum_xz2 += x*z2; sum_yz2 += y*z2;
        sum_x4 += x2*x2; sum_y4 += y2*y2; sum_z4 += z2*z2;
        sum_x2y2 += x2*y2; sum_x2z2 += x2*z2; sum_y2z2 += y2*z2;
        sum_b += b_val;
        sum_xb += x * b_val;
        sum_yb += y * b_val;
        sum_zb += z * b_val;
    }

    // Build matrix M (4x4) and vector v (4) from the normal equations.
    // Rows/cols correspond to variables: a=c_x, b=c_y, c=c_z, d=k
    // The equation for a row is sum over points of: [2x, 2y, 2z, 1] * [a,b,c,d]^T = |p|^2
    // Normal eqns: (sum 4x²) a + (sum 4xy) b + (sum 4xz) c + (sum 2x) d = (sum 2x |p|²)
    // etc.
    T A[4][4], rhs[4];
    for (int i=0;i<4;++i) for (int j=0;j<4;++j) A[i][j]=0;

    A[0][0] = 4*sum_x2;
    A[0][1] = 4*sum_xy;
    A[0][2] = 4*sum_xz;
    A[0][3] = 2*sum_x;

    A[1][1] = 4*sum_y2;
    A[1][2] = 4*sum_yz;
    A[1][3] = 2*sum_y;

    A[2][2] = 4*sum_z2;
    A[2][3] = 2*sum_z;

    A[3][3] = T(n);

    // fill symmetric
    A[1][0] = A[0][1];
    A[2][0] = A[0][2];
    A[2][1] = A[1][2];
    A[3][0] = A[0][3];
    A[3][1] = A[1][3];
    A[3][2] = A[2][3];

    rhs[0] = 2*sum_xb;
    rhs[1] = 2*sum_yb;
    rhs[2] = 2*sum_zb;
    rhs[3] = sum_b;

    // Solve 4x4 system using Gaussian elimination (no partial pivoting, but matrix is well-behaved)
    // We'll use a simple elimination since it's small.
    for (int k=0;k<4;++k) {
        // If pivot is zero, attempt row swap? skip for now (should not happen with enough points)
        T pivot = A[k][k];
        if (std::abs(pivot) < epsilon<T>) continue;
        // Normalize row k
        for (int j=k+1;j<4;++j) A[k][j] /= pivot;
        rhs[k] /= pivot;
        // Eliminate other rows
        for (int i=k+1;i<4;++i) {
            T factor = A[i][k];
            if (std::abs(factor) < epsilon<T>) continue;
            for (int j=k+1;j<4;++j) A[i][j] -= factor * A[k][j];
            rhs[i] -= factor * rhs[k];
        }
    }
    // Back substitution
    T solution[4];
    for (int i=3;i>=0;--i) {
        T s = rhs[i];
        for (int j=i+1;j<4;++j) s -= A[i][j] * solution[j];
        solution[i] = s; // row i was normalized
    }

    vec3<T> center(solution[0], solution[1], solution[2]);
    T k = solution[3];
    T radius2 = k + dot(center, center);
    if (radius2 < T(0)) radius2 = T(0);
    return sphere<T>(center, std::sqrt(radius2));
}

// ── Fit 2D circle to points using algebraic least squares ──────────────
// Similar to 3D sphere but using 2D coordinates.
// Returns (center, radius).
template <typename T>
std::pair<vec2<T>, T> fit_circle_2d(const std::vector<vec2<T>>& points) {
    if (points.size() < 3) return {vec2<T>(T(0)), T(0)};
    const size_t n = points.size();
    T sum_x=0, sum_y=0, sum_x2=0, sum_y2=0, sum_xy=0;
    T sum_x3=0, sum_y3=0, sum_x2y=0, sum_xy2=0;
    T sum_b=0, sum_xb=0, sum_yb=0;

    for (const auto& p : points) {
        T x = p.x, y = p.y;
        T b_val = x*x + y*y;
        T x2 = x*x, y2 = y*y;
        sum_x += x; sum_y += y;
        sum_x2 += x2; sum_y2 += y2;
        sum_xy += x*y;
        sum_x3 += x2*x; sum_y3 += y2*y;
        sum_x2y += x2*y; sum_xy2 += x*y2;
        sum_b += b_val;
        sum_xb += x*b_val;
        sum_yb += y*b_val;
    }

    // Matrix 3x3: unknowns a=c_x, b=c_y, d=k = r² - |c|².
    T A[3][3] = {}, rhs[3] = {};
    A[0][0] = 4*sum_x2;
    A[0][1] = 4*sum_xy;
    A[0][2] = 2*sum_x;
    A[1][1] = 4*sum_y2;
    A[1][2] = 2*sum_y;
    A[2][2] = T(n);
    A[1][0] = A[0][1];
    A[2][0] = A[0][2];
    A[2][1] = A[1][2];

    rhs[0] = 2*sum_xb;
    rhs[1] = 2*sum_yb;
    rhs[2] = sum_b;

    // Gaussian elimination 3x3
    for (int k=0;k<3;++k) {
        T pivot = A[k][k];
        if (std::abs(pivot) < epsilon<T>) continue;
        for (int j=k+1;j<3;++j) A[k][j] /= pivot;
        rhs[k] /= pivot;
        for (int i=k+1;i<3;++i) {
            T factor = A[i][k];
            if (std::abs(factor) < epsilon<T>) continue;
            for (int j=k+1;j<3;++j) A[i][j] -= factor * A[k][j];
            rhs[i] -= factor * rhs[k];
        }
    }
    T sol[3];
    for (int i=2;i>=0;--i) {
        T s = rhs[i];
        for (int j=i+1;j<3;++j) s -= A[i][j] * sol[j];
        sol[i] = s;
    }
    vec2<T> center(sol[0], sol[1]);
    T k = sol[2];
    T r2 = k + dot(center, center);
    if (r2 < T(0)) r2 = T(0);
    return {center, std::sqrt(r2)};
}

// ── Fit 3D plane using RANSAC (outlier rejection) ──────────────────────
template <typename T>
plane<T> fit_plane_ransac(const std::vector<vec3<T>>& points,
                          T distance_threshold, int max_iterations = 100) {
    if (points.size() < 3) return plane<T>();
    // Use random sampling (simple LCG)
    auto rand_int = [&](int lo, int hi) -> int {
        static uint64 state = 123456789;
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return lo + int(state % uint64(hi - lo + 1));
    };

    plane<T> best_plane;
    int best_inliers = 0;

    for (int iter = 0; iter < max_iterations; ++iter) {
        int i0 = rand_int(0, int(points.size())-1);
        int i1 = rand_int(0, int(points.size())-1);
        int i2 = rand_int(0, int(points.size())-1);
        if (i0 == i1 || i0 == i2 || i1 == i2) continue;

        const vec3<T>& p0 = points[i0];
        const vec3<T>& p1 = points[i1];
        const vec3<T>& p2 = points[i2];
        vec3<T> normal = normalize(cross(p1 - p0, p2 - p0));
        if (length_sq(normal) < epsilon<T>) continue;
        plane<T> candidate(normal, -dot(normal, p0));

        int inliers = 0;
        for (const auto& pt : points) {
            if (std::abs(candidate.distance(pt)) <= distance_threshold)
                ++inliers;
        }
        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_plane = candidate;
        }
    }
    return best_plane;
}

} // namespace wp