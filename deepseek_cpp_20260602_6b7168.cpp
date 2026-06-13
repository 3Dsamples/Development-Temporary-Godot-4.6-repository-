// system name : Octree Spatial Master
//File 0017 : core/math/fixed_statistics.h
//Fixed‑point statistics with tensor operations: mean, covariance, PCA via SVD, Mahalanobis distance, Gaussian probability, batch updates using SIMD
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "core/math/fixed_tensor.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace fixed_math {

// ---------------------------------------------------------------------------
// Sample mean of an array of 3D points
// ---------------------------------------------------------------------------
inline fvec3 mean_point(const fvec3* points, size_t count) noexcept {
    if (count == 0) return {0,0,0};
    fvec3 sum = {0,0,0};
    for (size_t i = 0; i < count; ++i) {
        sum.x += points[i].x; sum.y += points[i].y; sum.z += points[i].z;
    }
    fixed64_t inv_n = fixed_rcp(static_cast<fixed64_t>(count) << FRAC_BITS);
    return fvec3_scale(sum, inv_n);
}

// ---------------------------------------------------------------------------
// Covariance matrix of 3D points (sample covariance, N-1 denominator)
// ---------------------------------------------------------------------------
inline fmat3 covariance_matrix(const fvec3* points, size_t count) noexcept {
    if (count < 2) return fmat3_identity();
    fvec3 mean = mean_point(points, count);
    // Accumulate outer products
    fixed64_t c00=0, c01=0, c02=0, c11=0, c12=0, c22=0;
    for (size_t i = 0; i < count; ++i) {
        fvec3 d = fvec3_sub(points[i], mean);
        c00 += fixed_mul(d.x, d.x);
        c01 += fixed_mul(d.x, d.y);
        c02 += fixed_mul(d.x, d.z);
        c11 += fixed_mul(d.y, d.y);
        c12 += fixed_mul(d.y, d.z);
        c22 += fixed_mul(d.z, d.z);
    }
    fixed64_t inv_nm1 = fixed_rcp(static_cast<fixed64_t>(count-1) << FRAC_BITS);
    fmat3 cov;
    cov.rows[0] = {fixed_mul(c00, inv_nm1), fixed_mul(c01, inv_nm1), fixed_mul(c02, inv_nm1)};
    cov.rows[1] = {fixed_mul(c01, inv_nm1), fixed_mul(c11, inv_nm1), fixed_mul(c12, inv_nm1)};
    cov.rows[2] = {fixed_mul(c02, inv_nm1), fixed_mul(c12, inv_nm1), fixed_mul(c22, inv_nm1)};
    return cov;
}

// ---------------------------------------------------------------------------
// Principal Component Analysis (PCA) – eigenvalues sorted descending
// ---------------------------------------------------------------------------
inline void pca(const fvec3* points, size_t count, fmat3& eigenvectors, fixed64_t eigenvalues[3]) noexcept {
    fmat3 cov = covariance_matrix(points, count);
    fmat3 V; // eigenvectors matrix (columns)
    fmat3 U;
    fmat3_svd(cov, U, V, eigenvalues);
    // SVD of symmetric matrix: U == V, eigenvalues are singular values squared? Actually for SVD on cov, we get U * S * V^T, eigenvalues = S^2? Wait: cov is symmetric positive semi-definite. SVD of cov gives U*S*V^T. Because cov is symmetric, V = U, and singular values are the eigenvalues. So S[i] = eigenvalues[i]. Our fmat3_svd returns S[3] as singular values directly. So we can use those.
    eigenvectors = V;
    // Sort eigenvalues descending by swapping columns of V and eigenvalues.
    for (int i = 0; i < 2; ++i) {
        for (int j = i+1; j < 3; ++j) {
            if (eigenvalues[j] > eigenvalues[i]) {
                std::swap(eigenvalues[i], eigenvalues[j]);
                for (int r = 0; r < 3; ++r) {
                    std::swap(*(&eigenvectors.rows[0].x + r*3 + i),
                              *(&eigenvectors.rows[0].x + r*3 + j));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Mahalanobis distance squared: (x - mu)^T * inv(cov) * (x - mu)
// ---------------------------------------------------------------------------
inline fixed64_t mahalanobis_sq(const fvec3& x, const fvec3& mu, const fmat3& cov_inv) noexcept {
    fvec3 d = fvec3_sub(x, mu);
    fvec3 prod = fmat3_mul_vec3(cov_inv, d);
    return fvec3_dot(d, prod);
}

// ---------------------------------------------------------------------------
// Multivariate Gaussian probability density (3D)
//   pdf = (1 / sqrt((2π)^3 |cov|)) * exp(-0.5 * mahalanobis_sq)
// ---------------------------------------------------------------------------
inline fixed64_t gaussian_pdf(const fvec3& x, const fvec3& mu, const fmat3& cov) noexcept {
    fixed64_t det = fmat3_det(cov);
    fixed64_t norm = fixed_mul(fixed_mul(2 * FIXED64_PI, 2 * FIXED64_PI), 2 * FIXED64_PI); // (2π)^3
    norm = fixed_mul(norm, det);
    fixed64_t inv_norm = fixed_div(FIXED64_ONE, fixed_sqrt(norm));
    fmat3 cov_inv = fmat3_inverse(cov);
    fixed64_t mahal_sq = mahalanobis_sq(x, mu, cov_inv);
    return fixed_mul(inv_norm, fixed_exp(-fixed_mul(mahal_sq, FIXED64_HALF)));
}

// ---------------------------------------------------------------------------
// Batch update of mean and covariance (Welford's online algorithm)
// ---------------------------------------------------------------------------
struct RunningStats {
    size_t count = 0;
    fvec3 mean = {0,0,0};
    // Accumulates sum of outer products of (x - mean_old)*(x - mean_new) recursively
    fmat3 M2 = fmat3_identity(); // initialized to identity? Actually we start with zero matrix.
    // We'll store the M2 as a 3x3 matrix accumulating the sum of squared differences.
    // Initialize M2 to zero matrix.
    void add(const fvec3& x) noexcept {
        ++count;
        fvec3 delta = fvec3_sub(x, mean);
        fvec3 delta_div_n = fvec3_scale(delta, fixed_rcp(static_cast<fixed64_t>(count) << FRAC_BITS));
        mean = fvec3_add(mean, delta_div_n);
        // update M2 = M2 + delta * (delta - delta_div_n)^T? Actually Welford uses M2 += delta * (x - mean_new)
        fvec3 delta2 = fvec3_sub(x, mean);
        // Add outer product delta * delta2^T to M2
        fmat3 rank1;
        for (int r=0;r<3;++r)
            for (int c=0;c<3;++c)
                *(&rank1.rows[0].x + r*3 + c) = fixed_mul(*(&delta.x + r), *(&delta2.x + c));
        for (int r=0;r<3;++r)
            for (int c=0;c<3;++c)
                *(&M2.rows[0].x + r*3 + c) += *(&rank1.rows[0].x + r*3 + c);
    }
    fmat3 covariance() const noexcept {
        if (count < 2) return fmat3_identity();
        fixed64_t inv_nm1 = fixed_rcp(static_cast<fixed64_t>(count-1) << FRAC_BITS);
        fmat3 cov = M2;
        for (int r=0;r<3;++r)
            for (int c=0;c<3;++c)
                *(&cov.rows[0].x + r*3 + c) = fixed_mul(*(&M2.rows[0].x + r*3 + c), inv_nm1);
        return cov;
    }
};

// ---------------------------------------------------------------------------
// SIMD 4‑lane Mahalanobis distance for 4 sample points against same mean & inverse cov
// ---------------------------------------------------------------------------
inline __m256i simd4_mahalanobis_sq(const fvec3* samples, const fvec3& mu, const fmat3& cov_inv) noexcept {
    alignas(32) int64_t res[4];
    for (int i = 0; i < 4; ++i) {
        res[i] = mahalanobis_sq(samples[i], mu, cov_inv);
    }
    return _mm256_load_si256((__m256i*)res);
}

} // namespace fixed_math