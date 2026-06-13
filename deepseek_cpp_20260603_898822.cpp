// system name : Octree Spatial Master
//File 0022 : core/math/fixed_spectral_decomposition.h
//Spectral decomposition of symmetric 3×3 matrices: eigenvalues, eigenvectors, sorting, matrix exponential/power, SIMD batch, condition number
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
// Jacobi eigenvalue decomposition for symmetric 3×3 matrix A
//   returns eigenvalues in lambda[3] and eigenvectors in columns of V (V^T * A * V = diag(lambda))
// ---------------------------------------------------------------------------
inline void symmetric_eigen_decomposition(const fmat3& A, fmat3& V, fixed64_t lambda[3]) noexcept {
    // Initialize V as identity, working copy M = A
    fmat3 M = A;
    V = fmat3_identity();
    const int MAX_ITER = 32;
    const fixed64_t TOL = 1; // convergence tolerance

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        // Find largest off-diagonal element
        int p = 0, q = 1;
        fixed64_t max_off = 0;
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                fixed64_t val = fixed_abs(*(&M.rows[0].x + i*3 + j));
                if (val > max_off) { max_off = val; p = i; q = j; }
            }
        }
        if (max_off <= TOL) break;

        // Compute Jacobi rotation angle
        fixed64_t app = *(&M.rows[0].x + p*3 + p);
        fixed64_t aqq = *(&M.rows[0].x + q*3 + q);
        fixed64_t apq = *(&M.rows[0].x + p*3 + q);
        fixed64_t theta;
        if (fixed_abs(app - aqq) < 1) { // difference negligible, use 45°
            theta = FIXED64_PI / 4;
        } else {
            fixed64_t num = 2 * apq;
            fixed64_t den = app - aqq;
            theta = fixed_atan2(num, den) >> 1; // half angle
        }
        fixed64_t c = fixed_cos(theta);
        fixed64_t s = fixed_sin(theta);

        // Apply rotation to M: M = J^T * M * J
        // We update rows/cols p and q
        // Save old values
        fixed64_t mpp = *(&M.rows[0].x + p*3 + p);
        fixed64_t mqq = *(&M.rows[0].x + q*3 + q);
        fixed64_t mpq = *(&M.rows[0].x + p*3 + q);

        // Update diagonal entries
        fixed64_t c2 = fixed_mul(c, c);
        fixed64_t s2 = fixed_mul(s, s);
        fixed64_t cs = fixed_mul(c, s);
        *(&M.rows[0].x + p*3 + p) = fixed_add(fixed_mul(c2, mpp), fixed_add(fixed_mul(s2, mqq), 2 * fixed_mul(cs, mpq)));
        *(&M.rows[0].x + q*3 + q) = fixed_add(fixed_mul(s2, mpp), fixed_add(fixed_mul(c2, mqq), -2 * fixed_mul(cs, mpq)));
        *(&M.rows[0].x + p*3 + q) = 0;
        *(&M.rows[0].x + q*3 + p) = 0;

        // Update off-diagonal elements in rows/cols p and q
        for (int i = 0; i < 3; ++i) {
            if (i == p || i == q) continue;
            fixed64_t mip = *(&M.rows[0].x + i*3 + p);
            fixed64_t miq = *(&M.rows[0].x + i*3 + q);
            *(&M.rows[0].x + i*3 + p) = fixed_add(fixed_mul(c, mip), fixed_mul(s, miq));
            *(&M.rows[0].x + p*3 + i) = *(&M.rows[0].x + i*3 + p);
            *(&M.rows[0].x + i*3 + q) = fixed_sub(fixed_mul(c, miq), fixed_mul(s, mip));
            *(&M.rows[0].x + q*3 + i) = *(&M.rows[0].x + i*3 + q);
        }

        // Update eigenvector matrix V: V = V * J
        for (int i = 0; i < 3; ++i) {
            fixed64_t vip = *(&V.rows[0].x + i*3 + p);
            fixed64_t viq = *(&V.rows[0].x + i*3 + q);
            *(&V.rows[0].x + i*3 + p) = fixed_add(fixed_mul(c, vip), fixed_mul(s, viq));
            *(&V.rows[0].x + i*3 + q) = fixed_sub(fixed_mul(c, viq), fixed_mul(s, vip));
        }
    }

    // Extract eigenvalues from diagonal of M
    lambda[0] = *(&M.rows[0].x + 0*3 + 0);
    lambda[1] = *(&M.rows[0].x + 1*3 + 1);
    lambda[2] = *(&M.rows[0].x + 2*3 + 2);
}

// ---------------------------------------------------------------------------
// Sort eigenvalues in descending absolute value (or descending) and permute eigenvectors
// ---------------------------------------------------------------------------
inline void sort_eigen_descending(fmat3& V, fixed64_t lambda[3]) noexcept {
    for (int i = 0; i < 2; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            if (lambda[j] > lambda[i]) {
                std::swap(lambda[i], lambda[j]);
                for (int r = 0; r < 3; ++r) {
                    std::swap(*(&V.rows[0].x + r*3 + i),
                              *(&V.rows[0].x + r*3 + j));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reconstruct symmetric matrix from its spectral decomposition: A = V * diag(lambda) * V^T
// ---------------------------------------------------------------------------
inline fmat3 reconstruct_from_spectral(const fmat3& V, const fixed64_t lambda[3]) noexcept {
    fmat3 A;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            fixed64_t sum = 0;
            for (int k = 0; k < 3; ++k) {
                sum += fixed_mul(fixed_mul(*(&V.rows[0].x + r*3 + k), lambda[k]),
                                 *(&V.rows[0].x + c*3 + k)); // V[r][k] * lambda[k] * V[c][k] (transposed)
            }
            *(&A.rows[0].x + r*3 + c) = sum;
        }
    }
    return A;
}

// ---------------------------------------------------------------------------
// Spectral norm (largest absolute eigenvalue)
// ---------------------------------------------------------------------------
inline fixed64_t spectral_norm(const fmat3& A) noexcept {
    fmat3 V;
    fixed64_t lambda[3];
    symmetric_eigen_decomposition(A, V, lambda);
    return fixed_max(fixed_abs(lambda[0]), fixed_max(fixed_abs(lambda[1]), fixed_abs(lambda[2])));
}

// ---------------------------------------------------------------------------
// Condition number (ratio of largest to smallest absolute eigenvalue)
// ---------------------------------------------------------------------------
inline fixed64_t condition_number(const fmat3& A) noexcept {
    fmat3 V;
    fixed64_t lambda[3];
    symmetric_eigen_decomposition(A, V, lambda);
    fixed64_t max_val = fixed_max(fixed_abs(lambda[0]), fixed_max(fixed_abs(lambda[1]), fixed_abs(lambda[2])));
    fixed64_t min_val = fixed_min(fixed_abs(lambda[0]), fixed_min(fixed_abs(lambda[1]), fixed_abs(lambda[2])));
    if (min_val == 0) return std::numeric_limits<fixed64_t>::max();
    return fixed_div(max_val, min_val);
}

// ---------------------------------------------------------------------------
// Matrix square root of symmetric positive definite matrix A = V * sqrt(lambda) * V^T
// ---------------------------------------------------------------------------
inline fmat3 matrix_sqrt_sympd(const fmat3& A) noexcept {
    fmat3 V;
    fixed64_t lambda[3];
    symmetric_eigen_decomposition(A, V, lambda);
    // Ensure non‑negative
    for (int i = 0; i < 3; ++i) if (lambda[i] < 0) lambda[i] = 0;
    // Compute sqrt(lambda)
    fixed64_t sqrt_lambda[3] = {fixed_sqrt(lambda[0]), fixed_sqrt(lambda[1]), fixed_sqrt(lambda[2])};
    return reconstruct_from_spectral(V, sqrt_lambda);
}

// ---------------------------------------------------------------------------
// Matrix exponential of symmetric matrix: exp(A) = V * diag(exp(lambda)) * V^T
// ---------------------------------------------------------------------------
inline fmat3 matrix_exponential_symmetric(const fmat3& A) noexcept {
    fmat3 V;
    fixed64_t lambda[3];
    symmetric_eigen_decomposition(A, V, lambda);
    fixed64_t exp_lambda[3] = {fixed_exp(lambda[0]), fixed_exp(lambda[1]), fixed_exp(lambda[2])};
    return reconstruct_from_spectral(V, exp_lambda);
}

// ---------------------------------------------------------------------------
// Matrix power (for positive eigenvalues) A^p = V * diag(lambda^p) * V^T
// ---------------------------------------------------------------------------
inline fmat3 matrix_power_sympd(const fmat3& A, fixed64_t p) noexcept {
    fmat3 V;
    fixed64_t lambda[3];
    symmetric_eigen_decomposition(A, V, lambda);
    for (int i = 0; i < 3; ++i) {
        if (lambda[i] < 0) lambda[i] = 0;
        lambda[i] = fixed_pow(lambda[i], p);
    }
    return reconstruct_from_spectral(V, lambda);
}

// ---------------------------------------------------------------------------
// SIMD batch: spectral decomposition for 4 matrices, output eigenvalues SoA and eigenvectors as fmat3[4]
// ---------------------------------------------------------------------------
inline void symmetric_eigen_decomposition_batch(const fmat3 A[4], fmat3 V[4], fixed64_t lambda[4][3]) noexcept {
    for (int i = 0; i < 4; ++i) {
        symmetric_eigen_decomposition(A[i], V[i], lambda[i]);
    }
}

} // namespace fixed_math

// End of File 0022
// Next file: File 0023 – core/math/fixed_plasticity.h
// Description: Plasticity models (von Mises, Drucker-Prager) with return mapping, yield criteria, and hardening, using tensor spectral decomposition and elasticity.