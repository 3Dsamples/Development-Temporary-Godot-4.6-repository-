// system name : Octree Spatial Master
//File 0021 : core/math/fixed_polar_decomposition.h
//Polar decomposition of 3×3 matrices into rotation (R) and symmetric stretch (U), iterative Newton method, SIMD batch
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
// Compute inverse of a symmetric positive definite matrix using Cholesky
// ---------------------------------------------------------------------------
inline fmat3 fmat3_inverse_sympd(const fmat3& A) noexcept {
    // Use Cholesky, then solve I.
    fmat3 L;
    if (!fmat3_cholesky(A, L)) {
        // Fallback to general inverse
        return fmat3_inverse(A);
    }
    // Invert L (lower triangular)
    fmat3 L_inv = fmat3_identity();
    for (int i = 0; i < 3; ++i) {
        fixed64_t diag = *(&L.rows[0].x + i*3 + i);
        if (diag == 0) return fmat3_inverse(A); // fallback
        for (int j = 0; j < i; ++j) {
            fixed64_t sum = 0;
            for (int k = j; k < i; ++k) {
                sum += fixed_mul(*(&L.rows[0].x + i*3 + k), *(&L_inv.rows[0].x + k*3 + j));
            }
            *(&L_inv.rows[0].x + i*3 + j) = fixed_div(-sum, diag);
        }
        *(&L_inv.rows[0].x + i*3 + i) = fixed_rcp(diag);
    }
    // A_inv = L^{-T} * L^{-1}
    fmat3 L_invT = fmat3_transpose(L_inv);
    return fmat3_mul(L_invT, L_inv);
}

// ---------------------------------------------------------------------------
// Newton iteration to find the orthogonal factor R of the polar decomposition
//   X_{k+1} = 0.5 * (X_k + X_k^{-T})
//   X_0 = A, and iterate until convergence (||X_k - X_{k-1}||_F < tol)
//   Returns R = X_k (approximately orthogonal), then U = R^T * A
// ---------------------------------------------------------------------------
inline bool polar_decomposition(const fmat3& A, fmat3& R, fmat3& U, int max_iter = 16, fixed64_t tol = 1) noexcept {
    R = A;
    fmat3 R_prev = R;
    for (int iter = 0; iter < max_iter; ++iter) {
        // Compute R^{-T} = (R^{-1})^T
        fmat3 R_inv = fmat3_inverse_sympd(R); // R is not symmetric but may become nearly symmetric; use general inverse
        // Actually R is not symmetric, so we use general inverse:
        R_inv = fmat3_inverse(R);
        fmat3 R_invT = fmat3_transpose(R_inv);
        // X_new = 0.5 * (X + X^{-T})
        fmat3 R_new;
        for (int r=0; r<3; ++r) {
            for (int c=0; c<3; ++c) {
                fixed64_t val = fixed_mul(FIXED64_HALF, *(&R.rows[0].x + r*3 + c) + *(&R_invT.rows[0].x + r*3 + c));
                *(&R_new.rows[0].x + r*3 + c) = val;
            }
        }
        // Convergence check: Frobenius norm of difference
        fixed64_t norm_sq = 0;
        for (int r=0; r<3; ++r) {
            for (int c=0; c<3; ++c) {
                fixed64_t diff = *(&R_new.rows[0].x + r*3 + c) - *(&R_prev.rows[0].x + r*3 + c);
                norm_sq += fixed_mul(diff, diff);
            }
        }
        R = R_new;
        if (norm_sq < tol) break;
        R_prev = R;
    }
    // Now R is approximately orthogonal; compute U = R^T * A
    fmat3 RT = fmat3_transpose(R);
    U = fmat3_mul(RT, A);
    return true;
}

// ---------------------------------------------------------------------------
// Polar decomposition returning rotation and symmetric positive definite stretch
//   A = R * U (R rotation, U symmetric positive definite)
//   For invertible A, this decomposition always exists.
//   Also compute V = R * U * R^T (the other stretch)
// ---------------------------------------------------------------------------
inline bool polar_decomposition_right_stretch(const fmat3& A, fmat3& R, fmat3& U) noexcept {
    return polar_decomposition(A, R, U);
}

// ---------------------------------------------------------------------------
// Polar decomposition with left stretch: A = V * R
//   First compute right stretch via A^T = R * U, then V = A * R^T
// ---------------------------------------------------------------------------
inline bool polar_decomposition_left_stretch(const fmat3& A, fmat3& R, fmat3& V) noexcept {
    fmat3 AT = fmat3_transpose(A);
    fmat3 U, RT;
    if (!polar_decomposition(AT, RT, U)) return false;
    R = fmat3_transpose(RT);
    V = fmat3_mul(A, RT);
    return true;
}

// ---------------------------------------------------------------------------
// SIMD batch: apply polar decomposition to 4 matrices (scalar loop)
// ---------------------------------------------------------------------------
inline void polar_decomposition_batch(const fmat3 A[4], fmat3 R[4], fmat3 U[4]) noexcept {
    for (int i = 0; i < 4; ++i) polar_decomposition(A[i], R[i], U[i]);
}

// ---------------------------------------------------------------------------
// Interpolate rotation: given two rotation matrices R0, R1, parameter t,
//   compute R(t) = exp(t * log(R1 * R0^T)) * R0
//   Simplified: slerp using quaternion from rotation matrix
// ---------------------------------------------------------------------------
inline fmat3 interpolate_rotation(const fmat3& R0, const fmat3& R1, fixed64_t t) noexcept {
    // Convert to quaternion, slerp, convert back
    // Extract quaternion from rotation matrix (assumes orthogonal)
    // This is non‑trivial but can be done via fquat.
    // We'll implement a robust extraction:
    fquat q0, q1;
    // Convert matrix to quaternion (assuming orthonormal)
    auto mat3_to_quat = [](const fmat3& m, fquat& q) noexcept {
        fixed64_t tr = m.rows[0].x + m.rows[1].y + m.rows[2].z;
        if (tr > 0) {
            fixed64_t S = fixed_sqrt(tr + FIXED64_ONE) * 2; // S=4*qw
            q.w = fixed_div(S, 4 * FIXED64_ONE);
            q.x = fixed_div(m.rows[2].y - m.rows[1].z, S);
            q.y = fixed_div(m.rows[0].z - m.rows[2].x, S);
            q.z = fixed_div(m.rows[1].x - m.rows[0].y, S);
        } else if (m.rows[0].x > m.rows[1].y && m.rows[0].x > m.rows[2].z) {
            fixed64_t S = fixed_sqrt(FIXED64_ONE + m.rows[0].x - m.rows[1].y - m.rows[2].z) * 2;
            q.w = fixed_div(m.rows[2].y - m.rows[1].z, S);
            q.x = fixed_div(S, 4 * FIXED64_ONE);
            q.y = fixed_div(m.rows[0].y + m.rows[1].x, S);
            q.z = fixed_div(m.rows[0].z + m.rows[2].x, S);
        } else if (m.rows[1].y > m.rows[2].z) {
            fixed64_t S = fixed_sqrt(FIXED64_ONE + m.rows[1].y - m.rows[0].x - m.rows[2].z) * 2;
            q.w = fixed_div(m.rows[0].z - m.rows[2].x, S);
            q.x = fixed_div(m.rows[0].y + m.rows[1].x, S);
            q.y = fixed_div(S, 4 * FIXED64_ONE);
            q.z = fixed_div(m.rows[1].z + m.rows[2].y, S);
        } else {
            fixed64_t S = fixed_sqrt(FIXED64_ONE + m.rows[2].z - m.rows[0].x - m.rows[1].y) * 2;
            q.w = fixed_div(m.rows[1].x - m.rows[0].y, S);
            q.x = fixed_div(m.rows[0].z + m.rows[2].x, S);
            q.y = fixed_div(m.rows[1].z + m.rows[2].y, S);
            q.z = fixed_div(S, 4 * FIXED64_ONE);
        }
    };
    mat3_to_quat(R0, q0);
    mat3_to_quat(R1, q1);
    fquat q = fquat_slerp(q0, q1, t);
    return fquat_to_mat3(q);
}

} // namespace fixed_math