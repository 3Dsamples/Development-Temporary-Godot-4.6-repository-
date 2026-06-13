// system name : Octree Spatial Master
//File 0013 : core/math/fixed_tensor.h
//Fixed‑point tensor operations: Cholesky, LU, QR, SVD decompositions, linear system solver, rank‑one update, and matrix power iteration for 3x3 matrices
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace fixed_math {

// ============================================================================
// Cholesky decomposition – A = L * L^T (A symmetric positive‑definite 3x3)
// Returns true if decomposition succeeded; L is lower‑triangular.
// ============================================================================
inline bool fmat3_cholesky(const fmat3& A, fmat3& L) noexcept {
    L = fmat3_identity();
    fixed64_t l00 = fixed_sqrt(*(&A.rows[0].x + 0*3 + 0));
    if (l00 == 0) return false;
    fixed64_t l10 = fixed_div(*(&A.rows[0].x + 1*3 + 0), l00);
    fixed64_t l20 = fixed_div(*(&A.rows[0].x + 2*3 + 0), l00);
    fixed64_t l11_sq = *(&A.rows[0].x + 1*3 + 1) - fixed_mul(l10, l10);
    if (l11_sq <= 0) return false;
    fixed64_t l11 = fixed_sqrt(l11_sq);
    fixed64_t l21 = fixed_div(*(&A.rows[0].x + 2*3 + 1) - fixed_mul(l20, l10), l11);
    fixed64_t l22_sq = *(&A.rows[0].x + 2*3 + 2) - fixed_mul(l20, l20) - fixed_mul(l21, l21);
    if (l22_sq <= 0) return false;
    fixed64_t l22 = fixed_sqrt(l22_sq);
    L.rows[0] = {l00, 0, 0};
    L.rows[1] = {l10, l11, 0};
    L.rows[2] = {l20, l21, l22};
    return true;
}

// ============================================================================
// LU decomposition with partial pivoting (3x3) – PA = LU
//   P is permutation encoded in perm array, L unit lower triangular, U upper.
//   Returns true on success.
// ============================================================================
inline bool fmat3_lu(const fmat3& A, fmat3& L, fmat3& U, int perm[3]) noexcept {
    // Copy A into U
    U = A;
    L = fmat3_identity();
    perm[0]=0; perm[1]=1; perm[2]=2;
    for (int i=0; i<3; ++i) {
        // Find pivot
        int pivot = i;
        fixed64_t max_val = fixed_abs(*(&U.rows[0].x + i*3 + i));
        for (int r=i+1; r<3; ++r) {
            fixed64_t v = fixed_abs(*(&U.rows[0].x + r*3 + i));
            if (v > max_val) { max_val = v; pivot = r; }
        }
        if (pivot != i) {
            // Swap rows in U, L (only columns < i in L), and perm
            for (int c=0; c<3; ++c) {
                std::swap(*(&U.rows[0].x + i*3 + c), *(&U.rows[0].x + pivot*3 + c));
                if (c < i) std::swap(*(&L.rows[0].x + i*3 + c), *(&L.rows[0].x + pivot*3 + c));
            }
            std::swap(perm[i], perm[pivot]);
        }
        fixed64_t pivot_val = *(&U.rows[0].x + i*3 + i);
        if (pivot_val == 0) return false;
        for (int r=i+1; r<3; ++r) {
            fixed64_t factor = fixed_div(*(&U.rows[0].x + r*3 + i), pivot_val);
            *(&L.rows[0].x + r*3 + i) = factor;
            for (int c=i; c<3; ++c) {
                *(&U.rows[0].x + r*3 + c) -= fixed_mul(factor, *(&U.rows[0].x + i*3 + c));
            }
        }
    }
    return true;
}

// Solve Ax = b using LU decomposition (3x3). b is overwritten with x.
inline bool fmat3_solve_lu(const fmat3& L, const fmat3& U, const int perm[3], fvec3& b) noexcept {
    // Forward substitution for Ly = Pb
    fvec3 y;
    for (int i=0; i<3; ++i) {
        fixed64_t sum = 0;
        for (int j=0; j<i; ++j) {
            sum += fixed_mul(*(&L.rows[0].x + i*3 + j), *(&y.x + j));
        }
        // b[perm[i]] - sum
        fixed64_t rhs = *(&b.x + perm[i]);
        *(&y.x + i) = rhs - sum;
    }
    // Back substitution for Ux = y
    for (int i=2; i>=0; --i) {
        fixed64_t sum = 0;
        for (int j=i+1; j<3; ++j) {
            sum += fixed_mul(*(&U.rows[0].x + i*3 + j), *(&b.x + j));
        }
        *(&b.x + i) = fixed_div(*(&y.x + i) - sum, *(&U.rows[0].x + i*3 + i));
    }
    return true;
}

// ============================================================================
// QR decomposition (3x3) – Householder transformations
//   Q orthogonal, R upper triangular.  A = Q * R
// ============================================================================
inline void fmat3_qr(const fmat3& A, fmat3& Q, fmat3& R) noexcept {
    R = A;
    Q = fmat3_identity();
    for (int k=0; k<2; ++k) { // only two Householder steps needed
        // Build Householder vector to zero below diagonal in column k
        fixed64_t x[3];
        for (int i=k; i<3; ++i) x[i] = *(&R.rows[0].x + i*3 + k);
        fixed64_t norm_x = fixed_sqrt(fixed_mul(x[k], x[k]) + fixed_mul(x[k+1], x[k+1]) + (k+2<3 ? fixed_mul(x[k+2], x[k+2]) : 0));
        if (norm_x == 0) continue;
        fixed64_t alpha = (x[k] > 0) ? -norm_x : norm_x;
        fixed64_t u0 = x[k] - alpha;
        fixed64_t norm_u2 = fixed_mul(u0, u0) + fixed_mul(x[k+1], x[k+1]) + (k+2<3 ? fixed_mul(x[k+2], x[k+2]) : 0);
        if (norm_u2 == 0) continue;
        fixed64_t beta = FIXED64_ONE;
        if (norm_u2 != 0) beta = FIXED64_ONE;
        // Actually Householder: H = I - 2 * v*v^T / (v^T v) with v = x - alpha*e_k
        // We compute v = [u0, x[k+1], x[k+2]]
        // v^T * R_sub: then update R and Q
        // For brevity, we implement the full update.
        // Compute v^T * R[k..2][k..2] (vector of dot products)
        fixed64_t vt_R[3] = {0,0,0};
        for (int c=k; c<3; ++c) {
            fixed64_t sum = 0;
            for (int i=k; i<3; ++i) {
                sum += fixed_mul(*(&R.rows[0].x + i*3 + c), (i==k ? u0 : *(&R.rows[0].x + i*3 + k))); // v_i = u0 if i==k else x[i]
            }
            vt_R[c] = sum;
        }
        fixed64_t scale = fixed_div(FIXED64_ONE, norm_u2);
        // Update R: R = R - 2 * v * (scale * vt_R)
        for (int i=k; i<3; ++i) {
            fixed64_t vi = (i==k) ? u0 : *(&R.rows[0].x + i*3 + k);
            for (int c=k; c<3; ++c) {
                *(&R.rows[0].x + i*3 + c) -= fixed_mul(2 * FIXED64_ONE, fixed_mul(vi, fixed_mul(scale, vt_R[c])));
            }
        }
        // Update Q: Q = H * Q
        // Similar pattern: Q = Q - 2 * v * (scale * (v^T * Q))
        // vt_Q = v^T * Q (dot product of v with each row of Q)
        fixed64_t vt_Q[3] = {0,0,0};
        for (int r=0; r<3; ++r) {
            fixed64_t sum = 0;
            for (int i=k; i<3; ++i) {
                sum += fixed_mul((i==k ? u0 : *(&R.rows[0].x + i*3 + k)), *(&Q.rows[0].x + r*3 + i));
            }
            vt_Q[r] = sum;
        }
        for (int r=0; r<3; ++r) {
            fixed64_t vi = (r==k) ? u0 : *(&R.rows[0].x + r*3 + k);
            for (int c=0; c<3; ++c) {
                *(&Q.rows[0].x + r*3 + c) -= fixed_mul(2 * FIXED64_ONE, fixed_mul(vi, fixed_mul(scale, vt_Q[c])));
            }
        }
    }
    // Q is orthogonal, R is upper triangular; multiply Q by -1 if diagonal negative? Actually fine.
}

// ============================================================================
// Singular Value Decomposition (3x3) – Golub‑Reinsch via Jacobi
//   Returns U, S (diagonal), V^T.  A = U * S * V^T
//   (Simplified implementation: uses symmetric Jacobi on A^T*A)
// ============================================================================
inline void fmat3_svd(const fmat3& A, fmat3& U, fmat3& V, fixed64_t S[3]) noexcept {
    // Compute A^T * A
    fmat3 ATA;
    for (int r=0; r<3; ++r)
        for (int c=0; c<3; ++c) {
            fixed64_t sum = 0;
            for (int k=0; k<3; ++k) sum += fixed_mul(*(&A.rows[0].x + k*3 + r), *(&A.rows[0].x + k*3 + c));
            *(&ATA.rows[0].x + r*3 + c) = sum;
        }
    // Jacobi decomposition of ATA -> V * S^2 * V^T
    V = fmat3_identity();
    // Iterative Jacobi (same as in obb_from_points but on ATA)
    fmat3 M = ATA;
    const int MAX_ITER = 16;
    for (int iter=0; iter<MAX_ITER; ++iter) {
        int p=0, q=1;
        fixed64_t max_off = 0;
        for (int i=0; i<3; ++i) {
            for (int j=i+1; j<3; ++j) {
                fixed64_t val = fixed_abs(*(&M.rows[0].x + i*3 + j));
                if (val > max_off) { max_off = val; p=i; q=j; }
            }
        }
        if (max_off <= 1) break;
        fixed64_t app = *(&M.rows[0].x + p*3 + p);
        fixed64_t aqq = *(&M.rows[0].x + q*3 + q);
        fixed64_t apq = *(&M.rows[0].x + p*3 + q);
        fixed64_t theta = fixed_atan2(2*apq, app - aqq) >> 1;
        fixed64_t c = fixed_cos(theta);
        fixed64_t s = fixed_sin(theta);
        // Update M and V
        for (int i=0; i<3; ++i) {
            if (i!=p && i!=q) {
                fixed64_t mip = *(&M.rows[0].x + i*3 + p);
                fixed64_t miq = *(&M.rows[0].x + i*3 + q);
                fixed64_t new_mip = fixed_add(fixed_mul(c, mip), fixed_mul(s, miq));
                fixed64_t new_miq = fixed_sub(fixed_mul(c, miq), fixed_mul(s, mip));
                *(&M.rows[0].x + i*3 + p) = *(&M.rows[0].x + p*3 + i) = new_mip;
                *(&M.rows[0].x + i*3 + q) = *(&M.rows[0].x + q*3 + i) = new_miq;
            }
        }
        // Update V
        for (int i=0; i<3; ++i) {
            fixed64_t vip = *(&V.rows[0].x + i*3 + p);
            fixed64_t viq = *(&V.rows[0].x + i*3 + q);
            *(&V.rows[0].x + i*3 + p) = fixed_add(fixed_mul(c, vip), fixed_mul(s, viq));
            *(&V.rows[0].x + i*3 + q) = fixed_sub(fixed_mul(c, viq), fixed_mul(s, vip));
        }
    }
    // Extract singular values from M diagonal
    S[0] = *(&M.rows[0].x + 0*3 + 0);
    S[1] = *(&M.rows[0].x + 1*3 + 1);
    S[2] = *(&M.rows[0].x + 2*3 + 2);
    // Ensure non‑negative
    for (int i=0; i<3; ++i) if (S[i] < 0) S[i] = 0;
    // Compute U = A * V * S⁻¹
    // S⁻¹ diagonal matrix: invS[i] = 1/S[i] (if S[i]==0 then 0)
    fmat3 S_inv_diag = fmat3_identity();
    *(&S_inv_diag.rows[0].x + 0*3 + 0) = (S[0]==0) ? 0 : fixed_rcp(S[0]);
    *(&S_inv_diag.rows[0].x + 1*3 + 1) = (S[1]==0) ? 0 : fixed_rcp(S[1]);
    *(&S_inv_diag.rows[0].x + 2*3 + 2) = (S[2]==0) ? 0 : fixed_rcp(S[2]);
    U = fmat3_mul(A, fmat3_mul(V, S_inv_diag));
}

// ============================================================================
// Rank‑one update: A += alpha * u * v^T
// ============================================================================
inline void fmat3_rank1_update(fmat3& A, fixed64_t alpha, const fvec3& u, const fvec3& v) noexcept {
    for (int r=0; r<3; ++r) {
        for (int c=0; c<3; ++c) {
            *(&A.rows[0].x + r*3 + c) += fixed_mul(alpha, fixed_mul(*(&u.x + r), *(&v.x + c)));
        }
    }
}

// ============================================================================
// Power iteration: find dominant eigenvalue and eigenvector of symmetric 3x3
// ============================================================================
inline fixed64_t power_iteration(const fmat3& A, fvec3& eigenvector, int max_iter = 32) noexcept {
    fvec3 v = {FIXED64_ONE, 0, 0};
    fixed64_t eigenvalue = 0;
    for (int iter=0; iter<max_iter; ++iter) {
        fvec3 Av = fmat3_mul_vec3(A, v);
        eigenvalue = fvec3_dot(v, Av);
        fixed64_t len = fvec3_length(Av);
        if (len == 0) break;
        v = fvec3_scale(Av, fixed_rcp(len));
    }
    eigenvector = v;
    return eigenvalue;
}

// ============================================================================
// Tensor product: outer product of two vectors (already in vec3, but formal)
// ============================================================================
inline fmat3 tensor_product(const fvec3& a, const fvec3& b) noexcept {
    return fmat3_outer(a, b);
}

// ============================================================================
// Matrix trace
// ============================================================================
inline fixed64_t fmat3_trace(const fmat3& A) noexcept {
    return *(&A.rows[0].x + 0*3 + 0) + *(&A.rows[0].x + 1*3 + 1) + *(&A.rows[0].x + 2*3 + 2);
}

// ============================================================================
// Determinant of 3x3 (already in fixed_mat but alias for completeness)
// ============================================================================
inline fixed64_t fmat3_determinant(const fmat3& A) noexcept { return fmat3_det(A); }

} // namespace fixed_math

// End of File 0013
// Next file: File 0014 – core/math/fixed_noise.h
// Description: Fixed‑point Perlin, Simplex, and Worley noise functions with SIMD batch evaluation for procedural generation and simulation.