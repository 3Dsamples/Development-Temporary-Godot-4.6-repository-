// system name : Octree Spatial Master
//File 0024 : core/math/fixed_tensor_contraction.h
//Tensor contractions: double contraction A:B, Frobenius norm, stress power, Voigt dot products, SIMD batch, high‑order tensor inner products
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "core/math/fixed_tensor.h"
#include "core/math/fixed_elasticity_tensor.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace fixed_math {

// ---------------------------------------------------------------------------
// Double contraction (inner product) of two symmetric 3×3 tensors: A : B = Σ A_ij * B_ij
// ---------------------------------------------------------------------------
inline fixed64_t double_contraction(const fmat3& A, const fmat3& B) noexcept {
    fixed64_t sum = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            sum += fixed_mul(*(&A.rows[0].x + i*3 + j), *(&B.rows[0].x + i*3 + j));
    return sum;
}

// ---------------------------------------------------------------------------
// Frobenius norm squared of a 3×3 matrix: ||A||_F^2 = A : A
// ---------------------------------------------------------------------------
inline fixed64_t frobenius_norm_sq(const fmat3& A) noexcept {
    return double_contraction(A, A);
}

// ---------------------------------------------------------------------------
// Frobenius norm: sqrt(A : A)
// ---------------------------------------------------------------------------
inline fixed64_t frobenius_norm(const fmat3& A) noexcept {
    return fixed_sqrt(frobenius_norm_sq(A));
}

// ---------------------------------------------------------------------------
// L2 norm (alias for Frobenius)
// ---------------------------------------------------------------------------
inline fixed64_t tensor_norm(const fmat3& A) noexcept { return frobenius_norm(A); }

// ---------------------------------------------------------------------------
// Deviatoric norm squared: ||dev(A)||^2
// ---------------------------------------------------------------------------
inline fixed64_t deviatoric_norm_sq(const fmat3& A) noexcept {
    fmat3 dev = devatoric(A);
    return frobenius_norm_sq(dev);
}

// ---------------------------------------------------------------------------
// Volumetric norm squared: ||(tr(A)/3)*I||^2
// ---------------------------------------------------------------------------
inline fixed64_t volumetric_norm_sq(const fmat3& A) noexcept {
    fixed64_t p = fixed_div(stress_I1(A), 3 * FIXED64_ONE);
    return 3 * fixed_mul(p, p); // I:I = 3
}

// ---------------------------------------------------------------------------
// Stress power: σ : ε_dot (rate of mechanical work)
// ---------------------------------------------------------------------------
inline fixed64_t stress_power(const fmat3& sigma, const fmat3& epsilon_dot) noexcept {
    return double_contraction(sigma, epsilon_dot);
}

// ---------------------------------------------------------------------------
// Inner product in Voigt notation: voigt_a · voigt_b
// ---------------------------------------------------------------------------
inline fixed64_t voigt_dot(const fixed64_t a[6], const fixed64_t b[6]) noexcept {
    fixed64_t sum = 0;
    for (int i = 0; i < 6; ++i) sum += fixed_mul(a[i], b[i]);
    return sum;
}

// ---------------------------------------------------------------------------
// Double contraction using Voigt vectors (more efficient for 6x6 stiffness)
//   A : B = Σ voigt(A)_i * voigt(B)_i  (but note: shear terms have factor 2 for energy conjugate)
//   We use standard Voigt mapping for strain/stress that already incorporates factors.
// ---------------------------------------------------------------------------
inline fixed64_t double_contraction_voigt(const fmat3& A, const fmat3& B) noexcept {
    fixed64_t a[6], b[6];
    symmetric_3x3_to_voigt(A, a);
    symmetric_3x3_to_voigt(B, b);
    // Standard Voigt for energy: multiply shear components by 2 because the off-diagonal terms appear twice in A:B
    // Actually, if we use engineering shear strain (γ=2ε), then the contraction is Σ a_i * b_i. But for stress-strain energy,
    // it's σ:ε = σ_xx ε_xx + σ_yy ε_yy + σ_zz ε_zz + 2*(σ_yz ε_yz + σ_xz ε_xz + σ_xy ε_xy).
    // We'll just compute the full sum directly from the matrix to avoid confusion. Use the scalar version.
    return double_contraction(A, B);
}

// ---------------------------------------------------------------------------
// Voigt matrix-vector product: C_voigt * v_voigt = w_voigt
// ---------------------------------------------------------------------------
inline void voigt_matrix_apply(const fixed64_t C[6][6], const fixed64_t v[6], fixed64_t w[6]) noexcept {
    for (int i = 0; i < 6; ++i) {
        w[i] = 0;
        for (int j = 0; j < 6; ++j) {
            w[i] += fixed_mul(C[i][j], v[j]);
        }
    }
}

// ---------------------------------------------------------------------------
// Normalised tensor: A / ||A|| (if norm > 0)
// ---------------------------------------------------------------------------
inline fmat3 normalize_tensor(const fmat3& A) noexcept {
    fixed64_t n = frobenius_norm(A);
    if (n == 0) return A;
    fmat3 result;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            *(&result.rows[0].x + i*3 + j) = fixed_div(*(&A.rows[0].x + i*3 + j), n);
    return result;
}

// ---------------------------------------------------------------------------
// Cosine similarity between two tensors: (A:B) / (||A|| * ||B||)
// ---------------------------------------------------------------------------
inline fixed64_t tensor_cosine(const fmat3& A, const fmat3& B) noexcept {
    fixed64_t num = double_contraction(A, B);
    fixed64_t den = frobenius_norm(A) * frobenius_norm(B);
    if (den == 0) return 0;
    return fixed_div(num, den);
}

// ---------------------------------------------------------------------------
// Angle between two tensors (in radians)
// ---------------------------------------------------------------------------
inline fixed64_t tensor_angle(const fmat3& A, const fmat3& B) noexcept {
    fixed64_t cos_sim = tensor_cosine(A, B);
    return fixed_acos(cos_sim);
}

// ---------------------------------------------------------------------------
// Third‑order tensor operations: double contraction of a 3rd‑order tensor (3 matrices) with a 2nd‑order tensor
//   Result[i] = Σ_jk C_i_jk * B_jk
//   We represent 3rd‑order tensor as an array of three fmat3 (each is the i‑th component over j,k).
// ---------------------------------------------------------------------------
inline fvec3 third_order_double_contraction(const fmat3 C[3], const fmat3& B) noexcept {
    return {
        double_contraction(C[0], B),
        double_contraction(C[1], B),
        double_contraction(C[2], B)
    };
}

// ---------------------------------------------------------------------------
// Fourth‑order tensor operation: apply 6x6 stiffness matrix to a strain tensor and return stress tensor
//   Already covered by apply_stiffness. Provide an alias.
// ---------------------------------------------------------------------------
inline fmat3 apply_fourth_order(const StiffnessMatrix6x6& C, const fmat3& strain) noexcept {
    return apply_stiffness(C, strain);
}

// ---------------------------------------------------------------------------
// Compute elasticity tensor from Lame constants as 6x6 (already in elasticity_tensor), here we add a dynamic creation
// ---------------------------------------------------------------------------
inline StiffnessMatrix6x6 make_isotropic_stiffness_from_lame(fixed64_t lambda, fixed64_t mu) noexcept {
    return isotropic_stiffness_from_lame(lambda, mu);
}

// ---------------------------------------------------------------------------
// SIMD 4‑lane double contraction: result = Σ A_i : B_i for i=0..3, returns 4 values
// ---------------------------------------------------------------------------
inline __m256i simd4_double_contraction(const fmat3 A[4], const fmat3 B[4]) noexcept {
    alignas(32) fixed64_t res[4];
    for (int i = 0; i < 4; ++i) res[i] = double_contraction(A[i], B[i]);
    return _mm256_load_si256((__m256i*)res);
}

// ---------------------------------------------------------------------------
// SIMD 4‑lane Frobenius norm
// ---------------------------------------------------------------------------
inline __m256i simd4_frobenius_norm(const fmat3 A[4]) noexcept {
    alignas(32) fixed64_t res[4];
    for (int i = 0; i < 4; ++i) res[i] = frobenius_norm(A[i]);
    return _mm256_load_si256((__m256i*)res);
}

} // namespace fixed_math