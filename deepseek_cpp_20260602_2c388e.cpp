// system name : Octree Spatial Master
//File 0019 : core/math/fixed_tensor_physics.h
//Tensor‑based physics operations: invariants, principal stresses, von Mises, octahedral stress, and strain energy density, with SIMD 4‑lane reduction
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "core/math/fixed_tensor.h"
#include "core/math/fixed_vector_field.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace fixed_math {

// ---------------------------------------------------------------------------
// Stress tensor invariants
// ---------------------------------------------------------------------------

// First invariant (trace)
inline fixed64_t stress_I1(const fmat3& sigma) noexcept {
    return fmat3_trace(sigma);
}

// Second invariant (sum of principal minors)
inline fixed64_t stress_I2(const fmat3& sigma) noexcept {
    fixed64_t sxx = sigma.rows[0].x, sxy = sigma.rows[0].y, sxz = sigma.rows[0].z;
    fixed64_t syy = sigma.rows[1].y, syz = sigma.rows[1].z;
    fixed64_t szz = sigma.rows[2].z;
    fixed64_t minor1 = fixed_sub(fixed_mul(syy, szz), fixed_mul(syz, syz));
    fixed64_t minor2 = fixed_sub(fixed_mul(sxx, szz), fixed_mul(sxz, sxz));
    fixed64_t minor3 = fixed_sub(fixed_mul(sxx, syy), fixed_mul(sxy, sxy));
    return minor1 + minor2 + minor3;
}

// Third invariant (determinant)
inline fixed64_t stress_I3(const fmat3& sigma) noexcept {
    return fmat3_det(sigma);
}

// ---------------------------------------------------------------------------
// Principal stresses (eigenvalues) – computed via cubic root of characteristic eqn
//   λ^3 - I1*λ^2 + I2*λ - I3 = 0
// We use trigonometric method for three real roots (since stress tensor is symmetric)
// ---------------------------------------------------------------------------
inline void principal_stresses(const fmat3& sigma, fixed64_t& s1, fixed64_t& s2, fixed64_t& s3) noexcept {
    fixed64_t I1 = stress_I1(sigma);
    fixed64_t I2 = stress_I2(sigma);
    fixed64_t I3 = stress_I3(sigma);
    fixed64_t p = fixed_sub(I2, fixed_div(fixed_mul(I1, I1), 3 * FIXED64_ONE));
    fixed64_t q = fixed_sub(fixed_add(fixed_mul(2 * FIXED64_ONE,
                               fixed_div(fixed_mul(fixed_mul(I1, I1), I1), 27 * FIXED64_ONE)),
                         fixed_div(fixed_mul(I1, I2), 3 * FIXED64_ONE)), I3);
    if (p == 0) {
        s1 = s2 = s3 = fixed_div(I1, 3 * FIXED64_ONE);
        return;
    }
    // discriminant: D = (q/2)^2 + (p/3)^3
    fixed64_t q_half = fixed_mul(q, FIXED64_HALF);
    fixed64_t p_third = fixed_div(p, 3 * FIXED64_ONE);
    fixed64_t D = fixed_add(fixed_mul(q_half, q_half), fixed_mul(p_third, fixed_mul(p_third, p_third)));
    if (D > 0) {
        // one real root (should not happen for real symmetric? but handle)
        fixed64_t sqrtD = fixed_sqrt(D);
        fixed64_t A = fixed_exp(fixed_div(fixed_log(fixed_abs(fixed_add(-q_half, sqrtD))), 3 * FIXED64_ONE));
        fixed64_t B = fixed_exp(fixed_div(fixed_log(fixed_abs(fixed_sub(-q_half, sqrtD))), 3 * FIXED64_ONE));
        s1 = fixed_add(A, B) + fixed_div(I1, 3 * FIXED64_ONE);
        s2 = s3 = 0;
    } else {
        // three real roots
        fixed64_t phi = fixed_acos(fixed_div(-q_half, fixed_sqrt(fixed_mul(-p_third, fixed_mul(-p_third, -p_third)))));
        fixed64_t two_sqrt_p3 = fixed_mul(2 * FIXED64_ONE, fixed_sqrt(-p_third));
        s1 = fixed_add(fixed_mul(two_sqrt_p3, fixed_cos(fixed_div(phi, 3 * FIXED64_ONE))), fixed_div(I1, 3 * FIXED64_ONE));
        s2 = fixed_add(fixed_mul(two_sqrt_p3, fixed_cos(fixed_div(fixed_add(phi, 2 * FIXED64_PI), 3 * FIXED64_ONE))), fixed_div(I1, 3 * FIXED64_ONE));
        s3 = fixed_add(fixed_mul(two_sqrt_p3, fixed_cos(fixed_div(fixed_add(phi, 4 * FIXED64_PI), 3 * FIXED64_ONE))), fixed_div(I1, 3 * FIXED64_ONE));
    }
}

// ---------------------------------------------------------------------------
// Von Mises stress: sqrt(0.5 * ((s1-s2)^2 + (s2-s3)^2 + (s3-s1)^2))
// ---------------------------------------------------------------------------
inline fixed64_t von_mises(const fmat3& sigma) noexcept {
    fixed64_t s1, s2, s3;
    principal_stresses(sigma, s1, s2, s3);
    fixed64_t d12 = fixed_sub(s1, s2), d23 = fixed_sub(s2, s3), d31 = fixed_sub(s3, s1);
    return fixed_sqrt(fixed_mul(FIXED64_HALF,
        fixed_add(fixed_add(fixed_mul(d12, d12), fixed_mul(d23, d23)), fixed_mul(d31, d31))));
}

// ---------------------------------------------------------------------------
// Octahedral shear stress: sqrt(2*J2/3)
// ---------------------------------------------------------------------------
inline fixed64_t octahedral_shear(const fmat3& sigma) noexcept {
    fixed64_t I1 = stress_I1(sigma);
    fixed64_t I2 = stress_I2(sigma);
    fixed64_t J2 = fixed_sub(fixed_div(fixed_mul(I1, I1), 3 * FIXED64_ONE), I2);
    if (J2 < 0) J2 = 0;
    return fixed_sqrt(fixed_div(fixed_mul(2 * FIXED64_ONE, J2), 3 * FIXED64_ONE));
}

// ---------------------------------------------------------------------------
// Strain energy density for isotropic linear elasticity: W = 0.5 * sigma_ij * epsilon_ij
//   epsilon = (1/E) * ((1+nu)*sigma - nu*tr(sigma)*I)
//   For given sigma and material constants, compute W
// ---------------------------------------------------------------------------
inline fixed64_t strain_energy_density(const fmat3& sigma, fixed64_t E, fixed64_t nu) noexcept {
    fixed64_t trace = stress_I1(sigma);
    fmat3 I = fmat3_identity();
    fmat3 epsilon;
    for (int r=0; r<3; ++r)
        for (int c=0; c<3; ++c) {
            fixed64_t delta = (r==c) ? FIXED64_ONE : 0;
            fixed64_t eps_ij = fixed_div(fixed_sub(fixed_mul(FIXED64_ONE + nu, *(&sigma.rows[0].x + r*3 + c)),
                                             fixed_mul(nu, fixed_mul(trace, delta))), E);
            *(&epsilon.rows[0].x + r*3 + c) = eps_ij;
        }
    // W = 0.5 * sigma_ij * epsilon_ij
    fixed64_t W = 0;
    for (int r=0; r<3; ++r)
        for (int c=0; c<3; ++c)
            W += fixed_mul(*(&sigma.rows[0].x + r*3 + c), *(&epsilon.rows[0].x + r*3 + c));
    return fixed_mul(FIXED64_HALF, W);
}

// ---------------------------------------------------------------------------
// SIMD 4‑lane von Mises for four stress tensors
// ---------------------------------------------------------------------------
inline __m256i simd4_von_mises(const fmat3* sigmas) noexcept {
    alignas(32) int64_t res[4];
    for (int i = 0; i < 4; ++i) res[i] = von_mises(sigmas[i]);
    return _mm256_load_si256((__m256i*)res);
}

} // namespace fixed_math