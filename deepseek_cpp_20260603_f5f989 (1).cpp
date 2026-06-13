// system name : Octree Spatial Master
//File 0027 : core/math/fixed_tensor_derivative.h
//Derivatives of tensor functions: Gateaux derivatives, tangent stiffness linearisation, Jacobian of constitutive models, objective rates, SIMD batch
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "core/math/fixed_tensor.h"
#include "core/math/fixed_elasticity_tensor.h"
#include "core/math/fixed_plasticity.h"
#include "core/math/fixed_spectral_decomposition.h"
#include "core/math/fixed_polar_decomposition.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <functional>

namespace fixed_math {

// ---------------------------------------------------------------------------
// Gateaux derivative of a tensor-valued function F(A) in direction H
//   dF/dA [H] = (F(A + eps*H) - F(A)) / eps, with eps = 2^-12 (approx 2.44e-4)
// ---------------------------------------------------------------------------
template<typename Func>
inline fmat3 gateaux_derivative(const Func& F, const fmat3& A, const fmat3& H) noexcept {
    constexpr fixed64_t EPS = FIXED64_ONE >> 12; // ~0.000244
    fmat3 A_plus = fmat3_add(A, fmat3_mul_scalar(H, EPS));
    fmat3 FA = F(A);
    fmat3 FA_plus = F(A_plus);
    fmat3 dF;
    for (int r=0; r<3; ++r)
        for (int c=0; c<3; ++c)
            *(&dF.rows[0].x + r*3 + c) = fixed_div(*(&FA_plus.rows[0].x + r*3 + c) - *(&FA.rows[0].x + r*3 + c), EPS);
    return dF;
}

// ---------------------------------------------------------------------------
// Directional derivative of a scalar invariant with respect to a tensor argument
//   dI/dA : H = (I(A + eps*H) - I(A)) / eps
// ---------------------------------------------------------------------------
template<typename InvFunc>
inline fixed64_t scalar_gateaux(const InvFunc& I, const fmat3& A, const fmat3& H) noexcept {
    constexpr fixed64_t EPS = FIXED64_ONE >> 12;
    fixed64_t I0 = I(A);
    fmat3 A_plus = fmat3_add(A, fmat3_mul_scalar(H, EPS));
    fixed64_t I1 = I(A_plus);
    return fixed_div(I1 - I0, EPS);
}

// ---------------------------------------------------------------------------
// Derivative of the trace: d(tr(A))/dA = I (identity matrix)
// ---------------------------------------------------------------------------
inline fmat3 d_trace_dA(const fmat3& /*A*/) noexcept {
    return fmat3_identity();
}

// ---------------------------------------------------------------------------
// Derivative of the determinant: d(det(A))/dA = det(A) * A^{-T}
// ---------------------------------------------------------------------------
inline fmat3 d_det_dA(const fmat3& A) noexcept {
    fixed64_t det = fmat3_det(A);
    fmat3 invT = fmat3_transpose(fmat3_inverse(A));
    return fmat3_mul_scalar(invT, det);
}

// ---------------------------------------------------------------------------
// Derivative of the second invariant I2(A) = (tr(A)^2 - tr(A^2))/2
//   dI2/dA = tr(A)*I - A
// ---------------------------------------------------------------------------
inline fmat3 d_I2_dA(const fmat3& A) noexcept {
    fixed64_t trA = fmat3_trace(A);
    fmat3 I = fmat3_identity();
    fmat3 term1 = fmat3_mul_scalar(I, trA);
    // subtract A from identity-scaled
    for (int r=0; r<3; ++r)
        for (int c=0; c<3; ++c)
            *(&term1.rows[0].x + r*3 + c) -= *(&A.rows[0].x + r*3 + c);
    return term1;
}

// ---------------------------------------------------------------------------
// Derivative of the Frobenius norm squared: d(||A||^2)/dA = 2*A
// ---------------------------------------------------------------------------
inline fmat3 d_frob2_dA(const fmat3& A) noexcept {
    return fmat3_mul_scalar(A, 2 * FIXED64_ONE);
}

// ---------------------------------------------------------------------------
// Derivative of the Frobenius norm: d||A||/dA = A / ||A|| (if A != 0)
// ---------------------------------------------------------------------------
inline fmat3 d_frob_dA(const fmat3& A) noexcept {
    fixed64_t norm = frobenius_norm(A);
    if (norm == 0) {
        fmat3 z;
        for (int r=0;r<3;++r) for (int c=0;c<3;++c) *(&z.rows[0].x + r*3 + c) = 0;
        return z;
    }
    return fmat3_mul_scalar(A, fixed_rcp(norm));
}

// ---------------------------------------------------------------------------
// Derivative of the von Mises stress with respect to stress tensor
//   dσ_eq / dσ = (3/σ_eq) * dev(sigma)  (if σ_eq > 0)
// ---------------------------------------------------------------------------
inline fmat3 d_von_mises_d_sigma(const fmat3& sigma) noexcept {
    fixed64_t seq = von_mises_eq_stress(sigma);
    if (seq == 0) {
        fmat3 z;
        for (int r=0;r<3;++r) for (int c=0;c<3;++c) *(&z.rows[0].x + r*3 + c) = 0;
        return z;
    }
    fmat3 dev = deviatoric(sigma);
    return fmat3_mul_scalar(dev, fixed_div(3 * FIXED64_ONE, seq));
}

// ---------------------------------------------------------------------------
// Derivative of the yield surface normal n = s/||s|| (deviatoric space)
//   dn/dσ = (I_dev - n⊗n) / ||s||, where I_dev is the deviatoric projection tensor.
//   We output a 6x6 matrix in Voigt form.
// ---------------------------------------------------------------------------
inline void d_yield_normal(const fmat3& sigma, StiffnessMatrix6x6& dn_dsigma) noexcept {
    fmat3 s = deviatoric(sigma);
    fixed64_t norm_s = fixed_sqrt(2 * J2(s));
    if (norm_s == 0) {
        dn_dsigma.zero();
        return;
    }
    // n_voigt = s_voigt / norm_s
    fixed64_t n_voigt[6];
    symmetric_3x3_to_voigt(s, n_voigt);
    for (int i=0;i<6;++i) n_voigt[i] = fixed_div(n_voigt[i], norm_s);

    // I_dev in Voigt (6x6)
    StiffnessMatrix6x6 I_dev;
    I_dev.zero();
    I_dev(0,0) = I_dev(1,1) = I_dev(2,2) = 2 * FIXED64_ONE / 3;
    I_dev(0,1) = I_dev(0,2) = I_dev(1,2) = -FIXED64_ONE / 3;
    I_dev(3,3) = I_dev(4,4) = I_dev(5,5) = FIXED64_HALF;

    // n⊗n in Voigt (6x6)
    for (int i=0;i<6;++i) {
        for (int j=0;j<6;++j) {
            fixed64_t outer_ij = fixed_mul(n_voigt[i], n_voigt[j]);
            fixed64_t dev_ij = I_dev(i,j);
            fixed64_t val = fixed_div(dev_ij - outer_ij, norm_s);
            dn_dsigma(i,j) = val;
        }
    }
}

// ---------------------------------------------------------------------------
// Consistent tangent stiffness for von Mises plasticity with isotropic hardening
//   Full implementation using trial stress, corrected stress, and hardening parameters.
// ---------------------------------------------------------------------------
inline StiffnessMatrix6x6 von_mises_consistent_tangent_full(
    const fmat3& sigma_trial, const fmat3& sigma_corrected,
    fixed64_t sigma_eq_trial, fixed64_t mu, fixed64_t K,
    fixed64_t delta_gamma, const StiffnessMatrix6x6& elastic) noexcept {
    StiffnessMatrix6x6 Cep = elastic;
    if (sigma_eq_trial <= 0 || delta_gamma <= 0) return Cep;

    fmat3 s_trial = deviatoric(sigma_trial);
    fixed64_t norm_s_trial = fixed_sqrt(2 * J2(s_trial));
    if (norm_s_trial == 0) return Cep;

    // n_voigt = s_trial_voigt / sigma_eq_trial
    fixed64_t n_voigt[6];
    symmetric_3x3_to_voigt(s_trial, n_voigt);
    for (int i=0;i<6;++i) n_voigt[i] = fixed_div(n_voigt[i], sigma_eq_trial);

    fixed64_t alpha = FIXED64_ONE - fixed_div(3 * mu * delta_gamma, sigma_eq_trial);
    fixed64_t beta = FIXED64_ONE - alpha - fixed_div(K, 3*mu + K);

    // I_dev in Voigt (6x6)
    StiffnessMatrix6x6 I_dev;
    I_dev.zero();
    I_dev(0,0) = I_dev(1,1) = I_dev(2,2) = 2 * FIXED64_ONE / 3;
    I_dev(0,1) = I_dev(0,2) = I_dev(1,2) = -FIXED64_ONE / 3;
    I_dev(3,3) = I_dev(4,4) = I_dev(5,5) = FIXED64_HALF;

    // n⊗n
    StiffnessMatrix6x6 N_outer;
    for (int i=0;i<6;++i) for (int j=0;j<6;++j) N_outer(i,j) = fixed_mul(n_voigt[i], n_voigt[j]);

    // Cep = C - 2μ * (α * I_dev + β * N_outer)
    fixed64_t two_mu = 2 * mu;
    for (int i=0;i<6;++i) {
        for (int j=0;j<6;++j) {
            fixed64_t dev_term = fixed_mul(two_mu, fixed_mul(alpha, I_dev(i,j)));
            fixed64_t outer_term = fixed_mul(two_mu, fixed_mul(beta, N_outer(i,j)));
            Cep(i,j) = Cep(i,j) - dev_term - outer_term;
        }
    }
    return Cep;
}

// ---------------------------------------------------------------------------
// Objective stress rates: Jaumann and Green‑Naghdi
// ---------------------------------------------------------------------------
inline fmat3 jaumann_rate(const fmat3& sigma, const fmat3& velocity_gradient) noexcept {
    fmat3 L = velocity_gradient;
    fmat3 LT = fmat3_transpose(L);
    fmat3 W = fmat3_mul_scalar(fmat3_sub(L, LT), FIXED64_HALF);
    fmat3 sigma_W = fmat3_mul(sigma, W);
    fmat3 W_sigma = fmat3_mul(W, sigma);
    return fmat3_sub(sigma_W, W_sigma);
}

inline fmat3 green_naghdi_rate(const fmat3& sigma, const fmat3& R_dot, const fmat3& R) noexcept {
    fmat3 RT = fmat3_transpose(R);
    fmat3 Omega = fmat3_mul(R_dot, RT);
    fmat3 sigma_Omega = fmat3_mul(sigma, Omega);
    fmat3 Omega_sigma = fmat3_mul(Omega, sigma);
    return fmat3_sub(sigma_Omega, Omega_sigma);
}

// ---------------------------------------------------------------------------
// Generic derivative of isotropic tensor function via spectral decomposition
//   Given F(A) = V * diag(f(λ_i)) * V^T, the derivative is computed by:
//   dF_ij / dA_kl = Σ_p Σ_q V_ip V_jp V_kq V_lq D_pq
//   where D_pq = (f(λ_p)-f(λ_q))/(λ_p-λ_q) if λ_p≠λ_q, else f'(λ_p)
//   Returns a 6x6 stiffness matrix in Voigt form.
// ---------------------------------------------------------------------------
inline StiffnessMatrix6x6 isotropic_tensor_function_derivative(const fmat3& A,
    const std::function<fixed64_t(fixed64_t)>& f,
    const std::function<fixed64_t(fixed64_t)>& f_prime) noexcept {
    fmat3 V;
    fixed64_t lambda[3];
    symmetric_eigen_decomposition(A, V, lambda);

    fixed64_t f_vals[3];
    for (int i=0;i<3;++i) f_vals[i] = f(lambda[i]);

    // D_pq matrix (3x3)
    fixed64_t Dpq[3][3];
    for (int p=0;p<3;++p) {
        for (int q=0;q<=p;++q) {
            if (fixed_abs(lambda[p] - lambda[q]) < 1) {
                Dpq[p][q] = f_prime(lambda[p]);
            } else {
                Dpq[p][q] = fixed_div(f_vals[p] - f_vals[q], lambda[p] - lambda[q]);
            }
            Dpq[q][p] = Dpq[p][q];
        }
    }

    const int idx[6][2] = {{0,0},{1,1},{2,2},{1,2},{0,2},{0,1}};
    StiffnessMatrix6x6 D;
    D.zero();

    for (int I=0; I<6; ++I) {
        int i = idx[I][0], j = idx[I][1];
        for (int J=0; J<6; ++J) {
            int k = idx[J][0], l = idx[J][1];
            fixed64_t sum = 0;
            for (int p=0; p<3; ++p) {
                for (int q=0; q<3; ++q) {
                    fixed64_t v_ip = *(&V.rows[0].x + i*3 + p);
                    fixed64_t v_jp = *(&V.rows[0].x + j*3 + p);
                    fixed64_t v_kq = *(&V.rows[0].x + k*3 + q);
                    fixed64_t v_lq = *(&V.rows[0].x + l*3 + q);
                    fixed64_t prod = fixed_mul(fixed_mul(fixed_mul(v_ip, v_jp), fixed_mul(v_kq, v_lq)), Dpq[p][q]);
                    sum += prod;
                }
            }
            D(I,J) = sum;
        }
    }
    return D;
}

// ---------------------------------------------------------------------------
// Specific derivatives of tensor functions (using the above generic function)
// ---------------------------------------------------------------------------
inline StiffnessMatrix6x6 d_sqrtA_dA(const fmat3& A) noexcept {
    auto f_sqrt = [](fixed64_t x) -> fixed64_t { return fixed_sqrt(x); };
    auto f_prime = [](fixed64_t x) -> fixed64_t { return fixed_rcp(2 * fixed_sqrt(x)); };
    return isotropic_tensor_function_derivative(A, f_sqrt, f_prime);
}

inline StiffnessMatrix6x6 d_logA_dA(const fmat3& A) noexcept {
    auto f_log = [](fixed64_t x) -> fixed64_t { return fixed_log(x); };
    auto f_prime = [](fixed64_t x) -> fixed64_t { return fixed_rcp(x); };
    return isotropic_tensor_function_derivative(A, f_log, f_prime);
}

inline StiffnessMatrix6x6 d_expA_dA(const fmat3& A) noexcept {
    auto f_exp = [](fixed64_t x) -> fixed64_t { return fixed_exp(x); };
    auto f_prime = [](fixed64_t x) -> fixed64_t { return fixed_exp(x); };
    return isotropic_tensor_function_derivative(A, f_exp, f_prime);
}

// ---------------------------------------------------------------------------
// Hencky (logarithmic) strain tangent stiffness
//   Implements the spatial tangent stiffness for the Hencky isotropic elasticity
//   model using the constant material stiffness λ (I⊗I) + 2μ I_sym, pushed forward
//   with the left stretch tensor V.
//   Input: V (left stretch, symmetric positive definite), Lamé constants λ and μ.
//   Output: spatial elasticity tensor c (6x6 in Voigt) at the current configuration.
//   The spatial tangent is defined by τ∇ = c : d, where τ is Kirchhoff stress,
//   ∇ the Jaumann rate, and d the rate of deformation.
//   Formula: c_ijkl = (1/J) * V_iI V_jJ V_kK V_lL C_IJKL_iso
//   with C_iso the material isotropic stiffness:
//   C_IJKL = λ δ_IJ δ_KL + μ (δ_IK δ_JL + δ_IL δ_JK)
// ---------------------------------------------------------------------------
inline StiffnessMatrix6x6 hencky_elastic_tangent(const fmat3& V, fixed64_t lambda, fixed64_t mu) noexcept {
    fixed64_t detV = fmat3_det(V); // J
    if (detV == 0) detV = FIXED64_ONE;
    fixed64_t invJ = fixed_rcp(detV);

    // Material stiffness tensor C_iso (Voigt 6x6)
    StiffnessMatrix6x6 C_iso;
    C_iso.zero();
    // Normal components
    C_iso(0,0) = C_iso(1,1) = C_iso(2,2) = lambda + 2*mu;
    C_iso(0,1) = C_iso(0,2) = C_iso(1,2) = lambda;
    // Shear components
    C_iso(3,3) = C_iso(4,4) = C_iso(5,5) = mu;

    // Push-forward: c_ijkl = (1/J) V_iI V_jJ V_kK V_lL C_IJKL
    // We'll compute the spatial stiffness by transforming each Voigt basis.
    // The mapping from material indices (IJ) to spatial (ij) uses V components.
    // We'll compute c(I,J) = (1/J) * Σ_{K,L} C_iso(IJKL) * (V_{iK} V_{jL}) ...
    // But this is exactly the operation of transforming a fourth-order tensor:
    // For Voigt pairs, the transformation is:
    // c_{ii} (normal) = (1/J) Σ_{p,q} V_ip V_iq C_pq_iso, where C_pq_iso is the 6x6 matrix
    // and the shear components similarly.
    // We'll implement the full transformation using nested loops over Cartesian indices.

    StiffnessMatrix6x6 c_spatial;
    c_spatial.zero();
    const int idx[6][2] = {{0,0},{1,1},{2,2},{1,2},{0,2},{0,1}};

    for (int I=0; I<6; ++I) { // spatial Voigt index
        int i = idx[I][0], j = idx[I][1];
        for (int J=0; J<6; ++J) {
            int k = idx[J][0], l = idx[J][1];
            fixed64_t sum = 0;
            // sum over material Voigt indices K,L (6 components)
            for (int K=0; K<6; ++K) {
                int p = idx[K][0], q = idx[K][1]; // material indices p,q
                for (int L=0; L<6; ++L) {
                    int r = idx[L][0], s = idx[L][1]; // material indices r,s
                    // C_iso(K,L) is the material stiffness in Voigt
                    fixed64_t C_KL = C_iso(K,L);
                    if (C_KL == 0) continue;
                    // Compute V_iI V_jJ V_kK V_lL? No, material indices are p,q,r,s.
                    // The push-forward is: c_ijkl = (1/J) V_ip V_jq V_kr V_ls C_pqrs
                    fixed64_t prod = fixed_mul(fixed_mul(V.rows[i].x + p, V.rows[j].x + q), // careful: V is stored row-major, but we need element V(i,p). 
                                               // rows[i] gives the i-th row; V(i,p) = *(&V.rows[0].x + i*3 + p)
                                               // Let's use a helper to access V element.
                                               *(&V.rows[0].x + i*3 + p) );
                    prod = fixed_mul(prod, *(&V.rows[0].x + j*3 + q));
                    prod = fixed_mul(prod, *(&V.rows[0].x + k*3 + r));
                    prod = fixed_mul(prod, *(&V.rows[0].x + l*3 + s));
                    sum += fixed_mul(prod, C_KL);
                }
            }
            c_spatial(I,J) = fixed_mul(sum, invJ);
        }
    }
    return c_spatial;
}

// ---------------------------------------------------------------------------
// SIMD batch for Gateaux derivative: apply same direction H to 4 matrices, compute dF/dA
// ---------------------------------------------------------------------------
template<typename Func>
inline void gateaux_derivative_batch(const Func& F, const fmat3 A[4], const fmat3 H[4], fmat3 dF[4]) noexcept {
    for (int i=0; i<4; ++i) {
        dF[i] = gateaux_derivative(F, A[i], H[i]);
    }
}

} // namespace fixed_math