// system name : Octree Spatial Master
//File 0026 : core/math/fixed_constitutive_integrator.h
//Constitutive integrators for inelastic materials: explicit/implicit Euler, viscoelastic, viscoplastic, damage, Newton-Raphson, SIMD batch
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "core/math/fixed_tensor.h"
#include "core/math/fixed_elasticity_tensor.h"
#include "core/math/fixed_plasticity.h"
#include "core/math/fixed_spectral_decomposition.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <functional>

namespace fixed_math {

// ============================================================================
// General state vector (array of fixed64_t) for internal variables
// ============================================================================
using StateVector = fixed64_t*; // pointer to array

// ---------------------------------------------------------------------------
// Forward Euler: state_new = state + dt * rate(state)
// ---------------------------------------------------------------------------
inline void forward_euler(const std::function<void(const fixed64_t*, fixed64_t*)>& rate,
                          const fixed64_t* state, fixed64_t* state_new,
                          int dim, fixed64_t dt) noexcept {
    rate(state, state_new); // rate is stored in state_new temporary
    for (int i = 0; i < dim; ++i) {
        state_new[i] = state[i] + fixed_mul(state_new[i], dt);
    }
}

// ---------------------------------------------------------------------------
// Backward Euler with Newton-Raphson: residual R = state_new - state - dt*rate(state_new) = 0
//   Jacobian provided by user; solves for state_new.
// ---------------------------------------------------------------------------
inline bool backward_euler_newton(
    const std::function<void(const fixed64_t*, fixed64_t*)>& rate,
    const std::function<void(const fixed64_t*, fixed64_t*, int, bool)>& jacobian_action, // J * dx = rate_perturb? We'll do finite diff internally.
    const fixed64_t* state, fixed64_t* state_new,
    int dim, fixed64_t dt, int max_iter=16, fixed64_t tol=1) noexcept
{
    // Initial guess: forward Euler
    forward_euler(rate, state, state_new, dim, dt);

    // Temporary arrays for residual and Jacobian (we'll use finite differences)
    fixed64_t* R = static_cast<fixed64_t*>(alloca(dim * sizeof(fixed64_t)));
    fixed64_t* rate_new = static_cast<fixed64_t*>(alloca(dim * sizeof(fixed64_t)));
    fixed64_t* rate_pert = static_cast<fixed64_t*>(alloca(dim * sizeof(fixed64_t)));
    fixed64_t* dx = static_cast<fixed64_t*>(alloca(dim * sizeof(fixed64_t)));
    fixed64_t perturbation = FIXED64_ONE / 1000; // ~1e-3 fixed point

    for (int iter=0; iter<max_iter; ++iter) {
        // Compute rate at current guess
        rate(state_new, rate_new);
        // Residual R = state_new - state - dt*rate_new
        for (int i=0; i<dim; ++i) {
            R[i] = state_new[i] - state[i] - fixed_mul(rate_new[i], dt);
        }
        // Check convergence
        fixed64_t norm_R = 0;
        for (int i=0; i<dim; ++i) norm_R += fixed_mul(R[i], R[i]);
        norm_R = fixed_sqrt(norm_R);
        if (norm_R < tol) return true;

        // Finite-difference Jacobian solve via iterative method (simplified: use diagonal approximation)
        // We'll solve J * dx = -R using a simple Jacobi iteration since dim is small.
        // Compute approximate Jacobian diagonal by perturbing each component
        fixed64_t diag[16]; // assume dim <= 16
        for (int i=0; i<dim; ++i) {
            fixed64_t save = state_new[i];
            state_new[i] += perturbation;
            rate(state_new, rate_pert);
            state_new[i] = save;
            // J_ii ≈ (rate_pert[i] - rate_new[i]) / perturbation
            fixed64_t drate = rate_pert[i] - rate_new[i];
            diag[i] = FIXED64_ONE - fixed_mul(dt, fixed_div(drate, perturbation));
            if (diag[i] == 0) diag[i] = FIXED64_ONE;
        }
        // Solve dx = -R / diag (Jacobi step)
        for (int i=0; i<dim; ++i) {
            dx[i] = -fixed_div(R[i], diag[i]);
            state_new[i] += dx[i];
        }
    }
    return false; // not converged
}

// ---------------------------------------------------------------------------
// Linear viscoelastic Maxwell model: stress evolution
//   dσ/dt = C : dε/dt - σ / τ   (τ = relaxation time)
//   rate = C : strain_rate - sigma / tau
// ---------------------------------------------------------------------------
struct MaxwellViscoelastic {
    StiffnessMatrix6x6 C;  // elastic stiffness
    fixed64_t tau;          // relaxation time

    void stress_rate(const fmat3& strain_rate, const fmat3& sigma, fmat3& dsigma_dt) const noexcept {
        fmat3 elastic_stress_rate = apply_stiffness(C, strain_rate);
        // -sigma / tau
        fmat3 relaxation = fmat3_mul_scalar(sigma, -fixed_rcp(tau));
        // dsigma_dt = elastic_stress_rate + relaxation
        for (int r=0; r<3; ++r)
            for (int c=0; c<3; ++c)
                *(&dsigma_dt.rows[0].x + r*3 + c) = *(&elastic_stress_rate.rows[0].x + r*3 + c)
                                                    + *(&relaxation.rows[0].x + r*3 + c);
    }

    // Forward Euler stress update
    void update_forward(const fmat3& strain_inc, fmat3& sigma, fixed64_t dt) const noexcept {
        fmat3 strain_rate = fmat3_mul_scalar(strain_inc, fixed_rcp(dt));
        fmat3 dsigma;
        stress_rate(strain_rate, sigma, dsigma);
        for (int r=0; r<3; ++r)
            for (int c=0; c<3; ++c)
                *(&sigma.rows[0].x + r*3 + c) += fixed_mul(*(&dsigma.rows[0].x + r*3 + c), dt);
    }
};

// ---------------------------------------------------------------------------
// Kelvin-Voigt viscoelastic: σ = C : ε + η * C : dε/dt
//   stress = elastic + viscous contribution
// ---------------------------------------------------------------------------
struct KelvinVoigtViscoelastic {
    StiffnessMatrix6x6 C;
    fixed64_t eta;  // viscosity

    void compute_stress(const fmat3& strain, const fmat3& strain_rate, fmat3& sigma) const noexcept {
        fmat3 elastic_stress = apply_stiffness(C, strain);
        fmat3 viscous_stress = fmat3_mul_scalar(apply_stiffness(C, strain_rate), eta);
        for (int r=0; r<3; ++r)
            for (int c=0; c<3; ++c)
                *(&sigma.rows[0].x + r*3 + c) = *(&elastic_stress.rows[0].x + r*3 + c) + *(&viscous_stress.rows[0].x + r*3 + c);
    }
};

// ---------------------------------------------------------------------------
// Perzyna viscoplasticity: dε_vp/dt = γ * <φ(F)> * ∂F/∂σ
//   where φ(F) = (F / σ_y)^N, and F = σ_eq - σ_y
// ---------------------------------------------------------------------------
struct PerzynaViscoplastic {
    fixed64_t gamma;     // fluidity parameter
    fixed64_t exponent_N; // rate sensitivity exponent
    fixed64_t sigma_y;   // yield stress (constant for simplicity, could be hardening)

    // Compute viscoplastic strain rate given stress sigma
    void plastic_strain_rate(const fmat3& sigma, fmat3& rate) const noexcept {
        fmat3 s = deviatoric(sigma);
        fixed64_t sigma_eq = von_mises_eq_stress(sigma);
        fixed64_t F = sigma_eq - sigma_y;
        if (F <= 0) {
            // no plastic flow
            for (int r=0; r<3; ++r) for (int c=0; c<3; ++c) *(&rate.rows[0].x + r*3 + c) = 0;
            return;
        }
        fixed64_t phi = fixed_pow(fixed_div(F, sigma_y), exponent_N);
        fixed64_t flow_factor = fixed_mul(gamma, phi);
        // flow direction = 0.5 * s / sigma_eq (associative)
        if (sigma_eq == 0) {
            for (int r=0; r<3; ++r) for (int c=0; c<3; ++c) *(&rate.rows[0].x + r*3 + c) = 0;
            return;
        }
        for (int r=0; r<3; ++r)
            for (int c=0; c<3; ++c)
                *(&rate.rows[0].x + r*3 + c) = fixed_mul(flow_factor, fixed_div(*(&s.rows[0].x + r*3 + c), 2 * sigma_eq));
    }

    // Forward Euler: total strain = elastic + plastic, elastic = total - plastic
    void update_forward(const fmat3& total_strain_inc, fmat3& sigma, fmat3& plastic_strain, fixed64_t dt) const noexcept {
        // trial elastic stress = C : (total_strain_inc - plastic_strain_inc)
        fmat3 plastic_rate;
        plastic_strain_rate(sigma, plastic_rate);
        fmat3 plastic_inc = fmat3_mul_scalar(plastic_rate, dt);
        // update plastic strain
        plastic_strain = fmat3_add(plastic_strain, plastic_inc);
        // effective elastic strain = total_strain_inc - plastic_inc
        fmat3 elastic_inc;
        for (int r=0; r<3; ++r) for (int c=0; c<3; ++c)
            *(&elastic_inc.rows[0].x + r*3 + c) = *(&total_strain_inc.rows[0].x + r*3 + c) - *(&plastic_inc.rows[0].x + r*3 + c);
        // stress increment = C : elastic_inc
        fmat3 stress_inc = apply_stiffness(C, elastic_inc); // need C member; we'll add later
        sigma = fmat3_add(sigma, stress_inc);
    }
};

// ---------------------------------------------------------------------------
// Lemaitre damage model: effective stress = σ / (1 - D), damage evolution D rate = (Y/S)^s * p_dot
//   where Y = -1/2 σ : C^-1 : σ / (1-D)^2 (strain energy release rate)
// ---------------------------------------------------------------------------
struct LemaitreDamage {
    fixed64_t S; // damage strength
    fixed64_t s; // damage exponent
    fixed64_t Dc; // critical damage
    StiffnessMatrix6x6 compliance; // C^-1

    // Compute damage rate given stress sigma and plastic strain rate p_dot (scalar)
    fixed64_t damage_rate(const fmat3& sigma, fixed64_t D, fixed64_t p_dot) const noexcept {
        if (D >= Dc) return 0;
        // Y = -1/(2*(1-D)^2) * σ : S : σ
        fixed64_t one_minus_D = FIXED64_ONE - D;
        fixed64_t inv_one_minus_D2 = fixed_div(FIXED64_ONE, fixed_mul(one_minus_D, one_minus_D));
        fixed64_t Y = fixed_mul(-FIXED64_HALF, inv_one_minus_D2);
        Y = fixed_mul(Y, double_contraction(sigma, apply_compliance(compliance, sigma)));
        if (Y < 0) Y = 0;
        // D_dot = (Y/S)^s * p_dot
        fixed64_t term = fixed_pow(fixed_div(Y, S), s);
        return fixed_mul(term, p_dot);
    }
};

// ---------------------------------------------------------------------------
// General explicit integrator for stress and internal variables
//   dσ/dt = C : (dε/dt - dε_p/dt),  dκ/dt = h(σ, κ)
// ---------------------------------------------------------------------------
template<typename PlasticModel>
inline void explicit_constitutive_integrate(
    const fmat3& strain_inc, fixed64_t dt,
    PlasticModel& model,
    fmat3& sigma, fmat3& plastic_strain, fixed64_t& kappa) noexcept
{
    // Evaluate plastic strain rate and hardening rate at current state
    fmat3 plastic_rate;
    fixed64_t kappa_rate;
    model.plastic_rate(sigma, kappa, plastic_rate, kappa_rate);

    // Plastic strain increment
    fmat3 plastic_inc = fmat3_mul_scalar(plastic_rate, dt);
    plastic_strain = fmat3_add(plastic_strain, plastic_inc);

    // Hardening variable increment
    kappa += fixed_mul(kappa_rate, dt);

    // Elastic strain increment
    fmat3 elastic_inc;
    for (int r=0; r<3; ++r)
        for (int c=0; c<3; ++c)
            *(&elastic_inc.rows[0].x + r*3 + c) = *(&strain_inc.rows[0].x + r*3 + c) - *(&plastic_inc.rows[0].x + r*3 + c);

    // Stress increment using elasticity stiffness
    fmat3 stress_inc = apply_stiffness(model.elastic_stiffness, elastic_inc);
    sigma = fmat3_add(sigma, stress_inc);
}

// ---------------------------------------------------------------------------
// SIMD batch integrator for 4 states simultaneously (calls scalar)
// ---------------------------------------------------------------------------
inline void explicit_constitutive_batch(
    const fmat3 strain_inc[4], fixed64_t dt,
    const StiffnessMatrix6x6& C,
    const fmat3 plastic_rate[4],
    fmat3 sigma[4], fmat3 plastic_strain[4]) noexcept
{
    for (int i=0; i<4; ++i) {
        fmat3 plastic_inc = fmat3_mul_scalar(plastic_rate[i], dt);
        plastic_strain[i] = fmat3_add(plastic_strain[i], plastic_inc);
        fmat3 elastic_inc;
        for (int r=0; r<3; ++r)
            for (int c=0; c<3; ++c)
                *(&elastic_inc.rows[0].x + r*3 + c) = *(&strain_inc[i].rows[0].x + r*3 + c) - *(&plastic_inc.rows[0].x + r*3 + c);
        fmat3 stress_inc = apply_stiffness(C, elastic_inc);
        sigma[i] = fmat3_add(sigma[i], stress_inc);
    }
}

} // namespace fixed_math