// system name : Octree Spatial Master
//File 0039 : core/math/fixed_thermo_mechanics.h
//Thermo‑mechanics: thermal expansion, heat equation, thermo‑elastic/plastic, entropy, SIMD batch, perceptual colour
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "core/math/fixed_tensor.h"
#include "core/math/fixed_elasticity_tensor.h"
#include "core/math/fixed_plasticity.h"
#include "core/math/fixed_sparse_solver.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>

namespace fixed_math {

struct ThermoMaterial {
    fixed64_t E, nu;          // elastic constants
    fixed64_t alpha;          // thermal expansion coefficient (1/K)
    fixed64_t k;              // thermal conductivity (W/(m·K))
    fixed64_t rho_c;          // density * specific heat capacity
    fixed64_t yield_stress0;  // initial yield stress (for plasticity)
    fixed64_t hardening_mod;  // linear hardening
    fixed64_t softening_param;// temperature softening factor (yield = y0 * exp(-beta * T))
};

// Thermal strain tensor: ε_th = α (T - T0) * I
inline fmat3 thermal_strain(fixed64_t alpha, fixed64_t T, fixed64_t T0) noexcept {
    fixed64_t dT = T - T0;
    fixed64_t strain_val = fixed_mul(alpha, dT);
    fmat3 e;
    e.rows[0] = {strain_val, 0, 0};
    e.rows[1] = {0, strain_val, 0};
    e.rows[2] = {0, 0, strain_val};
    return e;
}

// Update temperature field using explicit finite difference (heat equation)
// T_new = T + dt * k/(rho_c) * laplacian(T) + heat_source
inline void heat_explicit_step(std::vector<fixed64_t>& T, int nx, int ny, int nz, fixed64_t h,
                               const ThermoMaterial& mat, fixed64_t dt, fixed64_t heat_source=0) noexcept {
    fixed64_t coeff = fixed_div(mat.k, mat.rho_c);
    std::vector<fixed64_t> T_new(T.size());
    for (int k=0;k<nz;++k) for (int j=0;j<ny;++j) for (int i=0;i<nx;++i) {
        int idx = (k*ny + j)*nx + i;
        fixed64_t lap = 0;
        if (i>0) lap += T[idx-1] - 2*T[idx] + (i<nx-1 ? T[idx+1] : T[idx]) - T[idx];
        else lap += (i<nx-1 ? T[idx+1] : T[idx]) - T[idx];
        if (j>0) lap += T[idx-nx] - 2*T[idx] + (j<ny-1 ? T[idx+nx] : T[idx]) - T[idx];
        else lap += (j<ny-1 ? T[idx+nx] : T[idx]) - T[idx];
        if (k>0) lap += T[idx-nx*ny] - 2*T[idx] + (k<nz-1 ? T[idx+nx*ny] : T[idx]) - T[idx];
        else lap += (k<nz-1 ? T[idx+nx*ny] : T[idx]) - T[idx];
        lap = fixed_mul(lap, fixed_rcp(h*h));
        T_new[idx] = T[idx] + fixed_mul(coeff, lap)*dt + fixed_mul(heat_source, dt);
    }
    T.swap(T_new);
}

// Thermo‑elastic stress: σ = C : (ε_total - ε_th)
inline fmat3 thermo_elastic_stress(const fmat3& strain_total, const fmat3& strain_th,
                                   const StiffnessMatrix6x6& C) noexcept {
    fmat3 strain_mech;
    for (int r=0;r<3;++r) for (int c=0;c<3;++c)
        *(&strain_mech.rows[0].x + r*3 + c) = *(&strain_total.rows[0].x + r*3 + c) - *(&strain_th.rows[0].x + r*3 + c);
    return apply_stiffness(C, strain_mech);
}

// Temperature‑dependent yield stress (exponential softening)
inline fixed64_t temp_dependent_yield(const ThermoMaterial& mat, fixed64_t T) noexcept {
    fixed64_t factor = fixed_exp(fixed_mul(-mat.softening_param, T));
    return fixed_mul(mat.yield_stress0, factor);
}

// Thermo‑plastic radial return with temperature‑dependent yield
inline bool thermo_plastic_radial_return(const fmat3& sigma_trial, fixed64_t eq_plastic_old,
                                         fixed64_t T, const ThermoMaterial& mat,
                                         const StiffnessMatrix6x6& C, fmat3& sigma,
                                         fixed64_t& eq_plastic_new) noexcept {
    fixed64_t mu = C(3,3); // shear modulus (isotropic assumption)
    fixed64_t sigma_y = temp_dependent_yield(mat, T);
    fmat3 s_trial = deviatoric(sigma_trial);
    fixed64_t sigma_eq_trial = von_mises_eq_stress(sigma_trial);
    fixed64_t f_trial = sigma_eq_trial - sigma_y;
    if (f_trial <= 0) { sigma = sigma_trial; eq_plastic_new = eq_plastic_old; return true; }
    // Newton for delta_gamma
    fixed64_t dg = fixed_div(f_trial, 3*mu + mat.hardening_mod);
    for (int iter=0;iter<8;++iter) {
        fixed64_t yield = temp_dependent_yield(mat, T) + fixed_mul(mat.hardening_mod, eq_plastic_old + dg);
        fixed64_t seq = sigma_eq_trial - 3*mu*dg;
        fixed64_t residual = seq - yield;
        if (fixed_abs(residual) < 1) break;
        fixed64_t dres = -3*mu - mat.hardening_mod;
        if (dres==0) break;
        dg -= fixed_div(residual, dres);
    }
    eq_plastic_new = eq_plastic_old + dg;
    fixed64_t factor = FIXED64_ONE - fixed_div(3*mu*dg, sigma_eq_trial);
    fmat3 s_new = fmat3_mul_scalar(s_trial, factor);
    fixed64_t p = fixed_div(stress_I1(sigma_trial), 3*FIXED64_ONE);
    sigma = s_new;
    for (int i=0;i<3;++i) *(&sigma.rows[0].x + i*3 + i) += p;
    return true;
}

// Entropy production rate from plastic dissipation: s_dot = (σ : ε_dot_p) / T
inline fixed64_t entropy_production(const fmat3& sigma, const fmat3& plastic_strain_rate, fixed64_t T) noexcept {
    fixed64_t diss = double_contraction(sigma, plastic_strain_rate);
    if (T<=0) return 0;
    return fixed_div(diss, T);
}

// Plastic dissipation (heat source): ω = β * σ : ε_dot_p
inline fixed64_t plastic_heat_source(const fmat3& sigma, const fmat3& plastic_strain_rate, fixed64_t taylor_quinney=0.9) noexcept {
    return fixed_mul(taylor_quinney, double_contraction(sigma, plastic_strain_rate));
}

// SIMD batch: compute 4 thermal strains
inline void thermal_strain_batch(const fvec3 T[4], fixed64_t alpha, fixed64_t T0, fmat3 out[4]) noexcept {
    for (int i=0;i<4;++i) out[i] = thermal_strain(alpha, T[i], T0);
}

// Perceptual colour for temperature (blue=cold, red=hot)
inline fvec3 temperature_color(fixed64_t T, fixed64_t T_min, fixed64_t T_max) noexcept {
    fixed64_t range = T_max - T_min;
    fixed64_t t = (range>0) ? fixed_div(T - T_min, range) : FIXED64_HALF;
    if (t<0) t=0; if (t>FIXED64_ONE) t=FIXED64_ONE;
    fvec3 linear = {t, 0, FIXED64_ONE - t};
    return perceptual_color::linear_srgb_to_oklab(linear);
}

} // namespace fixed_math