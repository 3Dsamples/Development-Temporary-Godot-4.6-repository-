// system name : Octree Spatial Master
//File 0034 : core/math/fixed_fracture_mechanics.h
//Cohesive zone models, crack propagation, stress intensity factors, material separation, and energy release rate using fixed‑point tensors
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "core/math/fixed_tensor.h"
#include "core/math/fixed_elasticity_tensor.h"
#include "core/math/fixed_plasticity.h"
#include "core/math/fixed_geometry.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>

namespace fixed_math {

// Cohesive zone traction‑separation law parameters
struct CohesiveLaw {
    fixed64_t sigma_max;   // peak traction
    fixed64_t delta_c;     // critical separation
    fixed64_t delta_n;     // normal separation at peak
    fixed64_t delta_t;     // tangential separation at peak
};

// Compute cohesive traction from separation vector (delta) in local crack plane coordinates
inline fvec3 cohesive_traction(const CohesiveLaw& law, const fvec3& delta) noexcept {
    fixed64_t dn = delta.z; // normal component
    fixed64_t dt = fixed_sqrt(delta.x*delta.x + delta.y*delta.y); // tangential magnitude
    if (dn < 0) return {0,0,0}; // no resistance in compression
    fixed64_t tn = 0, tt = 0;
    if (dn < law.delta_n) {
        tn = fixed_div(law.sigma_max * dn, law.delta_n);
    } else if (dn < law.delta_c) {
        tn = fixed_mul(law.sigma_max, fixed_div(law.delta_c - dn, law.delta_c - law.delta_n));
    }
    if (dt < law.delta_t) {
        tt = fixed_div(law.sigma_max * dt, law.delta_t);
    } else if (dt < law.delta_c) {
        tt = fixed_mul(law.sigma_max, fixed_div(law.delta_c - dt, law.delta_c - law.delta_t));
    }
    fixed64_t inv_dt = (dt > 0) ? fixed_rcp(dt) : 0;
    return {fixed_mul(tt, delta.x * inv_dt), fixed_mul(tt, delta.y * inv_dt), tn};
}

// Cohesive tangent stiffness matrix (3x3) for implicit integration
inline fmat3 cohesive_tangent(const CohesiveLaw& law, const fvec3& delta) noexcept {
    fmat3 K = fmat3_identity(); // initialize to zero
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) *(&K.rows[0].x + r*3 + c) = 0;
    fixed64_t dn = delta.z;
    fixed64_t dt = fixed_sqrt(delta.x*delta.x + delta.y*delta.y);
    if (dn < 0) return K; // no stiffness in compression
    // Normal direction stiffness
    if (dn < law.delta_n) K.rows[2].z = fixed_div(law.sigma_max, law.delta_n);
    else if (dn < law.delta_c) K.rows[2].z = -fixed_div(law.sigma_max, law.delta_c - law.delta_n);
    // Tangential direction (simplified isotropic tangent)
    if (dt > 0 && dt < law.delta_t) {
        fixed64_t kt = fixed_div(law.sigma_max, law.delta_t);
        fvec3 d = {delta.x, delta.y, 0};
        fmat3 nxn = tensor_product(d, d);
        fmat3 I = fmat3_identity();
        // K_t = kt * ( I - n⊗n )   (projection to tangent plane) approximated
        for (int r=0;r<2;++r) for (int c=0;c<2;++c)
            K.rows[r].c[c] = kt * ((r==c ? FIXED64_ONE : 0) - fixed_mul(d.r, d.c) / (dt*dt));
    }
    return K;
}

// Stress intensity factor KI for a through‑crack in infinite plate (2D plane strain)
inline fixed64_t stress_intensity_factor_I(fixed64_t sigma_inf, fixed64_t crack_length) noexcept {
    return fixed_mul(sigma_inf, fixed_sqrt(FIXED64_PI * crack_length));
}

// Energy release rate G for plane strain (KI)
inline fixed64_t energy_release_rate_KI(fixed64_t KI, fixed64_t E, fixed64_t nu) noexcept {
    return fixed_div(fixed_mul(KI, KI), E) * (FIXED64_ONE - fixed_mul(nu, nu));
}

// Critical fracture toughness check
inline bool fracture_criterion(fixed64_t KI, fixed64_t KIC) noexcept {
    return KI >= KIC;
}

// Calculate the J‑integral using contour integration over a path of points
inline fixed64_t j_integral(const std::vector<fvec3>& contour, const std::vector<fmat3>& stress,
                            const std::vector<fvec3>& displacement, const fvec3& crack_dir) noexcept {
    // simplified: J = ∫ ( W dy - T · ∂u/∂x ds )   W = 0.5 σ:ε
    // We'll compute using midpoint rule.
    if (contour.size() < 2) return 0;
    fixed64_t J = 0;
    for (size_t i=0; i<contour.size()-1; ++i) {
        fvec3 mid = fvec3_scale(fvec3_add(contour[i], contour[i+1]), FIXED64_HALF);
        fvec3 ds = fvec3_sub(contour[i+1], contour[i]);
        fixed64_t dy = ds.y;
        fmat3 sig = stress[i]; // average stress (simplified)
        fvec3 du = fvec3_sub(displacement[i+1], displacement[i]);
        fixed64_t W = fixed_mul(FIXED64_HALF, double_contraction(sig, sig)); // approximate
        J += fixed_mul(W, dy) - fixed_mul(fvec3_dot(fmat3_mul_vec3(sig, crack_dir), du), ds.x);
    }
    return J;
}

// Perceptual colour for damage parameter D
inline fvec3 damage_color(fixed64_t D) noexcept {
    if (D < 0) D = 0; if (D > FIXED64_ONE) D = FIXED64_ONE;
    fvec3 linear = {D, FIXED64_ONE - D, 0};
    return perceptual_color::linear_srgb_to_oklab(linear);
}

} // namespace fixed_math