// system name : Octree Spatial Master
//File 0020 : core/math/fixed_elasticity_tensor.h
//Fourth‑order elasticity tensor: stiffness/compliance matrices, isotropic/anisotropic models, Voigt mapping, SIMD batch application
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "core/math/fixed_tensor.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace fixed_math {

// ---------------------------------------------------------------------------
// Voigt notation: map 3×3 symmetric tensor to 6‑component vector
//   Order: (xx, yy, zz, yz, xz, xy) – standard mechanics
// ---------------------------------------------------------------------------
inline void symmetric_3x3_to_voigt(const fmat3& m, fixed64_t v[6]) noexcept {
    v[0] = m.rows[0].x;                // xx
    v[1] = m.rows[1].y;                // yy
    v[2] = m.rows[2].z;                // zz
    v[3] = *(&m.rows[0].x + 1*3 + 2); // yz
    v[4] = *(&m.rows[0].x + 0*3 + 2); // xz
    v[5] = *(&m.rows[0].x + 0*3 + 1); // xy
}

inline void voigt_to_symmetric_3x3(const fixed64_t v[6], fmat3& m) noexcept {
    m.rows[0] = {v[0], v[5], v[4]};  // xx, xy, xz
    m.rows[1] = {v[5], v[1], v[3]};  // xy, yy, yz
    m.rows[2] = {v[4], v[3], v[2]};  // xz, yz, zz
}

// ---------------------------------------------------------------------------
// 6×6 stiffness matrix (symmetric, 21 independent entries)
// ---------------------------------------------------------------------------
struct StiffnessMatrix6x6 {
    // stored in Voigt order: row 0..5, column 0..5, only lower triangle used
    // We'll store full 36 entries for simplicity (performance is adequate)
    fixed64_t data[6][6];

    // Set all entries to zero
    void zero() noexcept { std::memset(data, 0, sizeof(data)); }

    // Set entry (r,c) and (c,r) to maintain symmetry
    void set(int r, int c, fixed64_t v) noexcept { data[r][c] = v; data[c][r] = v; }

    // Access entry
    fixed64_t& operator()(int r, int c) noexcept { return data[r][c]; }
    const fixed64_t& operator()(int r, int c) const noexcept { return data[r][c]; }

    // Multiply stiffness matrix by Voigt strain vector to get stress vector
    void apply(const fixed64_t strain[6], fixed64_t stress[6]) const noexcept {
        for (int i = 0; i < 6; ++i) {
            fixed64_t sum = 0;
            for (int j = 0; j < 6; ++j) sum += fixed_mul(data[i][j], strain[j]);
            stress[i] = sum;
        }
    }
};

// ---------------------------------------------------------------------------
// Create isotropic stiffness matrix from Lamé constants λ and μ
//   σ_ij = λ ε_kk δ_ij + 2μ ε_ij
// ---------------------------------------------------------------------------
inline StiffnessMatrix6x6 isotropic_stiffness_from_lame(fixed64_t lambda, fixed64_t mu) noexcept {
    StiffnessMatrix6x6 C;
    C.zero();
    // Row xx: λ+2μ, λ, λ, 0,0,0
    C.set(0,0, lambda + 2*mu); C.set(0,1, lambda); C.set(0,2, lambda);
    // Row yy: λ, λ+2μ, λ, 0,0,0
    C.set(1,1, lambda + 2*mu); C.set(1,2, lambda);
    // Row zz: λ+2μ
    C.set(2,2, lambda + 2*mu);
    // Shear terms: μ on diagonal for yz, xz, xy
    C.set(3,3, mu);  // yz
    C.set(4,4, mu);  // xz
    C.set(5,5, mu);  // xy
    return C;
}

// ---------------------------------------------------------------------------
// Create isotropic stiffness matrix from Young's modulus E and Poisson's ratio ν
//   λ = Eν / ((1+ν)(1-2ν)),  μ = E / (2(1+ν))
// ---------------------------------------------------------------------------
inline StiffnessMatrix6x6 isotropic_stiffness(fixed64_t E, fixed64_t nu) noexcept {
    fixed64_t one_plus_nu = FIXED64_ONE + nu;
    fixed64_t one_minus_2nu = FIXED64_ONE - 2*nu;
    fixed64_t lambda = fixed_div(fixed_mul(E, nu), fixed_mul(one_plus_nu, one_minus_2nu));
    fixed64_t mu = fixed_div(E, 2 * one_plus_nu);
    return isotropic_stiffness_from_lame(lambda, mu);
}

// ---------------------------------------------------------------------------
// Isotropic compliance matrix (inverse of stiffness) from E and ν
//   ε_ij = (1+ν)/E σ_ij - ν/E σ_kk δ_ij
// ---------------------------------------------------------------------------
inline StiffnessMatrix6x6 isotropic_compliance(fixed64_t E, fixed64_t nu) noexcept {
    fixed64_t one_plus_nu = FIXED64_ONE + nu;
    StiffnessMatrix6x6 S;
    S.zero();
    fixed64_t factor = fixed_div(FIXED64_ONE, E);
    // Normal terms
    S.set(0,0, factor); S.set(1,1, factor); S.set(2,2, factor);
    // Poisson contraction
    S.set(0,1, -fixed_mul(nu, factor)); S.set(0,2, -fixed_mul(nu, factor));
    S.set(1,2, -fixed_mul(nu, factor));
    // Shear terms: (1+ν)/E = 1/(2μ)
    fixed64_t shear = fixed_div(one_plus_nu, E);
    S.set(3,3, shear); S.set(4,4, shear); S.set(5,5, shear);
    return S;
}

// ---------------------------------------------------------------------------
// Compute stress from strain using stiffness matrix
// ---------------------------------------------------------------------------
inline fmat3 apply_stiffness(const StiffnessMatrix6x6& C, const fmat3& strain) noexcept {
    fixed64_t v_strain[6], v_stress[6];
    symmetric_3x3_to_voigt(strain, v_strain);
    C.apply(v_strain, v_stress);
    fmat3 stress;
    voigt_to_symmetric_3x3(v_stress, stress);
    return stress;
}

// ---------------------------------------------------------------------------
// Compute strain from stress using compliance matrix
// ---------------------------------------------------------------------------
inline fmat3 apply_compliance(const StiffnessMatrix6x6& S, const fmat3& stress) noexcept {
    fixed64_t v_stress[6], v_strain[6];
    symmetric_3x3_to_voigt(stress, v_stress);
    S.apply(v_stress, v_strain);
    fmat3 strain;
    voigt_to_symmetric_3x3(v_strain, strain);
    return strain;
}

// ---------------------------------------------------------------------------
// Strain energy density from strain: W = 0.5 * ε_ij * C_ijkl * ε_kl
//   = 0.5 * ε_voigt · C · ε_voigt
// ---------------------------------------------------------------------------
inline fixed64_t strain_energy_density_stiffness(const fmat3& strain, const StiffnessMatrix6x6& C) noexcept {
    fixed64_t v_strain[6], v_stress[6];
    symmetric_3x3_to_voigt(strain, v_strain);
    C.apply(v_strain, v_stress);
    fixed64_t energy = 0;
    for (int i = 0; i < 6; ++i) energy += fixed_mul(v_strain[i], v_stress[i]);
    return fixed_mul(FIXED64_HALF, energy);
}

// ---------------------------------------------------------------------------
// SIMD batch: apply stiffness to 4 strain tensors, output 4 stress tensors (SoA)
// ---------------------------------------------------------------------------
inline void simd4_apply_stiffness(const StiffnessMatrix6x6& C,
                                  const fmat3 strains[4], fmat3 stresses[4]) noexcept {
    fixed64_t strain_voigt[4][6], stress_voigt[4][6];
    for (int i = 0; i < 4; ++i) symmetric_3x3_to_voigt(strains[i], strain_voigt[i]);
    for (int i = 0; i < 4; ++i) C.apply(strain_voigt[i], stress_voigt[i]);
    for (int i = 0; i < 4; ++i) voigt_to_symmetric_3x3(stress_voigt[i], stresses[i]);
}

} // namespace fixed_math

// End of File 0020
// Next file: File 0021 – core/math/fixed_polar_decomposition.h
// Description: Polar decomposition of 3×3 matrices into rotation and stretch, used for large‑deformation mechanics and inverse kinematics.