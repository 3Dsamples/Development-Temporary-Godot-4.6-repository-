// system name : Octree Spatial Master
//File 0009 : core/math/fixed_sph_kernel.h
//SPH kernel functions (Poly6, Spiky, Viscosity) in fixed‑point with scalar and SIMD 4‑lane evaluations for smooth particle hydrodynamics
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <algorithm>

namespace fixed_math {

// ============================================================================
// SPH Kernel normalisation constants for 3D (Poly6, Spiky, Viscosity)
// ============================================================================

// Pre‑computed scaling factors (Q32.32) – note: h must be provided externally and normalised
// Poly6: W(r,h) = 315/(64*pi*h^9) * (h^2 - r^2)^3   for r in [0,h]
// For fixed‑point we compute (h^2 - r^2)^3 and multiply by a global kernel_scale.
// The caller must pre‑multiply kernel_scale = 315/(64*pi*h^9) in fixed‑point.
// Below, we provide the unnormalised polynomial values.

// Poly6 kernel value (unnormalised) – returns (h² - r²)³
inline fixed64_t sph_poly6_unnorm(fixed64_t r2, fixed64_t h2) noexcept {
    if (r2 >= h2) return 0;
    fixed64_t diff = h2 - r2;
    fixed64_t diff2 = fixed_mul(diff, diff);
    return fixed_mul(diff2, diff);
}

// Gradient magnitude of Poly6 (unnormalised): |∇W| = -6 * r * (h² - r²)²
inline fixed64_t sph_poly6_grad_unnorm(fixed64_t r, fixed64_t r2, fixed64_t h2) noexcept {
    if (r2 >= h2 || r == 0) return 0;
    fixed64_t diff = h2 - r2;
    fixed64_t diff_sq = fixed_mul(diff, diff);
    return fixed_mul(-6 * FIXED64_ONE * r, diff_sq);
}

// Laplacian of Poly6 (unnormalised): ΔW = -945/(32π h⁹) * (h² - r²)*(3h² - 7r²)
// Returns the numerator: (h² - r²) * (3h² - 7r²)
inline fixed64_t sph_poly6_laplacian_unnorm(fixed64_t r2, fixed64_t h2) noexcept {
    if (r2 >= h2) return 0;
    fixed64_t term1 = h2 - r2;
    fixed64_t term2 = fixed_mul(3 * FIXED64_ONE, h2) - fixed_mul(7 * FIXED64_ONE, r2);
    return fixed_mul(term1, term2);
}

// ---------------------------------------------------------------------------
// Spiky kernel (pressure gradient): W_spiky(r,h) = 15/(pi*h^6) * (h - r)^3
// Gradient magnitude: -45/(pi*h^6) * (h - r)^2
// We provide unnormalised versions.
// ---------------------------------------------------------------------------
inline fixed64_t sph_spiky_unnorm(fixed64_t r, fixed64_t h) noexcept {
    if (r >= h) return 0;
    fixed64_t diff = h - r;
    fixed64_t diff2 = fixed_mul(diff, diff);
    return fixed_mul(diff2, diff);
}

inline fixed64_t sph_spiky_grad_mag_unnorm(fixed64_t r, fixed64_t h) noexcept {
    if (r >= h) return 0;
    fixed64_t diff = h - r;
    return fixed_mul(diff, diff); // (h-r)^2, gradient direction is -r_ij/r
}

// ---------------------------------------------------------------------------
// Viscosity kernel (often uses the Laplacian of the smoothing kernel)
// ΔW(r) = 45/(π h^6) * (h - r)
// We provide unnormalised: (h - r)
// ---------------------------------------------------------------------------
inline fixed64_t sph_visc_laplacian_unnorm(fixed64_t r, fixed64_t h) noexcept {
    if (r >= h) return 0;
    return h - r;
}

// ============================================================================
// Complete kernel evaluation (requires caller to multiply by precomputed normalisation)
// ============================================================================

// Full Poly6 value: kernel_scale * (h^2 - r^2)^3
inline fixed64_t sph_poly6(fixed64_t r2, fixed64_t h2, fixed64_t kernel_scale) noexcept {
    return fixed_mul(kernel_scale, sph_poly6_unnorm(r2, h2));
}

// Full Spiky gradient magnitude: kernel_scale * (h - r)^2
inline fixed64_t sph_spiky_grad_mag(fixed64_t r, fixed64_t h, fixed64_t kernel_scale) noexcept {
    return fixed_mul(kernel_scale, sph_spiky_grad_mag_unnorm(r, h));
}

// ============================================================================
// SIMD 4‑lane SPH kernels (evaluate 4 particle pairs at once)
// ============================================================================

// 4‑lane Poly6 unnormalised
inline __m256i simd4_poly6_unnorm(__m256i r2, __m256i h2) noexcept {
    __m256i diff = simd4_sub_epi64(h2, r2);
    // clamp diff to >=0 via blend (if diff < 0, set diff = 0)
    __m256i zero = _mm256_setzero_si256();
    __m256i neg_mask = _mm256_cmpgt_epi64(zero, diff);
    __m256i clamped = _mm256_blendv_epi8(diff, zero, neg_mask);
    __m256i clamped2 = simd4_mul_epi64(clamped, clamped);
    return simd4_mul_epi64(clamped2, clamped);
}

// 4‑lane Spiky gradient magnitude unnormalised
inline __m256i simd4_spiky_grad_mag_unnorm(__m256i r, __m256i h) noexcept {
    __m256i diff = simd4_sub_epi64(h, r);
    __m256i zero = _mm256_setzero_si256();
    __m256i neg_mask = _mm256_cmpgt_epi64(zero, diff);
    __m256i clamped = _mm256_blendv_epi8(diff, zero, neg_mask);
    return simd4_mul_epi64(clamped, clamped);
}

// 4‑lane Laplacian unnormalised (viscosity)
inline __m256i simd4_visc_laplacian_unnorm(__m256i r, __m256i h) noexcept {
    __m256i diff = simd4_sub_epi64(h, r);
    __m256i zero = _mm256_setzero_si256();
    __m256i neg_mask = _mm256_cmpgt_epi64(zero, diff);
    return _mm256_blendv_epi8(diff, zero, neg_mask);
}

// ============================================================================
// Utility: compute normalisation constants from h in fixed‑point
// ============================================================================
inline fixed64_t sph_poly6_norm(fixed64_t h) noexcept {
    // 315 / (64 * pi * h^9)  -> we approximate by using integer arithmetic and division
    // Since fixed division of pi and high powers is costly, we pre‑compute a lookup?
    // We'll compute using double and convert back, preserving fixed accuracy.
    double hd = double_from_fixed(h);
    double norm_d = 315.0 / (64.0 * 3.14159265358979323846 * pow(hd, 9));
    return fixed_from_double(norm_d);
}

inline fixed64_t sph_spiky_norm(fixed64_t h) noexcept {
    double hd = double_from_fixed(h);
    double norm_d = 15.0 / (3.14159265358979323846 * pow(hd, 6));
    return fixed_from_double(norm_d);
}

inline fixed64_t sph_visc_norm(fixed64_t h) noexcept {
    double hd = double_from_fixed(h);
    double norm_d = 45.0 / (3.14159265358979323846 * pow(hd, 6));
    return fixed_from_double(norm_d);
}

// ============================================================================
// SPH pressure force helper (for completeness, using vector operations)
// ============================================================================
inline fvec3 sph_pressure_force(const fvec3& pi, const fvec3& pj,
                                fixed64_t mi, fixed64_t mj,
                                fixed64_t rho_i, fixed64_t rho_j,
                                fixed64_t press_i, fixed64_t press_j,
                                const fvec3& grad_W) noexcept {
    fixed64_t coeff = fixed_mul(mj, fixed_add(fixed_div(press_i, fixed_mul(rho_i, rho_i)),
                                              fixed_div(press_j, fixed_mul(rho_j, rho_j))));
    return fvec3_scale(grad_W, -coeff);
}

} // namespace fixed_math

// End of File 0009
// Next file: File 0010 – core/math/fixed_color.h
// Description: Fixed‑point colour spaces: linear sRGB ↔ OkLab conversion, luminance extraction, and perceptual colour blending using unified conversions.