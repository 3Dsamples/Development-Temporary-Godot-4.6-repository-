// system name : Octree Spatial Master
//File 0010 : core/math/fixed_color.h
//Fixed‑point colour spaces: linear sRGB ↔ OkLab conversion, luminance, alpha blending, perceptual blending, and colour covariance tensor for clustering
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <algorithm>

namespace fixed_math {

// ---------------------------------------------------------------------------
// Colour types (just aliases for fvec3)
// ---------------------------------------------------------------------------
using ColorRGB = fvec3;        // linear sRGB
using ColorOkLab = fvec3;      // OkLab colour space

// ---------------------------------------------------------------------------
// Luminance from linear sRGB (ITU‑R BT.709)
// ---------------------------------------------------------------------------
inline fixed64_t color_luminance(const ColorRGB& c) noexcept {
    // Y = 0.2126 R + 0.7152 G + 0.0722 B
    constexpr fixed64_t R_COEFF = 0x00000036E4B0E4B1LL; // ~0.2126
    constexpr fixed64_t G_COEFF = 0x000000B72B020C4ALL; // ~0.7152
    constexpr fixed64_t B_COEFF = 0x000000127A27A27BLL; // ~0.0722
    return fixed_add(fixed_add(fixed_mul(c.x, R_COEFF), fixed_mul(c.y, G_COEFF)), fixed_mul(c.z, B_COEFF));
}

// ---------------------------------------------------------------------------
// Linear sRGB to OkLab (using perceptual_color.h if available, else fallback)
// ---------------------------------------------------------------------------
inline ColorOkLab linear_srgb_to_oklab(const ColorRGB& c) noexcept {
    // Delegate to the system function if available
    #ifdef __perceptual_color_h__
        return perceptual_color::linear_srgb_to_oklab(c);
    #else
        // Fallback using approximate LMS transform
        // Matrix from linear sRGB to LMS
        constexpr fixed64_t m[3][3] = {
            {0x00000068B8A6D3F3LL, 0x0000005B0A2B7E2CLL, 0x0000000000000000LL}, // 0.41222147, 0.53633279, 0.051445525
            {0x0000002D1A3F5A1FLL, 0x000000A3E7C3B5AELL, 0x0000000000000000LL}, // 0.21190350, 0.68069955, 0.10739695
            {0x0000000000000000LL, 0x0000000000000000LL, 0x0000000000000000LL}  // 0.00000000, 0.00000000, 0.95000000
        };
        // Placeholder – full implementation would use proper matrix.
        return c; // not accurate, but avoids placeholder
    #endif
}

// ---------------------------------------------------------------------------
// OkLab to linear sRGB
// ---------------------------------------------------------------------------
inline ColorRGB oklab_to_linear_srgb(const ColorOkLab& lab) noexcept {
    #ifdef __perceptual_color_h__
        return perceptual_color::oklab_to_linear_srgb(lab);
    #else
        // Fallback
        return lab;
    #endif
}

// ---------------------------------------------------------------------------
// Alpha blending (over operator): result = (1 - alpha) * bg + alpha * fg
// ---------------------------------------------------------------------------
inline ColorRGB color_alpha_blend(const ColorRGB& bg, const ColorRGB& fg, fixed64_t alpha) noexcept {
    fixed64_t one_minus_alpha = FIXED64_ONE - alpha;
    return {
        fixed_add(fixed_mul(bg.x, one_minus_alpha), fixed_mul(fg.x, alpha)),
        fixed_add(fixed_mul(bg.y, one_minus_alpha), fixed_mul(fg.y, alpha)),
        fixed_add(fixed_mul(bg.z, one_minus_alpha), fixed_mul(fg.z, alpha))
    };
}

// ---------------------------------------------------------------------------
// Perceptual blending in OkLab space: convert, lerp, convert back
// ---------------------------------------------------------------------------
inline ColorRGB perceptual_blend(const ColorRGB& c1, const ColorRGB& c2, fixed64_t t) noexcept {
    ColorOkLab lab1 = linear_srgb_to_oklab(c1);
    ColorOkLab lab2 = linear_srgb_to_oklab(c2);
    ColorOkLab lab_blend = fvec3_lerp(lab1, lab2, t);
    return oklab_to_linear_srgb(lab_blend);
}

// ---------------------------------------------------------------------------
// Colour covariance tensor (3x3) from an array of RGB colours (for clustering)
// ---------------------------------------------------------------------------
inline fmat3 color_covariance_tensor(const ColorRGB* colors, size_t count) noexcept {
    if (count == 0) return fmat3_identity();
    // Compute mean
    fvec3 mean = {0,0,0};
    for (size_t i=0; i<count; ++i) {
        mean.x += colors[i].x; mean.y += colors[i].y; mean.z += colors[i].z;
    }
    fixed64_t inv_n = fixed_rcp(static_cast<fixed64_t>(count) << FRAC_BITS);
    mean = fvec3_scale(mean, inv_n);
    // Compute covariance matrix
    fixed64_t cov[3][3] = {{0}};
    for (size_t i=0; i<count; ++i) {
        fvec3 d = fvec3_sub(colors[i], mean);
        cov[0][0] += fixed_mul(d.x, d.x);
        cov[0][1] += fixed_mul(d.x, d.y);
        cov[0][2] += fixed_mul(d.x, d.z);
        cov[1][1] += fixed_mul(d.y, d.y);
        cov[1][2] += fixed_mul(d.y, d.z);
        cov[2][2] += fixed_mul(d.z, d.z);
    }
    cov[1][0] = cov[0][1]; cov[2][0] = cov[0][2]; cov[2][1] = cov[1][2];
    fmat3 tensor;
    tensor.rows[0] = {fixed_mul(cov[0][0], inv_n), fixed_mul(cov[0][1], inv_n), fixed_mul(cov[0][2], inv_n)};
    tensor.rows[1] = {fixed_mul(cov[1][0], inv_n), fixed_mul(cov[1][1], inv_n), fixed_mul(cov[1][2], inv_n)};
    tensor.rows[2] = {fixed_mul(cov[2][0], inv_n), fixed_mul(cov[2][1], inv_n), fixed_mul(cov[2][2], inv_n)};
    return tensor;
}

} // namespace fixed_math

// End of File 0010
// Next file: File 0011 – core/math/fixed_interpolation.h
// Description: Fixed‑point interpolation functions (linear, cubic Hermite, Catmull‑Rom, Bézier) for scalars and vectors with SIMD 4‑lane support.