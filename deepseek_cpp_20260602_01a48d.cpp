// system name : Octree Spatial Master
//File 0002 : core/math/fixed_trig.h
//Fixed‑point trigonometric functions: sin, cos, atan2, tan via CORDIC, with full scalar and SIMD 4‑lane implementations
#pragma once
#include "core/math/fixed_scalar.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <array>
#include <algorithm>

namespace fixed_math {

// Precomputed atan(2^-i) in Q32.32 for i = 0..31
constexpr std::array<fixed64_t, 32> ATAN_TABLE = {{
    0x1921FB54442D18LL, 0x0ED633823E0F60LL, 0x07D6DD7E4F4C70LL,
    0x03FAB753A79290LL, 0x01FF55BB7282F0LL, 0x00FFEAAADDDB60LL,
    0x007FFD555BBBA4LL, 0x003FFFAAAB7750LL, 0x001FFFD555BBBL,
    0x000FFFAAAAB78LL,  0x0007FFFD556LL,    0x0003FFFFAAALL,
    0x0001FFFFD56LL,    0x0000FFFFFABLL,    0x00007FFFFD5LL,
    0x00003FFFFFALL,    0x00001FFFFFDLL,    0x00000FFFFFFLL,
    0x000007FFFFFLL,    0x000003FFFFFLL,    0x00000200000LL,
    0x00000100000LL,    0x00000080000LL,    0x00000040000LL,
    0x00000020000LL,    0x00000010000LL,    0x00000008000LL,
    0x00000004000LL,    0x00000002000LL,    0x00000001000LL,
    0x00000000800LL,    0x00000000400LL
}};

// CORDIC rotation (sin/cos) for a single angle
inline void cordic_rotate(fixed64_t theta, fixed64_t& cos_out, fixed64_t& sin_out) noexcept {
    constexpr fixed64_t TWO_PI = 2 * FIXED64_PI;
    theta = theta % TWO_PI;
    if (theta > FIXED64_PI) theta -= TWO_PI;
    else if (theta < -FIXED64_PI) theta += TWO_PI;

    fixed64_t x = 0x09B74EDA843C20LL; // K gain (~0.607252935)
    fixed64_t y = 0;
    fixed64_t z = theta;
    for (int i = 0; i < 32; ++i) {
        int64_t d = (z >= 0) ? 1 : -1;
        fixed64_t x_new = x - d * (y >> i);
        fixed64_t y_new = y + d * (x >> i);
        z -= d * ATAN_TABLE[i];
        x = x_new;
        y = y_new;
    }
    cos_out = x;
    sin_out = y;
}

// Sine of fixed‑point angle
inline fixed64_t fixed_sin(fixed64_t x) noexcept {
    fixed64_t c, s;
    cordic_rotate(x, c, s);
    return s;
}

// Cosine of fixed‑point angle
inline fixed64_t fixed_cos(fixed64_t x) noexcept {
    fixed64_t c, s;
    cordic_rotate(x, c, s);
    return c;
}

// Simultaneous sin and cos
inline void fixed_sincos(fixed64_t x, fixed64_t& s, fixed64_t& c) noexcept {
    cordic_rotate(x, c, s);
}

// Arctangent of y/x (vectoring mode CORDIC)
inline fixed64_t fixed_atan2(fixed64_t y, fixed64_t x) noexcept {
    if (x == 0 && y == 0) return 0;
    int64_t sign = 1;
    if (x < 0) { x = -x; sign = -sign; }
    fixed64_t z = 0;
    for (int i = 0; i < 32; ++i) {
        int64_t d = (y >= 0) ? -1 : 1;
        fixed64_t x_new = x - d * (y >> i);
        fixed64_t y_new = y + d * (x >> i);
        z -= d * ATAN_TABLE[i];
        x = x_new;
        y = y_new;
    }
    if (sign < 0) z = (z > 0 ? z - FIXED64_PI : z + FIXED64_PI);
    return z;
}

// Tangent = sin / cos
inline fixed64_t fixed_tan(fixed64_t x) noexcept {
    fixed64_t s, c;
    fixed_sincos(x, s, c);
    return (c == 0) ? (s >= 0 ? 0x7FFFFFFFFFFFFFFFLL : -0x7FFFFFFFFFFFFFFFLL)
                     : fixed_div(s, c);
}

// Arcsine via atan2
inline fixed64_t fixed_asin(fixed64_t x) noexcept {
    if (x > FIXED64_ONE || x < -FIXED64_ONE) return 0;
    fixed64_t sq = fixed_mul(x, x);
    fixed64_t den = fixed_sqrt(FIXED64_ONE - sq);
    return fixed_atan2(x, den);
}

// Arccosine via arcsine
inline fixed64_t fixed_acos(fixed64_t x) noexcept {
    return FIXED64_PI/2 - fixed_asin(x);
}

// -----------------------------------------------------------------------
// SIMD 4‑lane CORDIC rotation (sin/cos) using scalar extraction
// -----------------------------------------------------------------------
inline void simd4_cordic_rotate(__m256i theta, __m256i& cos_out, __m256i& sin_out) noexcept {
    alignas(32) int64_t th[4], c[4], s[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(th), theta);
    for (int i = 0; i < 4; ++i) {
        cordic_rotate(th[i], c[i], s[i]);
    }
    cos_out = _mm256_load_si256(reinterpret_cast<__m256i*>(c));
    sin_out = _mm256_load_si256(reinterpret_cast<__m256i*>(s));
}

// SIMD 4‑lane sin
inline __m256i simd4_sin_epi64(__m256i x) noexcept {
    __m256i c, s;
    simd4_cordic_rotate(x, c, s);
    return s;
}

// SIMD 4‑lane cos
inline __m256i simd4_cos_epi64(__m256i x) noexcept {
    __m256i c, s;
    simd4_cordic_rotate(x, c, s);
    return c;
}

// SIMD 4‑lane atan2
inline __m256i simd4_atan2_epi64(__m256i y, __m256i x) noexcept {
    alignas(32) int64_t yv[4], xv[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(yv), y);
    _mm256_store_si256(reinterpret_cast<__m256i*>(xv), x);
    for (int i = 0; i < 4; ++i) {
        yv[i] = fixed_atan2(yv[i], xv[i]);
    }
    return _mm256_load_si256(reinterpret_cast<__m256i*>(yv));
}

// SIMD 4‑lane tan
inline __m256i simd4_tan_epi64(__m256i x) noexcept {
    __m256i s = simd4_sin_epi64(x);
    __m256i c = simd4_cos_epi64(x);
    return simd4_div_epi64(s, c);
}

} // namespace fixed_math

// End of File 0002
// Next file: File 0003 – core/math/fixed_exp_log.h
// Description: Fixed‑point exponential and logarithm functions (exp, log, pow) with minimax polynomial approximations and SIMD 4‑lane support.