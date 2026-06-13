// system name : Octree Spatial Master
//File 0003 : core/math/fixed_exp_log.h
//Fixed‑point exponential, logarithm, power, and hyperbolic functions via minimax polynomials, with SIMD 4‑lane support
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_trig.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <algorithm>

namespace fixed_math {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr fixed64_t LN2       = 0x0B17217F7D1CF7LL; // ln(2) in Q32.32
constexpr fixed64_t INV_LN2   = 0x171547652B82FELL; // 1/ln(2)
constexpr fixed64_t LOG2_E    = 0x171547652B82FELL; // log2(e)
constexpr fixed64_t ONE_THIRD = FIXED64_ONE / 3;
constexpr fixed64_t ONE_SIXTH = FIXED64_ONE / 6;

// ---------------------------------------------------------------------------
// Exponential function exp(x) (scalar)
//   Uses range reduction: exp(x) = 2^(x/ln2) = 2^int_part * exp(frac_part * ln2)
//   Fractional part evaluated with minimax polynomial on [0, ln2)
// ---------------------------------------------------------------------------
inline fixed64_t fixed_exp(fixed64_t x) noexcept {
    if (x == 0) return FIXED64_ONE;
    fixed64_t k = fixed_mul(x, INV_LN2);
    int64_t int_part = k >> FRAC_BITS;
    fixed64_t frac = k - (int_part << FRAC_BITS); // fractional part in Q32.32
    // Evaluate exp(frac * ln2) using polynomial: e^y = 1 + y + y^2/2 + y^3/6 + y^4/24 + y^5/120
    fixed64_t y = fixed_mul(frac, LN2) >> FRAC_BITS; // frac * ln2
    // Horner form for e^y on [0, ln2) using 5th order
    constexpr fixed64_t C5 = FIXED64_ONE / 120;
    constexpr fixed64_t C4 = FIXED64_ONE / 24;
    constexpr fixed64_t C3 = ONE_SIXTH;
    constexpr fixed64_t C2 = FIXED64_HALF;
    fixed64_t poly = C5;
    poly = fixed_mul(poly, y) + C4;
    poly = fixed_mul(poly, y) + C3;
    poly = fixed_mul(poly, y) + C2;
    poly = fixed_mul(poly, y) + FIXED64_ONE;
    poly = fixed_mul(poly, y) + FIXED64_ONE;
    // Multiply by 2^int_part
    return (int_part >= 0) ? (poly << int_part) : (poly >> (-int_part));
}

// ---------------------------------------------------------------------------
// Natural logarithm ln(x) (scalar)
//   Normalize x to [1, 2) and use series: ln(1 + y) = 2 * atanh(z) where z = y/(y+2)
//   or minimax on [1,2]. We'll use a minimax polynomial of degree 5.
// ---------------------------------------------------------------------------
inline fixed64_t fixed_log(fixed64_t x) noexcept {
    if (x <= 0) return -0x7FFFFFFFFFFFFFFFLL;
    // Normalize to [1,2)
    int m = 0;
    fixed64_t man = x;
    if (man >= FIXED64_ONE) {
        while (man >= (FIXED64_ONE << 1)) { man >>= 1; ++m; }
    } else {
        while (man < FIXED64_ONE) { man <<= 1; --m; }
    }
    // Now man in [1, 2)
    // Series: ln(man) using minimax on [1,2] for ln(t)
    fixed64_t t = man - FIXED64_ONE; // t in [0, 1)
    constexpr fixed64_t L5 = 0x1999999999999ALL; // 0.1
    constexpr fixed64_t L4 = 0x2000000000000000LL; // 0.2
    constexpr fixed64_t L3 = 0x2AAAAAAAAAAAAAALL; // 0.33333333
    constexpr fixed64_t L2 = FIXED64_HALF;
    fixed64_t poly = L5;
    poly = fixed_mul(poly, t) - L4;
    poly = fixed_mul(poly, t) + L3;
    poly = fixed_mul(poly, t) - L2;
    poly = fixed_mul(poly, t) + FIXED64_ONE;
    poly = fixed_mul(poly, t);
    return poly + (m * LN2);
}

// ---------------------------------------------------------------------------
// Power function pow(base, exp) = exp(exp * ln(base))
// ---------------------------------------------------------------------------
inline fixed64_t fixed_pow(fixed64_t base, fixed64_t exp) noexcept {
    return fixed_exp(fixed_mul(exp, fixed_log(base)));
}

// ---------------------------------------------------------------------------
// Hyperbolic functions
// ---------------------------------------------------------------------------
inline fixed64_t fixed_sinh(fixed64_t x) noexcept {
    fixed64_t e = fixed_exp(x);
    fixed64_t inv_e = fixed_rcp(e);
    return (e - inv_e) >> 1;
}
inline fixed64_t fixed_cosh(fixed64_t x) noexcept {
    fixed64_t e = fixed_exp(x);
    fixed64_t inv_e = fixed_rcp(e);
    return (e + inv_e) >> 1;
}
inline fixed64_t fixed_tanh(fixed64_t x) noexcept {
    fixed64_t e2 = fixed_exp(2 * x);
    return fixed_div(e2 - FIXED64_ONE, e2 + FIXED64_ONE);
}

// ---------------------------------------------------------------------------
// SIMD 4‑lane exponential (scalar fallback)
// ---------------------------------------------------------------------------
inline __m256i simd4_exp_epi64(__m256i x) noexcept {
    alignas(32) int64_t v[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(v), x);
    for (int i = 0; i < 4; ++i) v[i] = fixed_exp(v[i]);
    return _mm256_load_si256(reinterpret_cast<__m256i*>(v));
}

// SIMD 4‑lane natural log
inline __m256i simd4_log_epi64(__m256i x) noexcept {
    alignas(32) int64_t v[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(v), x);
    for (int i = 0; i < 4; ++i) v[i] = fixed_log(v[i]);
    return _mm256_load_si256(reinterpret_cast<__m256i*>(v));
}

// SIMD 4‑lane pow
inline __m256i simd4_pow_epi64(__m256i base, __m256i exp) noexcept {
    alignas(32) int64_t bv[4], ev[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(bv), base);
    _mm256_store_si256(reinterpret_cast<__m256i*>(ev), exp);
    for (int i = 0; i < 4; ++i) bv[i] = fixed_pow(bv[i], ev[i]);
    return _mm256_load_si256(reinterpret_cast<__m256i*>(bv));
}

} // namespace fixed_math

// End of File 0003
// Next file: File 0004 – core/math/fixed_vec3.h
// Description: 3D vector operations (add, sub, dot, cross, length, normalize) in fixed‑point with SIMD 4‑lane support and SoA helper conversions.