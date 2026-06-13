//File 0001 : core/math/fixed_scalar.h
//Fixed‑point Q32.32 scalar type with full scalar/SIMD arithmetic, conversions, and transcendental helpers
#pragma once

#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <type_traits>
#include <limits>
#include <bit>
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// Fixed‑point type and constant definitions
// ---------------------------------------------------------------------------
using fixed64_t = int64_t;
constexpr int FRAC_BITS = 32;
constexpr fixed64_t FIXED64_ONE  = 1LL << FRAC_BITS;
constexpr fixed64_t FIXED64_HALF = FIXED64_ONE >> 1;
constexpr fixed64_t FIXED64_PI   = 0x3243F6A8885A3LL; // pi in Q32.32
constexpr fixed64_t FIXED64_E    = 0x2B7E151628AED2LL; // e in Q32.32

// ---------------------------------------------------------------------------
// Scalar conversion functions
// ---------------------------------------------------------------------------
inline constexpr fixed64_t fixed_from_float(float x) noexcept {
    return static_cast<fixed64_t>(x * static_cast<float>(FIXED64_ONE));
}
inline constexpr fixed64_t fixed_from_double(double x) noexcept {
    return static_cast<fixed64_t>(x * static_cast<double>(FIXED64_ONE));
}
inline constexpr float float_from_fixed(fixed64_t x) noexcept {
    return static_cast<float>(x) / static_cast<float>(FIXED64_ONE);
}
inline constexpr double double_from_fixed(fixed64_t x) noexcept {
    return static_cast<double>(x) / static_cast<double>(FIXED64_ONE);
}

// ---------------------------------------------------------------------------
// Scalar arithmetic (exact Q32.32)
// ---------------------------------------------------------------------------
inline fixed64_t fixed_add(fixed64_t a, fixed64_t b) noexcept { return a + b; }
inline fixed64_t fixed_sub(fixed64_t a, fixed64_t b) noexcept { return a - b; }
inline fixed64_t fixed_mul(fixed64_t a, fixed64_t b) noexcept {
    __int128 prod = static_cast<__int128>(a) * static_cast<__int128>(b);
    return static_cast<fixed64_t>(prod >> FRAC_BITS);
}
inline fixed64_t fixed_div(fixed64_t a, fixed64_t b) noexcept {
    __int128 num = static_cast<__int128>(a) << FRAC_BITS;
    return static_cast<fixed64_t>(num / b);
}
inline fixed64_t fixed_abs(fixed64_t x) noexcept { return (x < 0) ? -x : x; }
inline int fixed_sign(fixed64_t x) noexcept { return (x > 0) - (x < 0); }
inline fixed64_t fixed_min(fixed64_t a, fixed64_t b) noexcept { return a < b ? a : b; }
inline fixed64_t fixed_max(fixed64_t a, fixed64_t b) noexcept { return a > b ? a : b; }
inline fixed64_t fixed_clamp(fixed64_t v, fixed64_t lo, fixed64_t hi) noexcept {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ---------------------------------------------------------------------------
// Scalar integer square root (full 64‑bit)
// ---------------------------------------------------------------------------
inline uint64_t int_sqrt(uint64_t n) noexcept {
    if (n == 0) return 0;
    uint64_t x = n;
    uint64_t y = (x + n / x) >> 1;
    while (y < x) {
        x = y;
        y = (x + n / x) >> 1;
    }
    return x;
}
inline fixed64_t fixed_sqrt(fixed64_t x) noexcept {
    if (x <= 0) return 0;
    uint64_t n = static_cast<uint64_t>(x);
    uint64_t root = int_sqrt(n);
    return static_cast<fixed64_t>(root << 16); // scales to Q32.32
}

// ---------------------------------------------------------------------------
// Scalar reciprocal and reciprocal square root
// ---------------------------------------------------------------------------
inline fixed64_t fixed_rcp(fixed64_t x) noexcept {
    return fixed_div(FIXED64_ONE, x);
}
inline fixed64_t fixed_rsqrt(fixed64_t x) noexcept {
    fixed64_t y = fixed_rcp(fixed_sqrt(x));
    // Newton‑Raphson refinement
    fixed64_t three = 3LL << FRAC_BITS;
    fixed64_t half  = FIXED64_HALF;
    y = fixed_mul(y, fixed_sub(three, fixed_mul(x, fixed_mul(y, y))));
    y = fixed_mul(y, half);
    return y;
}

// ---------------------------------------------------------------------------
// SIMD 4‑lane arithmetic (AVX2 / AVX‑512)
// ---------------------------------------------------------------------------
inline __m256i simd4_add_epi64(__m256i a, __m256i b) noexcept { return _mm256_add_epi64(a, b); }
inline __m256i simd4_sub_epi64(__m256i a, __m256i b) noexcept { return _mm256_sub_epi64(a, b); }

// Full 64×64 -> high 64 bits multiplication for Q32.32 (returns result >> 32)
inline __m256i simd4_mul_epi64(__m256i a, __m256i b) noexcept {
    __m256i a_hi = _mm256_srli_epi64(a, 32);
    __m256i a_lo = _mm256_and_si256(a, _mm256_set1_epi64x(0xFFFFFFFF));
    __m256i b_hi = _mm256_srli_epi64(b, 32);
    __m256i b_lo = _mm256_and_si256(b, _mm256_set1_epi64x(0xFFFFFFFF));
    __m256i prod_hi_hi = _mm256_mul_epu32(a_hi, b_hi);
    __m256i prod_hi_lo = _mm256_mul_epu32(a_hi, b_lo);
    __m256i prod_lo_hi = _mm256_mul_epu32(a_lo, b_hi);
    __m256i prod_lo_lo = _mm256_mul_epu32(a_lo, b_lo);
    __m256i mid_sum = _mm256_add_epi64(_mm256_slli_epi64(prod_hi_hi, 32),
                                        _mm256_add_epi64(prod_hi_lo, prod_lo_hi));
    __m256i carry = _mm256_srli_epi64(_mm256_add_epi64(prod_lo_lo, mid_sum), 32);
    return _mm256_add_epi64(_mm256_srli_epi64(prod_lo_lo, 32),
                            _mm256_add_epi64(mid_sum, carry));
}

// Division of 4 fixed64 lanes – scalar fallback
inline __m256i simd4_div_epi64(__m256i a, __m256i b) noexcept {
    alignas(32) int64_t av[4], bv[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(av), a);
    _mm256_store_si256(reinterpret_cast<__m256i*>(bv), b);
    for (int i = 0; i < 4; ++i) av[i] = fixed_div(av[i], bv[i]);
    return _mm256_load_si256(reinterpret_cast<__m256i*>(av));
}

// Square root of 4 fixed64 lanes – scalar fallback
inline __m256i simd4_sqrt_epi64(__m256i x) noexcept {
    alignas(32) int64_t v[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(v), x);
    for (int i = 0; i < 4; ++i) v[i] = fixed_sqrt(v[i]);
    return _mm256_load_si256(reinterpret_cast<__m256i*>(v));
}

// Reciprocal of 4 lanes
inline __m256i simd4_rcp_epi64(__m256i x) noexcept {
    __m256i one = _mm256_set1_epi64x(FIXED64_ONE);
    return simd4_div_epi64(one, x);
}

// ---------------------------------------------------------------------------
// Conversions between fixed64_t and SIMD double (4‑lane)
// ---------------------------------------------------------------------------
inline __m256d simd4_fixed_to_double(__m256i v) noexcept {
    __m256d d = _mm256_cvtepi64_pd(v);
    return _mm256_mul_pd(d, _mm256_set1_pd(1.0 / double(FIXED64_ONE)));
}
inline __m256i simd4_double_to_fixed(__m256d d) noexcept {
    d = _mm256_mul_pd(d, _mm256_set1_pd(double(FIXED64_ONE)));
    return _mm256_cvtpd_epi64(d);
}

// End of File 0001
// Next file: File 0002 – core/math/fixed_trig.h
// Description: Fixed‑point trigonometric functions: sin, cos, atan2, tan via CORDIC, with full scalar and SIMD 4‑lane implementations.