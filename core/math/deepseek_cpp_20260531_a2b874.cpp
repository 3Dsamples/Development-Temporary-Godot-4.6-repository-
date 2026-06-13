// File 0045 : core/math/fast_math.h
// High‑performance approximations for reciprocal sqrt, sin/cos, exp, log using minimax polynomials.

#pragma once

#include "constants.h"
#include <cstdint>
#include <cmath>

namespace wp {

// ── Fast reciprocal square root (Newton‑Raphson refinement) ─────────
inline float fast_rsqrt(float x) {
    const float xhalf = 0.5f * x;
    int32 i = *reinterpret_cast<int32*>(&x);
    i = 0x5f3759df - (i >> 1);
    float y = *reinterpret_cast<float*>(&i);
    y = y * (1.5f - xhalf * y * y);   // one Newton iteration
    return y;
}

inline double fast_rsqrt(double x) {
    double xhalf = 0.5 * x;
    int64 i = *reinterpret_cast<int64*>(&x);
    i = 0x5fe6eb50c7b537a9 - (i >> 1);
    double y = *reinterpret_cast<double*>(&i);
    y = y * (1.5 - xhalf * y * y);
    return y;
}

// ── Fast sine/cosine (range reduction + polynomial) ────────────────
// Uses Horner evaluation of minimax polynomial for sin(x), x in [-pi, pi].
inline float fast_sin(float x) {
    constexpr float pi = 3.14159265358979323846f;
    constexpr float inv_pi = 0.31830988618379067154f;
    constexpr float half_pi = 1.57079632679489661923f;
    // Range reduction to [-pi/2, pi/2]
    int32 k = static_cast<int32>(x * inv_pi + (x >= 0 ? 0.5f : -0.5f));
    x = x - k * pi;
    // Polynomial for sin on [-pi/2, pi/2]
    float x2 = x * x;
    float sin = x * (1.0f - x2 * (0.16666666666666666f - x2 * (0.008333333333333333f - x2 * 0.0001984126984126984f)));
    return (k & 1) ? -sin : sin;
}

inline float fast_cos(float x) {
    return fast_sin(x + 1.57079632679489661923f);  // shift by pi/2
}

// ── Fast exp (2^x) using bit manipulation ───────────────────────────
inline float fast_exp2(float x) {
    // Approximate 2^x for x in [-126, 127]
    constexpr float factor = 8388608.0f; // 2^23
    constexpr float offset = 1065353216;  // 127 * 2^23
    int32 i = static_cast<int32>(x * factor + offset);
    if (i < 0) i = 0;
    if (i > 0x7f800000) i = 0x7f800000;
    return *reinterpret_cast<float*>(&i);
}

inline float fast_exp(float x) {
    // exp(x) = 2^(x / ln(2))
    constexpr float inv_ln2 = 1.4426950408889634f;
    return fast_exp2(x * inv_ln2);
}

// ── Fast log2 using bit manipulation ────────────────────────────────
inline float fast_log2(float x) {
    // log2(x) = (exponent + mantissa approx)
    int32 i = *reinterpret_cast<int32*>(&x);
    int32 exp = (i >> 23) - 127;
    i = (i & 0x7fffff) | (127 << 23); // set mantissa to 1..2
    float y = *reinterpret_cast<float*>(&i);
    // Approximate log2(y) for y in [1,2] with a polynomial
    y = (y - 1.0f) * (1.0f - (y - 1.0f) * 0.5f); // simple quadratic
    return static_cast<float>(exp) + y;
}

inline float fast_log(float x) {
    constexpr float ln2 = 0.6931471805599453f;
    return fast_log2(x) * ln2;
}

// ── Half‑precision float conversion helpers ────────────────────────
struct half {
    uint16 u;
    half() noexcept : u(0) {}
    explicit half(float f) noexcept {
        uint32 x = *reinterpret_cast<uint32*>(&f);
        uint32 sign = (x >> 16) & 0x8000;
        int32 exp = ((x >> 23) & 0xff) - 127 + 15;
        uint32 mant = (x >> 13) & 0x3ff;
        if (exp <= 0) u = 0;
        else if (exp >= 31) u = sign | 0x7c00;
        else u = sign | (exp << 10) | mant;
    }
    operator float() const noexcept {
        uint32 sign = (u >> 15) & 0x1;
        uint32 exp  = (u >> 10) & 0x1f;
        uint32 mant = u & 0x3ff;
        if (exp == 0) return 0.0f;
        if (exp == 31) return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        uint32 f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
        return *reinterpret_cast<float*>(&f);
    }
};

} // namespace wp