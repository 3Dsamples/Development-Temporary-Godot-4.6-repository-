// SPDX-FileCopyrightText: Copyright (c) 2025 – C++17 Math/Physics Library
// SPDX-License-Identifier: MIT
// core/math/constants.h : fundamental constants, type aliases and scalar utilities
#pragma once

#include <cstdint>
#include <cfloat>
#include <cmath>
#include <limits>
#include <type_traits>

#if defined(__CUDACC__)
  #define WP_HOST_DEVICE __host__ __device__
#else
  #define WP_HOST_DEVICE
#endif

namespace wp {

using float32  = float;
using float64  = double;
using int8     = std::int8_t;
using uint8    = std::uint8_t;
using int16    = std::int16_t;
using uint16   = std::uint16_t;
using int32    = std::int32_t;
using uint32   = std::uint32_t;
using int64    = std::int64_t;
using uint64   = std::uint64_t;

template <typename T> struct Constants {
    static constexpr T pi            = T(3.14159265358979323846L);
    static constexpr T two_pi        = T(6.28318530717958647692L);
    static constexpr T half_pi       = T(1.57079632679489661923L);
    static constexpr T quarter_pi    = T(0.78539816339744830962L);
    static constexpr T e             = T(2.71828182845904523536L);
    static constexpr T sqrt_two      = T(1.41421356237309504880L);
    static constexpr T inv_sqrt_two  = T(0.70710678118654752440L);
    static constexpr T sqrt_three    = T(1.73205080756887729353L);
    static constexpr T deg_to_rad    = T(0.01745329251994329577L);
    static constexpr T rad_to_deg    = T(57.29577951308232087679L);
    static constexpr T epsilon       = std::numeric_limits<T>::epsilon();
    static constexpr T infinity      = std::numeric_limits<T>::infinity();
    static constexpr T max           = std::numeric_limits<T>::max();
    static constexpr T min           = std::numeric_limits<T>::lowest();
};

// -- half precision storage (arithmetic via float)
struct half {
    uint16 u;
    constexpr half() noexcept : u(0) {}
    explicit WP_HOST_DEVICE half(float f) noexcept {
        // simple truncation for storage; real implementation uses half-float conversion
        uint32 bits; std::memcpy(&bits, &f, sizeof(bits));
        uint32 sign = (bits >> 16) & 0x8000;
        int32 exponent = static_cast<int32>((bits >> 23) & 0xFF) - 112;
        uint32 mantissa = bits & 0x007FFFFF;
        if (exponent <= 0) { /* flush to zero */ u = 0; return; }
        if (exponent >= 0x1F) { u = static_cast<uint16>(sign | 0x7C00 | ((mantissa>>13)&0x3FF)); return; }
        u = static_cast<uint16>(sign | (exponent<<10) | (mantissa>>13));
    }
    WP_HOST_DEVICE operator float() const noexcept {
        uint32 sign = (u & 0x8000) << 16;
        uint32 exponent = (u >> 10) & 0x1F;
        uint32 mantissa = u & 0x3FF;
        if (exponent == 0) { if (mantissa==0) { uint32 r = sign; float res; std::memcpy(&res, &r, 4); return res; }
                             float res = 0; return res; }
        if (exponent == 0x1F) { uint32 r = sign | 0x7F800000 | (mantissa<<13); float res; std::memcpy(&res, &r, 4); return res; }
        exponent += 112;
        uint32 bits = sign | (exponent<<23) | (mantissa<<13);
        float res;
        std::memcpy(&res, &bits, 4);
        return res;
    }
};

// -- scalar math utilities
template <typename T> constexpr T sqr(T x) noexcept { return x * x; }
template <typename T> constexpr T cube(T x) noexcept { return x * x * x; }
template <typename T> constexpr T lerp(T a, T b, T t) noexcept { return a + t * (b - a); }
template <typename T> constexpr T clamp(T x, T lo, T hi) noexcept { return (x < lo) ? lo : (hi < x) ? hi : x; }
template <typename T> constexpr T sign(T x) noexcept { return (T(0) < x) - (x < T(0)); }
template <typename T> constexpr T smoothstep(T edge0, T edge1, T x) noexcept {
    T t = clamp((x - edge0) / (edge1 - edge0), T(0), T(1));
    return t * t * (T(3) - T(2) * t);
}
template <typename T> constexpr T radians(T deg) noexcept { return deg * Constants<T>::deg_to_rad; }
template <typename T> constexpr T degrees(T rad) noexcept { return rad * Constants<T>::rad_to_deg; }
template <typename T> constexpr T safe_rcp(T x, T eps = Constants<T>::epsilon) noexcept {
    return (std::abs(x) > eps) ? T(1) / x : T(0);
}

} // namespace wp