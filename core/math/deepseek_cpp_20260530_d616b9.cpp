// File 0001 : core/math/constants.h
// Mathematical constants, scalar utility functions, and fundamental type aliases.

#pragma once

#include <cstdint>
#include <cfloat>
#include <cmath>
#include <limits>
#include <type_traits>

namespace wp {

// ---------- floating point constants ----------
template <typename T> struct MathConst {};

template <> struct MathConst<float> {
    static constexpr float pi            = 3.14159265358979323846f;
    static constexpr float two_pi        = 6.28318530717958647692f;
    static constexpr float half_pi       = 1.57079632679489661923f;
    static constexpr float quarter_pi    = 0.78539816339744830962f;
    static constexpr float e             = 2.71828182845904523536f;
    static constexpr float sqrt_two      = 1.41421356237309504880f;
    static constexpr float inv_sqrt_two  = 0.70710678118654752440f;
    static constexpr float sqrt_three    = 1.73205080756887729353f;
    static constexpr float deg_to_rad    = 0.01745329251994329577f;
    static constexpr float rad_to_deg    = 57.29577951308232087679f;
    static constexpr float epsilon       = 1.192092896e-07f;
    static constexpr float infinity      = std::numeric_limits<float>::infinity();
    static constexpr float max           = std::numeric_limits<float>::max();
    static constexpr float min           = std::numeric_limits<float>::lowest();
};

template <> struct MathConst<double> {
    static constexpr double pi            = 3.14159265358979323846;
    static constexpr double two_pi        = 6.28318530717958647692;
    static constexpr double half_pi       = 1.57079632679489661923;
    static constexpr double quarter_pi    = 0.78539816339744830962;
    static constexpr double e             = 2.71828182845904523536;
    static constexpr double sqrt_two      = 1.41421356237309504880;
    static constexpr double inv_sqrt_two  = 0.70710678118654752440;
    static constexpr double sqrt_three    = 1.73205080756887729353;
    static constexpr double deg_to_rad    = 0.01745329251994329577;
    static constexpr double rad_to_deg    = 57.29577951308232087679;
    static constexpr double epsilon       = 2.2204460492503131e-16;
    static constexpr double infinity      = std::numeric_limits<double>::infinity();
    static constexpr double max           = std::numeric_limits<double>::max();
    static constexpr double min           = std::numeric_limits<double>::lowest();
};

// Convenient aliases
template <typename T> constexpr T pi = MathConst<T>::pi;
template <typename T> constexpr T two_pi = MathConst<T>::two_pi;
template <typename T> constexpr T half_pi = MathConst<T>::half_pi;
template <typename T> constexpr T quarter_pi = MathConst<T>::quarter_pi;
template <typename T> constexpr T e_const = MathConst<T>::e;
template <typename T> constexpr T sqrt_two = MathConst<T>::sqrt_two;
template <typename T> constexpr T inv_sqrt_two = MathConst<T>::inv_sqrt_two;
template <typename T> constexpr T sqrt_three = MathConst<T>::sqrt_three;
template <typename T> constexpr T deg_to_rad = MathConst<T>::deg_to_rad;
template <typename T> constexpr T rad_to_deg = MathConst<T>::rad_to_deg;
template <typename T> constexpr T epsilon = MathConst<T>::epsilon;
template <typename T> constexpr T infinity = MathConst<T>::infinity;
template <typename T> constexpr T max_val = MathConst<T>::max;
template <typename T> constexpr T min_val = MathConst<T>::min;

// ---------- fundamental numeric types ----------
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

// ---------- scalar utility functions ----------
template <typename T> constexpr T sqr(T x) noexcept { return x * x; }
template <typename T> constexpr T cube(T x) noexcept { return x * x * x; }
template <typename T> constexpr T lerp(T a, T b, T t) noexcept { return a + t * (b - a); }
template <typename T> constexpr T clamp(T x, T lo, T hi) noexcept { return (x < lo) ? lo : (hi < x) ? hi : x; }
template <typename T> constexpr T sign(T x) noexcept { return (T(0) < x) - (x < T(0)); }
template <typename T> constexpr T smoothstep(T edge0, T edge1, T x) noexcept {
    T t = clamp((x - edge0) / (edge1 - edge0), T(0), T(1));
    return t * t * (T(3) - T(2) * t);
}
template <typename T> constexpr T radians(T deg) noexcept { return deg * MathConst<T>::deg_to_rad; }
template <typename T> constexpr T degrees(T rad) noexcept { return rad * MathConst<T>::rad_to_deg; }
template <typename T> constexpr T safe_rcp(T x, T eps = MathConst<T>::epsilon) noexcept { return (std::abs(x) > eps) ? T(1) / x : T(0); }

// ---------- integer hashing ----------
constexpr uint64 splitmix64(uint64 x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
constexpr uint32 hash_uint32(uint32 x) noexcept {
    x = ((x >> 16) ^ x) * 0x45D9F3B;
    x = ((x >> 16) ^ x) * 0x45D9F3B;
    x = (x >> 16) ^ x;
    return x;
}

} // namespace wp