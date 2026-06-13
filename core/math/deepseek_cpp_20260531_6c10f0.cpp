// SPDX-FileCopyrightText: Copyright (c) 2025 – C++17 Math/Physics Library
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <cfloat>
#include <cmath>
#include <limits>
#include <type_traits>

// ── Platform-independent calling convention ──
#if defined(__CUDACC__)
  #define WP_HOST_DEVICE __host__ __device__
  #define WP_DEVICE      __device__
#else
  #define WP_HOST_DEVICE
  #define WP_DEVICE
#endif

// ── Tile block dimension (tunable) ──
#ifndef WP_TILE_BLOCK_DIM
  #define WP_TILE_BLOCK_DIM 256
#endif

// ── Mathematical constants (constexpr where possible) ──
namespace wp {

template <typename T> struct Constants {
    static constexpr T pi            = T(3.14159265358979323846);
    static constexpr T two_pi        = T(6.28318530717958647692);
    static constexpr T half_pi       = T(1.57079632679489661923);
    static constexpr T quarter_pi    = T(0.78539816339744830962);
    static constexpr T e             = T(2.71828182845904523536);
    static constexpr T sqrt_two      = T(1.41421356237309504880);
    static constexpr T inv_sqrt_two  = T(0.70710678118654752440);
    static constexpr T sqrt_three    = T(1.73205080756887729353);
    static constexpr T deg_to_rad    = T(0.01745329251994329577);
    static constexpr T rad_to_deg    = T(57.29577951308232087679);
    static constexpr T epsilon       = std::numeric_limits<T>::epsilon();
    static constexpr T infinity      = std::numeric_limits<T>::infinity();
    static constexpr T max           = std::numeric_limits<T>::max();
    static constexpr T min           = std::numeric_limits<T>::lowest();
};

// ── Fundamental numeric aliases (Warp‑compatible) ──
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
using str      = const char*;

// ── Half‑precision type (storage only, arithmetic via float) ──
struct half {
    uint16 u;
    constexpr half() noexcept : u(0) {}
    explicit WP_HOST_DEVICE half(float f) noexcept;
    WP_HOST_DEVICE operator float() const noexcept;
};

// ── Forward declarations of all vector / matrix / quaternion types ──
template <int Length, typename Type> struct vec_t;
template <int Rows, int Cols, typename Type> struct mat_t;
template <typename Type> struct quat_t;

// ── Core scalar math functions (C++17 constexpr where possible) ──
template <typename T> constexpr T sqr(T x) noexcept { return x * x; }
template <typename T> constexpr T cube(T x) noexcept { return x * x * x; }

template <typename T> constexpr T lerp(T a, T b, T t) noexcept {
    return a + t * (b - a);
}

template <typename T> constexpr T clamp(T x, T lo, T hi) noexcept {
    return (x < lo) ? lo : (hi < x) ? hi : x;
}

template <typename T> constexpr T sign(T x) noexcept {
    return (T(0) < x) - (x < T(0));
}

template <typename T> constexpr T smoothstep(T edge0, T edge1, T x) noexcept {
    T t = clamp((x - edge0) / (edge1 - edge0), T(0), T(1));
    return t * t * (T(3) - T(2) * t);
}

template <typename T> constexpr T radians(T deg) noexcept {
    return deg * Constants<T>::deg_to_rad;
}

template <typename T> constexpr T degrees(T rad) noexcept {
    return rad * Constants<T>::rad_to_deg;
}

// ── Safe reciprocal ──
template <typename T> constexpr T safe_rcp(T x, T eps = Constants<T>::epsilon) noexcept {
    return (std::abs(x) > eps) ? T(1) / x : T(0);
}

// ── Integer hashing for spatial grids ──
constexpr uint64 splitmix64(uint64 x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

constexpr uint32 hash(uint32 x) noexcept {
    x = ((x >> 16) ^ x) * 0x45D9F3B;
    x = ((x >> 16) ^ x) * 0x45D9F3B;
    x = (x >> 16) ^ x;
    return x;
}

} // namespace wp