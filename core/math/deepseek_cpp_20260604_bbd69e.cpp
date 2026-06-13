// system name : onetbb-warp
// File 0002 : core/math/scalar.h
// Description : High‑performance scalar math functions for real‑time simulation.

#ifndef __TBB_WARP_CORE_MATH_SCALAR_H
#define __TBB_WARP_CORE_MATH_SCALAR_H

#include "core/math/constants.h"
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <algorithm>
#include <initializer_list>
#include <functional>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Min / Max / Clamp
// ============================================================

template<typename T>
constexpr const T& min(const T& a, const T& b) noexcept { return (b < a) ? b : a; }

template<typename T, typename... Args>
constexpr const T& min(const T& a, const T& b, const Args&... args) noexcept {
    return min(min(a, b), args...);
}

template<typename T>
constexpr const T& max(const T& a, const T& b) noexcept { return (a < b) ? b : a; }

template<typename T, typename... Args>
constexpr const T& max(const T& a, const T& b, const Args&... args) noexcept {
    return max(max(a, b), args...);
}

template<typename T>
constexpr T clamp(const T& value, const T& low, const T& high) noexcept {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

// ============================================================
// Sign and step
// ============================================================

template<typename T>
constexpr int sign(const T& val) noexcept {
    return (T(0) < val) - (val < T(0));
}

template<typename T>
constexpr T signum(const T& val) noexcept {
    return (val > T(0)) ? T(1) : ((val < T(0)) ? T(-1) : T(0));
}

template<typename T>
constexpr T step(const T& edge, const T& x) noexcept {
    return (x < edge) ? T(0) : T(1);
}

template<typename T>
constexpr T select(bool condition, const T& a, const T& b) noexcept {
    return condition ? a : b;
}

// ============================================================
// Linear interpolation and remap
// ============================================================

template<typename T, typename U>
constexpr T lerp(const T& a, const T& b, U t) noexcept {
    return a + static_cast<T>((b - a) * t);
}

template<typename T>
constexpr T remap(const T& value, const T& in_min, const T& in_max,
                  const T& out_min, const T& out_max) noexcept {
    return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min);
}

template<typename T>
constexpr T smoothstep(const T& edge0, const T& edge1, const T& x) noexcept {
    T t = clamp((x - edge0) / (edge1 - edge0), T(0), T(1));
    return t * t * (T(3) - T(2) * t);
}

template<typename T>
constexpr T smootherstep(const T& edge0, const T& edge1, const T& x) noexcept {
    T t = clamp((x - edge0) / (edge1 - edge0), T(0), T(1));
    return t * t * t * (t * (t * T(6) - T(15)) + T(10));
}

// ============================================================
// Cubic Hermite interpolation
// ============================================================

template<typename T>
constexpr T hermite(const T& p0, const T& m0, const T& p1, const T& m1, T t) noexcept {
    T t2 = t * t;
    T t3 = t2 * t;
    return (T(2) * t3 - T(3) * t2 + T(1)) * p0 +
           (t3 - T(2) * t2 + t) * m0 +
           (T(-2) * t3 + T(3) * t2) * p1 +
           (t3 - t2) * m1;
}

// ============================================================
// Angle utilities
// ============================================================

template<typename T>
constexpr T degrees_to_radians(T degrees) noexcept { return degrees * T(PI_D / 180.0); }

template<typename T>
constexpr T radians_to_degrees(T radians) noexcept { return radians * T(180.0 / PI_D); }

template<typename T>
constexpr T wrap_angle(T angle, T period = T(TAU_D)) noexcept {
    angle = std::fmod(angle, period);
    if (angle < T(0)) angle += period;
    return angle;
}

template<typename T>
constexpr T normalize_angle(T angle) noexcept {
    angle = std::fmod(angle, T(TAU_D));
    if (angle > T(PI_D)) angle -= T(TAU_D);
    else if (angle < -T(PI_D)) angle += T(TAU_D);
    return angle;
}

// ============================================================
// Integer math
// ============================================================

template<typename T>
constexpr bool is_power_of_two(T value) noexcept {
    static_assert(std::is_integral_v<T>, "is_power_of_two requires integral type");
    return value && !(value & (value - 1));
}

template<typename T>
constexpr T next_power_of_two(T value) noexcept {
    static_assert(std::is_integral_v<T>, "next_power_of_two requires integral type");
    if (value <= 0) return 1;
    --value;
    for (int i = 1; i < static_cast<int>(sizeof(T) * 8); i <<= 1)
        value |= value >> i;
    return value + 1;
}

template<typename T>
constexpr T align_up(T value, std::size_t alignment) noexcept {
    static_assert(std::is_integral_v<T>, "align_up requires integral type");
    auto v = static_cast<std::uintptr_t>(value);
    auto a = static_cast<std::uintptr_t>(alignment);
    return static_cast<T>((v + a - 1) & ~(a - 1));
}

template<typename T>
constexpr bool is_aligned(T value, std::size_t alignment) noexcept {
    return (static_cast<std::uintptr_t>(value) & (alignment - 1)) == 0;
}

constexpr int log2(std::uint64_t x) noexcept {
    if (x == 0) return -1;
    int result = 0;
    while (x >>= 1) ++result;
    return result;
}

// ============================================================
// Fast reciprocal (Newton‑Raphson)
// ============================================================

template<typename T>
constexpr T fast_reciprocal(T x) noexcept {
    static_assert(std::is_floating_point_v<T>, "fast_reciprocal requires floating point");
    T y = T(1) / x;
    return y * (T(2) - x * y);
}

// ============================================================
// Damping / Smooth decay
// ============================================================

template<typename T>
constexpr T damp(const T& current, const T& target, T decay, T dt) noexcept {
    return current + (target - current) * (T(1) - std::exp(-decay * dt));
}

// ============================================================
// Oscillation
// ============================================================

template<typename T>
constexpr T sin_oscillate(T time, T frequency, T amplitude, T phase = T(0)) noexcept {
    return amplitude * std::sin(T(TAU_D) * frequency * time + phase);
}

template<typename T>
constexpr T damped_oscillation(T time, T frequency, T amplitude, T decay, T phase = T(0)) noexcept {
    return amplitude * std::exp(-decay * time) * std::sin(T(TAU_D) * frequency * time + phase);
}

// ============================================================
// Exponential moving average
// ============================================================

template<typename T>
constexpr T exp_moving_average(const T& current_avg, const T& new_value, T alpha) noexcept {
    return alpha * new_value + (T(1) - alpha) * current_avg;
}

// ============================================================
// Bezier interpolation (cubic)
// ============================================================

template<typename T>
constexpr T cubic_bezier(const T& p0, const T& p1, const T& p2, const T& p3, T t) noexcept {
    T u = T(1) - t;
    T u2 = u * u;
    T u3 = u2 * u;
    T t2 = t * t;
    T t3 = t2 * t;
    return u3 * p0 + T(3) * u2 * t * p1 + T(3) * u * t2 * p2 + t3 * p3;
}

// ============================================================
// Catmull‑Rom interpolation
// ============================================================

template<typename T>
constexpr T catmull_rom(const T& p0, const T& p1, const T& p2, const T& p3, T t, T alpha = T(0.5)) noexcept {
    T t2 = t * t;
    T t3 = t2 * t;
    return T(0.5) * ((T(2) * p1) +
                     (-p0 + p2) * t +
                     (T(2) * p0 - T(5) * p1 + T(4) * p2 - p3) * t2 +
                     (-p0 + T(3) * p1 - T(3) * p2 + p3) * t3);
}

// ============================================================
// Factorial and binomial (constexpr when possible)
// ============================================================

constexpr std::uint64_t factorial(int n) noexcept {
    std::uint64_t result = 1;
    for (int i = 2; i <= n; ++i) result *= i;
    return result;
}

constexpr std::uint64_t binomial(int n, int k) noexcept {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    k = std::min(k, n - k);
    std::uint64_t result = 1;
    for (int i = 1; i <= k; ++i) {
        result *= (n - k + i);
        result /= i;
    }
    return result;
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_SCALAR_H