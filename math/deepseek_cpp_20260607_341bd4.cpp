//File group name : OrthoTree Math
//File 0050 : core/math/basic/scalar.h
//Basic scalar utilities: constants (pi, e, etc.), type traits, clamping, lerping, and compile‑time arithmetic helpers.

#ifndef ORTHOTREE_CORE_MATH_BASIC_SCALAR_H_INCLUDED
#define ORTHOTREE_CORE_MATH_BASIC_SCALAR_H_INCLUDED

#include "../../build_config.h"
#include <cmath>
#include <limits>
#include <type_traits>

namespace OrthoTree {
namespace Math {
namespace Basic {

// ============================================================================
//  Mathematical constants (constexpr, double precision)
// ============================================================================
template<typename T = double>
struct Constants {
    static constexpr T pi() noexcept { return T(3.14159265358979323846264338327950288419716939937510); }
    static constexpr T twoPi() noexcept { return T(2) * pi<T>(); }
    static constexpr T halfPi() noexcept { return pi<T>() / T(2); }
    static constexpr T quarterPi() noexcept { return pi<T>() / T(4); }
    static constexpr T e() noexcept { return T(2.71828182845904523536028747135266249775724709369995); }
    static constexpr T sqrt2() noexcept { return T(1.41421356237309504880168872420969807856967187537694); }
    static constexpr T sqrt3() noexcept { return T(1.73205080756887729352744634150587236694280525381038); }
    static constexpr T goldenRatio() noexcept { return T(1.61803398874989484820458683436563811772030917980576); }
};

// ----------------------------------------------------------------------------
//  Convenience accessors for double and float
// ----------------------------------------------------------------------------
inline constexpr double pi_d() { return Constants<double>::pi(); }
inline constexpr float pi_f() { return Constants<float>::pi(); }
inline constexpr double twoPi_d() { return Constants<double>::twoPi(); }
inline constexpr float twoPi_f() { return Constants<float>::twoPi(); }

// ============================================================================
//  Type traits for scalar arithmetic (promotion, common type)
// ============================================================================
template<typename T>
struct is_scalar : std::integral_constant<bool,
    std::is_floating_point_v<T> || std::is_integral_v<T>> {};

template<typename T>
inline constexpr bool is_scalar_v = is_scalar<T>::value;

// Promote two scalars to a common type (float, double, or long double)
template<typename T, typename U>
using promote_t = std::common_type_t<T, U>;

// ============================================================================
//  Clamping, smoothing, and interpolation (branchless)
// ============================================================================
template<typename T>
constexpr T clamp(T x, T lo, T hi) noexcept {
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

template<typename T>
constexpr T lerp(T a, T b, T t) noexcept {
    return a + (b - a) * t;
}

template<typename T>
constexpr T inverseLerp(T a, T b, T value) noexcept {
    return (value - a) / (b - a);
}

template<typename T>
constexpr T smoothStep(T edge0, T edge1, T x) noexcept {
    T t = clamp((x - edge0) / (edge1 - edge0), T(0), T(1));
    return t * t * (T(3) - T(2) * t);
}

template<typename T>
constexpr T smootherStep(T edge0, T edge1, T x) noexcept {
    T t = clamp((x - edge0) / (edge1 - edge0), T(0), T(1));
    return t * t * t * (t * (t * T(6) - T(15)) + T(10));
}

// ----------------------------------------------------------------------------
//  Sign, absolute, and integer rounding (fast)
// ----------------------------------------------------------------------------
template<typename T>
constexpr T abs(T x) noexcept { return (x < T(0)) ? -x : x; }

template<typename T>
constexpr int sign(T x) noexcept {
    return (T(0) < x) - (x < T(0));
}

inline int fastRound(float x) noexcept {
    return static_cast<int>(std::floor(x + 0.5f));
}
inline long fastRound(double x) noexcept {
    return static_cast<long>(std::floor(x + 0.5));
}

// ----------------------------------------------------------------------------
//  Comparison with epsilon (robust)
// ----------------------------------------------------------------------------
template<typename T>
bool nearlyEqual(T a, T b, T eps = std::numeric_limits<T>::epsilon()) noexcept {
    T diff = abs(a - b);
    return diff <= eps * std::max(abs(a), abs(b)) ||
           diff <= eps;
}

template<typename T>
bool nearlyZero(T x, T eps = std::numeric_limits<T>::epsilon()) noexcept {
    return abs(x) <= eps;
}

// ----------------------------------------------------------------------------
//  Power of two / integer bit utilities
// ----------------------------------------------------------------------------
template<typename Int>
constexpr bool isPowerOfTwo(Int x) noexcept {
    static_assert(std::is_integral_v<Int>);
    return (x > 0) && ((x & (x - 1)) == 0);
}

template<typename Int>
constexpr Int nextPowerOfTwo(Int x) noexcept {
    if (x <= 1) return 1;
    --x;
    for (size_t i = 1; i < sizeof(Int) * 8; i <<= 1) {
        x |= x >> i;
    }
    return ++x;
}

// ----------------------------------------------------------------------------
//  Compile‑time arithmetic (for templates)
// ----------------------------------------------------------------------------
template<typename T, T x, T y>
struct StaticMax { static constexpr T value = (x > y) ? x : y; };
template<typename T, T x, T y>
struct StaticMin { static constexpr T value = (x < y) ? x : y; };

} // namespace Basic
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_BASIC_SCALAR_H_INCLUDED