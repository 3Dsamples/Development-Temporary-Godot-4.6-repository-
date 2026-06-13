/* MIT License Copyright (c) 2021-2026 Attila Csikós & Contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * @file common.h
 * @brief Common utilities, macros, and helper functions used throughout OrthoTree.
 *
 * This file provides portable compiler intrinsics for branch prediction,
 * unreachable code annotations, fast integer operations, and lightweight
 * debugging helpers. It also includes constexpr math utilities for compile‑time
 * computations and generic min/max/clamp functions that work with arithmetic
 * types and custom scalar types.
 *
 * All functions are header‑only and designed for maximum inlining.
 */

#ifndef ORTHOTREE_DETAIL_COMMON_H_INCLUDED
#define ORTHOTREE_DETAIL_COMMON_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include <algorithm>
#include <cstdlib>
#include <type_traits>

namespace OrthoTree {
namespace detail {

// ============================================================================
//  Branch prediction hints
// ============================================================================

/**
 * @brief Hint that a condition is likely true.
 */
#if defined(ORTHOTREE_COMPILER_GCC) || defined(ORTHOTREE_COMPILER_CLANG)
    #define ORTHOTREE_LIKELY(cond)   __builtin_expect(!!(cond), 1)
    #define ORTHOTREE_UNLIKELY(cond) __builtin_expect(!!(cond), 0)
#else
    #define ORTHOTREE_LIKELY(cond)   (cond)
    #define ORTHOTREE_UNLIKELY(cond) (cond)
#endif

// ============================================================================
//  Unreachable code annotation
// ============================================================================

/**
 * @brief Marks unreachable code (e.g., after a switch that covers all cases).
 */
#ifdef __GNUC__
    #define ORTHOTREE_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
    #define ORTHOTREE_UNREACHABLE() __assume(false)
#else
    #define ORTHOTREE_UNREACHABLE() do { } while (true)
#endif

// ============================================================================
//  Compile‑time assertions
// ============================================================================

/**
 * @brief Static assertion with a message (C++17 has static_assert with message).
 */
#define ORTHOTREE_STATIC_ASSERT(cond, msg) static_assert(cond, msg)

// ============================================================================
//  Math helpers (constexpr where possible)
// ============================================================================

/**
 * @brief Constexpr version of std::min for two values.
 */
template <typename T>
constexpr const T& min(const T& a, const T& b) noexcept {
    return (a < b) ? a : b;
}

/**
 * @brief Constexpr version of std::max.
 */
template <typename T>
constexpr const T& max(const T& a, const T& b) noexcept {
    return (a > b) ? a : b;
}

/**
 * @brief Constexpr clamp: value clamped to [low, high].
 */
template <typename T>
constexpr T clamp(const T& value, const T& low, const T& high) noexcept {
    return (value < low) ? low : ((value > high) ? high : value);
}

/**
 * @brief Linear interpolation: a + t * (b - a). Constexpr for floating types.
 */
template <typename T>
constexpr T lerp(const T& a, const T& b, T t) noexcept {
    return a + t * (b - a);
}

/**
 * @brief Inverse linear interpolation: t where value = lerp(a,b,t).
 */
template <typename T>
constexpr T invLerp(const T& a, const T& b, const T& value) noexcept {
    return (value - a) / (b - a);
}

/**
 * @brief Square of a value (x*x).
 */
template <typename T>
constexpr T sqr(const T& x) noexcept {
    return x * x;
}

/**
 * @brief Cube of a value.
 */
template <typename T>
constexpr T cube(const T& x) noexcept {
    return x * x * x;
}

/**
 * @brief Sign of a value: -1, 0, or 1.
 */
template <typename T>
constexpr int sign(T x) noexcept {
    return (T(0) < x) - (x < T(0));
}

/**
 * @brief Fast approximation of 1/sqrt(x) using Newton‑Raphson (for floats).
 *        Useful when high precision is not critical.
 */
inline float fastInvSqrt(float x) noexcept {
    float xhalf = 0.5f * x;
    int i = *(int*)&x;
    i = 0x5f3759df - (i >> 1);
    x = *(float*)&i;
    x = x * (1.5f - xhalf * x * x);
    return x;
}

// ============================================================================
//  Alignment and padding helpers
// ============================================================================

/**
 * @brief Align a value `x` upward to the nearest multiple of `alignment`.
 *        Alignment must be a power of two.
 */
template <typename T>
constexpr T alignUp(T x, size_t alignment) noexcept {
    return (x + (alignment - 1)) & ~(alignment - 1);
}

/**
 * @brief Align a value downward.
 */
template <typename T>
constexpr T alignDown(T x, size_t alignment) noexcept {
    return x & ~(alignment - 1);
}

/**
 * @brief Check if a pointer is aligned to a given alignment.
 */
template <typename T>
constexpr bool isAligned(const T* ptr, size_t alignment) noexcept {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

// ============================================================================
//  Hash combining (for custom hash maps)
// ============================================================================

/**
 * @brief Combine two hash values into one (boost::hash_combine style).
 */
inline void hashCombine(size_t& seed, size_t value) noexcept {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// ============================================================================
//  Tag dispatching helpers
// ============================================================================

/**
 * @brief Empty tag type for policy‑based overload selection.
 */
struct PriorityTag {};

/**
 * @brief Tag for selecting a specific overload via priority.
 */
template <int N>
struct Priority : PriorityTag {
    static constexpr int value = N;
    using next = Priority<N - 1>;
};
template <>
struct Priority<0> : PriorityTag {};

// ============================================================================
//  Debug/assert helpers (runtime)
// ============================================================================

/**
 * @brief Runtime assertion with a message, only active in debug mode.
 */
#if ORTHOTREE_DEBUG
    #define ORTHOTREE_ASSERT_MSG(expr, msg) \
        do { if (!(expr)) { std::fprintf(stderr, "Assertion failed: %s\n", msg); std::abort(); } } while(0)
#else
    #define ORTHOTREE_ASSERT_MSG(expr, msg) ((void)0)
#endif

/**
 * @brief Runtime assertion without message.
 */
#define ORTHOTREE_ASSERT(expr) ORTHOTREE_ASSERT_MSG(expr, #expr)

// ============================================================================
//  Compile‑time detection of whether a type is a container with size() and data()
// ============================================================================

template <typename T, typename = void>
struct has_data_method : std::false_type {};

template <typename T>
struct has_data_method<T, std::void_t<decltype(std::declval<T>().data())>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_data_method_v = has_data_method<T>::value;

template <typename T, typename = void>
struct has_size_method : std::false_type {};

template <typename T>
struct has_size_method<T, std::void_t<decltype(std::declval<T>().size())>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_size_method_v = has_size_method<T>::value;

// ============================================================================
//  Safe integer casts (with overflow checks in debug)
// ============================================================================

template <typename To, typename From>
constexpr To safe_cast(From value) noexcept {
    static_assert(std::is_integral_v<To> && std::is_integral_v<From>,
                  "safe_cast only for integrals");
    To result = static_cast<To>(value);
#if ORTHOTREE_DEBUG
    if (static_cast<From>(result) != value)
        std::abort(); // overflow in debug
#endif
    return result;
}

// ============================================================================
//  Fast rounding of float to nearest integer (banker's rounding not required)
// ============================================================================

inline int32_t fastRound(float x) noexcept {
    return static_cast<int32_t>(std::floor(x + 0.5f));
}

inline int64_t fastRound(double x) noexcept {
    return static_cast<int64_t>(std::floor(x + 0.5));
}

// ============================================================================
//  Swap bytes (for endianness handling) – compile‑time
// ============================================================================

constexpr uint32_t bswap32(uint32_t x) noexcept {
    return ((x & 0x000000FF) << 24) |
           ((x & 0x0000FF00) << 8)  |
           ((x & 0x00FF0000) >> 8)  |
           ((x & 0xFF000000) >> 24);
}

constexpr uint64_t bswap64(uint64_t x) noexcept {
    return ((x & 0x00000000000000FFULL) << 56) |
           ((x & 0x000000000000FF00ULL) << 40) |
           ((x & 0x0000000000FF0000ULL) << 24) |
           ((x & 0x00000000FF000000ULL) << 8)  |
           ((x & 0x000000FF00000000ULL) >> 8)  |
           ((x & 0x0000FF0000000000ULL) >> 24) |
           ((x & 0x00FF000000000000ULL) >> 40) |
           ((x & 0xFF00000000000000ULL) >> 56);
}

// ============================================================================
//  Simple scope guard (RAII for deferring actions)
// ============================================================================

template <typename F>
struct ScopeGuard {
    explicit ScopeGuard(F&& f) : func(std::move(f)), active(true) {}
    ~ScopeGuard() { if (active) func(); }
    void dismiss() { active = false; }
    ScopeGuard(ScopeGuard&& other) noexcept : func(std::move(other.func)), active(other.active) {
        other.dismiss();
    }
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
private:
    F func;
    bool active;
};

template <typename F>
ScopeGuard<F> makeScopeGuard(F&& f) { return ScopeGuard<F>(std::forward<F>(f)); }

} // namespace detail
} // namespace OrthoTree

#endif // ORTHOTREE_DETAIL_COMMON_H_INCLUDED