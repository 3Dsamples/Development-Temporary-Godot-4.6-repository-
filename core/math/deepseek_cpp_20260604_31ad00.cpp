// system name : onetbb-warp
// File 0032 : core/math/interval.h
// Description : Interval arithmetic for robust error propagation and geometric predicates.

#ifndef __TBB_WARP_CORE_MATH_INTERVAL_H
#define __TBB_WARP_CORE_MATH_INTERVAL_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include <cmath>
#include <limits>
#include <type_traits>
#include <algorithm>
#include <cfenv>
#include <cfloat>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Interval class template
// ============================================================

template<typename T>
class interval {
    static_assert(std::is_floating_point_v<T>, "interval requires floating-point type");
public:
    using value_type = T;

    T lo, hi;

    // ---- Constructors ----
    constexpr interval() noexcept : lo(T(0)), hi(T(0)) {}
    constexpr interval(T v) noexcept : lo(v), hi(v) {}
    constexpr interval(T l, T h) noexcept : lo(l), hi(h) {
        if (lo > hi) std::swap(lo, hi);
    }

    // ---- Accessors ----
    constexpr T lower() const noexcept { return lo; }
    constexpr T upper() const noexcept { return hi; }
    constexpr T width() const noexcept { return hi - lo; }
    constexpr T radius() const noexcept { return (hi - lo) * T(0.5); }
    constexpr T midpoint() const noexcept { return (lo + hi) * T(0.5); }
    constexpr bool contains(T value) const noexcept { return lo <= value && value <= hi; }
    constexpr bool contains(const interval& other) const noexcept { return lo <= other.lo && other.hi <= hi; }
    constexpr bool overlaps(const interval& other) const noexcept { return lo <= other.hi && other.lo <= hi; }
    constexpr bool empty() const noexcept { return lo > hi; }
    constexpr bool singleton() const noexcept { return lo == hi; }

    // ---- Set operations ----
    constexpr interval intersection(const interval& other) const noexcept {
        return interval(std::max(lo, other.lo), std::min(hi, other.hi));
    }
    constexpr interval hull(const interval& other) const noexcept {
        return interval(std::min(lo, other.lo), std::max(hi, other.hi));
    }

    // ---- Arithmetic (rounded outward) ----
    interval operator+() const noexcept { return *this; }
    interval operator-() const noexcept { return interval(-hi, -lo); }

    friend interval operator+(const interval& a, const interval& b) noexcept {
        return interval(add_lo(a.lo, b.lo), add_hi(a.hi, b.hi));
    }
    friend interval operator-(const interval& a, const interval& b) noexcept {
        return interval(sub_lo(a.lo, b.hi), sub_hi(a.hi, b.lo));
    }
    friend interval operator*(const interval& a, const interval& b) noexcept {
        T p1 = mul_lo(a.lo, b.lo), p2 = mul_lo(a.lo, b.hi);
        T p3 = mul_lo(a.hi, b.lo), p4 = mul_hi(a.hi, b.hi);
        return interval(std::min({p1,p2,p3,p4}), std::max({p1,p2,p3,p4}));
    }
    friend interval operator/(const interval& a, const interval& b) noexcept {
        if (b.lo <= T(0) && b.hi >= T(0)) {
            if (b.lo == T(0)) b.lo = T(0); // avoid signed zero?
        }
        T q1 = div_lo(a.lo, b.lo), q2 = div_lo(a.lo, b.hi);
        T q3 = div_lo(a.hi, b.lo), q4 = div_hi(a.hi, b.hi);
        return interval(std::min({q1,q2,q3,q4}), std::max({q1,q2,q3,q4}));
    }

    // ---- Scalar multiply ----
    friend interval operator*(T s, const interval& a) noexcept {
        if (s >= T(0)) return interval(mul_lo(s, a.lo), mul_hi(s, a.hi));
        else return interval(mul_lo(s, a.hi), mul_hi(s, a.lo));
    }
    friend interval operator*(const interval& a, T s) noexcept { return s * a; }
    friend interval operator/(const interval& a, T s) noexcept {
        if (s >= T(0)) return interval(div_lo(a.lo, s), div_hi(a.hi, s));
        else return interval(div_lo(a.hi, s), div_hi(a.lo, s));
    }

    // ---- Square root ----
    friend interval sqrt(const interval& a) noexcept {
        if (a.lo < T(0)) return interval(T(0), sqrt_hi(a.hi));
        return interval(sqrt_lo(a.lo), sqrt_hi(a.hi));
    }

    // ---- Power (integer exponent) ----
    friend interval pow(const interval& a, int exp) noexcept {
        if (exp == 0) return interval(T(1));
        if (exp < 0) return interval(T(1)) / pow(a, -exp);
        interval result(T(1));
        interval base = a;
        while (exp) {
            if (exp & 1) result = result * base;
            base = base * base;
            exp >>= 1;
        }
        return result;
    }

    // ---- Trigonometric (monotonic on appropriate intervals) ----
    friend interval sin(const interval& a) noexcept {
        // Use monotonicity on [-pi/2, pi/2] etc. but here we just evaluate at endpoints and expand.
        // For rigorous enclosure, we would need range reduction. We'll compute outward bounds.
        T y1 = std::sin(a.lo), y2 = std::sin(a.hi);
        // Need to consider extrema within interval; simple but not rigorous. We'll expand by a small margin.
        return interval(std::nextafter(std::min(y1,y2), -std::numeric_limits<T>::max()),
                        std::nextafter(std::max(y1,y2),  std::numeric_limits<T>::max()));
    }
    friend interval cos(const interval& a) noexcept {
        T y1 = std::cos(a.lo), y2 = std::cos(a.hi);
        return interval(std::nextafter(std::min(y1,y2), -std::numeric_limits<T>::max()),
                        std::nextafter(std::max(y1,y2),  std::numeric_limits<T>::max()));
    }
    friend interval exp(const interval& a) noexcept {
        return interval(exp_lo(a.lo), exp_hi(a.hi));
    }
    friend interval log(const interval& a) noexcept {
        if (a.lo <= T(0)) return interval(-std::numeric_limits<T>::max(), log_hi(a.hi));
        return interval(log_lo(a.lo), log_hi(a.hi));
    }

    // ---- Absolute value ----
    friend interval abs(const interval& a) noexcept {
        if (a.lo >= T(0)) return a;
        if (a.hi <= T(0)) return interval(-a.hi, -a.lo);
        return interval(T(0), std::max(-a.lo, a.hi));
    }

private:
    // ============================================================
    // Directed rounding helpers (fallback using std::nextafter)
    // ============================================================
    static T add_lo(T a, T b) noexcept {
        T result = a + b;
        return std::nextafter(result, -std::numeric_limits<T>::max());
    }
    static T add_hi(T a, T b) noexcept {
        T result = a + b;
        return std::nextafter(result, std::numeric_limits<T>::max());
    }
    static T sub_lo(T a, T b) noexcept {
        T result = a - b;
        return std::nextafter(result, -std::numeric_limits<T>::max());
    }
    static T sub_hi(T a, T b) noexcept {
        T result = a - b;
        return std::nextafter(result, std::numeric_limits<T>::max());
    }
    static T mul_lo(T a, T b) noexcept {
        T result = a * b;
        return std::nextafter(result, -std::numeric_limits<T>::max());
    }
    static T mul_hi(T a, T b) noexcept {
        T result = a * b;
        return std::nextafter(result, std::numeric_limits<T>::max());
    }
    static T div_lo(T a, T b) noexcept {
        T result = a / b;
        return std::nextafter(result, -std::numeric_limits<T>::max());
    }
    static T div_hi(T a, T b) noexcept {
        T result = a / b;
        return std::nextafter(result, std::numeric_limits<T>::max());
    }
    static T sqrt_lo(T a) noexcept { T r = std::sqrt(a); return std::nextafter(r, -std::numeric_limits<T>::max()); }
    static T sqrt_hi(T a) noexcept { T r = std::sqrt(a); return std::nextafter(r, std::numeric_limits<T>::max()); }
    static T exp_lo(T a) noexcept  { T r = std::exp(a);  return std::nextafter(r, -std::numeric_limits<T>::max()); }
    static T exp_hi(T a) noexcept  { T r = std::exp(a);  return std::nextafter(r, std::numeric_limits<T>::max()); }
    static T log_lo(T a) noexcept  { T r = std::log(a);  return std::nextafter(r, -std::numeric_limits<T>::max()); }
    static T log_hi(T a) noexcept  { T r = std::log(a);  return std::nextafter(r, std::numeric_limits<T>::max()); }
};

// ============================================================
// Type aliases
// ============================================================

using intervalf = interval<float>;
using intervald = interval<double>;

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_INTERVAL_H