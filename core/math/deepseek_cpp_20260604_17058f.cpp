// system name : onetbb-warp
// File 0003 : core/math/vector2.h
// Description : 2D vector type with full arithmetic and geometric operations.

#ifndef __TBB_WARP_CORE_MATH_VECTOR2_H
#define __TBB_WARP_CORE_MATH_VECTOR2_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include <cmath>
#include <type_traits>
#include <array>
#include <initializer_list>
#include <functional>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Vector2 class template
// ============================================================

template<typename T>
struct vector2 {
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;

    T x, y;

    // ---- Constructors ----
    constexpr vector2() noexcept : x(T(0)), y(T(0)) {}
    constexpr vector2(T v) noexcept : x(v), y(v) {}
    constexpr vector2(T x_, T y_) noexcept : x(x_), y(y_) {}
    template<typename U>
    constexpr explicit vector2(const vector2<U>& v) noexcept : x(static_cast<T>(v.x)), y(static_cast<T>(v.y)) {}
    constexpr vector2(const std::array<T,2>& arr) noexcept : x(arr[0]), y(arr[1]) {}

    // ---- Access ----
    constexpr T& operator[](std::size_t i) noexcept { return (&x)[i]; }
    constexpr const T& operator[](std::size_t i) const noexcept { return (&x)[i]; }

    // ---- Compound assignment ----
    constexpr vector2& operator+=(const vector2& v) noexcept { x+=v.x; y+=v.y; return *this; }
    constexpr vector2& operator-=(const vector2& v) noexcept { x-=v.x; y-=v.y; return *this; }
    constexpr vector2& operator*=(T s) noexcept { x*=s; y*=s; return *this; }
    constexpr vector2& operator/=(T s) noexcept { x/=s; y/=s; return *this; }

    // ---- Unary ----
    constexpr vector2 operator+() const noexcept { return *this; }
    constexpr vector2 operator-() const noexcept { return vector2(-x, -y); }

    // ---- Conversion ----
    constexpr operator std::array<T,2>() const noexcept { return {x, y}; }
    explicit constexpr operator bool() const noexcept { return x!=T(0) || y!=T(0); }
};

// ============================================================
// Binary operators
// ============================================================

template<typename T> constexpr vector2<T> operator+(const vector2<T>& a, const vector2<T>& b) noexcept { return {a.x+b.x, a.y+b.y}; }
template<typename T> constexpr vector2<T> operator-(const vector2<T>& a, const vector2<T>& b) noexcept { return {a.x-b.x, a.y-b.y}; }
template<typename T> constexpr vector2<T> operator*(const vector2<T>& v, T s) noexcept { return {v.x*s, v.y*s}; }
template<typename T> constexpr vector2<T> operator*(T s, const vector2<T>& v) noexcept { return {v.x*s, v.y*s}; }
template<typename T> constexpr vector2<T> operator/(const vector2<T>& v, T s) noexcept { return {v.x/s, v.y/s}; }
template<typename T> constexpr bool operator==(const vector2<T>& a, const vector2<T>& b) noexcept { return a.x==b.x && a.y==b.y; }
template<typename T> constexpr bool operator!=(const vector2<T>& a, const vector2<T>& b) noexcept { return !(a==b); }

// ============================================================
// Geometric operations
// ============================================================

template<typename T>
constexpr T dot(const vector2<T>& a, const vector2<T>& b) noexcept {
    return a.x*b.x + a.y*b.y;
}

template<typename T>
constexpr T cross(const vector2<T>& a, const vector2<T>& b) noexcept {
    return a.x*b.y - a.y*b.x;
}

template<typename T>
constexpr T length_sq(const vector2<T>& v) noexcept {
    return v.x*v.x + v.y*v.y;
}

template<typename T>
T length(const vector2<T>& v) noexcept {
    return std::sqrt(length_sq(v));
}

template<typename T>
vector2<T> normalize(const vector2<T>& v) noexcept {
    T len = length(v);
    if (len < T(FLOAT_EPSILON)) return vector2<T>(T(0));
    return v / len;
}

template<typename T>
vector2<T> safe_normalize(const vector2<T>& v, const vector2<T>& fallback = vector2<T>(T(1),T(0))) noexcept {
    T len = length(v);
    if (len < T(FLOAT_EPSILON)) return fallback;
    return v / len;
}

template<typename T>
T distance(const vector2<T>& a, const vector2<T>& b) noexcept {
    return length(b - a);
}

template<typename T>
constexpr T distance_sq(const vector2<T>& a, const vector2<T>& b) noexcept {
    return length_sq(b - a);
}

template<typename T>
T manhattan_distance(const vector2<T>& a, const vector2<T>& b) noexcept {
    return std::abs(a.x-b.x) + std::abs(a.y-b.y);
}

template<typename T>
T chebyshev_distance(const vector2<T>& a, const vector2<T>& b) noexcept {
    return std::max(std::abs(a.x-b.x), std::abs(a.y-b.y));
}

template<typename T>
constexpr vector2<T> lerp(const vector2<T>& a, const vector2<T>& b, T t) noexcept {
    return a + (b - a) * t;
}

template<typename T>
constexpr vector2<T> reflect(const vector2<T>& incident, const vector2<T>& normal) noexcept {
    return incident - normal * T(2) * dot(incident, normal);
}

template<typename T>
vector2<T> refract(const vector2<T>& incident, const vector2<T>& normal, T eta) noexcept {
    T ndoti = dot(incident, normal);
    T k = T(1) - eta * eta * (T(1) - ndoti * ndoti);
    if (k < T(0)) return vector2<T>(T(0));
    return incident * eta - normal * (eta * ndoti + std::sqrt(k));
}

template<typename T>
constexpr vector2<T> project(const vector2<T>& a, const vector2<T>& b) noexcept {
    T b2 = dot(b,b);
    if (b2 < T(FLOAT_EPSILON)) return vector2<T>(T(0));
    return b * (dot(a,b) / b2);
}

template<typename T>
constexpr vector2<T> perpendicular(const vector2<T>& v, bool clockwise = true) noexcept {
    return clockwise ? vector2<T>(v.y, -v.x) : vector2<T>(-v.y, v.x);
}

template<typename T>
constexpr vector2<T> rotate(const vector2<T>& v, T angle_rad) noexcept {
    T c = std::cos(angle_rad), s = std::sin(angle_rad);
    return vector2<T>(v.x*c - v.y*s, v.x*s + v.y*c);
}

template<typename T>
T angle(const vector2<T>& from, const vector2<T>& to) noexcept {
    return std::atan2(cross(from, to), dot(from, to));
}

template<typename T>
T angle_unsigned(const vector2<T>& a, const vector2<T>& b) noexcept {
    return std::acos(clamp(dot(a,b) / (length(a)*length(b) + T(FLOAT_EPSILON)), T(-1), T(1)));
}

// ============================================================
// Component‑wise min / max / abs / clamp
// ============================================================

template<typename T>
constexpr vector2<T> abs(const vector2<T>& v) noexcept { return vector2<T>(std::abs(v.x), std::abs(v.y)); }
template<typename T>
constexpr vector2<T> min(const vector2<T>& a, const vector2<T>& b) noexcept { return vector2<T>(min(a.x,b.x), min(a.y,b.y)); }
template<typename T>
constexpr vector2<T> max(const vector2<T>& a, const vector2<T>& b) noexcept { return vector2<T>(max(a.x,b.x), max(a.y,b.y)); }
template<typename T>
constexpr vector2<T> clamp(const vector2<T>& v, const vector2<T>& lo, const vector2<T>& hi) noexcept {
    return vector2<T>(clamp(v.x, lo.x, hi.x), clamp(v.y, lo.y, hi.y));
}
template<typename T>
constexpr T sum(const vector2<T>& v) noexcept { return v.x + v.y; }
template<typename T>
constexpr T product(const vector2<T>& v) noexcept { return v.x * v.y; }
template<typename T>
constexpr vector2<T> floor(const vector2<T>& v) noexcept { return vector2<T>(std::floor(v.x), std::floor(v.y)); }
template<typename T>
constexpr vector2<T> ceil(const vector2<T>& v) noexcept { return vector2<T>(std::ceil(v.x), std::ceil(v.y)); }
template<typename T>
constexpr vector2<T> round(const vector2<T>& v) noexcept { return vector2<T>(std::round(v.x), std::round(v.y)); }
template<typename T>
constexpr vector2<T> fract(const vector2<T>& v) noexcept { return v - floor(v); }

// ============================================================
// Comparison
// ============================================================

template<typename T>
constexpr bool is_zero(const vector2<T>& v) noexcept { return v.x==T(0) && v.y==T(0); }
template<typename T>
constexpr bool is_normalized(const vector2<T>& v, T epsilon = T(FLOAT_EPSILON)) noexcept {
    return std::abs(length_sq(v) - T(1)) < epsilon;
}
template<typename T>
bool is_finite(const vector2<T>& v) noexcept { return std::isfinite(v.x) && std::isfinite(v.y); }
template<typename T>
bool is_nan(const vector2<T>& v) noexcept { return std::isnan(v.x) || std::isnan(v.y); }

// ============================================================
// Conversion between types
// ============================================================

template<typename U, typename T>
constexpr vector2<U> vector_cast(const vector2<T>& v) noexcept {
    return vector2<U>(static_cast<U>(v.x), static_cast<U>(v.y));
}

// ============================================================
// Polar coordinates
// ============================================================

template<typename T>
constexpr vector2<T> from_polar(T radius, T angle_rad) noexcept {
    return vector2<T>(radius * std::cos(angle_rad), radius * std::sin(angle_rad));
}
template<typename T>
constexpr T to_radius(const vector2<T>& v) noexcept { return length(v); }
template<typename T>
T to_angle(const vector2<T>& v) noexcept { return std::atan2(v.y, v.x); }

// ============================================================
// Smooth dynamics (damping, critical, spring)
// ============================================================

template<typename T>
vector2<T> smooth_damp(const vector2<T>& current, const vector2<T>& target,
                       vector2<T>& velocity, T smooth_time, T max_speed, T dt) {
    smooth_time = std::max(T(0.0001), smooth_time);
    T omega = T(2) / smooth_time;
    T x = omega * dt;
    T exp = T(1) / (T(1) + x + T(0.48)*x*x + T(0.235)*x*x*x);
    vector2<T> change = current - target;
    vector2<T> original_to = target;
    T max_change = max_speed * smooth_time;
    change = clamp(change, vector2<T>(-max_change), vector2<T>(max_change));
    vector2<T> temp = (velocity + change * omega) * dt;
    velocity = (velocity - temp * omega) * exp;
    vector2<T> output = (current - change) + (change + temp) * exp;
    if (dot(original_to - current, output - original_to) > T(0)) {
        output = original_to;
        velocity = vector2<T>(T(0));
    }
    return output;
}

template<typename T>
vector2<T> critical_damp(const vector2<T>& current, const vector2<T>& target,
                         vector2<T>& velocity, T frequency, T dt) {
    T w = T(TAU_D) * frequency;
    T d = T(1) + T(2) * dt * w;
    T w2 = w * w;
    T inv = T(1) / (T(1) + T(2) * dt * w + dt * dt * w2);
    vector2<T> output = (current * T(1) + velocity * dt + target * (dt * dt * w2)) * inv;
    velocity = (velocity + (target - output) * (dt * w2)) * inv;
    return output;
}

// ============================================================
// Type aliases
// ============================================================

using vector2f = vector2<float>;
using vector2d = vector2<double>;
using vector2i = vector2<std::int32_t>;
using vector2u = vector2<std::uint32_t>;

// ============================================================
// Fold expression helpers
// ============================================================

template<typename T, typename... Rest>
constexpr vector2<T> sum_vectors(const vector2<T>& first, const Rest&... rest) noexcept {
    return (first + ... + rest);
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_VECTOR2_H