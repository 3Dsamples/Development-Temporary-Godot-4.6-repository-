// system name : onetbb-warp
// File 0005 : core/math/vector4.h
// Description : 4D vector type for homogeneous coordinates and RGBA colors.

#ifndef __TBB_WARP_CORE_MATH_VECTOR4_H
#define __TBB_WARP_CORE_MATH_VECTOR4_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include <cmath>
#include <type_traits>
#include <array>
#include <initializer_list>
#include <functional>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Vector4 class template
// ============================================================

template<typename T>
struct vector4 {
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;

    T x, y, z, w;

    // ---- Constructors ----
    constexpr vector4() noexcept : x(T(0)), y(T(0)), z(T(0)), w(T(0)) {}
    constexpr vector4(T v) noexcept : x(v), y(v), z(v), w(v) {}
    constexpr vector4(T x_, T y_, T z_, T w_) noexcept : x(x_), y(y_), z(z_), w(w_) {}
    template<typename U>
    constexpr explicit vector4(const vector4<U>& v) noexcept : x(static_cast<T>(v.x)), y(static_cast<T>(v.y)), z(static_cast<T>(v.z)), w(static_cast<T>(v.w)) {}
    constexpr vector4(const std::array<T,4>& arr) noexcept : x(arr[0]), y(arr[1]), z(arr[2]), w(arr[3]) {}
    constexpr vector4(const vector2<T>& v, T z_ = T(0), T w_ = T(0)) noexcept : x(v.x), y(v.y), z(z_), w(w_) {}
    constexpr vector4(const vector3<T>& v, T w_ = T(0)) noexcept : x(v.x), y(v.y), z(v.z), w(w_) {}

    // ---- Access ----
    constexpr T& operator[](std::size_t i) noexcept { return (&x)[i]; }
    constexpr const T& operator[](std::size_t i) const noexcept { return (&x)[i]; }

    // ---- Compound assignment ----
    constexpr vector4& operator+=(const vector4& v) noexcept { x+=v.x; y+=v.y; z+=v.z; w+=v.w; return *this; }
    constexpr vector4& operator-=(const vector4& v) noexcept { x-=v.x; y-=v.y; z-=v.z; w-=v.w; return *this; }
    constexpr vector4& operator*=(T s) noexcept { x*=s; y*=s; z*=s; w*=s; return *this; }
    constexpr vector4& operator/=(T s) noexcept { x/=s; y/=s; z/=s; w/=s; return *this; }

    // ---- Unary ----
    constexpr vector4 operator+() const noexcept { return *this; }
    constexpr vector4 operator-() const noexcept { return vector4(-x, -y, -z, -w); }

    // ---- Conversion ----
    constexpr operator std::array<T,4>() const noexcept { return {x, y, z, w}; }
    explicit constexpr operator bool() const noexcept { return x!=T(0) || y!=T(0) || z!=T(0) || w!=T(0); }
};

// ============================================================
// Binary operators
// ============================================================

template<typename T> constexpr vector4<T> operator+(const vector4<T>& a, const vector4<T>& b) noexcept { return {a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w}; }
template<typename T> constexpr vector4<T> operator-(const vector4<T>& a, const vector4<T>& b) noexcept { return {a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w}; }
template<typename T> constexpr vector4<T> operator*(const vector4<T>& v, T s) noexcept { return {v.x*s, v.y*s, v.z*s, v.w*s}; }
template<typename T> constexpr vector4<T> operator*(T s, const vector4<T>& v) noexcept { return {v.x*s, v.y*s, v.z*s, v.w*s}; }
template<typename T> constexpr vector4<T> operator/(const vector4<T>& v, T s) noexcept { return {v.x/s, v.y/s, v.z/s, v.w/s}; }
template<typename T> constexpr bool operator==(const vector4<T>& a, const vector4<T>& b) noexcept { return a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w; }
template<typename T> constexpr bool operator!=(const vector4<T>& a, const vector4<T>& b) noexcept { return !(a==b); }

// ============================================================
// Dot product
// ============================================================

template<typename T>
constexpr T dot(const vector4<T>& a, const vector4<T>& b) noexcept {
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

// ============================================================
// Length
// ============================================================

template<typename T>
constexpr T length_sq(const vector4<T>& v) noexcept {
    return v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
}

template<typename T>
T length(const vector4<T>& v) noexcept {
    return std::sqrt(length_sq(v));
}

template<typename T>
vector4<T> normalize(const vector4<T>& v) noexcept {
    T len = length(v);
    if (len < T(FLOAT_EPSILON)) return vector4<T>(T(0));
    return v / len;
}

template<typename T>
T distance(const vector4<T>& a, const vector4<T>& b) noexcept {
    return length(b - a);
}

template<typename T>
constexpr T distance_sq(const vector4<T>& a, const vector4<T>& b) noexcept {
    return length_sq(b - a);
}

// ============================================================
// Interpolation
// ============================================================

template<typename T>
constexpr vector4<T> lerp(const vector4<T>& a, const vector4<T>& b, T t) noexcept {
    return a + (b - a) * t;
}

template<typename T>
constexpr vector4<T> bilinear(const vector4<T>& v00, const vector4<T>& v10,
                              const vector4<T>& v01, const vector4<T>& v11,
                              T u, T v) noexcept {
    return lerp(lerp(v00, v10, u), lerp(v01, v11, u), v);
}

// ============================================================
// Component‑wise operations
// ============================================================

template<typename T> constexpr vector4<T> abs(const vector4<T>& v) noexcept { return vector4<T>(std::abs(v.x), std::abs(v.y), std::abs(v.z), std::abs(v.w)); }
template<typename T> constexpr vector4<T> min(const vector4<T>& a, const vector4<T>& b) noexcept { return vector4<T>(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w)); }
template<typename T> constexpr vector4<T> max(const vector4<T>& a, const vector4<T>& b) noexcept { return vector4<T>(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w)); }
template<typename T> constexpr vector4<T> clamp(const vector4<T>& v, const vector4<T>& lo, const vector4<T>& hi) noexcept {
    return vector4<T>(clamp(v.x, lo.x, hi.x), clamp(v.y, lo.y, hi.y), clamp(v.z, lo.z, hi.z), clamp(v.w, lo.w, hi.w));
}
template<typename T> constexpr T sum(const vector4<T>& v) noexcept { return v.x+v.y+v.z+v.w; }
template<typename T> constexpr T product(const vector4<T>& v) noexcept { return v.x*v.y*v.z*v.w; }
template<typename T> constexpr vector4<T> floor(const vector4<T>& v) noexcept { return vector4<T>(std::floor(v.x), std::floor(v.y), std::floor(v.z), std::floor(v.w)); }
template<typename T> constexpr vector4<T> ceil(const vector4<T>& v) noexcept { return vector4<T>(std::ceil(v.x), std::ceil(v.y), std::ceil(v.z), std::ceil(v.w)); }
template<typename T> constexpr vector4<T> round(const vector4<T>& v) noexcept { return vector4<T>(std::round(v.x), std::round(v.y), std::round(v.z), std::round(v.w)); }
template<typename T> constexpr vector4<T> fract(const vector4<T>& v) noexcept { return v - floor(v); }

// ============================================================
// Color‑specific operations (RGBA)
// ============================================================

template<typename T>
constexpr T luminance_rgba(const vector4<T>& color) noexcept {
    return T(0.2126) * color.x + T(0.7152) * color.y + T(0.0722) * color.z;
}

template<typename T>
constexpr vector4<T> sRGB_to_linear(const vector4<T>& c) noexcept {
    auto convert = [](T ch) -> T {
        if (ch <= T(0.04045)) return ch / T(12.92);
        return std::pow((ch + T(0.055)) / T(1.055), T(2.4));
    };
    return vector4<T>(convert(c.x), convert(c.y), convert(c.z), c.w);
}

template<typename T>
constexpr vector4<T> linear_to_sRGB(const vector4<T>& c) noexcept {
    auto convert = [](T ch) -> T {
        if (ch <= T(0.0031308)) return ch * T(12.92);
        return T(1.055) * std::pow(ch, T(1.0/2.4)) - T(0.055);
    };
    return vector4<T>(convert(c.x), convert(c.y), convert(c.z), c.w);
}

template<typename T>
constexpr vector4<T> hue_rotate(const vector4<T>& color, T angle) noexcept {
    T c = std::cos(angle), s = std::sin(angle);
    T r = color.x, g = color.y, b = color.z;
    T lum_r = T(0.213), lum_g = T(0.715), lum_b = T(0.072);
    return vector4<T>(
        r * (c + (T(1)-c)*lum_r) + g * ((T(1)-c)*lum_g - s*lum_b) + b * ((T(1)-c)*lum_b + s*lum_g),
        r * ((T(1)-c)*lum_r + s*lum_b) + g * (c + (T(1)-c)*lum_g) + b * ((T(1)-c)*lum_b - s*lum_r),
        r * ((T(1)-c)*lum_r - s*lum_g) + g * ((T(1)-c)*lum_g + s*lum_r) + b * (c + (T(1)-c)*lum_b),
        color.w
    );
}

// ============================================================
// Comparison
// ============================================================

template<typename T> constexpr bool is_zero(const vector4<T>& v) noexcept { return v.x==T(0) && v.y==T(0) && v.z==T(0) && v.w==T(0); }
template<typename T> bool is_finite(const vector4<T>& v) noexcept { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(v.w); }
template<typename T> bool is_nan(const vector4<T>& v) noexcept { return std::isnan(v.x) || std::isnan(v.y) || std::isnan(v.z) || std::isnan(v.w); }

// ============================================================
// Conversion between types
// ============================================================

template<typename U, typename T>
constexpr vector4<U> vector_cast(const vector4<T>& v) noexcept {
    return vector4<U>(static_cast<U>(v.x), static_cast<U>(v.y), static_cast<U>(v.z), static_cast<U>(v.w));
}

// ============================================================
// Type aliases
// ============================================================

using vector4f = vector4<float>;
using vector4d = vector4<double>;
using vector4i = vector4<std::int32_t>;
using vector4u = vector4<std::uint32_t>;

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_VECTOR4_H