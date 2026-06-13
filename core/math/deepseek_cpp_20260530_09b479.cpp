// File 0004 : core/math/vec4.h
// Four‑component vector (float/double) for homogeneous coordinates, colors, and quaternion storage.

#pragma once

#include "constants.h"
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <type_traits>

namespace wp {

template <typename T>
struct alignas(sizeof(T) * 4) vec4 {
    T x, y, z, w;

    constexpr vec4() noexcept : x(T(0)), y(T(0)), z(T(0)), w(T(0)) {}
    constexpr explicit vec4(T s) noexcept : x(s), y(s), z(s), w(s) {}
    constexpr vec4(T x_, T y_, T z_, T w_) noexcept : x(x_), y(y_), z(z_), w(w_) {}
    template <typename U> constexpr explicit vec4(const vec4<U>& o) noexcept : x(static_cast<T>(o.x)), y(static_cast<T>(o.y)), z(static_cast<T>(o.z)), w(static_cast<T>(o.w)) {}
    constexpr vec4(std::initializer_list<T> il) noexcept {
        auto it = il.begin();
        x = (it != il.end()) ? *it++ : T(0);
        y = (it != il.end()) ? *it++ : T(0);
        z = (it != il.end()) ? *it++ : T(0);
        w = (it != il.end()) ? *it   : T(0);
    }

    constexpr T  operator[](int i) const noexcept { assert(i>=0&&i<4); return (&x)[i]; }
    constexpr T& operator[](int i)       noexcept { assert(i>=0&&i<4); return (&x)[i]; }

    constexpr vec4 operator+() const noexcept { return *this; }
    constexpr vec4 operator-() const noexcept { return vec4(-x, -y, -z, -w); }

    constexpr vec4& operator+=(const vec4& o) noexcept { x+=o.x; y+=o.y; z+=o.z; w+=o.w; return *this; }
    constexpr vec4& operator-=(const vec4& o) noexcept { x-=o.x; y-=o.y; z-=o.z; w-=o.w; return *this; }
    constexpr vec4& operator*=(T s) noexcept { x*=s; y*=s; z*=s; w*=s; return *this; }
    constexpr vec4& operator/=(T s) noexcept { x/=s; y/=s; z/=s; w/=s; return *this; }
};

template <typename T> constexpr vec4<T> operator+(const vec4<T>& a, const vec4<T>& b) noexcept { return vec4<T>(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
template <typename T> constexpr vec4<T> operator-(const vec4<T>& a, const vec4<T>& b) noexcept { return vec4<T>(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }
template <typename T> constexpr vec4<T> operator*(const vec4<T>& a, T s) noexcept { return vec4<T>(a.x*s, a.y*s, a.z*s, a.w*s); }
template <typename T> constexpr vec4<T> operator*(T s, const vec4<T>& a) noexcept { return vec4<T>(s*a.x, s*a.y, s*a.z, s*a.w); }
template <typename T> constexpr vec4<T> operator/(const vec4<T>& a, T s) noexcept { return vec4<T>(a.x/s, a.y/s, a.z/s, a.w/s); }
template <typename T> constexpr bool operator==(const vec4<T>& a, const vec4<T>& b) noexcept { return a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w; }
template <typename T> constexpr bool operator!=(const vec4<T>& a, const vec4<T>& b) noexcept { return !(a==b); }

template <typename T> constexpr T dot(const vec4<T>& a, const vec4<T>& b) noexcept { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
template <typename T> constexpr T length_sq(const vec4<T>& v) noexcept { return dot(v,v); }
template <typename T> T length(const vec4<T>& v) noexcept { return std::sqrt(length_sq(v)); }
template <typename T> vec4<T> normalize(const vec4<T>& v) noexcept {
    T l = length(v);
    return (l > MathConst<T>::epsilon) ? v / l : vec4<T>(T(0));
}
template <typename T> constexpr T distance_sq(const vec4<T>& a, const vec4<T>& b) noexcept { return length_sq(a-b); }
template <typename T> T distance(const vec4<T>& a, const vec4<T>& b) noexcept { return length(a-b); }
template <typename T> constexpr vec4<T> min(const vec4<T>& a, const vec4<T>& b) noexcept { return vec4<T>(a.x<b.x?a.x:b.x, a.y<b.y?a.y:b.y, a.z<b.z?a.z:b.z, a.w<b.w?a.w:b.w); }
template <typename T> constexpr vec4<T> max(const vec4<T>& a, const vec4<T>& b) noexcept { return vec4<T>(a.x>b.x?a.x:b.x, a.y>b.y?a.y:b.y, a.z>b.z?a.z:b.z, a.w>b.w?a.w:b.w); }
template <typename T> constexpr vec4<T> abs(const vec4<T>& v) noexcept { return vec4<T>(std::abs(v.x), std::abs(v.y), std::abs(v.z), std::abs(v.w)); }
template <typename T> constexpr vec4<T> lerp(const vec4<T>& a, const vec4<T>& b, T t) noexcept { return a + (b - a) * t; }

template <typename T> constexpr vec4<T> perspective_divide(const vec4<T>& v) noexcept { return v / v.w; }

using vec4f = vec4<float>;
using vec4d = vec4<double>;

} // namespace wp