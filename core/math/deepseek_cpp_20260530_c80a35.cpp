// File 0003 : core/math/vec3.h
// Three‑component vector (float/double) with arithmetic, dot/cross/normalize, and utility functions.

#pragma once

#include "constants.h"
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <type_traits>

namespace wp {

template <typename T>
struct alignas(sizeof(T) * 3) vec3 {
    T x, y, z;

    constexpr vec3() noexcept : x(T(0)), y(T(0)), z(T(0)) {}
    constexpr explicit vec3(T s) noexcept : x(s), y(s), z(s) {}
    constexpr vec3(T x_, T y_, T z_) noexcept : x(x_), y(y_), z(z_) {}
    template <typename U> constexpr explicit vec3(const vec3<U>& o) noexcept : x(static_cast<T>(o.x)), y(static_cast<T>(o.y)), z(static_cast<T>(o.z)) {}
    constexpr vec3(std::initializer_list<T> il) noexcept {
        auto it = il.begin();
        x = (it != il.end()) ? *it++ : T(0);
        y = (it != il.end()) ? *it++ : T(0);
        z = (it != il.end()) ? *it   : T(0);
    }

    constexpr T  operator[](int i) const noexcept { assert(i>=0&&i<3); return (&x)[i]; }
    constexpr T& operator[](int i)       noexcept { assert(i>=0&&i<3); return (&x)[i]; }

    constexpr vec3 operator+() const noexcept { return *this; }
    constexpr vec3 operator-() const noexcept { return vec3(-x, -y, -z); }

    constexpr vec3& operator+=(const vec3& o) noexcept { x+=o.x; y+=o.y; z+=o.z; return *this; }
    constexpr vec3& operator-=(const vec3& o) noexcept { x-=o.x; y-=o.y; z-=o.z; return *this; }
    constexpr vec3& operator*=(T s) noexcept { x*=s; y*=s; z*=s; return *this; }
    constexpr vec3& operator/=(T s) noexcept { x/=s; y/=s; z/=s; return *this; }
};

template <typename T> constexpr vec3<T> operator+(const vec3<T>& a, const vec3<T>& b) noexcept { return vec3<T>(a.x+b.x, a.y+b.y, a.z+b.z); }
template <typename T> constexpr vec3<T> operator-(const vec3<T>& a, const vec3<T>& b) noexcept { return vec3<T>(a.x-b.x, a.y-b.y, a.z-b.z); }
template <typename T> constexpr vec3<T> operator*(const vec3<T>& a, T s) noexcept { return vec3<T>(a.x*s, a.y*s, a.z*s); }
template <typename T> constexpr vec3<T> operator*(T s, const vec3<T>& a) noexcept { return vec3<T>(s*a.x, s*a.y, s*a.z); }
template <typename T> constexpr vec3<T> operator/(const vec3<T>& a, T s) noexcept { return vec3<T>(a.x/s, a.y/s, a.z/s); }
template <typename T> constexpr bool operator==(const vec3<T>& a, const vec3<T>& b) noexcept { return a.x==b.x && a.y==b.y && a.z==b.z; }
template <typename T> constexpr bool operator!=(const vec3<T>& a, const vec3<T>& b) noexcept { return !(a==b); }

template <typename T> constexpr T dot(const vec3<T>& a, const vec3<T>& b) noexcept { return a.x*b.x + a.y*b.y + a.z*b.z; }
template <typename T> constexpr vec3<T> cross(const vec3<T>& a, const vec3<T>& b) noexcept { return vec3<T>(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); }
template <typename T> constexpr T length_sq(const vec3<T>& v) noexcept { return dot(v,v); }
template <typename T> T length(const vec3<T>& v) noexcept { return std::sqrt(length_sq(v)); }
template <typename T> vec3<T> normalize(const vec3<T>& v) noexcept {
    T l = length(v);
    return (l > MathConst<T>::epsilon) ? v / l : vec3<T>(T(0));
}
template <typename T> constexpr T distance_sq(const vec3<T>& a, const vec3<T>& b) noexcept { return length_sq(a-b); }
template <typename T> T distance(const vec3<T>& a, const vec3<T>& b) noexcept { return length(a-b); }
template <typename T> constexpr vec3<T> min(const vec3<T>& a, const vec3<T>& b) noexcept { return vec3<T>(a.x<b.x?a.x:b.x, a.y<b.y?a.y:b.y, a.z<b.z?a.z:b.z); }
template <typename T> constexpr vec3<T> max(const vec3<T>& a, const vec3<T>& b) noexcept { return vec3<T>(a.x>b.x?a.x:b.x, a.y>b.y?a.y:b.y, a.z>b.z?a.z:b.z); }
template <typename T> constexpr vec3<T> abs(const vec3<T>& v) noexcept { return vec3<T>(std::abs(v.x), std::abs(v.y), std::abs(v.z)); }
template <typename T> constexpr vec3<T> lerp(const vec3<T>& a, const vec3<T>& b, T t) noexcept { return a + (b - a) * t; }
template <typename T> constexpr vec3<T> project(const vec3<T>& a, const vec3<T>& b) noexcept { return b * (dot(a,b) / dot(b,b)); }
template <typename T> constexpr vec3<T> reflect(const vec3<T>& v, const vec3<T>& n) noexcept { return v - n * (T(2) * dot(v,n)); }
template <typename T> constexpr vec3<T> refract(const vec3<T>& v, const vec3<T>& n, T eta) noexcept {
    T ndotv = dot(n,v);
    T k = T(1) - eta * eta * (T(1) - ndotv * ndotv);
    if (k < T(0)) return vec3<T>(T(0));
    return v * eta - n * (eta * ndotv + std::sqrt(k));
}
template <typename T> constexpr vec3<T> orthonormalize(const vec3<T>& a, const vec3<T>& b) noexcept {
    vec3<T> u = a;
    vec3<T> v = b - project(b, a);
    return normalize(u);
}
template <typename T> constexpr vec3<T> perpendicular(const vec3<T>& v) noexcept {
    if (std::abs(v.x) < std::abs(v.y) && std::abs(v.x) < std::abs(v.z)) return vec3<T>(T(0), -v.z, v.y);
    if (std::abs(v.y) < std::abs(v.z)) return vec3<T>(-v.z, T(0), v.x);
    return vec3<T>(-v.y, v.x, T(0));
}
template <typename T> constexpr T angle_between(const vec3<T>& a, const vec3<T>& b) noexcept {
    return std::acos(clamp(dot(normalize(a), normalize(b)), T(-1), T(1)));
}

using vec3f = vec3<float>;
using vec3d = vec3<double>;

} // namespace wp