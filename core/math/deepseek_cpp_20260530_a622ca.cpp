// File 0002 : core/math/vec2.h
// Two‑component vector (float and double) with arithmetic, geometric operations, and SIMD‑friendly layout.

#pragma once

#include "constants.h"
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <type_traits>

namespace wp {

template <typename T>
struct alignas(sizeof(T) * 2) vec2 {
    T x, y;

    constexpr vec2() noexcept : x(T(0)), y(T(0)) {}
    constexpr explicit vec2(T s) noexcept : x(s), y(s) {}
    constexpr vec2(T x_, T y_) noexcept : x(x_), y(y_) {}
    template <typename U> constexpr explicit vec2(const vec2<U>& o) noexcept : x(static_cast<T>(o.x)), y(static_cast<T>(o.y)) {}
    constexpr vec2(std::initializer_list<T> il) noexcept { auto it = il.begin(); if (it != il.end()) x = *it++; if (it != il.end()) y = *it; else y = T(0); }

    constexpr T  operator[](int i) const noexcept { assert(i>=0&&i<2); return i==0?x:y; }
    constexpr T& operator[](int i)       noexcept { assert(i>=0&&i<2); return i==0?x:y; }

    constexpr vec2 operator+() const noexcept { return *this; }
    constexpr vec2 operator-() const noexcept { return vec2(-x, -y); }

    constexpr vec2& operator+=(const vec2& o) noexcept { x+=o.x; y+=o.y; return *this; }
    constexpr vec2& operator-=(const vec2& o) noexcept { x-=o.x; y-=o.y; return *this; }
    constexpr vec2& operator*=(T s) noexcept { x*=s; y*=s; return *this; }
    constexpr vec2& operator/=(T s) noexcept { x/=s; y/=s; return *this; }
};

template <typename T> constexpr vec2<T> operator+(const vec2<T>& a, const vec2<T>& b) noexcept { return vec2<T>(a.x+b.x, a.y+b.y); }
template <typename T> constexpr vec2<T> operator-(const vec2<T>& a, const vec2<T>& b) noexcept { return vec2<T>(a.x-b.x, a.y-b.y); }
template <typename T> constexpr vec2<T> operator*(const vec2<T>& a, T s) noexcept { return vec2<T>(a.x*s, a.y*s); }
template <typename T> constexpr vec2<T> operator*(T s, const vec2<T>& a) noexcept { return vec2<T>(s*a.x, s*a.y); }
template <typename T> constexpr vec2<T> operator/(const vec2<T>& a, T s) noexcept { return vec2<T>(a.x/s, a.y/s); }
template <typename T> constexpr bool operator==(const vec2<T>& a, const vec2<T>& b) noexcept { return a.x==b.x && a.y==b.y; }
template <typename T> constexpr bool operator!=(const vec2<T>& a, const vec2<T>& b) noexcept { return !(a==b); }

template <typename T> constexpr T dot(const vec2<T>& a, const vec2<T>& b) noexcept { return a.x*b.x + a.y*b.y; }
template <typename T> constexpr T cross(const vec2<T>& a, const vec2<T>& b) noexcept { return a.x*b.y - a.y*b.x; }
template <typename T> constexpr T length_sq(const vec2<T>& v) noexcept { return dot(v,v); }
template <typename T> T length(const vec2<T>& v) noexcept { return std::sqrt(length_sq(v)); }
template <typename T> vec2<T> normalize(const vec2<T>& v) noexcept {
    T l = length(v);
    return (l > MathConst<T>::epsilon) ? v / l : vec2<T>(T(0));
}
template <typename T> constexpr T distance_sq(const vec2<T>& a, const vec2<T>& b) noexcept { return length_sq(a-b); }
template <typename T> T distance(const vec2<T>& a, const vec2<T>& b) noexcept { return length(a-b); }
template <typename T> constexpr vec2<T> min(const vec2<T>& a, const vec2<T>& b) noexcept { return vec2<T>(a.x<b.x?a.x:b.x, a.y<b.y?a.y:b.y); }
template <typename T> constexpr vec2<T> max(const vec2<T>& a, const vec2<T>& b) noexcept { return vec2<T>(a.x>b.x?a.x:b.x, a.y>b.y?a.y:b.y); }
template <typename T> constexpr vec2<T> abs(const vec2<T>& v) noexcept { return vec2<T>(std::abs(v.x), std::abs(v.y)); }
template <typename T> constexpr vec2<T> lerp(const vec2<T>& a, const vec2<T>& b, T t) noexcept { return a + (b - a) * t; }
template <typename T> constexpr vec2<T> project(const vec2<T>& a, const vec2<T>& b) noexcept { return b * (dot(a,b) / dot(b,b)); }
template <typename T> constexpr vec2<T> reflect(const vec2<T>& v, const vec2<T>& n) noexcept { return v - n * (T(2) * dot(v,n)); }
template <typename T> constexpr vec2<T> refract(const vec2<T>& v, const vec2<T>& n, T eta) noexcept {
    T ndotv = dot(n,v);
    T k = T(1) - eta * eta * (T(1) - ndotv * ndotv);
    if (k < T(0)) return vec2<T>(T(0));
    return v * eta - n * (eta * ndotv + std::sqrt(k));
}

template <typename T> vec2<T> rotate(const vec2<T>& v, T angle) noexcept {
    T c = std::cos(angle), s = std::sin(angle);
    return vec2<T>(v.x*c - v.y*s, v.x*s + v.y*c);
}
template <typename T> T angle(const vec2<T>& v) noexcept { return std::atan2(v.y, v.x); }
template <typename T> T angle_between(const vec2<T>& a, const vec2<T>& b) noexcept {
    return std::atan2(cross(a,b), dot(a,b));
}

using vec2f = vec2<float>;
using vec2d = vec2<double>;

} // namespace wp