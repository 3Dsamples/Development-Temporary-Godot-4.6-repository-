// File 0017 : core/math/interpolation.h
// Interpolation functions (linear, bilinear, cubic, smoothstep, Catmull-Rom, Bezier) for scalar and vector types.

#pragma once

#include "vec2.h"
#include "vec3.h"
#include "vec4.h"
#include "constants.h"
#include <cmath>
#include <algorithm>

namespace wp {

// ---------- Scalar interpolation ----------
template <typename T> constexpr T lerp(T a, T b, T t) noexcept { return a + t * (b - a); }
template <typename T> constexpr T bilerp(T a0, T a1, T b0, T b1, T t) noexcept { return lerp(lerp(a0, a1, t), lerp(b0, b1, t), t); }
template <typename T> constexpr T trilerp(T a0, T a1, T b0, T b1, T c0, T c1, T t) noexcept { return lerp(bilerp(a0,a1,b0,b1,t), bilerp(b0,b1,c0,c1,t), t); }

template <typename T> constexpr T smoothstep(T edge0, T edge1, T x) noexcept {
    T t = clamp((x - edge0) / (edge1 - edge0), T(0), T(1));
    return t * t * (T(3) - T(2) * t);
}
template <typename T> constexpr T smootherstep(T edge0, T edge1, T x) noexcept {
    T t = clamp((x - edge0) / (edge1 - edge0), T(0), T(1));
    return t * t * t * (t * (t * T(6) - T(15)) + T(10));
}

template <typename T> T cosine_interp(T a, T b, T t) noexcept {
    T f = (T(1) - std::cos(t * pi<T>)) * T(0.5);
    return lerp(a, b, f);
}

// ---------- Vector interpolation ----------
template <typename T> constexpr vec2<T> lerp(const vec2<T>& a, const vec2<T>& b, T t) noexcept { return a + (b - a) * t; }
template <typename T> constexpr vec3<T> lerp(const vec3<T>& a, const vec3<T>& b, T t) noexcept { return a + (b - a) * t; }
template <typename T> constexpr vec4<T> lerp(const vec4<T>& a, const vec4<T>& b, T t) noexcept { return a + (b - a) * t; }

template <typename T> constexpr vec2<T> smoothstep(const vec2<T>& e0, const vec2<T>& e1, const vec2<T>& x) noexcept {
    return vec2<T>(smoothstep(e0.x, e1.x, x.x), smoothstep(e0.y, e1.y, x.y));
}
template <typename T> constexpr vec3<T> smoothstep(const vec3<T>& e0, const vec3<T>& e1, const vec3<T>& x) noexcept {
    return vec3<T>(smoothstep(e0.x, e1.x, x.x), smoothstep(e0.y, e1.y, x.y), smoothstep(e0.z, e1.z, x.z));
}

// ---------- Bilinear / Trilinear for textures (uv) ----------
template <typename T> constexpr T bilinear(const vec2<T>& uv, const T* values, int w, int h) noexcept {
    int x0 = static_cast<int>(uv.x);
    int y0 = static_cast<int>(uv.y);
    int x1 = std::min(x0 + 1, w - 1);
    int y1 = std::min(y0 + 1, h - 1);
    T fx = uv.x - x0;
    T fy = uv.y - y0;
    T v00 = values[y0 * w + x0];
    T v10 = values[y0 * w + x1];
    T v01 = values[y1 * w + x0];
    T v11 = values[y1 * w + x1];
    return lerp(lerp(v00, v10, fx), lerp(v01, v11, fx), fy);
}

// ---------- Cubic Hermite interpolation ----------
template <typename T> T cubic_hermite(T a, T b, T tan_a, T tan_b, T t) noexcept {
    T t2 = t * t;
    T t3 = t2 * t;
    return (T(2) * t3 - T(3) * t2 + T(1)) * a +
           (t3 - T(2) * t2 + t) * tan_a +
           (-T(2) * t3 + T(3) * t2) * b +
           (t3 - t2) * tan_b;
}

template <typename T> vec3<T> cubic_hermite(const vec3<T>& a, const vec3<T>& b, const vec3<T>& ta, const vec3<T>& tb, T t) noexcept {
    return vec3<T>(cubic_hermite(a.x, b.x, ta.x, tb.x, t),
                   cubic_hermite(a.y, b.y, ta.y, tb.y, t),
                   cubic_hermite(a.z, b.z, ta.z, tb.z, t));
}

// ---------- Catmull-Rom spline (4 points, return t in [0,1] along p1-p2) ----------
template <typename T> T catmull_rom(T p0, T p1, T p2, T p3, T t) noexcept {
    T t2 = t * t;
    T t3 = t2 * t;
    return T(0.5) * ((T(2) * p1) +
                     (-p0 + p2) * t +
                     (T(2) * p0 - T(5) * p1 + T(4) * p2 - p3) * t2 +
                     (-p0 + T(3) * p1 - T(3) * p2 + p3) * t3);
}
template <typename T> vec3<T> catmull_rom(const vec3<T>& p0, const vec3<T>& p1, const vec3<T>& p2, const vec3<T>& p3, T t) noexcept {
    return vec3<T>(catmull_rom(p0.x, p1.x, p2.x, p3.x, t),
                   catmull_rom(p0.y, p1.y, p2.y, p3.y, t),
                   catmull_rom(p0.z, p1.z, p2.z, p3.z, t));
}

// ---------- Bezier curves (quadratic, cubic) ----------
template <typename T> T bezier_quadratic(T p0, T p1, T p2, T t) noexcept {
    T mt = T(1) - t;
    return mt * mt * p0 + T(2) * mt * t * p1 + t * t * p2;
}
template <typename T> vec3<T> bezier_quadratic(const vec3<T>& p0, const vec3<T>& p1, const vec3<T>& p2, T t) noexcept {
    return vec3<T>(bezier_quadratic(p0.x, p1.x, p2.x, t),
                   bezier_quadratic(p0.y, p1.y, p2.y, t),
                   bezier_quadratic(p0.z, p1.z, p2.z, t));
}

template <typename T> T bezier_cubic(T p0, T p1, T p2, T p3, T t) noexcept {
    T mt = T(1) - t;
    T mt2 = mt * mt;
    T t2 = t * t;
    return mt2 * mt * p0 + T(3) * mt2 * t * p1 + T(3) * mt * t2 * p2 + t2 * t * p3;
}
template <typename T> vec3<T> bezier_cubic(const vec3<T>& p0, const vec3<T>& p1, const vec3<T>& p2, const vec3<T>& p3, T t) noexcept {
    return vec3<T>(bezier_cubic(p0.x, p1.x, p2.x, p3.x, t),
                   bezier_cubic(p0.y, p1.y, p2.y, p3.y, t),
                   bezier_cubic(p0.z, p1.z, p2.z, p3.z, t));
}

// ---------- Step function (discrete interpolation) ----------
template <typename T> constexpr T step(T edge, T x) noexcept { return (x >= edge) ? T(1) : T(0); }

// ---------- Easing functions (Penner-like) ----------
template <typename T> T ease_in_quad(T t) noexcept { return t * t; }
template <typename T> T ease_out_quad(T t) noexcept { return t * (T(2) - t); }
template <typename T> T ease_in_out_quad(T t) noexcept { return (t < T(0.5)) ? T(2)*t*t : T(1) - sqr(-T(2)*t + T(2)) * T(0.5); }

template <typename T> T ease_in_cubic(T t) noexcept { return t * t * t; }
template <typename T> T ease_out_cubic(T t) noexcept { T t1 = t - T(1); return t1 * t1 * t1 + T(1); }
template <typename T> T ease_in_out_cubic(T t) noexcept { return (t < T(0.5)) ? T(4)*t*t*t : T(1) - sqr(-T(2)*t + T(2)) * T(0.5); }

template <typename T> T ease_in_expo(T t) noexcept { return (t == T(0)) ? T(0) : std::pow(T(2), T(10) * (t - T(1))); }
template <typename T> T ease_out_expo(T t) noexcept { return (t == T(1)) ? T(1) : T(1) - std::pow(T(2), T(-10) * t); }
template <typename T> T ease_in_out_expo(T t) noexcept {
    if (t == T(0)) return T(0);
    if (t == T(1)) return T(1);
    if (t < T(0.5)) return T(0.5) * std::pow(T(2), T(20) * t - T(10));
    return T(0.5) * (T(2) - std::pow(T(2), -T(20) * t + T(10)));
}

template <typename T> T ease_in_back(T t) noexcept { constexpr T s = T(1.70158); return t * t * ((s + T(1)) * t - s); }
template <typename T> T ease_out_back(T t) noexcept { constexpr T s = T(1.70158); T t1 = t - T(1); return t1 * t1 * ((s + T(1)) * t1 + s) + T(1); }
template <typename T> T ease_in_out_back(T t) noexcept {
    constexpr T s = T(1.70158) * T(1.525);
    if (t < T(0.5)) { T t2 = t * T(2); return T(0.5) * t2 * t2 * ((s + T(1)) * t2 - s); }
    else { T t2 = (t - T(1)) * T(2); return T(0.5) * (t2 * t2 * ((s + T(1)) * t2 + s) + T(2)); }
}

template <typename T> T ease_in_elastic(T t) noexcept {
    if (t == T(0) || t == T(1)) return t;
    return -std::pow(T(2), T(10) * (t - T(1))) * std::sin((t - T(1.075)) * two_pi<T> / T(0.3));
}
template <typename T> T ease_out_elastic(T t) noexcept {
    if (t == T(0) || t == T(1)) return t;
    return std::pow(T(2), T(-10) * t) * std::sin((t - T(0.075)) * two_pi<T> / T(0.3)) + T(1);
}
template <typename T> T ease_in_out_elastic(T t) noexcept {
    if (t == T(0) || t == T(1)) return t;
    if (t < T(0.5)) return T(-0.5) * std::pow(T(2), T(20) * t - T(10)) * std::sin((T(20) * t - T(11.125)) * two_pi<T> / T(4.5));
    return std::pow(T(2), T(-20) * t + T(10)) * std::sin((T(20) * t - T(11.125)) * two_pi<T> / T(4.5)) * T(0.5) + T(1);
}

} // namespace wp