// File 0005 : core/math/mat2.h
// 2×2 matrix (float/double) with arithmetic, determinant, inverse, transpose, and transformation utilities.

#pragma once

#include "vec2.h"
#include <cassert>
#include <cmath>
#include <type_traits>

namespace wp {

template <typename T>
struct alignas(sizeof(T) * 4) mat2 {
    union {
        T data[2][2];
        struct { T m00, m01, m10, m11; };
    };

    constexpr mat2() noexcept : m00(T(1)), m01(T(0)), m10(T(0)), m11(T(1)) {}
    constexpr explicit mat2(T s) noexcept : m00(s), m01(T(0)), m10(T(0)), m11(s) {}
    constexpr mat2(T m00_, T m01_, T m10_, T m11_) noexcept : m00(m00_), m01(m01_), m10(m10_), m11(m11_) {}
    constexpr mat2(const vec2<T>& col0, const vec2<T>& col1) noexcept : m00(col0.x), m10(col0.y), m01(col1.x), m11(col1.y) {}
    template <typename U> constexpr explicit mat2(const mat2<U>& o) noexcept : m00(static_cast<T>(o.m00)), m01(static_cast<T>(o.m01)), m10(static_cast<T>(o.m10)), m11(static_cast<T>(o.m11)) {}

    constexpr T  operator()(int row, int col) const noexcept { assert(row>=0&&row<2&&col>=0&&col<2); return data[row][col]; }
    constexpr T& operator()(int row, int col)       noexcept { assert(row>=0&&row<2&&col>=0&&col<2); return data[row][col]; }

    constexpr vec2<T> col(int i) const noexcept { assert(i>=0&&i<2); return vec2<T>(data[0][i], data[1][i]); }
    constexpr vec2<T> row(int i) const noexcept { assert(i>=0&&i<2); return vec2<T>(data[i][0], data[i][1]); }

    constexpr mat2 operator+() const noexcept { return *this; }
    constexpr mat2 operator-() const noexcept { return mat2(-m00, -m01, -m10, -m11); }

    constexpr mat2& operator+=(const mat2& o) noexcept { m00+=o.m00; m01+=o.m01; m10+=o.m10; m11+=o.m11; return *this; }
    constexpr mat2& operator-=(const mat2& o) noexcept { m00-=o.m00; m01-=o.m01; m10-=o.m10; m11-=o.m11; return *this; }
    constexpr mat2& operator*=(T s) noexcept { m00*=s; m01*=s; m10*=s; m11*=s; return *this; }
    constexpr mat2& operator/=(T s) noexcept { m00/=s; m01/=s; m10/=s; m11/=s; return *this; }
};

template <typename T> constexpr mat2<T> operator+(const mat2<T>& a, const mat2<T>& b) noexcept { return mat2<T>(a.m00+b.m00, a.m01+b.m01, a.m10+b.m10, a.m11+b.m11); }
template <typename T> constexpr mat2<T> operator-(const mat2<T>& a, const mat2<T>& b) noexcept { return mat2<T>(a.m00-b.m00, a.m01-b.m01, a.m10-b.m10, a.m11-b.m11); }
template <typename T> constexpr mat2<T> operator*(const mat2<T>& a, T s) noexcept { return mat2<T>(a.m00*s, a.m01*s, a.m10*s, a.m11*s); }
template <typename T> constexpr mat2<T> operator*(T s, const mat2<T>& a) noexcept { return mat2<T>(s*a.m00, s*a.m01, s*a.m10, s*a.m11); }
template <typename T> constexpr mat2<T> operator/(const mat2<T>& a, T s) noexcept { return mat2<T>(a.m00/s, a.m01/s, a.m10/s, a.m11/s); }
template <typename T> constexpr bool operator==(const mat2<T>& a, const mat2<T>& b) noexcept { return a.m00==b.m00&&a.m01==b.m01&&a.m10==b.m10&&a.m11==b.m11; }
template <typename T> constexpr bool operator!=(const mat2<T>& a, const mat2<T>& b) noexcept { return !(a==b); }

template <typename T> constexpr mat2<T> mul(const mat2<T>& a, const mat2<T>& b) noexcept {
    return mat2<T>(a.m00*b.m00 + a.m01*b.m10, a.m00*b.m01 + a.m01*b.m11,
                   a.m10*b.m00 + a.m11*b.m10, a.m10*b.m01 + a.m11*b.m11);
}
template <typename T> constexpr vec2<T> mul(const mat2<T>& m, const vec2<T>& v) noexcept {
    return vec2<T>(m.m00*v.x + m.m01*v.y, m.m10*v.x + m.m11*v.y);
}
template <typename T> constexpr vec2<T> mul(const vec2<T>& v, const mat2<T>& m) noexcept {
    return vec2<T>(v.x*m.m00 + v.y*m.m10, v.x*m.m01 + v.y*m.m11);
}

template <typename T> constexpr mat2<T> transpose(const mat2<T>& m) noexcept { return mat2<T>(m.m00, m.m10, m.m01, m.m11); }
template <typename T> constexpr T determinant(const mat2<T>& m) noexcept { return m.m00*m.m11 - m.m01*m.m10; }
template <typename T> constexpr T trace(const mat2<T>& m) noexcept { return m.m00 + m.m11; }
template <typename T> mat2<T> inverse(const mat2<T>& m) noexcept {
    T det = determinant(m);
    if (std::abs(det) <= MathConst<T>::epsilon) return mat2<T>(T(1));
    T inv_det = T(1) / det;
    return mat2<T>( m.m11*inv_det, -m.m01*inv_det,
                   -m.m10*inv_det,  m.m00*inv_det);
}
template <typename T> constexpr mat2<T> adjugate(const mat2<T>& m) noexcept { return mat2<T>(m.m11, -m.m01, -m.m10, m.m00); }
template <typename T> constexpr mat2<T> identity2() noexcept { return mat2<T>(T(1), T(0), T(0), T(1)); }
template <typename T> constexpr mat2<T> scaling2(const vec2<T>& s) noexcept { return mat2<T>(s.x, T(0), T(0), s.y); }
template <typename T> mat2<T> rotation2(T angle) noexcept {
    T c = std::cos(angle), s = std::sin(angle);
    return mat2<T>(c, -s, s, c);
}
template <typename T> constexpr mat2<T> shear2(T shear_x, T shear_y) noexcept { return mat2<T>(T(1), shear_x, shear_y, T(1)); }
template <typename T> constexpr mat2<T> reflect2(const vec2<T>& normal) noexcept {
    vec2<T> n = normalize(normal);
    T xx = T(1) - T(2) * n.x * n.x;
    T yy = T(1) - T(2) * n.y * n.y;
    T xy = -T(2) * n.x * n.y;
    return mat2<T>(xx, xy, xy, yy);
}

using mat2f = mat2<float>;
using mat2d = mat2<double>;

} // namespace wp