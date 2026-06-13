// File 0006 : core/math/mat3.h
// 3×3 matrix (float/double) with arithmetic, determinant, inverse, transpose, transformations, and linear algebra utilities.

#pragma once

#include "vec3.h"
#include <cassert>
#include <cmath>
#include <type_traits>

namespace wp {

template <typename T>
struct alignas(sizeof(T) * 9) mat3 {
    union {
        T data[3][3];
        T m[9];
        struct { T m00, m01, m02, m10, m11, m12, m20, m21, m22; };
    };

    constexpr mat3() noexcept : m00(T(1)), m01(T(0)), m02(T(0)),
                                 m10(T(0)), m11(T(1)), m12(T(0)),
                                 m20(T(0)), m21(T(0)), m22(T(1)) {}
    constexpr explicit mat3(T s) noexcept : m00(s), m01(T(0)), m02(T(0)),
                                             m10(T(0)), m11(s), m12(T(0)),
                                             m20(T(0)), m21(T(0)), m22(s) {}
    constexpr mat3(T m00_, T m01_, T m02_,
                   T m10_, T m11_, T m12_,
                   T m20_, T m21_, T m22_) noexcept
        : m00(m00_), m01(m01_), m02(m02_),
          m10(m10_), m11(m11_), m12(m12_),
          m20(m20_), m21(m21_), m22(m22_) {}
    constexpr mat3(const vec3<T>& col0, const vec3<T>& col1, const vec3<T>& col2) noexcept
        : m00(col0.x), m10(col0.y), m20(col0.z),
          m01(col1.x), m11(col1.y), m21(col1.z),
          m02(col2.x), m12(col2.y), m22(col2.z) {}
    template <typename U> constexpr explicit mat3(const mat3<U>& o) noexcept
        : m00(static_cast<T>(o.m00)), m01(static_cast<T>(o.m01)), m02(static_cast<T>(o.m02)),
          m10(static_cast<T>(o.m10)), m11(static_cast<T>(o.m11)), m12(static_cast<T>(o.m12)),
          m20(static_cast<T>(o.m20)), m21(static_cast<T>(o.m21)), m22(static_cast<T>(o.m22)) {}

    constexpr T  operator()(int row, int col) const noexcept { assert(row>=0&&row<3&&col>=0&&col<3); return data[row][col]; }
    constexpr T& operator()(int row, int col)       noexcept { assert(row>=0&&row<3&&col>=0&&col<3); return data[row][col]; }

    constexpr vec3<T> col(int i) const noexcept { assert(i>=0&&i<3); return vec3<T>(data[0][i], data[1][i], data[2][i]); }
    constexpr vec3<T> row(int i) const noexcept { assert(i>=0&&i<3); return vec3<T>(data[i][0], data[i][1], data[i][2]); }

    constexpr mat3 operator+() const noexcept { return *this; }
    constexpr mat3 operator-() const noexcept { return mat3(-m00, -m01, -m02, -m10, -m11, -m12, -m20, -m21, -m22); }

    constexpr mat3& operator+=(const mat3& o) noexcept {
        m00+=o.m00; m01+=o.m01; m02+=o.m02;
        m10+=o.m10; m11+=o.m11; m12+=o.m12;
        m20+=o.m20; m21+=o.m21; m22+=o.m22;
        return *this;
    }
    constexpr mat3& operator-=(const mat3& o) noexcept {
        m00-=o.m00; m01-=o.m01; m02-=o.m02;
        m10-=o.m10; m11-=o.m11; m12-=o.m12;
        m20-=o.m20; m21-=o.m21; m22-=o.m22;
        return *this;
    }
    constexpr mat3& operator*=(T s) noexcept {
        m00*=s; m01*=s; m02*=s;
        m10*=s; m11*=s; m12*=s;
        m20*=s; m21*=s; m22*=s;
        return *this;
    }
    constexpr mat3& operator/=(T s) noexcept {
        m00/=s; m01/=s; m02/=s;
        m10/=s; m11/=s; m12/=s;
        m20/=s; m21/=s; m22/=s;
        return *this;
    }
};

template <typename T> constexpr mat3<T> operator+(const mat3<T>& a, const mat3<T>& b) noexcept {
    return mat3<T>(a.m00+b.m00, a.m01+b.m01, a.m02+b.m02,
                   a.m10+b.m10, a.m11+b.m11, a.m12+b.m12,
                   a.m20+b.m20, a.m21+b.m21, a.m22+b.m22);
}
template <typename T> constexpr mat3<T> operator-(const mat3<T>& a, const mat3<T>& b) noexcept {
    return mat3<T>(a.m00-b.m00, a.m01-b.m01, a.m02-b.m02,
                   a.m10-b.m10, a.m11-b.m11, a.m12-b.m12,
                   a.m20-b.m20, a.m21-b.m21, a.m22-b.m22);
}
template <typename T> constexpr mat3<T> operator*(const mat3<T>& a, T s) noexcept {
    return mat3<T>(a.m00*s, a.m01*s, a.m02*s,
                   a.m10*s, a.m11*s, a.m12*s,
                   a.m20*s, a.m21*s, a.m22*s);
}
template <typename T> constexpr mat3<T> operator*(T s, const mat3<T>& a) noexcept {
    return mat3<T>(s*a.m00, s*a.m01, s*a.m02,
                   s*a.m10, s*a.m11, s*a.m12,
                   s*a.m20, s*a.m21, s*a.m22);
}
template <typename T> constexpr mat3<T> operator/(const mat3<T>& a, T s) noexcept {
    return mat3<T>(a.m00/s, a.m01/s, a.m02/s,
                   a.m10/s, a.m11/s, a.m12/s,
                   a.m20/s, a.m21/s, a.m22/s);
}
template <typename T> constexpr bool operator==(const mat3<T>& a, const mat3<T>& b) noexcept {
    return a.m00==b.m00 && a.m01==b.m01 && a.m02==b.m02 &&
           a.m10==b.m10 && a.m11==b.m11 && a.m12==b.m12 &&
           a.m20==b.m20 && a.m21==b.m21 && a.m22==b.m22;
}
template <typename T> constexpr bool operator!=(const mat3<T>& a, const mat3<T>& b) noexcept { return !(a==b); }

template <typename T> constexpr mat3<T> mul(const mat3<T>& a, const mat3<T>& b) noexcept {
    return mat3<T>(
        a.m00*b.m00 + a.m01*b.m10 + a.m02*b.m20,
        a.m00*b.m01 + a.m01*b.m11 + a.m02*b.m21,
        a.m00*b.m02 + a.m01*b.m12 + a.m02*b.m22,
        a.m10*b.m00 + a.m11*b.m10 + a.m12*b.m20,
        a.m10*b.m01 + a.m11*b.m11 + a.m12*b.m21,
        a.m10*b.m02 + a.m11*b.m12 + a.m12*b.m22,
        a.m20*b.m00 + a.m21*b.m10 + a.m22*b.m20,
        a.m20*b.m01 + a.m21*b.m11 + a.m22*b.m21,
        a.m20*b.m02 + a.m21*b.m12 + a.m22*b.m22
    );
}
template <typename T> constexpr vec3<T> mul(const mat3<T>& m, const vec3<T>& v) noexcept {
    return vec3<T>(
        m.m00*v.x + m.m01*v.y + m.m02*v.z,
        m.m10*v.x + m.m11*v.y + m.m12*v.z,
        m.m20*v.x + m.m21*v.y + m.m22*v.z
    );
}
template <typename T> constexpr vec3<T> mul(const vec3<T>& v, const mat3<T>& m) noexcept {
    return vec3<T>(
        v.x*m.m00 + v.y*m.m10 + v.z*m.m20,
        v.x*m.m01 + v.y*m.m11 + v.z*m.m21,
        v.x*m.m02 + v.y*m.m12 + v.z*m.m22
    );
}

template <typename T> constexpr mat3<T> transpose(const mat3<T>& m) noexcept {
    return mat3<T>(m.m00, m.m10, m.m20,
                   m.m01, m.m11, m.m21,
                   m.m02, m.m12, m.m22);
}
template <typename T> constexpr T trace(const mat3<T>& m) noexcept { return m.m00 + m.m11 + m.m22; }
template <typename T> constexpr T determinant(const mat3<T>& m) noexcept {
    return m.m00 * (m.m11*m.m22 - m.m12*m.m21)
         - m.m01 * (m.m10*m.m22 - m.m12*m.m20)
         + m.m02 * (m.m10*m.m21 - m.m11*m.m20);
}
template <typename T> mat3<T> inverse(const mat3<T>& m) noexcept {
    T det = determinant(m);
    if (std::abs(det) <= MathConst<T>::epsilon) return mat3<T>(T(1));
    T inv_det = T(1) / det;
    return mat3<T>(
        (m.m11*m.m22 - m.m12*m.m21) * inv_det,
        (m.m02*m.m21 - m.m01*m.m22) * inv_det,
        (m.m01*m.m12 - m.m02*m.m11) * inv_det,
        (m.m12*m.m20 - m.m10*m.m22) * inv_det,
        (m.m00*m.m22 - m.m02*m.m20) * inv_det,
        (m.m02*m.m10 - m.m00*m.m12) * inv_det,
        (m.m10*m.m21 - m.m11*m.m20) * inv_det,
        (m.m01*m.m20 - m.m00*m.m21) * inv_det,
        (m.m00*m.m11 - m.m01*m.m10) * inv_det
    );
}
template <typename T> constexpr mat3<T> adjugate(const mat3<T>& m) noexcept {
    return mat3<T>(
         m.m11*m.m22 - m.m12*m.m21,  m.m02*m.m21 - m.m01*m.m22,  m.m01*m.m12 - m.m02*m.m11,
         m.m12*m.m20 - m.m10*m.m22,  m.m00*m.m22 - m.m02*m.m20,  m.m02*m.m10 - m.m00*m.m12,
         m.m10*m.m21 - m.m11*m.m20,  m.m01*m.m20 - m.m00*m.m21,  m.m00*m.m11 - m.m01*m.m10
    );
}
template <typename T> constexpr mat3<T> outer_product(const vec3<T>& a, const vec3<T>& b) noexcept {
    return mat3<T>(a.x*b.x, a.x*b.y, a.x*b.z,
                   a.y*b.x, a.y*b.y, a.y*b.z,
                   a.z*b.x, a.z*b.y, a.z*b.z);
}
template <typename T> constexpr mat3<T> scaling3(const vec3<T>& s) noexcept {
    return mat3<T>(s.x, T(0), T(0),
                   T(0), s.y, T(0),
                   T(0), T(0), s.z);
}
template <typename T> mat3<T> rotation_x(T angle) noexcept {
    T c = std::cos(angle), s = std::sin(angle);
    return mat3<T>(T(1), T(0), T(0),
                   T(0),    c,   -s,
                   T(0),    s,    c);
}
template <typename T> mat3<T> rotation_y(T angle) noexcept {
    T c = std::cos(angle), s = std::sin(angle);
    return mat3<T>(   c, T(0),    s,
                   T(0), T(1), T(0),
                    -s, T(0),    c);
}
template <typename T> mat3<T> rotation_z(T angle) noexcept {
    T c = std::cos(angle), s = std::sin(angle);
    return mat3<T>(   c,   -s, T(0),
                      s,    c, T(0),
                   T(0), T(0), T(1));
}
template <typename T> mat3<T> rotation_axis_angle(const vec3<T>& axis, T angle) noexcept {
    vec3<T> u = normalize(axis);
    T c = std::cos(angle), s = std::sin(angle), t = T(1) - c;
    return mat3<T>(
        t*u.x*u.x + c,        t*u.x*u.y - s*u.z,   t*u.x*u.z + s*u.y,
        t*u.x*u.y + s*u.z,    t*u.y*u.y + c,        t*u.y*u.z - s*u.x,
        t*u.x*u.z - s*u.y,    t*u.y*u.z + s*u.x,    t*u.z*u.z + c
    );
}
template <typename T> constexpr mat3<T> reflect3(const vec3<T>& normal) noexcept {
    vec3<T> n = normalize(normal);
    T xx = T(1) - T(2) * n.x * n.x;
    T yy = T(1) - T(2) * n.y * n.y;
    T zz = T(1) - T(2) * n.z * n.z;
    T xy = -T(2) * n.x * n.y;
    T xz = -T(2) * n.x * n.z;
    T yz = -T(2) * n.y * n.z;
    return mat3<T>(xx, xy, xz,
                   xy, yy, yz,
                   xz, yz, zz);
}
template <typename T> constexpr mat3<T> shear3(T shear_xy, T shear_xz, T shear_yx, T shear_yz, T shear_zx, T shear_zy) noexcept {
    return mat3<T>(T(1), shear_yx, shear_zx,
                   shear_xy, T(1), shear_zy,
                   shear_xz, shear_yz, T(1));
}
template <typename T> constexpr mat3<T> identity3() noexcept { return mat3<T>(T(1), T(0), T(0), T(0), T(1), T(0), T(0), T(0), T(1)); }
template <typename T> constexpr mat3<T> zero3() noexcept { return mat3<T>(T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0)); }

// QR decomposition using Gram-Schmidt (return Q and R)
template <typename T> void qr_decompose(const mat3<T>& m, mat3<T>& Q, mat3<T>& R) {
    vec3<T> a0 = m.col(0);
    vec3<T> a1 = m.col(1);
    vec3<T> a2 = m.col(2);
    vec3<T> u0 = a0;
    vec3<T> e0 = normalize(u0);
    vec3<T> u1 = a1 - project(a1, e0);
    vec3<T> e1 = normalize(u1);
    vec3<T> u2 = a2 - project(a2, e0) - project(a2, e1);
    vec3<T> e2 = normalize(u2);
    Q = mat3<T>(e0, e1, e2);
    R = mat3<T>(dot(e0,a0), dot(e0,a1), dot(e0,a2),
                T(0),         dot(e1,a1), dot(e1,a2),
                T(0),         T(0),         dot(e2,a2));
}
template <typename T> mat3<T> gram_schmidt(const mat3<T>& m) noexcept {
    vec3<T> c0 = m.col(0);
    vec3<T> c1 = m.col(1);
    vec3<T> c2 = m.col(2);
    vec3<T> u0 = c0;
    vec3<T> e0 = normalize(u0);
    vec3<T> u1 = c1 - project(c1, e0);
    vec3<T> e1 = normalize(u1);
    vec3<T> u2 = c2 - project(c2, e0) - project(c2, e1);
    vec3<T> e2 = normalize(u2);
    return mat3<T>(e0, e1, e2);
}
template <typename T> constexpr mat3<T> basis_from_z(const vec3<T>& z) noexcept {
    vec3<T> zz = normalize(z);
    vec3<T> x;
    if (std::abs(zz.x) < std::abs(zz.y) && std::abs(zz.x) < std::abs(zz.z))
        x = vec3<T>(T(0), -zz.z, zz.y);
    else if (std::abs(zz.y) < std::abs(zz.z))
        x = vec3<T>(-zz.z, T(0), zz.x);
    else
        x = vec3<T>(-zz.y, zz.x, T(0));
    x = normalize(x);
    vec3<T> y = cross(zz, x);
    return mat3<T>(x, y, zz);
}

using mat3f = mat3<float>;
using mat3d = mat3<double>;

} // namespace wp