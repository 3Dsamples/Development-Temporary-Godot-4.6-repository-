// File 0007 : core/math/mat4.h
// 4×4 matrix (float/double) for 3D transformations, projections, and homogeneous coordinates.

#pragma once

#include "vec4.h"
#include "mat3.h"
#include <cassert>
#include <cmath>
#include <type_traits>

namespace wp {

template <typename T>
struct alignas(sizeof(T) * 16) mat4 {
    union {
        T data[4][4];
        T m[16];
        struct { T m00, m01, m02, m03,
                 m10, m11, m12, m13,
                 m20, m21, m22, m23,
                 m30, m31, m32, m33; };
    };

    constexpr mat4() noexcept
        : m00(T(1)), m01(T(0)), m02(T(0)), m03(T(0)),
          m10(T(0)), m11(T(1)), m12(T(0)), m13(T(0)),
          m20(T(0)), m21(T(0)), m22(T(1)), m23(T(0)),
          m30(T(0)), m31(T(0)), m32(T(0)), m33(T(1)) {}
    constexpr explicit mat4(T s) noexcept
        : m00(s),    m01(T(0)), m02(T(0)), m03(T(0)),
          m10(T(0)), m11(s),    m12(T(0)), m13(T(0)),
          m20(T(0)), m21(T(0)), m22(s),    m23(T(0)),
          m30(T(0)), m31(T(0)), m32(T(0)), m33(s) {}
    constexpr mat4(T m00_, T m01_, T m02_, T m03_,
                   T m10_, T m11_, T m12_, T m13_,
                   T m20_, T m21_, T m22_, T m23_,
                   T m30_, T m31_, T m32_, T m33_) noexcept
        : m00(m00_), m01(m01_), m02(m02_), m03(m03_),
          m10(m10_), m11(m11_), m12(m12_), m13(m13_),
          m20(m20_), m21(m21_), m22(m22_), m23(m23_),
          m30(m30_), m31(m31_), m32(m32_), m33(m33_) {}
    constexpr mat4(const vec4<T>& col0, const vec4<T>& col1, const vec4<T>& col2, const vec4<T>& col3) noexcept
        : m00(col0.x), m10(col0.y), m20(col0.z), m30(col0.w),
          m01(col1.x), m11(col1.y), m21(col1.z), m31(col1.w),
          m02(col2.x), m12(col2.y), m22(col2.z), m32(col2.w),
          m03(col3.x), m13(col3.y), m23(col3.z), m33(col3.w) {}
    template <typename U> constexpr explicit mat4(const mat4<U>& o) noexcept
        : m00(static_cast<T>(o.m00)), m01(static_cast<T>(o.m01)), m02(static_cast<T>(o.m02)), m03(static_cast<T>(o.m03)),
          m10(static_cast<T>(o.m10)), m11(static_cast<T>(o.m11)), m12(static_cast<T>(o.m12)), m13(static_cast<T>(o.m13)),
          m20(static_cast<T>(o.m20)), m21(static_cast<T>(o.m21)), m22(static_cast<T>(o.m22)), m23(static_cast<T>(o.m23)),
          m30(static_cast<T>(o.m30)), m31(static_cast<T>(o.m31)), m32(static_cast<T>(o.m32)), m33(static_cast<T>(o.m33)) {}

    constexpr T  operator()(int row, int col) const noexcept { assert(row>=0&&row<4&&col>=0&&col<4); return data[row][col]; }
    constexpr T& operator()(int row, int col)       noexcept { assert(row>=0&&row<4&&col>=0&&col<4); return data[row][col]; }

    constexpr vec4<T> col(int i) const noexcept { assert(i>=0&&i<4); return vec4<T>(data[0][i], data[1][i], data[2][i], data[3][i]); }
    constexpr vec4<T> row(int i) const noexcept { assert(i>=0&&i<4); return vec4<T>(data[i][0], data[i][1], data[i][2], data[i][3]); }

    constexpr mat4 operator+() const noexcept { return *this; }
    constexpr mat4 operator-() const noexcept {
        return mat4(-m00, -m01, -m02, -m03,
                    -m10, -m11, -m12, -m13,
                    -m20, -m21, -m22, -m23,
                    -m30, -m31, -m32, -m33);
    }

    constexpr mat4& operator+=(const mat4& o) noexcept {
        m00+=o.m00; m01+=o.m01; m02+=o.m02; m03+=o.m03;
        m10+=o.m10; m11+=o.m11; m12+=o.m12; m13+=o.m13;
        m20+=o.m20; m21+=o.m21; m22+=o.m22; m23+=o.m23;
        m30+=o.m30; m31+=o.m31; m32+=o.m32; m33+=o.m33;
        return *this;
    }
    constexpr mat4& operator-=(const mat4& o) noexcept {
        m00-=o.m00; m01-=o.m01; m02-=o.m02; m03-=o.m03;
        m10-=o.m10; m11-=o.m11; m12-=o.m12; m13-=o.m13;
        m20-=o.m20; m21-=o.m21; m22-=o.m22; m23-=o.m23;
        m30-=o.m30; m31-=o.m31; m32-=o.m32; m33-=o.m33;
        return *this;
    }
    constexpr mat4& operator*=(T s) noexcept {
        m00*=s; m01*=s; m02*=s; m03*=s;
        m10*=s; m11*=s; m12*=s; m13*=s;
        m20*=s; m21*=s; m22*=s; m23*=s;
        m30*=s; m31*=s; m32*=s; m33*=s;
        return *this;
    }
    constexpr mat4& operator/=(T s) noexcept {
        m00/=s; m01/=s; m02/=s; m03/=s;
        m10/=s; m11/=s; m12/=s; m13/=s;
        m20/=s; m21/=s; m22/=s; m23/=s;
        m30/=s; m31/=s; m32/=s; m33/=s;
        return *this;
    }
};

template <typename T> constexpr mat4<T> operator+(const mat4<T>& a, const mat4<T>& b) noexcept {
    return mat4<T>(a.m00+b.m00, a.m01+b.m01, a.m02+b.m02, a.m03+b.m03,
                   a.m10+b.m10, a.m11+b.m11, a.m12+b.m12, a.m13+b.m13,
                   a.m20+b.m20, a.m21+b.m21, a.m22+b.m22, a.m23+b.m23,
                   a.m30+b.m30, a.m31+b.m31, a.m32+b.m32, a.m33+b.m33);
}
template <typename T> constexpr mat4<T> operator-(const mat4<T>& a, const mat4<T>& b) noexcept {
    return mat4<T>(a.m00-b.m00, a.m01-b.m01, a.m02-b.m02, a.m03-b.m03,
                   a.m10-b.m10, a.m11-b.m11, a.m12-b.m12, a.m13-b.m13,
                   a.m20-b.m20, a.m21-b.m21, a.m22-b.m22, a.m23-b.m23,
                   a.m30-b.m30, a.m31-b.m31, a.m32-b.m32, a.m33-b.m33);
}
template <typename T> constexpr mat4<T> operator*(const mat4<T>& a, T s) noexcept {
    return mat4<T>(a.m00*s, a.m01*s, a.m02*s, a.m03*s,
                   a.m10*s, a.m11*s, a.m12*s, a.m13*s,
                   a.m20*s, a.m21*s, a.m22*s, a.m23*s,
                   a.m30*s, a.m31*s, a.m32*s, a.m33*s);
}
template <typename T> constexpr mat4<T> operator*(T s, const mat4<T>& a) noexcept {
    return mat4<T>(s*a.m00, s*a.m01, s*a.m02, s*a.m03,
                   s*a.m10, s*a.m11, s*a.m12, s*a.m13,
                   s*a.m20, s*a.m21, s*a.m22, s*a.m23,
                   s*a.m30, s*a.m31, s*a.m32, s*a.m33);
}
template <typename T> constexpr mat4<T> operator/(const mat4<T>& a, T s) noexcept {
    return mat4<T>(a.m00/s, a.m01/s, a.m02/s, a.m03/s,
                   a.m10/s, a.m11/s, a.m12/s, a.m13/s,
                   a.m20/s, a.m21/s, a.m22/s, a.m23/s,
                   a.m30/s, a.m31/s, a.m32/s, a.m33/s);
}
template <typename T> constexpr bool operator==(const mat4<T>& a, const mat4<T>& b) noexcept {
    return a.m00==b.m00 && a.m01==b.m01 && a.m02==b.m02 && a.m03==b.m03 &&
           a.m10==b.m10 && a.m11==b.m11 && a.m12==b.m12 && a.m13==b.m13 &&
           a.m20==b.m20 && a.m21==b.m21 && a.m22==b.m22 && a.m23==b.m23 &&
           a.m30==b.m30 && a.m31==b.m31 && a.m32==b.m32 && a.m33==b.m33;
}
template <typename T> constexpr bool operator!=(const mat4<T>& a, const mat4<T>& b) noexcept { return !(a==b); }

template <typename T> constexpr mat4<T> mul(const mat4<T>& a, const mat4<T>& b) noexcept {
    return mat4<T>(
        a.m00*b.m00 + a.m01*b.m10 + a.m02*b.m20 + a.m03*b.m30,
        a.m00*b.m01 + a.m01*b.m11 + a.m02*b.m21 + a.m03*b.m31,
        a.m00*b.m02 + a.m01*b.m12 + a.m02*b.m22 + a.m03*b.m32,
        a.m00*b.m03 + a.m01*b.m13 + a.m02*b.m23 + a.m03*b.m33,
        a.m10*b.m00 + a.m11*b.m10 + a.m12*b.m20 + a.m13*b.m30,
        a.m10*b.m01 + a.m11*b.m11 + a.m12*b.m21 + a.m13*b.m31,
        a.m10*b.m02 + a.m11*b.m12 + a.m12*b.m22 + a.m13*b.m32,
        a.m10*b.m03 + a.m11*b.m13 + a.m12*b.m23 + a.m13*b.m33,
        a.m20*b.m00 + a.m21*b.m10 + a.m22*b.m20 + a.m23*b.m30,
        a.m20*b.m01 + a.m21*b.m11 + a.m22*b.m21 + a.m23*b.m31,
        a.m20*b.m02 + a.m21*b.m12 + a.m22*b.m22 + a.m23*b.m32,
        a.m20*b.m03 + a.m21*b.m13 + a.m22*b.m23 + a.m23*b.m33,
        a.m30*b.m00 + a.m31*b.m10 + a.m32*b.m20 + a.m33*b.m30,
        a.m30*b.m01 + a.m31*b.m11 + a.m32*b.m21 + a.m33*b.m31,
        a.m30*b.m02 + a.m31*b.m12 + a.m32*b.m22 + a.m33*b.m32,
        a.m30*b.m03 + a.m31*b.m13 + a.m32*b.m23 + a.m33*b.m33
    );
}
template <typename T> constexpr vec4<T> mul(const mat4<T>& m, const vec4<T>& v) noexcept {
    return vec4<T>(
        m.m00*v.x + m.m01*v.y + m.m02*v.z + m.m03*v.w,
        m.m10*v.x + m.m11*v.y + m.m12*v.z + m.m13*v.w,
        m.m20*v.x + m.m21*v.y + m.m22*v.z + m.m23*v.w,
        m.m30*v.x + m.m31*v.y + m.m32*v.z + m.m33*v.w
    );
}
template <typename T> constexpr vec4<T> mul(const vec4<T>& v, const mat4<T>& m) noexcept {
    return vec4<T>(
        v.x*m.m00 + v.y*m.m10 + v.z*m.m20 + v.w*m.m30,
        v.x*m.m01 + v.y*m.m11 + v.z*m.m21 + v.w*m.m31,
        v.x*m.m02 + v.y*m.m12 + v.z*m.m22 + v.w*m.m32,
        v.x*m.m03 + v.y*m.m13 + v.z*m.m23 + v.w*m.m33
    );
}

template <typename T> constexpr mat4<T> transpose(const mat4<T>& m) noexcept {
    return mat4<T>(m.m00, m.m10, m.m20, m.m30,
                   m.m01, m.m11, m.m21, m.m31,
                   m.m02, m.m12, m.m22, m.m32,
                   m.m03, m.m13, m.m23, m.m33);
}
template <typename T> constexpr T trace(const mat4<T>& m) noexcept { return m.m00 + m.m11 + m.m22 + m.m33; }

template <typename T> T determinant(const mat4<T>& m) noexcept {
    T s0 = m.m00*m.m11 - m.m01*m.m10;
    T s1 = m.m00*m.m12 - m.m02*m.m10;
    T s2 = m.m00*m.m13 - m.m03*m.m10;
    T s3 = m.m01*m.m12 - m.m02*m.m11;
    T s4 = m.m01*m.m13 - m.m03*m.m11;
    T s5 = m.m02*m.m13 - m.m03*m.m12;
    T c0 = m.m20*m.m31 - m.m21*m.m30;
    T c1 = m.m20*m.m32 - m.m22*m.m30;
    T c2 = m.m20*m.m33 - m.m23*m.m30;
    T c3 = m.m21*m.m32 - m.m22*m.m31;
    T c4 = m.m21*m.m33 - m.m23*m.m31;
    T c5 = m.m22*m.m33 - m.m23*m.m32;
    return s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0;
}

template <typename T> mat4<T> inverse(const mat4<T>& m) noexcept {
    T s0 = m.m00*m.m11 - m.m01*m.m10;
    T s1 = m.m00*m.m12 - m.m02*m.m10;
    T s2 = m.m00*m.m13 - m.m03*m.m10;
    T s3 = m.m01*m.m12 - m.m02*m.m11;
    T s4 = m.m01*m.m13 - m.m03*m.m11;
    T s5 = m.m02*m.m13 - m.m03*m.m12;
    T c0 = m.m20*m.m31 - m.m21*m.m30;
    T c1 = m.m20*m.m32 - m.m22*m.m30;
    T c2 = m.m20*m.m33 - m.m23*m.m30;
    T c3 = m.m21*m.m32 - m.m22*m.m31;
    T c4 = m.m21*m.m33 - m.m23*m.m31;
    T c5 = m.m22*m.m33 - m.m23*m.m32;
    T det = s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0;
    if (std::abs(det) <= MathConst<T>::epsilon) return mat4<T>(T(1));
    T inv_det = T(1) / det;
    return mat4<T>(
        ( m.m11*c5 - m.m12*c4 + m.m13*c3) * inv_det,
        (-m.m01*c5 + m.m02*c4 - m.m03*c3) * inv_det,
        ( m.m31*s5 - m.m32*s4 + m.m33*s3) * inv_det,
        (-m.m21*s5 + m.m22*s4 - m.m23*s3) * inv_det,
        (-m.m10*c5 + m.m12*c2 - m.m13*c1) * inv_det,
        ( m.m00*c5 - m.m02*c2 + m.m03*c1) * inv_det,
        (-m.m30*s5 + m.m32*s2 - m.m33*s1) * inv_det,
        ( m.m20*s5 - m.m22*s2 + m.m23*s1) * inv_det,
        ( m.m10*c4 - m.m11*c2 + m.m13*c0) * inv_det,
        (-m.m00*c4 + m.m01*c2 - m.m03*c0) * inv_det,
        ( m.m30*s4 - m.m31*s2 + m.m33*s0) * inv_det,
        (-m.m20*s4 + m.m21*s2 - m.m23*s0) * inv_det,
        (-m.m10*c3 + m.m11*c1 - m.m12*c0) * inv_det,
        ( m.m00*c3 - m.m01*c1 + m.m02*c0) * inv_det,
        (-m.m30*s3 + m.m31*s1 - m.m32*s0) * inv_det,
        ( m.m20*s3 - m.m21*s1 + m.m22*s0) * inv_det
    );
}

template <typename T> mat4<T> inverse_affine(const mat4<T>& m) noexcept {
    T det3 = m.m00*(m.m11*m.m22 - m.m12*m.m21)
           - m.m01*(m.m10*m.m22 - m.m12*m.m20)
           + m.m02*(m.m10*m.m21 - m.m11*m.m20);
    if (std::abs(det3) <= MathConst<T>::epsilon) return mat4<T>(T(1));
    T inv_det3 = T(1) / det3;
    T r00 = (m.m11*m.m22 - m.m12*m.m21) * inv_det3;
    T r01 = (m.m02*m.m21 - m.m01*m.m22) * inv_det3;
    T r02 = (m.m01*m.m12 - m.m02*m.m11) * inv_det3;
    T r10 = (m.m12*m.m20 - m.m10*m.m22) * inv_det3;
    T r11 = (m.m00*m.m22 - m.m02*m.m20) * inv_det3;
    T r12 = (m.m02*m.m10 - m.m00*m.m12) * inv_det3;
    T r20 = (m.m10*m.m21 - m.m11*m.m20) * inv_det3;
    T r21 = (m.m01*m.m20 - m.m00*m.m21) * inv_det3;
    T r22 = (m.m00*m.m11 - m.m01*m.m10) * inv_det3;
    T tx = -(r00*m.m03 + r01*m.m13 + r02*m.m23);
    T ty = -(r10*m.m03 + r11*m.m13 + r12*m.m23);
    T tz = -(r20*m.m03 + r21*m.m13 + r22*m.m23);
    return mat4<T>(r00, r01, r02, tx,
                   r10, r11, r12, ty,
                   r20, r21, r22, tz,
                   T(0), T(0), T(0), T(1));
}

template <typename T> constexpr mat4<T> identity4() noexcept {
    return mat4<T>(T(1), T(0), T(0), T(0),
                   T(0), T(1), T(0), T(0),
                   T(0), T(0), T(1), T(0),
                   T(0), T(0), T(0), T(1));
}
template <typename T> constexpr mat4<T> zero4() noexcept {
    return mat4<T>(T(0), T(0), T(0), T(0),
                   T(0), T(0), T(0), T(0),
                   T(0), T(0), T(0), T(0),
                   T(0), T(0), T(0), T(0));
}

template <typename T> constexpr mat4<T> translation4(const vec3<T>& t) noexcept {
    return mat4<T>(T(1), T(0), T(0), t.x,
                   T(0), T(1), T(0), t.y,
                   T(0), T(0), T(1), t.z,
                   T(0), T(0), T(0), T(1));
}
template <typename T> constexpr mat4<T> scaling4(const vec3<T>& s) noexcept {
    return mat4<T>(s.x, T(0), T(0), T(0),
                   T(0), s.y, T(0), T(0),
                   T(0), T(0), s.z, T(0),
                   T(0), T(0), T(0), T(1));
}
template <typename T> mat4<T> rotation_x4(T angle) noexcept {
    T c = std::cos(angle), s = std::sin(angle);
    return mat4<T>(T(1), T(0), T(0), T(0),
                   T(0), c,   -s,   T(0),
                   T(0), s,    c,   T(0),
                   T(0), T(0), T(0), T(1));
}
template <typename T> mat4<T> rotation_y4(T angle) noexcept {
    T c = std::cos(angle), s = std::sin(angle);
    return mat4<T>(c,   T(0), s,   T(0),
                   T(0), T(1), T(0), T(0),
                   -s,   T(0), c,   T(0),
                   T(0), T(0), T(0), T(1));
}
template <typename T> mat4<T> rotation_z4(T angle) noexcept {
    T c = std::cos(angle), s = std::sin(angle);
    return mat4<T>(c,   -s,   T(0), T(0),
                   s,    c,   T(0), T(0),
                   T(0), T(0), T(1), T(0),
                   T(0), T(0), T(0), T(1));
}
template <typename T> mat4<T> rotation_axis_angle4(const vec3<T>& axis, T angle) noexcept {
    vec3<T> u = normalize(axis);
    T c = std::cos(angle), s = std::sin(angle), t = T(1) - c;
    return mat4<T>(
        t*u.x*u.x + c,        t*u.x*u.y - s*u.z,   t*u.x*u.z + s*u.y,   T(0),
        t*u.x*u.y + s*u.z,    t*u.y*u.y + c,        t*u.y*u.z - s*u.x,   T(0),
        t*u.x*u.z - s*u.y,    t*u.y*u.z + s*u.x,    t*u.z*u.z + c,        T(0),
        T(0), T(0), T(0), T(1)
    );
}
template <typename T> constexpr mat4<T> look_at(const vec3<T>& eye, const vec3<T>& center, const vec3<T>& up) noexcept {
    vec3<T> f = normalize(center - eye);
    vec3<T> s = normalize(cross(f, up));
    vec3<T> u = cross(s, f);
    return mat4<T>( s.x,  s.y,  s.z, -dot(s, eye),
                    u.x,  u.y,  u.z, -dot(u, eye),
                   -f.x, -f.y, -f.z,  dot(f, eye),
                   T(0), T(0), T(0), T(1));
}
template <typename T> mat4<T> perspective(T fovy, T aspect, T near, T far) noexcept {
    T f = T(1) / std::tan(fovy * T(0.5));
    T d = far - near;
    return mat4<T>(f/aspect, T(0), T(0), T(0),
                   T(0), f, T(0), T(0),
                   T(0), T(0), -(far+near)/d, -(T(2)*far*near)/d,
                   T(0), T(0), T(-1), T(0));
}
template <typename T> mat4<T> ortho(T left, T right, T bottom, T top, T near, T far) noexcept {
    return mat4<T>(T(2)/(right-left), T(0), T(0), -(right+left)/(right-left),
                   T(0), T(2)/(top-bottom), T(0), -(top+bottom)/(top-bottom),
                   T(0), T(0), T(-2)/(far-near), -(far+near)/(far-near),
                   T(0), T(0), T(0), T(1));
}
template <typename T> mat4<T> frustum(T left, T right, T bottom, T top, T near, T far) noexcept {
    T d = far - near;
    return mat4<T>(T(2)*near/(right-left), T(0), (right+left)/(right-left), T(0),
                   T(0), T(2)*near/(top-bottom), (top+bottom)/(top-bottom), T(0),
                   T(0), T(0), -(far+near)/d, -(T(2)*far*near)/d,
                   T(0), T(0), T(-1), T(0));
}

template <typename T> constexpr mat3<T> upper_left_3x3(const mat4<T>& m) noexcept {
    return mat3<T>(m.m00, m.m01, m.m02,
                   m.m10, m.m11, m.m12,
                   m.m20, m.m21, m.m22);
}
template <typename T> constexpr mat4<T> from_mat3_translation(const mat3<T>& m, const vec3<T>& t) noexcept {
    return mat4<T>(m.m00, m.m01, m.m02, t.x,
                   m.m10, m.m11, m.m12, t.y,
                   m.m20, m.m21, m.m22, t.z,
                   T(0), T(0), T(0), T(1));
}
template <typename T> constexpr mat4<T> from_mat3(const mat3<T>& m) noexcept {
    return mat4<T>(m.m00, m.m01, m.m02, T(0),
                   m.m10, m.m11, m.m12, T(0),
                   m.m20, m.m21, m.m22, T(0),
                   T(0), T(0), T(0), T(1));
}
template <typename T> constexpr void decompose_affine(const mat4<T>& m, vec3<T>& translation, mat3<T>& rotation, vec3<T>& scale) {
    rotation = upper_left_3x3(m);
    T sx = length(rotation.col(0));
    T sy = length(rotation.col(1));
    T sz = length(rotation.col(2));
    scale = vec3<T>(sx, sy, sz);
    rotation = mat3<T>(rotation.col(0)/sx, rotation.col(1)/sy, rotation.col(2)/sz);
    translation = vec3<T>(m.m03, m.m13, m.m23);
}

using mat4f = mat4<float>;
using mat4d = mat4<double>;

} // namespace wp