// system name : onetbb-warp
// File 0008 : core/math/matrix4.h
// Description : 4x4 matrix for affine and projective transformations.

#ifndef __TBB_WARP_CORE_MATH_MATRIX4_H
#define __TBB_WARP_CORE_MATH_MATRIX4_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector3.h"
#include "core/math/vector4.h"
#include "core/math/quaternion.h"
#include "core/math/matrix3.h"
#include <cmath>
#include <type_traits>
#include <array>
#include <initializer_list>
#include <algorithm>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Matrix4 class template (column‑major storage)
// ============================================================

template<typename T>
struct matrix4 {
    using value_type = T;
    using col_type = vector4<T>;
    using row_type = std::array<T,4>;

    col_type col[4];

    // ---- Constructors ----
    constexpr matrix4() noexcept
        : col{col_type(T(1),T(0),T(0),T(0)), col_type(T(0),T(1),T(0),T(0)),
              col_type(T(0),T(0),T(1),T(0)), col_type(T(0),T(0),T(0),T(1))} {}
    constexpr matrix4(const col_type& c0, const col_type& c1, const col_type& c2, const col_type& c3) noexcept
        : col{c0, c1, c2, c3} {}
    constexpr matrix4(
        T m00, T m01, T m02, T m03,
        T m10, T m11, T m12, T m13,
        T m20, T m21, T m22, T m23,
        T m30, T m31, T m32, T m33) noexcept
        : col{col_type(m00,m10,m20,m30), col_type(m01,m11,m21,m31),
              col_type(m02,m12,m22,m32), col_type(m03,m13,m23,m33)} {}
    constexpr matrix4(const matrix3<T>& m) noexcept
        : col{col_type(m.col[0],T(0)), col_type(m.col[1],T(0)),
              col_type(m.col[2],T(0)), col_type(T(0),T(0),T(0),T(1))} {}
    constexpr matrix4(const quaternion<T>& q) noexcept : matrix4(matrix3<T>(q)) {}
    constexpr matrix4(T diag) noexcept
        : col{col_type(diag,T(0),T(0),T(0)), col_type(T(0),diag,T(0),T(0)),
              col_type(T(0),T(0),diag,T(0)), col_type(T(0),T(0),T(0),diag)} {}
    template<typename U> constexpr explicit matrix4(const matrix4<U>& m) noexcept
        : col{col_type(m.col[0]), col_type(m.col[1]), col_type(m.col[2]), col_type(m.col[3])} {}

    // ---- Access ----
    constexpr col_type& operator[](std::size_t i) noexcept { return col[i]; }
    constexpr const col_type& operator[](std::size_t i) const noexcept { return col[i]; }
    constexpr T& operator()(std::size_t row, std::size_t col_idx) noexcept { return col[col_idx][row]; }
    constexpr const T& operator()(std::size_t row, std::size_t col_idx) const noexcept { return col[col_idx][row]; }

    // ---- Compound assignment ----
    constexpr matrix4& operator+=(const matrix4& m) noexcept { col[0]+=m.col[0]; col[1]+=m.col[1]; col[2]+=m.col[2]; col[3]+=m.col[3]; return *this; }
    constexpr matrix4& operator-=(const matrix4& m) noexcept { col[0]-=m.col[0]; col[1]-=m.col[1]; col[2]-=m.col[2]; col[3]-=m.col[3]; return *this; }
    constexpr matrix4& operator*=(T s) noexcept { col[0]*=s; col[1]*=s; col[2]*=s; col[3]*=s; return *this; }
    constexpr matrix4& operator/=(T s) noexcept { col[0]/=s; col[1]/=s; col[2]/=s; col[3]/=s; return *this; }

    // ---- Unary ----
    constexpr matrix4 operator+() const noexcept { return *this; }
    constexpr matrix4 operator-() const noexcept { return matrix4(-col[0], -col[1], -col[2], -col[3]); }

    // ---- Conversion ----
    constexpr operator std::array<std::array<T,4>,4>() const noexcept {
        return {{ {col[0].x,col[1].x,col[2].x,col[3].x}, {col[0].y,col[1].y,col[2].y,col[3].y},
                  {col[0].z,col[1].z,col[2].z,col[3].z}, {col[0].w,col[1].w,col[2].w,col[3].w} }};
    }
};

// ============================================================
// Binary operators
// ============================================================

template<typename T> constexpr matrix4<T> operator+(const matrix4<T>& a, const matrix4<T>& b) noexcept { return matrix4<T>(a.col[0]+b.col[0], a.col[1]+b.col[1], a.col[2]+b.col[2], a.col[3]+b.col[3]); }
template<typename T> constexpr matrix4<T> operator-(const matrix4<T>& a, const matrix4<T>& b) noexcept { return matrix4<T>(a.col[0]-b.col[0], a.col[1]-b.col[1], a.col[2]-b.col[2], a.col[3]-b.col[3]); }
template<typename T> constexpr matrix4<T> operator*(const matrix4<T>& m, T s) noexcept { return matrix4<T>(m.col[0]*s, m.col[1]*s, m.col[2]*s, m.col[3]*s); }
template<typename T> constexpr matrix4<T> operator*(T s, const matrix4<T>& m) noexcept { return m*s; }
template<typename T> constexpr bool operator==(const matrix4<T>& a, const matrix4<T>& b) noexcept { return a.col[0]==b.col[0] && a.col[1]==b.col[1] && a.col[2]==b.col[2] && a.col[3]==b.col[3]; }
template<typename T> constexpr bool operator!=(const matrix4<T>& a, const matrix4<T>& b) noexcept { return !(a==b); }

// ============================================================
// Matrix‑vector multiplication
// ============================================================

template<typename T>
constexpr vector4<T> operator*(const matrix4<T>& m, const vector4<T>& v) noexcept {
    return m.col[0]*v.x + m.col[1]*v.y + m.col[2]*v.z + m.col[3]*v.w;
}

template<typename T>
constexpr vector3<T> transform_point(const matrix4<T>& m, const vector3<T>& v) noexcept {
    vector4<T> r = m * vector4<T>(v.x, v.y, v.z, T(1));
    T inv_w = T(1) / r.w;
    return vector3<T>(r.x * inv_w, r.y * inv_w, r.z * inv_w);
}

template<typename T>
constexpr vector3<T> transform_vector(const matrix4<T>& m, const vector3<T>& v) noexcept {
    vector4<T> r = m * vector4<T>(v.x, v.y, v.z, T(0));
    return vector3<T>(r.x, r.y, r.z);
}

// ============================================================
// Matrix‑matrix multiplication
// ============================================================

template<typename T>
constexpr matrix4<T> operator*(const matrix4<T>& a, const matrix4<T>& b) noexcept {
    return matrix4<T>(a*b.col[0], a*b.col[1], a*b.col[2], a*b.col[3]);
}

// ============================================================
// Transpose
// ============================================================

template<typename T>
constexpr matrix4<T> transpose(const matrix4<T>& m) noexcept {
    return matrix4<T>(
        m(0,0), m(1,0), m(2,0), m(3,0),
        m(0,1), m(1,1), m(2,1), m(3,1),
        m(0,2), m(1,2), m(2,2), m(3,2),
        m(0,3), m(1,3), m(2,3), m(3,3)
    );
}

// ============================================================
// Trace
// ============================================================

template<typename T>
constexpr T trace(const matrix4<T>& m) noexcept {
    return m(0,0) + m(1,1) + m(2,2) + m(3,3);
}

// ============================================================
// Determinant
// ============================================================

template<typename T>
T determinant(const matrix4<T>& m) noexcept {
    T s0 = m(0,0)*m(1,1) - m(1,0)*m(0,1);
    T s1 = m(0,0)*m(1,2) - m(1,0)*m(0,2);
    T s2 = m(0,0)*m(1,3) - m(1,0)*m(0,3);
    T s3 = m(0,1)*m(1,2) - m(1,1)*m(0,2);
    T s4 = m(0,1)*m(1,3) - m(1,1)*m(0,3);
    T s5 = m(0,2)*m(1,3) - m(1,2)*m(0,3);
    T c0 = m(2,0)*m(3,1) - m(3,0)*m(2,1);
    T c1 = m(2,0)*m(3,2) - m(3,0)*m(2,2);
    T c2 = m(2,0)*m(3,3) - m(3,0)*m(2,3);
    T c3 = m(2,1)*m(3,2) - m(3,1)*m(2,2);
    T c4 = m(2,1)*m(3,3) - m(3,1)*m(2,3);
    T c5 = m(2,2)*m(3,3) - m(3,2)*m(2,3);
    return s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0;
}

// ============================================================
// Inverse (via cofactor expansion)
// ============================================================

template<typename T>
matrix4<T> inverse(const matrix4<T>& m) noexcept {
    T det = determinant(m);
    if (std::abs(det) < T(FLOAT_EPSILON)) return matrix4<T>(T(1));
    T inv_det = T(1) / det;
    T a0=m(0,0),a1=m(0,1),a2=m(0,2),a3=m(0,3);
    T b0=m(1,0),b1=m(1,1),b2=m(1,2),b3=m(1,3);
    T c0=m(2,0),c1=m(2,1),c2=m(2,2),c3=m(2,3);
    T d0=m(3,0),d1=m(3,1),d2=m(3,2),d3=m(3,3);
    T s0=a0*b1-a1*b0, s1=a0*b2-a2*b0, s2=a0*b3-a3*b0;
    T s3=a1*b2-a2*b1, s4=a1*b3-a3*b1, s5=a2*b3-a3*b2;
    T c5=c2*d3-c3*d2, c4=c1*d3-c3*d1, c3=c1*d2-c2*d1;
    T c2=c0*d3-c3*d0, c1=c0*d2-c2*d0, c0=c0*d1-c1*d0;
    return matrix4<T>(
        ( b1*c5 - b2*c4 + b3*c3) * inv_det,
        (-a1*c5 + a2*c4 - a3*c3) * inv_det,
        ( d1*s5 - d2*s4 + d3*s3) * inv_det,
        (-c1*s5 + c2*s4 - c3*s3) * inv_det,
        (-b0*c5 + b2*c2 - b3*c1) * inv_det,
        ( a0*c5 - a2*c2 + a3*c1) * inv_det,
        (-d0*s5 + d2*s2 - d3*s1) * inv_det,
        ( c0*s5 - c2*s2 + c3*s1) * inv_det,
        ( b0*c4 - b1*c2 + b3*c0) * inv_det,
        (-a0*c4 + a1*c2 - a3*c0) * inv_det,
        ( d0*s4 - d1*s2 + d3*s0) * inv_det,
        (-c0*s4 + c1*s2 - c3*s0) * inv_det,
        (-b0*c3 + b1*c1 - b2*c0) * inv_det,
        ( a0*c3 - a1*c1 + a2*c0) * inv_det,
        (-d0*s3 + d1*s1 - d2*s0) * inv_det,
        ( c0*s3 - c1*s1 + c2*s0) * inv_det
    );
}

// ============================================================
// Affine inverse (faster for TRS matrices)
// ============================================================

template<typename T>
matrix4<T> affine_inverse(const matrix4<T>& m) noexcept {
    matrix3<T> R(m(0,0),m(0,1),m(0,2), m(1,0),m(1,1),m(1,2), m(2,0),m(2,1),m(2,2));
    vector3<T> t(m(0,3), m(1,3), m(2,3));
    matrix3<T> Rt = transpose(R);
    vector3<T> t_inv = -(Rt * t);
    return matrix4<T>(Rt.col[0], Rt.col[1], Rt.col[2], vector4<T>(t_inv, T(1)));
}

// ============================================================
// Translation matrix
// ============================================================

template<typename T>
constexpr matrix4<T> translation(const vector3<T>& t) noexcept {
    matrix4<T> m;
    m(0,3)=t.x; m(1,3)=t.y; m(2,3)=t.z;
    return m;
}

template<typename T>
constexpr matrix4<T> translation(T x, T y, T z) noexcept {
    return translation(vector3<T>(x,y,z));
}

// ============================================================
// Scaling matrix
// ============================================================

template<typename T>
constexpr matrix4<T> scaling(const vector3<T>& s) noexcept {
    return matrix4<T>(s.x,T(0),T(0),T(0), T(0),s.y,T(0),T(0), T(0),T(0),s.z,T(0), T(0),T(0),T(0),T(1));
}

template<typename T>
constexpr matrix4<T> scaling(T sx, T sy, T sz) noexcept {
    return scaling(vector3<T>(sx,sy,sz));
}

// ============================================================
// Rotation matrices (axis)
// ============================================================

template<typename T>
constexpr matrix4<T> rotation(const quaternion<T>& q) noexcept {
    return matrix4<T>(matrix3<T>(q));
}

template<typename T>
constexpr matrix4<T> rotation_x(T angle_rad) noexcept {
    return matrix4<T>(rotation_x(angle_rad));
}

template<typename T>
constexpr matrix4<T> rotation_y(T angle_rad) noexcept {
    return matrix4<T>(rotation_y(angle_rad));
}

template<typename T>
constexpr matrix4<T> rotation_z(T angle_rad) noexcept {
    return matrix4<T>(rotation_z(angle_rad));
}

template<typename T>
constexpr matrix4<T> rotation_from_axis_angle(const vector3<T>& axis, T angle_rad) noexcept {
    return matrix4<T>(rotation_from_axis_angle(axis, angle_rad));
}

// ============================================================
// TRS composition
// ============================================================

template<typename T>
constexpr matrix4<T> TRS(const vector3<T>& translation, const quaternion<T>& rotation, const vector3<T>& scale) noexcept {
    return tbb::core::math::translation(translation) * tbb::core::math::rotation(rotation) * scaling(scale);
}

// ============================================================
// Decompose (affine)
// ============================================================

template<typename T>
void decompose(const matrix4<T>& m, vector3<T>& translation, quaternion<T>& rotation, vector3<T>& scale) noexcept {
    translation = vector3<T>(m(0,3), m(1,3), m(2,3));
    matrix3<T> R(m(0,0),m(0,1),m(0,2), m(1,0),m(1,1),m(1,2), m(2,0),m(2,1),m(2,2));
    scale.x = length(R.col[0]);
    scale.y = length(R.col[1]);
    scale.z = length(R.col[2]);
    if (scale.x > T(FLOAT_EPSILON)) R.col[0] /= scale.x;
    if (scale.y > T(FLOAT_EPSILON)) R.col[1] /= scale.y;
    if (scale.z > T(FLOAT_EPSILON)) R.col[2] /= scale.z;
    if (determinant(R) < T(0)) { R.col[2] = -R.col[2]; scale.z = -scale.z; }
    rotation = quaternion<T>(R);
}

// ============================================================
// LookAt
// ============================================================

template<typename T>
matrix4<T> look_at(const vector3<T>& eye, const vector3<T>& center, const vector3<T>& up) noexcept {
    vector3<T> f = normalize(center - eye);
    vector3<T> s = normalize(cross(f, up));
    vector3<T> u = cross(s, f);
    matrix4<T> m;
    m(0,0)=s.x; m(0,1)=s.y; m(0,2)=s.z; m(0,3)=-dot(s,eye);
    m(1,0)=u.x; m(1,1)=u.y; m(1,2)=u.z; m(1,3)=-dot(u,eye);
    m(2,0)=-f.x; m(2,1)=-f.y; m(2,2)=-f.z; m(2,3)=dot(f,eye);
    m(3,0)=T(0); m(3,1)=T(0); m(3,2)=T(0); m(3,3)=T(1);
    return m;
}

// ============================================================
// Perspective projection (right‑handed, zero‑to‑one depth)
// ============================================================

template<typename T>
matrix4<T> perspective(T fov_y_rad, T aspect, T near_z, T far_z) noexcept {
    T tan_half = std::tan(fov_y_rad * T(0.5));
    matrix4<T> m(T(0));
    m(0,0) = T(1) / (aspect * tan_half);
    m(1,1) = T(1) / tan_half;
    m(2,2) = far_z / (near_z - far_z);
    m(2,3) = (far_z * near_z) / (near_z - far_z);
    m(3,2) = T(-1);
    return m;
}

template<typename T>
matrix4<T> perspective_infinite(T fov_y_rad, T aspect, T near_z) noexcept {
    T tan_half = std::tan(fov_y_rad * T(0.5));
    matrix4<T> m(T(0));
    m(0,0) = T(1) / (aspect * tan_half);
    m(1,1) = T(1) / tan_half;
    m(2,2) = T(-1);
    m(2,3) = T(-2) * near_z;
    m(3,2) = T(-1);
    return m;
}

// ============================================================
// Orthographic projection
// ============================================================

template<typename T>
matrix4<T> orthographic(T left, T right, T bottom, T top, T near_z, T far_z) noexcept {
    matrix4<T> m(T(1));
    m(0,0) = T(2) / (right - left);
    m(1,1) = T(2) / (top - bottom);
    m(2,2) = T(1) / (near_z - far_z);
    m(0,3) = -(right + left) / (right - left);
    m(1,3) = -(top + bottom) / (top - bottom);
    m(2,3) = -near_z / (near_z - far_z);
    return m;
}

// ============================================================
// Component‑wise operations
// ============================================================

template<typename T> constexpr matrix4<T> min(const matrix4<T>& a, const matrix4<T>& b) noexcept {
    return matrix4<T>(min(a.col[0],b.col[0]), min(a.col[1],b.col[1]), min(a.col[2],b.col[2]), min(a.col[3],b.col[3]));
}
template<typename T> constexpr matrix4<T> max(const matrix4<T>& a, const matrix4<T>& b) noexcept {
    return matrix4<T>(max(a.col[0],b.col[0]), max(a.col[1],b.col[1]), max(a.col[2],b.col[2]), max(a.col[3],b.col[3]));
}
template<typename T> constexpr matrix4<T> abs(const matrix4<T>& m) noexcept {
    return matrix4<T>(abs(m.col[0]), abs(m.col[1]), abs(m.col[2]), abs(m.col[3]));
}

// ============================================================
// Interpolation (element‑wise lerp)
// ============================================================

template<typename T>
constexpr matrix4<T> lerp(const matrix4<T>& a, const matrix4<T>& b, T t) noexcept {
    return a + (b - a) * t;
}

// ============================================================
// Conversion between types
// ============================================================

template<typename U, typename T>
constexpr matrix4<U> matrix_cast(const matrix4<T>& m) noexcept {
    return matrix4<U>(vector_cast<U>(m.col[0]), vector_cast<U>(m.col[1]), vector_cast<U>(m.col[2]), vector_cast<U>(m.col[3]));
}

// ============================================================
// Type aliases
// ============================================================

using matrix4f = matrix4<float>;
using matrix4d = matrix4<double>;

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_MATRIX4_H