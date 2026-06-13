// File 0008 : core/math/quat.h
// Quaternion algebra (float/double) for rotations, slerp, conversion to/from matrices and vectors.

#pragma once

#include "vec3.h"
#include "mat3.h"
#include "mat4.h"
#include <cassert>
#include <cmath>
#include <type_traits>

namespace wp {

template <typename T>
struct alignas(sizeof(T) * 4) quat {
    T x, y, z, w;

    constexpr quat() noexcept : x(T(0)), y(T(0)), z(T(0)), w(T(1)) {}
    constexpr quat(T x_, T y_, T z_, T w_) noexcept : x(x_), y(y_), z(z_), w(w_) {}
    explicit constexpr quat(const vec3<T>& v, T w_ = T(0)) noexcept : x(v.x), y(v.y), z(v.z), w(w_) {}
    template <typename U> constexpr explicit quat(const quat<U>& o) noexcept : x(static_cast<T>(o.x)), y(static_cast<T>(o.y)), z(static_cast<T>(o.z)), w(static_cast<T>(o.w)) {}

    constexpr T  operator[](int i) const noexcept { assert(i>=0&&i<4); return (&x)[i]; }
    constexpr T& operator[](int i)       noexcept { assert(i>=0&&i<4); return (&x)[i]; }

    constexpr quat operator+() const noexcept { return *this; }
    constexpr quat operator-() const noexcept { return quat(-x, -y, -z, -w); }

    constexpr quat& operator+=(const quat& o) noexcept { x+=o.x; y+=o.y; z+=o.z; w+=o.w; return *this; }
    constexpr quat& operator-=(const quat& o) noexcept { x-=o.x; y-=o.y; z-=o.z; w-=o.w; return *this; }
    constexpr quat& operator*=(T s) noexcept { x*=s; y*=s; z*=s; w*=s; return *this; }
    constexpr quat& operator/=(T s) noexcept { x/=s; y/=s; z/=s; w/=s; return *this; }
};

template <typename T> constexpr quat<T> operator+(const quat<T>& a, const quat<T>& b) noexcept { return quat<T>(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
template <typename T> constexpr quat<T> operator-(const quat<T>& a, const quat<T>& b) noexcept { return quat<T>(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }
template <typename T> constexpr quat<T> operator*(const quat<T>& a, T s) noexcept { return quat<T>(a.x*s, a.y*s, a.z*s, a.w*s); }
template <typename T> constexpr quat<T> operator*(T s, const quat<T>& a) noexcept { return quat<T>(s*a.x, s*a.y, s*a.z, s*a.w); }
template <typename T> constexpr quat<T> operator/(const quat<T>& a, T s) noexcept { return quat<T>(a.x/s, a.y/s, a.z/s, a.w/s); }
template <typename T> constexpr bool operator==(const quat<T>& a, const quat<T>& b) noexcept { return a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w; }
template <typename T> constexpr bool operator!=(const quat<T>& a, const quat<T>& b) noexcept { return !(a==b); }

template <typename T> constexpr quat<T> mul(const quat<T>& a, const quat<T>& b) noexcept {
    return quat<T>(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
    );
}
template <typename T> constexpr T dot(const quat<T>& a, const quat<T>& b) noexcept { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
template <typename T> constexpr quat<T> conjugate(const quat<T>& q) noexcept { return quat<T>(-q.x, -q.y, -q.z, q.w); }
template <typename T> constexpr T norm_sq(const quat<T>& q) noexcept { return q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w; }
template <typename T> T norm(const quat<T>& q) noexcept { return std::sqrt(norm_sq(q)); }
template <typename T> quat<T> normalize(const quat<T>& q) noexcept {
    T n = norm(q);
    return (n > MathConst<T>::epsilon) ? q / n : quat<T>(T(0), T(0), T(0), T(1));
}
template <typename T> quat<T> inverse(const quat<T>& q) noexcept { return conjugate(q) / norm_sq(q); }
template <typename T> constexpr quat<T> negate(const quat<T>& q) noexcept { return quat<T>(-q.x, -q.y, -q.z, -q.w); }

template <typename T> vec3<T> rotate(const quat<T>& q, const vec3<T>& v) noexcept {
    quat<T> p(v, T(0));
    quat<T> r = mul(mul(q, p), conjugate(q));
    return vec3<T>(r.x, r.y, r.z);
}

template <typename T> quat<T> slerp(const quat<T>& qa, const quat<T>& qb, T t) noexcept {
    T cos_theta = dot(qa, qb);
    quat<T> qb2 = qb;
    if (cos_theta < T(0)) { cos_theta = -cos_theta; qb2 = -qb2; }
    T k0, k1;
    if (cos_theta > T(1) - MathConst<T>::epsilon) {
        k0 = T(1) - t; k1 = t;
    } else {
        T sin_theta = std::sqrt(T(1) - cos_theta * cos_theta);
        T theta = std::atan2(sin_theta, cos_theta);
        T inv_sin = T(1) / sin_theta;
        k0 = std::sin((T(1) - t) * theta) * inv_sin;
        k1 = std::sin(t * theta) * inv_sin;
    }
    return qa * k0 + qb2 * k1;
}

template <typename T> quat<T> lerp(const quat<T>& a, const quat<T>& b, T t) noexcept { return normalize(a + (b - a) * t); }

template <typename T> quat<T> from_axis_angle(const vec3<T>& axis, T angle) noexcept {
    T half = angle * T(0.5);
    T s = std::sin(half);
    return quat<T>(axis.x * s, axis.y * s, axis.z * s, std::cos(half));
}
template <typename T> void to_axis_angle(const quat<T>& q, vec3<T>& axis, T& angle) noexcept {
    if (std::abs(q.w) > T(1) - MathConst<T>::epsilon) {
        axis = vec3<T>(T(1), T(0), T(0));
        angle = T(0);
        return;
    }
    T s = std::sqrt(T(1) - q.w * q.w);
    T inv_s = T(1) / s;
    axis = vec3<T>(q.x * inv_s, q.y * inv_s, q.z * inv_s);
    angle = T(2) * std::acos(q.w);
}

template <typename T> mat3<T> to_matrix3(const quat<T>& q) noexcept {
    T xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    T xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    T wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    return mat3<T>(
        T(1) - T(2) * (yy + zz),  T(2) * (xy - wz),          T(2) * (xz + wy),
        T(2) * (xy + wz),          T(1) - T(2) * (xx + zz),  T(2) * (yz - wx),
        T(2) * (xz - wy),          T(2) * (yz + wx),          T(1) - T(2) * (xx + yy)
    );
}
template <typename T> quat<T> from_matrix3(const mat3<T>& m) noexcept {
    T trace = m.m00 + m.m11 + m.m22;
    if (trace > T(0)) {
        T s = std::sqrt(trace + T(1)) * T(2);
        T inv_s = T(1) / s;
        return quat<T>((m.m12 - m.m21) * inv_s, (m.m20 - m.m02) * inv_s, (m.m01 - m.m10) * inv_s, s / T(4));
    } else if (m.m00 > m.m11 && m.m00 > m.m22) {
        T s = std::sqrt(T(1) + m.m00 - m.m11 - m.m22) * T(2);
        T inv_s = T(1) / s;
        return quat<T>(s / T(4), (m.m01 + m.m10) * inv_s, (m.m20 + m.m02) * inv_s, (m.m12 - m.m21) * inv_s);
    } else if (m.m11 > m.m22) {
        T s = std::sqrt(T(1) + m.m11 - m.m00 - m.m22) * T(2);
        T inv_s = T(1) / s;
        return quat<T>((m.m01 + m.m10) * inv_s, s / T(4), (m.m12 + m.m21) * inv_s, (m.m20 - m.m02) * inv_s);
    } else {
        T s = std::sqrt(T(1) + m.m22 - m.m00 - m.m11) * T(2);
        T inv_s = T(1) / s;
        return quat<T>((m.m20 + m.m02) * inv_s, (m.m12 + m.m21) * inv_s, s / T(4), (m.m01 - m.m10) * inv_s);
    }
}

template <typename T> mat4<T> to_matrix4(const quat<T>& q) noexcept {
    mat3<T> r = to_matrix3(q);
    return mat4<T>(r.m00, r.m01, r.m02, T(0),
                   r.m10, r.m11, r.m12, T(0),
                   r.m20, r.m21, r.m22, T(0),
                   T(0), T(0), T(0), T(1));
}
template <typename T> quat<T> from_matrix4(const mat4<T>& m) noexcept {
    return from_matrix3(mat3<T>(m.m00, m.m01, m.m02,
                                m.m10, m.m11, m.m12,
                                m.m20, m.m21, m.m22));
}

template <typename T> quat<T> from_euler_xyz(const vec3<T>& euler) noexcept {
    T cx = std::cos(euler.x * T(0.5)), sx = std::sin(euler.x * T(0.5));
    T cy = std::cos(euler.y * T(0.5)), sy = std::sin(euler.y * T(0.5));
    T cz = std::cos(euler.z * T(0.5)), sz = std::sin(euler.z * T(0.5));
    return quat<T>(
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
        cx * cy * cz + sx * sy * sz
    );
}
template <typename T> vec3<T> to_euler_xyz(const quat<T>& q) noexcept {
    T sqw = q.w * q.w;
    T sqx = q.x * q.x;
    T sqy = q.y * q.y;
    T sqz = q.z * q.z;
    T unit = sqx + sqy + sqz + sqw;
    T test = q.x * q.y + q.z * q.w;
    if (test > T(0.499) * unit) {
        return vec3<T>(T(2) * std::atan2(q.x, q.w), half_pi<T>, T(0));
    } else if (test < T(-0.499) * unit) {
        return vec3<T>(T(-2) * std::atan2(q.x, q.w), -half_pi<T>, T(0));
    }
    return vec3<T>(
        std::atan2(T(2) * (q.x * q.w - q.y * q.z), -sqx + sqy - sqz + sqw),
        std::asin(T(2) * test / unit),
        std::atan2(T(2) * (q.y * q.w - q.x * q.z),  sqx - sqy - sqz + sqw)
    );
}

template <typename T> quat<T> from_euler_yxz(const vec3<T>& euler) noexcept {
    T cx = std::cos(euler.x * T(0.5)), sx = std::sin(euler.x * T(0.5));
    T cy = std::cos(euler.y * T(0.5)), sy = std::sin(euler.y * T(0.5));
    T cz = std::cos(euler.z * T(0.5)), sz = std::sin(euler.z * T(0.5));
    return quat<T>(
        sx * cy * cz + cx * sy * sz,
        cx * sy * cz - sx * cy * sz,
        cx * cy * sz + sx * sy * cz,
        cx * cy * cz - sx * sy * sz
    );
}
template <typename T> vec3<T> to_euler_yxz(const quat<T>& q) noexcept {
    return to_euler_xyz(quat<T>(q.x, q.y, -q.z, q.w)); // conversion approach
}

template <typename T> constexpr quat<T> look_at_quat(const vec3<T>& dir, const vec3<T>& up) noexcept {
    vec3<T> f = normalize(dir);
    vec3<T> s = normalize(cross(f, up));
    vec3<T> u = cross(s, f);
    mat3<T> rot(s, u, -f);
    return from_matrix3(rot);
}

template <typename T> quat<T> angular_velocity_to_quat(const vec3<T>& omega, T dt) noexcept {
    T len = length(omega);
    if (len < MathConst<T>::epsilon) return quat<T>();
    T half_angle = len * dt * T(0.5);
    return quat<T>(omega.x / len * std::sin(half_angle),
                   omega.y / len * std::sin(half_angle),
                   omega.z / len * std::sin(half_angle),
                   std::cos(half_angle));
}

template <typename T> constexpr quat<T> identity_quat() noexcept { return quat<T>(T(0), T(0), T(0), T(1)); }

template <typename T> T angle_between(const quat<T>& a, const quat<T>& b) noexcept {
    T d = dot(a, b);
    return T(2) * std::acos(clamp(std::abs(d), T(-1), T(1)));
}

using quatf = quat<float>;
using quatd = quat<double>;

} // namespace wp