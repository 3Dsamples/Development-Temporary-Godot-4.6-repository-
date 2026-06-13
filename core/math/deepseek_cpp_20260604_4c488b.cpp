// system name : onetbb-warp
// File 0006 : core/math/quaternion.h
// Description : Quaternion algebra for 3D rotations, SLERP, and smooth dynamics.

#ifndef __TBB_WARP_CORE_MATH_QUATERNION_H
#define __TBB_WARP_CORE_MATH_QUATERNION_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector3.h"
#include "core/math/vector4.h"
#include <cmath>
#include <type_traits>
#include <array>
#include <initializer_list>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Quaternion class template
// ============================================================

template<typename T>
struct quaternion {
    using value_type = T;

    T x, y, z, w;

    // ---- Constructors ----
    constexpr quaternion() noexcept : x(T(0)), y(T(0)), z(T(0)), w(T(1)) {}
    constexpr quaternion(T x_, T y_, T z_, T w_) noexcept : x(x_), y(y_), z(z_), w(w_) {}
    constexpr quaternion(const vector3<T>& axis, T angle_rad) noexcept {
        T half = angle_rad * T(0.5);
        T s = std::sin(half);
        T c = std::cos(half);
        T len = length(axis);
        if (len < T(FLOAT_EPSILON)) { x=T(0); y=T(0); z=T(0); w=T(1); return; }
        T inv = T(1) / len;
        x = axis.x * inv * s;
        y = axis.y * inv * s;
        z = axis.z * inv * s;
        w = c;
    }
    explicit constexpr quaternion(const vector4<T>& v) noexcept : x(v.x), y(v.y), z(v.z), w(v.w) {}
    template<typename U>
    constexpr explicit quaternion(const quaternion<U>& q) noexcept : x(static_cast<T>(q.x)), y(static_cast<T>(q.y)), z(static_cast<T>(q.z)), w(static_cast<T>(q.w)) {}

    // ---- Access ----
    constexpr T& operator[](std::size_t i) noexcept { return (&x)[i]; }
    constexpr const T& operator[](std::size_t i) const noexcept { return (&x)[i]; }

    // ---- Compound assignment ----
    constexpr quaternion& operator+=(const quaternion& q) noexcept { x+=q.x; y+=q.y; z+=q.z; w+=q.w; return *this; }
    constexpr quaternion& operator-=(const quaternion& q) noexcept { x-=q.x; y-=q.y; z-=q.z; w-=q.w; return *this; }
    constexpr quaternion& operator*=(const quaternion& q) noexcept {
        T x_ = w*q.x + x*q.w + y*q.z - z*q.y;
        T y_ = w*q.y - x*q.z + y*q.w + z*q.x;
        T z_ = w*q.z + x*q.y - y*q.x + z*q.w;
        T w_ = w*q.w - x*q.x - y*q.y - z*q.z;
        x=x_; y=y_; z=z_; w=w_;
        return *this;
    }
    constexpr quaternion& operator*=(T s) noexcept { x*=s; y*=s; z*=s; w*=s; return *this; }

    // ---- Unary ----
    constexpr quaternion operator+() const noexcept { return *this; }
    constexpr quaternion operator-() const noexcept { return quaternion(-x, -y, -z, -w); }
};

// ============================================================
// Binary operators
// ============================================================

template<typename T> constexpr quaternion<T> operator+(const quaternion<T>& a, const quaternion<T>& b) noexcept { return quaternion<T>(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
template<typename T> constexpr quaternion<T> operator-(const quaternion<T>& a, const quaternion<T>& b) noexcept { return quaternion<T>(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }
template<typename T> constexpr quaternion<T> operator*(const quaternion<T>& a, const quaternion<T>& b) noexcept {
    return quaternion<T>(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
    );
}
template<typename T> constexpr quaternion<T> operator*(const quaternion<T>& q, T s) noexcept { return quaternion<T>(q.x*s, q.y*s, q.z*s, q.w*s); }
template<typename T> constexpr quaternion<T> operator*(T s, const quaternion<T>& q) noexcept { return q*s; }
template<typename T> constexpr bool operator==(const quaternion<T>& a, const quaternion<T>& b) noexcept { return a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w; }
template<typename T> constexpr bool operator!=(const quaternion<T>& a, const quaternion<T>& b) noexcept { return !(a==b); }

// ============================================================
// Basic properties
// ============================================================

template<typename T>
constexpr T dot(const quaternion<T>& a, const quaternion<T>& b) noexcept {
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

template<typename T>
constexpr T norm_sq(const quaternion<T>& q) noexcept {
    return dot(q, q);
}

template<typename T>
T norm(const quaternion<T>& q) noexcept {
    return std::sqrt(norm_sq(q));
}

template<typename T>
quaternion<T> normalize(const quaternion<T>& q) noexcept {
    T n = norm(q);
    if (n < T(FLOAT_EPSILON)) return quaternion<T>(T(0), T(0), T(0), T(1));
    T inv = T(1) / n;
    return quaternion<T>(q.x*inv, q.y*inv, q.z*inv, q.w*inv);
}

template<typename T>
constexpr quaternion<T> conjugate(const quaternion<T>& q) noexcept {
    return quaternion<T>(-q.x, -q.y, -q.z, q.w);
}

template<typename T>
quaternion<T> inverse(const quaternion<T>& q) noexcept {
    T nsq = norm_sq(q);
    if (nsq < T(FLOAT_EPSILON)) return quaternion<T>(T(0), T(0), T(0), T(1));
    T inv = T(1) / nsq;
    return quaternion<T>(-q.x*inv, -q.y*inv, -q.z*inv, q.w*inv);
}

// ============================================================
// Rotation of vectors
// ============================================================

template<typename T>
vector3<T> rotate(const quaternion<T>& q, const vector3<T>& v) noexcept {
    vector3<T> u(q.x, q.y, q.z);
    T uv = dot(u, v);
    T uu = dot(u, u);
    T qw2 = q.w * q.w;
    vector3<T> term1 = u * (T(2) * uv);
    vector3<T> term2 = v * (qw2 - uu);
    vector3<T> term3 = cross(u, v) * (T(2) * q.w);
    return term1 + term2 + term3;
}

template<typename T>
constexpr vector3<T> forward(const quaternion<T>& q) noexcept {
    return rotate(q, vector3<T>(T(0), T(0), T(-1)));
}

template<typename T>
constexpr vector3<T> up(const quaternion<T>& q) noexcept {
    return rotate(q, vector3<T>(T(0), T(1), T(0)));
}

template<typename T>
constexpr vector3<T> right(const quaternion<T>& q) noexcept {
    return rotate(q, vector3<T>(T(1), T(0), T(0)));
}

// ============================================================
// Interpolation (NLERP, SLERP)
// ============================================================

template<typename T>
quaternion<T> nlerp(const quaternion<T>& a, const quaternion<T>& b, T t) noexcept {
    T cos_theta = dot(a, b);
    quaternion<T> corrected_b = b;
    if (cos_theta < T(0)) { corrected_b = -corrected_b; cos_theta = -cos_theta; }
    quaternion<T> result = a * (T(1) - t) + corrected_b * t;
    return normalize(result);
}

template<typename T>
quaternion<T> slerp(const quaternion<T>& a, const quaternion<T>& b, T t) noexcept {
    T cos_theta = dot(a, b);
    quaternion<T> corrected_b = b;
    if (cos_theta < T(0)) { corrected_b = -corrected_b; cos_theta = -cos_theta; }
    if (cos_theta > T(0.9995)) return nlerp(a, corrected_b, t);
    T theta = std::acos(cos_theta);
    T sin_theta = std::sin(theta);
    T wa = std::sin((T(1) - t) * theta) / sin_theta;
    T wb = std::sin(t * theta) / sin_theta;
    return a * wa + corrected_b * wb;
}

template<typename T>
quaternion<T> squad(const quaternion<T>& q0, const quaternion<T>& a,
                    const quaternion<T>& b, const quaternion<T>& q1, T t) noexcept {
    return slerp(slerp(q0, q1, t), slerp(a, b, t), T(2) * t * (T(1) - t));
}

// ============================================================
// Euler angles conversion (ZYX intrinsic = yaw‑pitch‑roll)
// ============================================================

template<typename T>
quaternion<T> from_euler(T yaw, T pitch, T roll) noexcept {
    T half_yaw = yaw * T(0.5), half_pitch = pitch * T(0.5), half_roll = roll * T(0.5);
    T cy = std::cos(half_yaw), sy = std::sin(half_yaw);
    T cp = std::cos(half_pitch), sp = std::sin(half_pitch);
    T cr = std::cos(half_roll), sr = std::sin(half_roll);
    return quaternion<T>(
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
        cr*cp*cy + sr*sp*sy
    );
}

template<typename T>
vector3<T> to_euler(const quaternion<T>& q) noexcept {
    T sinr_cosp = T(2) * (q.w*q.x + q.y*q.z);
    T cosr_cosp = T(1) - T(2) * (q.x*q.x + q.y*q.y);
    T roll = std::atan2(sinr_cosp, cosr_cosp);
    T sinp = T(2) * (q.w*q.y - q.z*q.x);
    T pitch;
    if (std::abs(sinp) >= T(1)) pitch = std::copysign(T(HALF_PI_D), sinp);
    else pitch = std::asin(sinp);
    T siny_cosp = T(2) * (q.w*q.z + q.x*q.y);
    T cosy_cosp = T(1) - T(2) * (q.y*q.y + q.z*q.z);
    T yaw = std::atan2(siny_cosp, cosy_cosp);
    return vector3<T>(yaw, pitch, roll);
}

// ============================================================
// Rotation matrix conversion
// ============================================================

template<typename T>
quaternion<T> from_rotation_matrix(
    T m00, T m01, T m02,
    T m10, T m11, T m12,
    T m20, T m21, T m22) noexcept {
    T trace = m00 + m11 + m22;
    if (trace > T(0)) {
        T s = std::sqrt(trace + T(1)) * T(2);
        T inv = T(1) / s;
        return quaternion<T>(
            (m21 - m12) * inv,
            (m02 - m20) * inv,
            (m10 - m01) * inv,
            T(0.25) * s
        );
    }
    if (m00 > m11 && m00 > m22) {
        T s = std::sqrt(T(1) + m00 - m11 - m22) * T(2);
        T inv = T(1) / s;
        return quaternion<T>(
            T(0.25) * s,
            (m01 + m10) * inv,
            (m02 + m20) * inv,
            (m21 - m12) * inv
        );
    }
    if (m11 > m22) {
        T s = std::sqrt(T(1) + m11 - m00 - m22) * T(2);
        T inv = T(1) / s;
        return quaternion<T>(
            (m01 + m10) * inv,
            T(0.25) * s,
            (m12 + m21) * inv,
            (m02 - m20) * inv
        );
    }
    T s = std::sqrt(T(1) + m22 - m00 - m11) * T(2);
    T inv = T(1) / s;
    return quaternion<T>(
        (m02 + m20) * inv,
        (m12 + m21) * inv,
        T(0.25) * s,
        (m10 - m01) * inv
    );
}

template<typename T>
std::array<std::array<T,4>,4> to_rotation_matrix(const quaternion<T>& q) noexcept {
    T xx = q.x*q.x, yy = q.y*q.y, zz = q.z*q.z;
    T xy = q.x*q.y, xz = q.x*q.z, yz = q.y*q.z;
    T wx = q.w*q.x, wy = q.w*q.y, wz = q.w*q.z;
    std::array<std::array<T,4>,4> m;
    m[0][0] = T(1) - T(2)*(yy+zz); m[0][1] = T(2)*(xy - wz);    m[0][2] = T(2)*(xz + wy);    m[0][3] = T(0);
    m[1][0] = T(2)*(xy + wz);       m[1][1] = T(1) - T(2)*(xx+zz); m[1][2] = T(2)*(yz - wx);    m[1][3] = T(0);
    m[2][0] = T(2)*(xz - wy);       m[2][1] = T(2)*(yz + wx);       m[2][2] = T(1) - T(2)*(xx+yy); m[2][3] = T(0);
    m[3][0] = T(0);                  m[3][1] = T(0);                  m[3][2] = T(0);                  m[3][3] = T(1);
    return m;
}

// ============================================================
// Quaternion log / exp (for angular velocity integration)
// ============================================================

template<typename T>
quaternion<T> log(const quaternion<T>& q) noexcept {
    T n = norm(q);
    if (n < T(FLOAT_EPSILON)) return quaternion<T>(T(0), T(0), T(0), T(0));
    T w_norm = q.w / n;
    T theta = std::acos(clamp(w_norm, T(-1), T(1)));
    T sin_theta = std::sin(theta);
    if (std::abs(sin_theta) < T(FLOAT_EPSILON)) return quaternion<T>(q.x, q.y, q.z, T(0));
    T coeff = theta / (sin_theta * n);
    return quaternion<T>(q.x * coeff, q.y * coeff, q.z * coeff, T(0));
}

template<typename T>
quaternion<T> exp(const quaternion<T>& q) noexcept {
    T theta = std::sqrt(q.x*q.x + q.y*q.y + q.z*q.z);
    if (theta < T(FLOAT_EPSILON)) return quaternion<T>(T(0), T(0), T(0), T(1));
    T sin_theta = std::sin(theta);
    T coeff = sin_theta / theta;
    return quaternion<T>(q.x * coeff, q.y * coeff, q.z * coeff, std::cos(theta));
}

// ============================================================
// Quaternion from two vectors
// ============================================================

template<typename T>
quaternion<T> from_to_rotation(const vector3<T>& from, const vector3<T>& to) noexcept {
    T d = dot(from, to);
    if (d > T(0.999999)) return quaternion<T>(T(0), T(0), T(0), T(1));
    if (d < T(-0.999999)) {
        vector3<T> axis = cross(vector3<T>(T(1), T(0), T(0)), from);
        if (length_sq(axis) < T(FLOAT_EPSILON))
            axis = cross(vector3<T>(T(0), T(1), T(0)), from);
        axis = normalize(axis);
        return quaternion<T>(axis, T(PI_D));
    }
    vector3<T> axis = cross(from, to);
    T s = std::sqrt((T(1) + d) * T(2));
    T inv = T(1) / s;
    return quaternion<T>(axis.x * inv, axis.y * inv, axis.z * inv, s * T(0.5));
}

// ============================================================
// Angular velocity integration
// ============================================================

template<typename T>
quaternion<T> integrate_angular_velocity(const quaternion<T>& q, const vector3<T>& omega, T dt) noexcept {
    T len_sq = length_sq(omega);
    if (len_sq < T(FLOAT_EPSILON)) return q;
    T theta = std::sqrt(len_sq) * dt * T(0.5);
    T sin_theta = std::sin(theta);
    T cos_theta = std::cos(theta);
    T inv_len = T(1) / std::sqrt(len_sq);
    quaternion<T> delta(
        omega.x * inv_len * sin_theta,
        omega.y * inv_len * sin_theta,
        omega.z * inv_len * sin_theta,
        cos_theta
    );
    return normalize(q * delta);
}

// ============================================================
// Component‑wise operations
// ============================================================

template<typename T> constexpr quaternion<T> abs(const quaternion<T>& q) noexcept { return quaternion<T>(std::abs(q.x), std::abs(q.y), std::abs(q.z), std::abs(q.w)); }
template<typename T> constexpr bool is_identity(const quaternion<T>& q, T eps = T(FLOAT_EPSILON)) noexcept {
    return std::abs(q.x)<eps && std::abs(q.y)<eps && std::abs(q.z)<eps && std::abs(q.w-T(1))<eps;
}
template<typename T> bool is_finite(const quaternion<T>& q) noexcept { return std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z) && std::isfinite(q.w); }
template<typename T> bool is_nan(const quaternion<T>& q) noexcept { return std::isnan(q.x) || std::isnan(q.y) || std::isnan(q.z) || std::isnan(q.w); }

// ============================================================
// Smooth dynamics for quaternion
// ============================================================

template<typename T>
quaternion<T> smooth_damp(const quaternion<T>& current, const quaternion<T>& target,
                          vector3<T>& angular_velocity, T smooth_time, T max_speed, T dt) {
    smooth_time = std::max(T(0.0001), smooth_time);
    T omega = T(2) / smooth_time;
    T x = omega * dt;
    T exp = T(1) / (T(1) + x + T(0.48)*x*x + T(0.235)*x*x*x);
    quaternion<T> diff = inverse(current) * target;
    diff = normalize(diff);
    vector3<T> change = vector3<T>(diff.x, diff.y, diff.z) * T(2);
    if (diff.w < T(0)) change = -change;
    T change_len = length(change);
    if (change_len > max_speed * smooth_time)
        change = change * (max_speed * smooth_time / change_len);
    angular_velocity = (angular_velocity + change * omega) * exp;
    quaternion<T> delta(angular_velocity.x*dt*T(0.5), angular_velocity.y*dt*T(0.5), angular_velocity.z*dt*T(0.5), T(1));
    delta = normalize(delta);
    return normalize(current * delta);
}

// ============================================================
// Conversion between types
// ============================================================

template<typename U, typename T>
constexpr quaternion<U> quaternion_cast(const quaternion<T>& q) noexcept {
    return quaternion<U>(static_cast<U>(q.x), static_cast<U>(q.y), static_cast<U>(q.z), static_cast<U>(q.w));
}

// ============================================================
// Type aliases
// ============================================================

using quaternionf = quaternion<float>;
using quaterniond = quaternion<double>;

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_QUATERNION_H