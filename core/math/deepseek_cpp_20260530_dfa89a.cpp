// File 0020 : core/math/dual_quat.h
// Dual quaternion algebra for rigid body transformations, blending, and skinning (float/double).

#pragma once

#include "quat.h"
#include "vec3.h"
#include "mat4.h"
#include <cmath>

namespace wp {

template <typename T>
struct dual_quat {
    quat<T> real;   // rotation
    quat<T> dual;   // translation (screw)

    constexpr dual_quat() noexcept : real(), dual(T(0), T(0), T(0), T(0)) {}
    constexpr dual_quat(const quat<T>& r, const quat<T>& d) noexcept : real(r), dual(d) {}
    template <typename U> constexpr explicit dual_quat(const dual_quat<U>& o) noexcept
        : real(o.real), dual(o.dual) {}

    // Constructor from rotation quaternion and translation vector
    dual_quat(const quat<T>& rot, const vec3<T>& trans) noexcept
        : real(rot)
    {
        dual = quat<T>(trans.x * T(0.5), trans.y * T(0.5), trans.z * T(0.5), T(0));
        dual = mul(dual, rot);
    }

    // Identity
    static constexpr dual_quat identity() noexcept { return dual_quat(); }

    // Conjugate (both parts)
    constexpr dual_quat conjugate() const noexcept {
        return dual_quat(conjugate(real), conjugate(dual));
    }

    // Norm (real part norm, dual part derived)
    constexpr quat<T> norm_real() const noexcept { return real; }
    T norm() const noexcept { return wp::norm(real); }

    // Normalize
    dual_quat normalize() const noexcept {
        T n = norm();
        if (n < MathConst<T>::epsilon)
            return identity();
        T inv_n = T(1) / n;
        return dual_quat(real * inv_n, dual * inv_n);
    }

    // Transform point
    vec3<T> transform_point(const vec3<T>& p) const noexcept {
        // p' = real * p * conjugate(real) + 2 * dual * conjugate(real)  (simplified)
        quat<T> qp(p, T(0));
        quat<T> t = mul(dual, conjugate(real));
        t = t + t; // 2 * dual * conj(real)
        quat<T> rotated = mul(mul(real, qp), conjugate(real));
        return vec3<T>(rotated.x + t.x, rotated.y + t.y, rotated.z + t.z);
    }

    // Transform vector (rotation only)
    vec3<T> transform_vector(const vec3<T>& v) const noexcept {
        quat<T> qv(v, T(0));
        quat<T> r = mul(mul(real, qv), conjugate(real));
        return vec3<T>(r.x, r.y, r.z);
    }

    // Multiplication (dual quaternion multiplication)
    constexpr friend dual_quat operator*(const dual_quat& a, const dual_quat& b) noexcept {
        return dual_quat(mul(a.real, b.real),
                         mul(a.real, b.dual) + mul(a.dual, b.real));
    }

    // Translate (post‑multiply by translation along axis)
    dual_quat translate(const vec3<T>& t) const noexcept {
        return *this * dual_quat(quat<T>(), t);
    }

    // Rotate (post‑multiply by rotation)
    dual_quat rotate(const quat<T>& q) const noexcept {
        return *this * dual_quat(q, vec3<T>(T(0)));
    }

    // Conversion to 4x4 matrix
    mat4<T> to_matrix4() const noexcept {
        mat3<T> r = to_matrix3(real);
        vec3<T> t = transform_point(vec3<T>(T(0)));
        return mat4<T>(r.m00, r.m01, r.m02, t.x,
                       r.m10, r.m11, r.m12, t.y,
                       r.m20, r.m21, r.m22, t.z,
                       T(0), T(0), T(0), T(1));
    }

    constexpr bool operator==(const dual_quat& o) const noexcept { return real == o.real && dual == o.dual; }
    constexpr bool operator!=(const dual_quat& o) const noexcept { return !(*this == o); }
};

// Scalar multiplication
template <typename T> constexpr dual_quat<T> operator*(T s, const dual_quat<T>& dq) noexcept {
    return dual_quat<T>(dq.real * s, dq.dual * s);
}
template <typename T> constexpr dual_quat<T> operator*(const dual_quat<T>& dq, T s) noexcept {
    return dual_quat<T>(dq.real * s, dq.dual * s);
}
template <typename T> constexpr dual_quat<T> operator+(const dual_quat<T>& a, const dual_quat<T>& b) noexcept {
    return dual_quat<T>(a.real + b.real, a.dual + b.dual);
}
template <typename T> constexpr dual_quat<T> operator-(const dual_quat<T>& a, const dual_quat<T>& b) noexcept {
    return dual_quat<T>(a.real - b.real, a.dual - b.dual);
}

// Screw linear interpolation (ScLerp)
template <typename T>
dual_quat<T> sclerp(const dual_quat<T>& a, const dual_quat<T>& b, T t) noexcept {
    // difference quaternion
    dual_quat<T> diff = conjugate(a) * b;
    // extract rotation and translation from diff
    // Use quaternion log/exp for rotation, then blend
    vec3<T> axis;
    T angle;
    to_axis_angle(diff.real, axis, angle);
    T half_angle = angle * T(0.5);
    T s = std::sin(half_angle);
    // rotation part
    quat<T> rot_blend = from_axis_angle(axis, angle * t);
    // translation part: need screw motion decomposition.
    // For simplicity, use dual quaternion linear blend (DLB) if angle small
    if (std::abs(angle) < MathConst<T>::epsilon) {
        return a + t * (b - a);
    }
    // Compute screw pitch: translation dual part ratio
    // diff.dual = (0.5 * t_vec) * diff.real
    // So we can compute the translation vector
    quat<T> trans_q = mul(diff.dual, conjugate(diff.real));
    vec3<T> trans = vec3<T>(trans_q.x, trans_q.y, trans_q.z) * T(2);
    // The screw axis is the same as rotation axis
    // Interpolate translation linearly along the screw
    vec3<T> trans_blend = trans * t;
    dual_quat<T> inc(rot_blend, trans_blend);
    return a * inc;
}

// Linear blend (DLB) – simple nlerp of dual quaternions, then normalize
template <typename T>
dual_quat<T> blend(const dual_quat<T>& a, const dual_quat<T>& b, T t) noexcept {
    return (a + (b - a) * t).normalize();
}

// From rotation+translation (already constructor)
template <typename T>
dual_quat<T> from_rotation_translation(const quat<T>& rot, const vec3<T>& trans) noexcept {
    return dual_quat<T>(rot, trans);
}

using dual_quatf = dual_quat<float>;
using dual_quatd = dual_quat<double>;

} // namespace wp