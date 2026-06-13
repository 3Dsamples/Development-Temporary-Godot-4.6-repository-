// system name : onetbb-warp
// File 0009 : core/math/dual_quaternion.h
// Description : Dual quaternion algebra for rigid body transformations and skinning.

#ifndef __TBB_WARP_CORE_MATH_DUAL_QUATERNION_H
#define __TBB_WARP_CORE_MATH_DUAL_QUATERNION_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector3.h"
#include "core/math/vector4.h"
#include "core/math/quaternion.h"
#include "core/math/matrix4.h"
#include <cmath>
#include <type_traits>
#include <array>
#include <initializer_list>
#include <algorithm>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Dual quaternion class template
// ============================================================

template<typename T>
struct dual_quaternion {
    using value_type = T;

    quaternion<T> real;
    quaternion<T> dual;

    // ---- Constructors ----
    constexpr dual_quaternion() noexcept : real(), dual(T(0),T(0),T(0),T(0)) {}
    constexpr dual_quaternion(const quaternion<T>& r, const quaternion<T>& d) noexcept : real(r), dual(d) {}
    template<typename U>
    constexpr explicit dual_quaternion(const dual_quaternion<U>& dq) noexcept
        : real(static_cast<quaternion<T>>(dq.real)), dual(static_cast<quaternion<T>>(dq.dual)) {}

    // ---- Access ----
    constexpr quaternion<T>& operator[](std::size_t i) noexcept { return (&real)[i]; }
    constexpr const quaternion<T>& operator[](std::size_t i) const noexcept { return (&real)[i]; }

    // ---- Compound assignment ----
    constexpr dual_quaternion& operator+=(const dual_quaternion& dq) noexcept { real+=dq.real; dual+=dq.dual; return *this; }
    constexpr dual_quaternion& operator-=(const dual_quaternion& dq) noexcept { real-=dq.real; dual-=dq.dual; return *this; }
    constexpr dual_quaternion& operator*=(T s) noexcept { real*=s; dual*=s; return *this; }
    constexpr dual_quaternion& operator/=(T s) noexcept { real/=s; dual/=s; return *this; }

    // ---- Unary ----
    constexpr dual_quaternion operator+() const noexcept { return *this; }
    constexpr dual_quaternion operator-() const noexcept { return dual_quaternion(-real, -dual); }
};

// ============================================================
// Binary operators
// ============================================================

template<typename T> constexpr dual_quaternion<T> operator+(const dual_quaternion<T>& a, const dual_quaternion<T>& b) noexcept { return dual_quaternion<T>(a.real+b.real, a.dual+b.dual); }
template<typename T> constexpr dual_quaternion<T> operator-(const dual_quaternion<T>& a, const dual_quaternion<T>& b) noexcept { return dual_quaternion<T>(a.real-b.real, a.dual-b.dual); }
template<typename T> constexpr dual_quaternion<T> operator*(const dual_quaternion<T>& dq, T s) noexcept { return dual_quaternion<T>(dq.real*s, dq.dual*s); }
template<typename T> constexpr dual_quaternion<T> operator*(T s, const dual_quaternion<T>& dq) noexcept { return dq*s; }
template<typename T> constexpr dual_quaternion<T> operator/(const dual_quaternion<T>& dq, T s) noexcept { return dual_quaternion<T>(dq.real/s, dq.dual/s); }
template<typename T> constexpr bool operator==(const dual_quaternion<T>& a, const dual_quaternion<T>& b) noexcept { return a.real==b.real && a.dual==b.dual; }
template<typename T> constexpr bool operator!=(const dual_quaternion<T>& a, const dual_quaternion<T>& b) noexcept { return !(a==b); }

// ============================================================
// Dual quaternion multiplication (Grassmann product)
// ============================================================

template<typename T>
constexpr dual_quaternion<T> operator*(const dual_quaternion<T>& a, const dual_quaternion<T>& b) noexcept {
    return dual_quaternion<T>(
        a.real * b.real,
        a.real * b.dual + a.dual * b.real
    );
}

// ============================================================
// Conjugate (quaternion conjugate on both parts)
// ============================================================

template<typename T>
constexpr dual_quaternion<T> conjugate(const dual_quaternion<T>& dq) noexcept {
    return dual_quaternion<T>(conjugate(dq.real), conjugate(dq.dual));
}

// ============================================================
// Dual conjugate (real conjugate, negate dual)
// ============================================================

template<typename T>
constexpr dual_quaternion<T> dual_conjugate(const dual_quaternion<T>& dq) noexcept {
    return dual_quaternion<T>(conjugate(dq.real), -conjugate(dq.dual));
}

// ============================================================
// Norm and normalization
// ============================================================

template<typename T>
constexpr T norm_sq(const dual_quaternion<T>& dq) noexcept {
    return dot(dq.real, dq.real);
}

template<typename T>
T norm(const dual_quaternion<T>& dq) noexcept {
    return std::sqrt(norm_sq(dq));
}

template<typename T>
dual_quaternion<T> normalize(const dual_quaternion<T>& dq) noexcept {
    T n = norm(dq);
    if (n < T(FLOAT_EPSILON)) return dual_quaternion<T>(quaternion<T>(T(0),T(0),T(0),T(1)), quaternion<T>(T(0),T(0),T(0),T(0)));
    T inv = T(1) / n;
    return dual_quaternion<T>(dq.real * inv, dq.dual * inv);
}

// ============================================================
// Inverse
// ============================================================

template<typename T>
dual_quaternion<T> inverse(const dual_quaternion<T>& dq) noexcept {
    T nsq = norm_sq(dq);
    if (nsq < T(FLOAT_EPSILON)) return dual_quaternion<T>(quaternion<T>(T(0),T(0),T(0),T(1)), quaternion<T>(T(0),T(0),T(0),T(0)));
    T inv = T(1) / nsq;
    return dual_quaternion<T>(conjugate(dq.real)*inv, conjugate(dq.dual)*inv);
}

// ============================================================
// Construction from rotation and translation
// ============================================================

template<typename T>
constexpr dual_quaternion<T> from_rotation_translation(const quaternion<T>& rotation, const vector3<T>& translation) noexcept {
    quaternion<T> d;
    d.x = T(0.5) * ( translation.x*rotation.w + translation.y*rotation.z - translation.z*rotation.y );
    d.y = T(0.5) * (-translation.x*rotation.z + translation.y*rotation.w + translation.z*rotation.x );
    d.z = T(0.5) * ( translation.x*rotation.y - translation.y*rotation.x + translation.z*rotation.w );
    d.w = T(0.5) * (-translation.x*rotation.x - translation.y*rotation.y - translation.z*rotation.z );
    return dual_quaternion<T>(rotation, d);
}

// ============================================================
// Construction from TRS (translation, rotation, scale) with matrix
// ============================================================

template<typename T>
dual_quaternion<T> from_TRS(const vector3<T>& translation, const quaternion<T>& rotation, const vector3<T>& scale) noexcept {
    return from_rotation_translation(rotation, translation);
    // Note: non‑uniform scale cannot be represented exactly; this function works for uniform scale only.
}

// ============================================================
// Decompose into rotation and translation
// ============================================================

template<typename T>
void decompose(const dual_quaternion<T>& dq, quaternion<T>& rotation, vector3<T>& translation) noexcept {
    rotation = normalize(dq.real);
    quaternion<T> t = dq.dual * conjugate(rotation) * T(2);
    translation = vector3<T>(t.x, t.y, t.z);
}

// ============================================================
// Transform a point (rigid body transformation)
// ============================================================

template<typename T>
constexpr vector3<T> transform_point(const dual_quaternion<T>& dq, const vector3<T>& point) noexcept {
    // p' = dq * p * dual_conjugate(dq)  (only the real part acts on point, dual carries translation)
    // More efficient:
    vector3<T> rv = rotate(dq.real, point);
    quaternion<T> t = dq.dual * conjugate(dq.real) * T(2);
    return rv + vector3<T>(t.x, t.y, t.z);
}

template<typename T>
constexpr vector3<T> transform_vector(const dual_quaternion<T>& dq, const vector3<T>& v) noexcept {
    return rotate(dq.real, v);
}

// ============================================================
// Conversion to 4x4 matrix
// ============================================================

template<typename T>
matrix4<T> to_matrix(const dual_quaternion<T>& dq) noexcept {
    quaternion<T> r = normalize(dq.real);
    quaternion<T> t = dq.dual * conjugate(r) * T(2);
    matrix4<T> m = matrix4<T>(matrix3<T>(r));
    m(0,3) = t.x; m(1,3) = t.y; m(2,3) = t.z;
    return m;
}

// ============================================================
// Interpolation (DLB – Dual Linear Blending)
// ============================================================

template<typename T>
dual_quaternion<T> nlerp(const dual_quaternion<T>& a, const dual_quaternion<T>& b, T t) noexcept {
    T cos_theta = dot(a.real, b.real);
    dual_quaternion<T> corrected_b = b;
    if (cos_theta < T(0)) { corrected_b = -corrected_b; cos_theta = -cos_theta; }
    dual_quaternion<T> result = a * (T(1) - t) + corrected_b * t;
    return normalize(result);
}

template<typename T>
dual_quaternion<T> sclerp(const dual_quaternion<T>& a, const dual_quaternion<T>& b, T t) noexcept {
    // Screw linear interpolation: decompose, interpolate rotation and translation separately
    quaternion<T> ra, rb;
    vector3<T> ta, tb;
    decompose(a, ra, ta);
    decompose(b, rb, tb);
    quaternion<T> r = slerp(ra, rb, t);
    vector3<T> trans = lerp(ta, tb, t);
    return from_rotation_translation(r, trans);
}

// ============================================================
// ScLERP (screw linear interpolation) with proper geodesic
// ============================================================

template<typename T>
dual_quaternion<T> scLerp(const dual_quaternion<T>& a, const dual_quaternion<T>& b, T t) noexcept {
    dual_quaternion<T> diff = conjugate(a) * b;
    diff = normalize(diff);
    // Extract screw parameters
    T angle = T(2) * std::acos(clamp(diff.real.w, T(-1), T(1)));
    T pitch = T(-2) * diff.dual.w / (std::sin(angle * T(0.5)) + T(FLOAT_EPSILON));
    T s = std::sin(t * angle * T(0.5)) / (std::sin(angle * T(0.5)) + T(FLOAT_EPSILON));
    quaternion<T> delta_real = quaternion<T>(
        diff.real.x * s, diff.real.y * s, diff.real.z * s,
        std::cos(t * angle * T(0.5))
    );
    T pitch_factor = t * pitch;
    quaternion<T> delta_dual(
        delta_real.x * pitch_factor - delta_real.w * diff.dual.x,
        delta_real.y * pitch_factor - delta_real.w * diff.dual.y,
        delta_real.z * pitch_factor - delta_real.w * diff.dual.z,
        delta_real.w * pitch_factor + delta_real.x * diff.dual.x + delta_real.y * diff.dual.y + delta_real.z * diff.dual.z
    );
    return normalize(a * dual_quaternion<T>(delta_real, delta_dual));
}

// ============================================================
// Component‑wise operations
// ============================================================

template<typename T> constexpr dual_quaternion<T> abs(const dual_quaternion<T>& dq) noexcept { return dual_quaternion<T>(abs(dq.real), abs(dq.dual)); }
template<typename T> constexpr bool is_identity(const dual_quaternion<T>& dq, T eps = T(FLOAT_EPSILON)) noexcept { return is_identity(dq.real,eps) && is_zero(vector3<T>(dq.dual.x,dq.dual.y,dq.dual.z)); }
template<typename T> bool is_finite(const dual_quaternion<T>& dq) noexcept { return is_finite(dq.real) && is_finite(dq.dual); }
template<typename T> bool is_nan(const dual_quaternion<T>& dq) noexcept { return is_nan(dq.real) || is_nan(dq.dual); }

// ============================================================
// Smooth dynamics
// ============================================================

template<typename T>
dual_quaternion<T> smooth_damp(const dual_quaternion<T>& current, const dual_quaternion<T>& target,
                               vector3<T>& angular_velocity, T smooth_time, T max_speed, T dt) {
    quaternion<T> cur_rot, tgt_rot;
    vector3<T> cur_trans, tgt_trans;
    decompose(current, cur_rot, cur_trans);
    decompose(target, tgt_rot, tgt_trans);
    quaternion<T> new_rot = smooth_damp(cur_rot, tgt_rot, angular_velocity, smooth_time, max_speed, dt);
    vector3<T> new_trans = lerp(cur_trans, tgt_trans, clamp(dt * T(5) / smooth_time, T(0), T(1)));
    return from_rotation_translation(new_rot, new_trans);
}

// ============================================================
// Conversion between types
// ============================================================

template<typename U, typename T>
constexpr dual_quaternion<U> dual_quaternion_cast(const dual_quaternion<T>& dq) noexcept {
    return dual_quaternion<U>(quaternion_cast<U>(dq.real), quaternion_cast<U>(dq.dual));
}

// ============================================================
// Type aliases
// ============================================================

using dual_quaternionf = dual_quaternion<float>;
using dual_quaterniond = dual_quaternion<double>;

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_DUAL_QUATERNION_H