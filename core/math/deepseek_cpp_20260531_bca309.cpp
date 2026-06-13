//File 0062 : core/math/swing_twist.h
//Decompose a quaternion into swing and twist components around a given axis, reconstruct, and compute rotation limits; essential for inverse kinematics and animation.
#ifndef CORE_MATH_SWING_TWIST_H
#define CORE_MATH_SWING_TWIST_H

#include "vector_math.h"
#include "quaternion_math.h"
#include "math_constants.h"

namespace SimulationMath {
namespace swing_twist {

// -----------------------------------------------------------------------------
// 1. Decompose a rotation quaternion q into swing (rotation away from twist_axis) and twist (rotation around twist_axis)
//    q = swing * twist, where swing has no twist component (its log is perpendicular to twist_axis)
//    twist_axis must be unit length in local space (the axis around which twist occurs, e.g., upper arm direction)
// -----------------------------------------------------------------------------
inline void decompose(DirectX::FXMVECTOR q, DirectX::FXMVECTOR twist_axis,
                      DirectX::XMVECTOR& out_swing, DirectX::XMVECTOR& out_twist) noexcept {
    DirectX::XMVECTOR v = twist_axis;
    float a = vector_math::get_w(q);
    float b = vector_math::dot3_scalar(q, v);
    float len = std::sqrt(a*a + b*b);
    DirectX::XMVECTOR twist;
    if (len < 1e-12f) {
        twist = quaternion_math::identity();
    } else {
        float inv_len = 1.0f / len;
        twist = DirectX::XMVectorSet(b * vector_math::get_x(v) * inv_len,
                                      b * vector_math::get_y(v) * inv_len,
                                      b * vector_math::get_z(v) * inv_len,
                                      a * inv_len);
    }
    DirectX::XMVECTOR swing = quaternion_math::mul(q, quaternion_math::conjugate(twist));
    out_twist = twist;
    out_swing = swing;
}

// -----------------------------------------------------------------------------
// 2. Reconstruct a quaternion from swing and twist
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR combine(DirectX::FXMVECTOR swing, DirectX::FXMVECTOR twist) noexcept {
    return quaternion_math::mul(swing, twist);
}

// -----------------------------------------------------------------------------
// 3. Extract the twist angle around the given axis from a quaternion (in radians)
// -----------------------------------------------------------------------------
inline float twist_angle(DirectX::FXMVECTOR q, DirectX::FXMVECTOR twist_axis) noexcept {
    DirectX::XMVECTOR swing, twist;
    decompose(q, twist_axis, swing, twist);
    float w = vector_math::get_w(twist);
    w = std::max(-1.0f, std::min(1.0f, w));
    float angle = 2.0f * std::acos(w);
    float b = vector_math::dot3_scalar(twist, twist_axis);
    if (b < 0.0f) angle = -angle;
    return angle;
}

// -----------------------------------------------------------------------------
// 4. Limit the swing angle to a maximum value (clamp)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR limit_swing(DirectX::FXMVECTOR q, DirectX::FXMVECTOR twist_axis,
                                     float max_swing_angle) noexcept {
    DirectX::XMVECTOR swing, twist;
    decompose(q, twist_axis, swing, twist);
    DirectX::XMVECTOR swing_log = quaternion_math::log(swing);
    float angle = vector_math::length3_scalar(swing_log);
    if (angle > max_swing_angle * 0.5f) {
        swing_log = DirectX::XMVectorScale(swing_log, (max_swing_angle * 0.5f) / angle);
        swing = quaternion_math::exp(swing_log);
    }
    return quaternion_math::mul(swing, twist);
}

} // namespace swing_twist
} // namespace SimulationMath

#endif // CORE_MATH_SWING_TWIST_H