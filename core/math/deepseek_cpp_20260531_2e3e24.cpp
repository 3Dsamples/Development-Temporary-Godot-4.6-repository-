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
    // Project the quaternion onto the twist axis to extract the twist component.
    // Twist quaternion: (cos(θ), sin(θ)*twist_axis)
    // First, compute the rotation vector (log) of q in the plane perpendicular to twist_axis.
    // Approach: compute the component of the quaternion that is aligned with the twist axis.
    // We can compute the twist quaternion as:
    //   twist = normalize( a + b * twist_axis ) where a = q.w, b = component of q.xyz along twist_axis.
    // But that's for a pure quaternion only when rotation axis is twist_axis.
    // The general decomposition: given a unit quaternion q and a unit direction v (twist axis in local frame),
    //   swing = q * twist^{-1} where twist is the rotation about v that aligns the direction v with q*v?
    // Standard method from Game Programming Gems 4:
    //   Let v = twist_axis (unit vector).
    //   Compute a = q.w, b = dot(q.xyz, v).
    //   Twist quaternion: t = normalize( Quat(a, b * v) ) but careful: the imaginary part must be b * v.
    //   Indeed the twist is the rotation about v such that q = swing * twist.
    //   The twist quaternion is t = (a, b*v) / sqrt(a^2 + b^2).
    //   Then swing = q * t^{-1} = q * conjugate(t) (since t is unit).
    // Check: t * v * t* = v? Yes, rotation about v leaves v unchanged. So swing should have no twist about v.
    DirectX::XMVECTOR v = twist_axis;
    float a = vector_math::get_w(q);
    float b = vector_math::dot3_scalar(q, v); // dot of imaginary parts
    // Compute twist quaternion
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
    // Swing = q * conjugate(twist)
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
    // twist angle = 2 * acos(twist.w) * sign(twist dot twist_axis)
    float w = vector_math::get_w(twist);
    w = std::max(-1.0f, std::min(1.0f, w));
    float angle = 2.0f * std::acos(w);
    // Determine sign: the imaginary part of twist is b * twist_axis; b = sin(angle/2). So sign of b determines direction.
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
    // Convert swing to rotation vector (log) and limit its magnitude
    DirectX::XMVECTOR swing_log = quaternion_math::log(swing); // returns pure imaginary vector with magnitude = half-angle
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