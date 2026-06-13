//File 0028 : core/math/quaternion_math.h
//Complete quaternion algebra: identity, conjugate, inverse, normalize, multiply, axis‑angle creation, Euler conversion, SLERP/NLERP, rotation of vectors, and conversion between Godot/Eigen/GLM/DirectX representations.
#ifndef CORE_MATH_QUATERNION_MATH_H
#define CORE_MATH_QUATERNION_MATH_H

#include "vector_math.h"
#include "matrix_math.h"
#include <cmath>
#include <type_traits>

namespace SimulationMath {
namespace quaternion_math {

// -----------------------------------------------------------------------------
// 1. Identity quaternion
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR identity() noexcept { return DirectX::XMQuaternionIdentity(); }

// -----------------------------------------------------------------------------
// 2. Create quaternion from components (x, y, z, w)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR create(float x, float y, float z, float w) noexcept {
    return DirectX::XMVectorSet(x, y, z, w);
}
inline DirectX::XMVECTOR create(DirectX::FXMVECTOR xyz, float w) noexcept {
    return DirectX::XMVectorSetW(xyz, w);
}

// -----------------------------------------------------------------------------
// 3. Conjugate (inverse for unit quaternions)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR conjugate(DirectX::FXMVECTOR q) noexcept {
    return DirectX::XMQuaternionConjugate(q);
}

// -----------------------------------------------------------------------------
// 4. Inverse (works for non‑unit quaternions)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR inverse(DirectX::FXMVECTOR q) noexcept {
    float len_sq = vector_math::length_sq4_scalar(q);
    DirectX::XMVECTOR conj = DirectX::XMQuaternionConjugate(q);
    return DirectX::XMVectorScale(conj, 1.0f / len_sq);
}

// -----------------------------------------------------------------------------
// 5. Normalize (make unit length)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR normalize(DirectX::FXMVECTOR q) noexcept {
    return DirectX::XMQuaternionNormalize(q);
}
inline DirectX::XMVECTOR normalize_est(DirectX::FXMVECTOR q) noexcept {
    return DirectX::XMQuaternionNormalizeEst(q);
}

// -----------------------------------------------------------------------------
// 6. Dot product of two quaternions
// -----------------------------------------------------------------------------
inline float dot(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept {
    return vector_math::dot4_scalar(a, b);
}

// -----------------------------------------------------------------------------
// 7. Quaternion multiplication (compose rotations: q1 then q2 → q2 * q1)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR mul(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept {
    DirectX::XMVECTOR q1 = a;
    DirectX::XMVECTOR q2 = b;
    float w1 = vector_math::get_w(q1), x1 = vector_math::get_x(q1), y1 = vector_math::get_y(q1), z1 = vector_math::get_z(q1);
    float w2 = vector_math::get_w(q2), x2 = vector_math::get_x(q2), y2 = vector_math::get_y(q2), z2 = vector_math::get_z(q2);
    return DirectX::XMVectorSet(
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    );
}

// -----------------------------------------------------------------------------
// 8. Create quaternion from axis‑angle
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR from_axis_angle(DirectX::FXMVECTOR axis, float angle) noexcept {
    return DirectX::XMQuaternionRotationAxis(axis, angle);
}
inline DirectX::XMVECTOR from_axis_angle(float x, float y, float z, float angle) noexcept {
    DirectX::XMVECTOR axis = DirectX::XMVectorSet(x, y, z, 0.0f);
    return DirectX::XMQuaternionRotationAxis(axis, angle);
}

// -----------------------------------------------------------------------------
// 9. Create quaternion from rotation matrix
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR from_matrix(DirectX::FXMMATRIX m) noexcept {
    return DirectX::XMQuaternionRotationMatrix(m);
}

// -----------------------------------------------------------------------------
// 10. Create rotation matrix from quaternion
// -----------------------------------------------------------------------------
inline DirectX::XMMATRIX to_matrix(DirectX::FXMVECTOR q) noexcept {
    return DirectX::XMMatrixRotationQuaternion(q);
}

// -----------------------------------------------------------------------------
// 11. Rotate a vector by a quaternion (q * v * q⁻¹, assumes unit quaternion)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR rotate_vector(DirectX::FXMVECTOR q, DirectX::FXMVECTOR v) noexcept {
    DirectX::XMVECTOR t = DirectX::XMVector3Cross(q, v);
    t = DirectX::XMVectorAdd(t, t);
    return DirectX::XMVectorAdd(v, DirectX::XMVectorAdd(
        DirectX::XMVectorScale(t, vector_math::get_w(q)),
        DirectX::XMVector3Cross(q, t)));
}

// -----------------------------------------------------------------------------
// 12. SLERP (spherical linear interpolation) – constant angular speed
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR slerp(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b, float t) noexcept {
    float cos_omega = dot(a, b);
    DirectX::XMVECTOR flip_b = b;
    if (cos_omega < 0.0f) {
        flip_b = DirectX::XMVectorNegate(b);
        cos_omega = -cos_omega;
    }
    float k0, k1;
    if (cos_omega > 0.9999f) {
        k0 = 1.0f - t;
        k1 = t;
    } else {
        float sin_omega = std::sqrt(1.0f - cos_omega * cos_omega);
        float omega = std::atan2(sin_omega, cos_omega);
        float inv_sin = 1.0f / sin_omega;
        k0 = std::sin((1.0f - t) * omega) * inv_sin;
        k1 = std::sin(t * omega) * inv_sin;
    }
    return DirectX::XMVectorAdd(
        DirectX::XMVectorScale(a, k0),
        DirectX::XMVectorScale(flip_b, k1));
}

// -----------------------------------------------------------------------------
// 13. NLERP (normalized linear interpolation) – faster, non‑constant speed
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR nlerp(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b, float t) noexcept {
    DirectX::XMVECTOR result = DirectX::XMVectorLerp(a, b, t);
    if (dot(a, b) < 0.0f) {
        result = DirectX::XMVectorLerp(a, DirectX::XMVectorNegate(b), t);
    }
    return DirectX::XMQuaternionNormalize(result);
}

// -----------------------------------------------------------------------------
// 14. Squad (spherical cubic interpolation) – uses quaternion control points
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR squad(DirectX::FXMVECTOR q0, DirectX::FXMVECTOR q1,
                                DirectX::FXMVECTOR a, DirectX::FXMVECTOR b, float t) noexcept {
    return slerp(slerp(q0, q1, t), slerp(a, b, t), 2.0f * t * (1.0f - t));
}

// -----------------------------------------------------------------------------
// 15. Logarithm of a unit quaternion (for exponential map)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR log(DirectX::FXMVECTOR q) noexcept {
    float w = vector_math::get_w(q);
    float x = vector_math::get_x(q);
    float y = vector_math::get_y(q);
    float z = vector_math::get_z(q);
    float vec_len = std::sqrt(x*x + y*y + z*z);
    if (vec_len < 1e-12f) return DirectX::XMVectorZero();
    float theta = std::atan2(vec_len, w);
    float inv_len = theta / vec_len;
    return DirectX::XMVectorSet(x * inv_len, y * inv_len, z * inv_len, 0.0f);
}

// -----------------------------------------------------------------------------
// 16. Exponential map to unit quaternion
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR exp(DirectX::FXMVECTOR v) noexcept {
    float x = vector_math::get_x(v);
    float y = vector_math::get_y(v);
    float z = vector_math::get_z(v);
    float theta = std::sqrt(x*x + y*y + z*z);
    if (theta < 1e-12f) return identity();
    float sin_theta = std::sin(theta);
    float inv_theta = 1.0f / theta;
    return DirectX::XMVectorSet(x * sin_theta * inv_theta,
                                 y * sin_theta * inv_theta,
                                 z * sin_theta * inv_theta,
                                 std::cos(theta));
}

// -----------------------------------------------------------------------------
// 17. Euler angles (pitch, yaw, roll) to quaternion
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR from_euler(float pitch, float yaw, float roll) noexcept {
    return DirectX::XMQuaternionRotationRollPitchYaw(pitch, yaw, roll);
}
inline DirectX::XMVECTOR from_euler(DirectX::FXMVECTOR angles) noexcept {
    return from_euler(vector_math::get_x(angles), vector_math::get_y(angles), vector_math::get_z(angles));
}

// -----------------------------------------------------------------------------
// 18. Quaternion to Euler angles (ZYX order)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR to_euler(DirectX::FXMVECTOR q) noexcept {
    float w = vector_math::get_w(q), x = vector_math::get_x(q);
    float y = vector_math::get_y(q), z = vector_math::get_z(q);
    float sinp = 2.0f * (w * y - z * x);
    float pitch, yaw, roll;
    if (std::abs(sinp) >= 1.0f)
        pitch = std::copysign(3.14159265358979f * 0.5f, sinp);
    else
        pitch = std::asin(sinp);
    yaw  = std::atan2(2.0f * (w * x + y * z), 1.0f - 2.0f * (x*x + y*y));
    roll = std::atan2(2.0f * (w * z + x * y), 1.0f - 2.0f * (y*y + z*z));
    return DirectX::XMVectorSet(pitch, yaw, roll, 0.0f);
}

// -----------------------------------------------------------------------------
// 19. Quaternion between two direction vectors (shortest arc)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR from_to(DirectX::FXMVECTOR from, DirectX::FXMVECTOR to) noexcept {
    DirectX::XMVECTOR f = DirectX::XMVector3Normalize(from);
    DirectX::XMVECTOR t = DirectX::XMVector3Normalize(to);
    float d = vector_math::dot3_scalar(f, t);
    if (d > 0.9999999f) return identity();
    if (d < -0.9999999f) {
        DirectX::XMVECTOR axis = DirectX::XMVector3Cross(DirectX::XMVectorSet(1,0,0,0), f);
        if (vector_math::length_sq3_scalar(axis) < 1e-6f)
            axis = DirectX::XMVector3Cross(DirectX::XMVectorSet(0,1,0,0), f);
        axis = DirectX::XMVector3Normalize(axis);
        return from_axis_angle(axis, 3.14159265358979f);
    }
    DirectX::XMVECTOR axis = DirectX::XMVector3Cross(f, t);
    float s = std::sqrt((1.0f + d) * 2.0f);
    float inv_s = 1.0f / s;
    return DirectX::XMVectorSet(
        vector_math::get_x(axis) * inv_s,
        vector_math::get_y(axis) * inv_s,
        vector_math::get_z(axis) * inv_s,
        s * 0.5f);
}

// -----------------------------------------------------------------------------
// 20. Angular velocity (time derivative) from quaternion and angular velocity vector omega
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR integrate_angular_velocity(DirectX::FXMVECTOR q, DirectX::FXMVECTOR omega, float dt) noexcept {
    DirectX::XMVECTOR w = DirectX::XMVectorScale(omega, 0.5f);
    DirectX::XMVECTOR qw = create(vector_math::get_x(w), vector_math::get_y(w), vector_math::get_z(w), 0.0f);
    DirectX::XMVECTOR dq = mul(qw, q);
    DirectX::XMVECTOR result = DirectX::XMVectorAdd(q, DirectX::XMVectorScale(dq, dt));
    return normalize(result);
}

// -----------------------------------------------------------------------------
// 21. Godot::Quaternion ↔ DirectX
// -----------------------------------------------------------------------------
inline Godot::Quaternion to_godot(DirectX::FXMVECTOR q) noexcept {
    return Godot::Quaternion(vector_math::get_x(q), vector_math::get_y(q), vector_math::get_z(q), vector_math::get_w(q));
}
inline DirectX::XMVECTOR to_directx(const Godot::Quaternion& q) noexcept {
    return DirectX::XMVectorSet(q.x, q.y, q.z, q.w);
}

// -----------------------------------------------------------------------------
// 22. Eigen::Quaternionf ↔ DirectX
// -----------------------------------------------------------------------------
inline Eigen::Quaternionf to_eigen(DirectX::FXMVECTOR q) noexcept {
    return Eigen::Quaternionf(vector_math::get_w(q), vector_math::get_x(q), vector_math::get_y(q), vector_math::get_z(q));
}
inline DirectX::XMVECTOR to_directx(const Eigen::Quaternionf& eq) noexcept {
    return DirectX::XMVectorSet(eq.x(), eq.y(), eq.z(), eq.w());
}

// -----------------------------------------------------------------------------
// 23. glm::quat ↔ DirectX
// -----------------------------------------------------------------------------
inline glm::quat to_glm(DirectX::FXMVECTOR q) noexcept {
    return glm::quat(vector_math::get_w(q), vector_math::get_x(q), vector_math::get_y(q), vector_math::get_z(q));
}
inline DirectX::XMVECTOR to_directx(const glm::quat& gq) noexcept {
    return DirectX::XMVectorSet(gq.x, gq.y, gq.z, gq.w);
}

} // namespace quaternion_math
} // namespace SimulationMath

#endif // CORE_MATH_QUATERNION_MATH_H