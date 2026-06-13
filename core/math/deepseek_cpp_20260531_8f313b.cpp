//File 0039 : core/math/dual_quaternion.h
//Dual quaternion algebra: creation from TRS/rotation‑translation, arithmetic, conjugate, norm, normalization, blending (DLB), rigid transformation of points, and conversion to matrices.
#ifndef CORE_MATH_DUAL_QUATERNION_H
#define CORE_MATH_DUAL_QUATERNION_H

#include "vector_math.h"
#include "quaternion_math.h"
#include "transform_math.h"
#include <cmath>

namespace SimulationMath {
namespace dual_quaternion {

// -----------------------------------------------------------------------------
// 1. Dual quaternion structure: real part (quat) + dual part (quat)
// -----------------------------------------------------------------------------
struct DualQuat {
    DirectX::XMVECTOR real;   // unit quaternion for rotation
    DirectX::XMVECTOR dual;   // translation quaternion: 0.5 * (t * real) where t is pure‑imaginary position vector

    DualQuat() noexcept : real(quaternion_math::identity()), dual(vector_math::zero()) {}
    DualQuat(DirectX::FXMVECTOR r, DirectX::FXMVECTOR d) noexcept : real(r), dual(d) {}
};

// -----------------------------------------------------------------------------
// 2. Identity
// -----------------------------------------------------------------------------
inline DualQuat identity() noexcept {
    return DualQuat(quaternion_math::identity(), vector_math::zero());
}

// -----------------------------------------------------------------------------
// 3. Create from rotation quaternion and translation vector
// -----------------------------------------------------------------------------
inline DualQuat from_rotation_translation(DirectX::FXMVECTOR rot, DirectX::FXMVECTOR translation) noexcept {
    DirectX::XMVECTOR t_vec = DirectX::XMVectorSetW(translation, 0.0f); // ensure w=0 for pure imaginary
    DirectX::XMVECTOR half_dual = quaternion_math::mul(t_vec, rot);
    DirectX::XMVECTOR dual = DirectX::XMVectorScale(half_dual, 0.5f);
    return DualQuat(rot, dual);
}

// -----------------------------------------------------------------------------
// 4. Create from TRS Transform (ignores scale; dual quaternion is rigid)
// -----------------------------------------------------------------------------
inline DualQuat from_transform(const transform_math::Transform& t) noexcept {
    return from_rotation_translation(t.rotation, t.position);
}

// -----------------------------------------------------------------------------
// 5. Create from 4x4 rigid matrix (rotation + translation)
// -----------------------------------------------------------------------------
inline DualQuat from_matrix(DirectX::FXMMATRIX m) noexcept {
    DirectX::XMVECTOR rot = quaternion_math::from_matrix(m);
    DirectX::XMVECTOR trans = matrix_math::extract_translation(m);
    return from_rotation_translation(rot, trans);
}

// -----------------------------------------------------------------------------
// 6. Convert back to 4x4 matrix
// -----------------------------------------------------------------------------
inline DirectX::XMMATRIX to_matrix(const DualQuat& dq) noexcept {
    DirectX::XMMATRIX R = DirectX::XMMatrixRotationQuaternion(dq.real);
    DirectX::XMVECTOR t = transform_rigid_point(dq, vector_math::zero()); // translation of origin
    DirectX::XMMATRIX T = DirectX::XMMatrixTranslationFromVector(t);
    return DirectX::XMMatrixMultiply(T, R);
}

// -----------------------------------------------------------------------------
// 7. Extract rotation quaternion
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR rotation(const DualQuat& dq) noexcept { return dq.real; }

// -----------------------------------------------------------------------------
// 8. Extract translation vector
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR translation(const DualQuat& dq) noexcept {
    DirectX::XMVECTOR conj = quaternion_math::conjugate(dq.real);
    DirectX::XMVECTOR t = quaternion_math::mul(dq.dual, conj);
    t = DirectX::XMVectorScale(t, 2.0f);
    return t; // w component will be 0
}

// -----------------------------------------------------------------------------
// 9. Conjugate (dual conjugate)
// -----------------------------------------------------------------------------
inline DualQuat conjugate(const DualQuat& dq) noexcept {
    return DualQuat(quaternion_math::conjugate(dq.real),
                    quaternion_math::conjugate(dq.dual));
}

// -----------------------------------------------------------------------------
// 10. Norm squared (should be 1 for unit dual quaternion)
// -----------------------------------------------------------------------------
inline float norm_squared(const DualQuat& dq) noexcept {
    return quaternion_math::dot(dq.real, dq.real);
}

// -----------------------------------------------------------------------------
// 11. Normalize (make unit)
// -----------------------------------------------------------------------------
inline DualQuat normalize(const DualQuat& dq) noexcept {
    float len = std::sqrt(quaternion_math::dot(dq.real, dq.real));
    if (len < 1e-12f) return identity();
    float inv_len = 1.0f / len;
    return DualQuat(DirectX::XMVectorScale(dq.real, inv_len),
                    DirectX::XMVectorScale(dq.dual, inv_len));
}

// -----------------------------------------------------------------------------
// 12. Multiplication (composition)
// -----------------------------------------------------------------------------
inline DualQuat mul(const DualQuat& a, const DualQuat& b) noexcept {
    DirectX::XMVECTOR real = quaternion_math::mul(a.real, b.real);
    DirectX::XMVECTOR dual = DirectX::XMVectorAdd(
        quaternion_math::mul(a.real, b.dual),
        quaternion_math::mul(a.dual, b.real));
    return DualQuat(real, dual);
}

// -----------------------------------------------------------------------------
// 13. Transform a 3D point (rigid body motion)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR transform_rigid_point(const DualQuat& dq, DirectX::FXMVECTOR point) noexcept {
    DirectX::XMVECTOR p_quat = DirectX::XMVectorSetW(point, 0.0f); // pure imaginary
    DirectX::XMVECTOR real_conj = quaternion_math::conjugate(dq.real);
    // rotated point: real * p * real*
    DirectX::XMVECTOR rotated = quaternion_math::mul(quaternion_math::mul(dq.real, p_quat), real_conj);
    // translation part: 2 * dual * real*
    DirectX::XMVECTOR t_part = DirectX::XMVectorScale(quaternion_math::mul(dq.dual, real_conj), 2.0f);
    // result = rotated + t_part, but need to ensure w stays 0; we'll return just xyz.
    DirectX::XMVECTOR result = DirectX::XMVectorAdd(rotated, t_part);
    return result; // w component is 0
}

// -----------------------------------------------------------------------------
// 14. Transform a 3D vector (rotation only)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR transform_rigid_vector(const DualQuat& dq, DirectX::FXMVECTOR vec) noexcept {
    return quaternion_math::rotate_vector(dq.real, vec);
}

// -----------------------------------------------------------------------------
// 15. Blending (Dual Linear Blending DLB) – average of two dual quaternions
// -----------------------------------------------------------------------------
inline DualQuat blend(const DualQuat& a, const DualQuat& b, float t) noexcept {
    // Simple linear blend followed by normalization (not optimal but works for small t)
    DualQuat result;
    result.real = vector_math::lerp(a.real, b.real, t);
    result.dual = vector_math::lerp(a.dual, b.dual, t);
    return normalize(result);
}

// -----------------------------------------------------------------------------
// 16. Better blending: if dot product negative, flip sign of b to take shortest path
// -----------------------------------------------------------------------------
inline DualQuat blend_shortest(const DualQuat& a, const DualQuat& b, float t) noexcept {
    float dot_val = quaternion_math::dot(a.real, b.real);
    DualQuat b_fixed = b;
    if (dot_val < 0.0f) {
        b_fixed.real = DirectX::XMVectorNegate(b.real);
        b_fixed.dual = DirectX::XMVectorNegate(b.dual);
    }
    DualQuat result;
    result.real = vector_math::lerp(a.real, b_fixed.real, t);
    result.dual = vector_math::lerp(a.dual, b_fixed.dual, t);
    return normalize(result);
}

// -----------------------------------------------------------------------------
// 17. Exponential map (from a screw motion: rotation axis * angle /2 + moment)
// -----------------------------------------------------------------------------
inline DualQuat exp(DirectX::FXMVECTOR real_part, DirectX::FXMVECTOR dual_part) noexcept {
    // real_part = 0.5 * angle * axis, dual_part = 0.5 * ( t_vec + ... ) but we'll just use quat exp.
    DirectX::XMVECTOR rot = quaternion_math::exp(real_part);
    DirectX::XMVECTOR t = vector_math::zero(); // simplified; full implementation requires more
    DualQuat dq;
    dq.real = rot;
    dq.dual = quaternion_math::mul(DirectX::XMVectorSetW(t, 0.0f), rot);
    dq.dual = DirectX::XMVectorScale(dq.dual, 0.5f);
    return dq;
}

// -----------------------------------------------------------------------------
// 18. Logarithm
// -----------------------------------------------------------------------------
inline void log(const DualQuat& dq, DirectX::XMVECTOR& out_rot_axis, float& out_angle,
                DirectX::XMVECTOR& out_translation) noexcept {
    out_rot_axis = quaternion_math::log(dq.real);
    out_angle = vector_math::length3_scalar(out_rot_axis);
    out_translation = translation(dq);
}

} // namespace dual_quaternion
} // namespace SimulationMath

#endif // CORE_MATH_DUAL_QUATERNION_H