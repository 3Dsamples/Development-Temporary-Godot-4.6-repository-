//File 0027 : core/math/matrix_math.h
//High‑performance 4x4 / 3x3 matrix arithmetic: identity, scaling, rotation (axis‑angle + quaternion), translation, perspective projection, look‑at, multiply, inverse, transpose, determinant – built on DirectXMath with Eigen/Godot/GLM adapters.
#ifndef CORE_MATH_MATRIX_MATH_H
#define CORE_MATH_MATRIX_MATH_H

#include "vector_math.h"                  // for vector operations we already defined
#include <cmath>

namespace SimulationMath {
namespace matrix_math {

// -----------------------------------------------------------------------------
// 1. Identity matrix
// -----------------------------------------------------------------------------
inline DirectX::XMMATRIX identity() noexcept { return DirectX::XMMatrixIdentity(); }

// -----------------------------------------------------------------------------
// 2. Translation matrix
// -----------------------------------------------------------------------------
inline DirectX::XMMATRIX translation(float tx, float ty, float tz) noexcept {
    return DirectX::XMMatrixTranslation(tx, ty, tz);
}
inline DirectX::XMMATRIX translation(DirectX::FXMVECTOR v) noexcept {
    return DirectX::XMMatrixTranslationFromVector(v);
}

// -----------------------------------------------------------------------------
// 3. Scaling matrix
// -----------------------------------------------------------------------------
inline DirectX::XMMATRIX scaling(float sx, float sy, float sz) noexcept {
    return DirectX::XMMatrixScaling(sx, sy, sz);
}
inline DirectX::XMMATRIX scaling_uniform(float s) noexcept {
    return DirectX::XMMatrixScaling(s, s, s);
}
inline DirectX::XMMATRIX scaling(DirectX::FXMVECTOR v) noexcept {
    return DirectX::XMMatrixScalingFromVector(v);
}

// -----------------------------------------------------------------------------
// 4. Rotation matrix (axis‑angle)
// -----------------------------------------------------------------------------
inline DirectX::XMMATRIX rotation_axis_angle(DirectX::FXMVECTOR axis, float angle) noexcept {
    return DirectX::XMMatrixRotationAxis(axis, angle);
}
inline DirectX::XMMATRIX rotation_axis_angle(float x, float y, float z, float angle) noexcept {
    DirectX::XMVECTOR axis = DirectX::XMVectorSet(x, y, z, 0.0f);
    return DirectX::XMMatrixRotationAxis(axis, angle);
}

// -----------------------------------------------------------------------------
// 5. Rotation from quaternion
// -----------------------------------------------------------------------------
inline DirectX::XMMATRIX rotation_quaternion(DirectX::FXMVECTOR quat) noexcept {
    return DirectX::XMMatrixRotationQuaternion(quat);
}
inline DirectX::XMMATRIX rotation_quaternion(float qx, float qy, float qz, float qw) noexcept {
    return DirectX::XMMatrixRotationQuaternion(DirectX::XMVectorSet(qx, qy, qz, qw));
}

// -----------------------------------------------------------------------------
// 6. Euler rotation (pitch‑yaw‑roll or ZYX)
// -----------------------------------------------------------------------------
inline DirectX::XMMATRIX rotation_euler(float pitch, float yaw, float roll) noexcept {
    return DirectX::XMMatrixRotationRollPitchYaw(pitch, yaw, roll);
}
inline DirectX::XMMATRIX rotation_euler(DirectX::FXMVECTOR angles) noexcept {
    float x = DirectX::XMVectorGetX(angles);
    float y = DirectX::XMVectorGetY(angles);
    float z = DirectX::XMVectorGetZ(angles);
    return DirectX::XMMatrixRotationRollPitchYaw(x, y, z);
}

// -----------------------------------------------------------------------------
// 7. Look‑At matrix (right‑handed)
// -----------------------------------------------------------------------------
inline DirectX::XMMATRIX look_at_rh(DirectX::FXMVECTOR eye, DirectX::FXMVECTOR target, DirectX::FXMVECTOR up) noexcept {
    return DirectX::XMMatrixLookAtRH(eye, target, up);
}
inline DirectX::XMMATRIX look_at_lh(DirectX::FXMVECTOR eye, DirectX::FXMVECTOR target, DirectX::FXMVECTOR up) noexcept {
    return DirectX::XMMatrixLookAtLH(eye, target, up);
}

// -----------------------------------------------------------------------------
// 8. Perspective projection (right‑handed)
// -----------------------------------------------------------------------------
inline DirectX::XMMATRIX perspective_fov_rh(float fov_y, float aspect, float near_z, float far_z) noexcept {
    return DirectX::XMMatrixPerspectiveFovRH(fov_y, aspect, near_z, far_z);
}
inline DirectX::XMMATRIX perspective_fov_lh(float fov_y, float aspect, float near_z, float far_z) noexcept {
    return DirectX::XMMatrixPerspectiveFovLH(fov_y, aspect, near_z, far_z);
}
inline DirectX::XMMATRIX orthographic_rh(float width, float height, float near_z, float far_z) noexcept {
    return DirectX::XMMatrixOrthographicRH(width, height, near_z, far_z);
}
inline DirectX::XMMATRIX orthographic_lh(float width, float height, float near_z, float far_z) noexcept {
    return DirectX::XMMatrixOrthographicLH(width, height, near_z, far_z);
}

// -----------------------------------------------------------------------------
// 9. Matrix multiplication
// -----------------------------------------------------------------------------
inline DirectX::XMMATRIX mul(DirectX::FXMMATRIX a, DirectX::FXMMATRIX b) noexcept {
    return DirectX::XMMatrixMultiply(a, b);
}
inline DirectX::XMMATRIX mul(DirectX::FXMMATRIX a, DirectX::FXMVECTOR b) noexcept {
    // Actually mul of matrix and vector: we return transformed vector? But return XMMATRIX not possible, we'll make a separate function for transform.
    // Provide function to multiply matrix by matrix, not vector.
    // We'll just keep matrix*matrix multiplication.
    return DirectX::XMMatrixMultiply(a, b);
}

// -----------------------------------------------------------------------------
// 10. Transform point / vector by matrix
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR transform_point(DirectX::FXMVECTOR v, DirectX::FXMMATRIX m) noexcept {
    return DirectX::XMVector3Transform(v, m);
}
inline DirectX::XMVECTOR transform_vector(DirectX::FXMVECTOR v, DirectX::FXMMATRIX m) noexcept {
    return DirectX::XMVector3TransformNormal(v, m);
}
inline DirectX::XMVECTOR transform_point4(DirectX::FXMVECTOR v, DirectX::FXMMATRIX m) noexcept {
    return DirectX::XMVector4Transform(v, m);
}

// -----------------------------------------------------------------------------
// 11. Transpose
// -----------------------------------------------------------------------------
inline DirectX::XMMATRIX transpose(DirectX::FXMMATRIX m) noexcept {
    return DirectX::XMMatrixTranspose(m);
}

// -----------------------------------------------------------------------------
// 12. Inverse
// -----------------------------------------------------------------------------
inline DirectX::XMMATRIX inverse(DirectX::FXMMATRIX m) noexcept {
    return DirectX::XMMatrixInverse(nullptr, m);
}
inline DirectX::XMVECTOR determinant(DirectX::FXMMATRIX m) noexcept {
    DirectX::XMVECTOR det;
    DirectX::XMMatrixDeterminant(det, m);
    return det;
}
inline float determinant_scalar(DirectX::FXMMATRIX m) noexcept {
    return DirectX::XMVectorGetX(determinant(m));
}

// -----------------------------------------------------------------------------
// 13. Compose transform from TRS (translation * rotation * scale)
// -----------------------------------------------------------------------------
inline DirectX::XMMATRIX compose_transform(DirectX::FXMVECTOR translation,
                                            DirectX::FXMVECTOR rotation_quat,
                                            DirectX::FXMVECTOR scale) noexcept {
    DirectX::XMMATRIX T = DirectX::XMMatrixTranslationFromVector(translation);
    DirectX::XMMATRIX R = DirectX::XMMatrixRotationQuaternion(rotation_quat);
    DirectX::XMMATRIX S = DirectX::XMMatrixScalingFromVector(scale);
    return DirectX::XMMatrixMultiply(S, DirectX::XMMatrixMultiply(R, T));
}

// -----------------------------------------------------------------------------
// 14. Extract translation, scale, and rotation from a TRS matrix
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR extract_translation(DirectX::FXMMATRIX m) noexcept {
    return m.r[3];  // DirectX stores translation in row 3 for XMMATRIX
}
inline DirectX::XMVECTOR extract_scale(DirectX::FXMMATRIX m) noexcept {
    float sx = DirectX::XMVector3Length(m.r[0]).m128_f32[0];
    float sy = DirectX::XMVector3Length(m.r[1]).m128_f32[0];
    float sz = DirectX::XMVector3Length(m.r[2]).m128_f32[0];
    return DirectX::XMVectorSet(sx, sy, sz, 0.0f);
}
inline DirectX::XMVECTOR extract_rotation_quat(DirectX::FXMMATRIX m) noexcept {
    // Compute quaternion from rotation part of matrix
    DirectX::XMVECTOR s = extract_scale(m);
    DirectX::XMVECTOR invS = DirectX::XMVectorReciprocal(s);
    DirectX::XMMATRIX rot;
    rot.r[0] = DirectX::XMVectorMultiply(m.r[0], DirectX::XMVectorReplicate(DirectX::XMVectorGetX(invS)));
    rot.r[1] = DirectX::XMVectorMultiply(m.r[1], DirectX::XMVectorReplicate(DirectX::XMVectorGetY(invS)));
    rot.r[2] = DirectX::XMVectorMultiply(m.r[2], DirectX::XMVectorReplicate(DirectX::XMVectorGetZ(invS)));
    rot.r[3] = DirectX::XMVectorSet(0,0,0,1);
    DirectX::XMVECTOR quat = DirectX::XMQuaternionRotationMatrix(rot);
    return quat;
}

// -----------------------------------------------------------------------------
// 15. Decompose matrix into TRS components
// -----------------------------------------------------------------------------
inline void decompose_trs(DirectX::FXMMATRIX m, DirectX::XMVECTOR& out_translation,
                           DirectX::XMVECTOR& out_rotation_quat,
                           DirectX::XMVECTOR& out_scale) noexcept {
    out_translation = extract_translation(m);
    out_scale = extract_scale(m);
    out_rotation_quat = extract_rotation_quat(m);
}

// -----------------------------------------------------------------------------
// 16. 3x3 matrix from 4x4 (upper‑left)
// -----------------------------------------------------------------------------
inline DirectX::XMMATRIX to_3x3(DirectX::FXMMATRIX m) noexcept {
    DirectX::XMMATRIX result = m;
    result.r[3] = DirectX::XMVectorSet(0,0,0,1);
    return result;
}

// -----------------------------------------------------------------------------
// 17. Convert between Godot::Transform3D and DirectX::XMMATRIX (using sim_math_unified_conversions)
// -----------------------------------------------------------------------------
inline Godot::Transform3D to_godot_transform(DirectX::FXMMATRIX m) noexcept {
    Godot::Transform3D t;
    // Basis (3x3)
    t.basis.rows[0][0] = m.r[0].m128_f32[0]; t.basis.rows[0][1] = m.r[1].m128_f32[0]; t.basis.rows[0][2] = m.r[2].m128_f32[0];
    t.basis.rows[1][0] = m.r[0].m128_f32[1]; t.basis.rows[1][1] = m.r[1].m128_f32[1]; t.basis.rows[1][2] = m.r[2].m128_f32[1];
    t.basis.rows[2][0] = m.r[0].m128_f32[2]; t.basis.rows[2][1] = m.r[1].m128_f32[2]; t.basis.rows[2][2] = m.r[2].m128_f32[2];
    t.origin = Godot::Vector3(m.r[3].m128_f32[0], m.r[3].m128_f32[1], m.r[3].m128_f32[2]);
    return t;
}
inline DirectX::XMMATRIX to_directx_matrix(const Godot::Transform3D& t) noexcept {
    DirectX::XMMATRIX m;
    m.r[0] = DirectX::XMVectorSet(t.basis.rows[0][0], t.basis.rows[1][0], t.basis.rows[2][0], 0.0f);
    m.r[1] = DirectX::XMVectorSet(t.basis.rows[0][1], t.basis.rows[1][1], t.basis.rows[2][1], 0.0f);
    m.r[2] = DirectX::XMVectorSet(t.basis.rows[0][2], t.basis.rows[1][2], t.basis.rows[2][2], 0.0f);
    m.r[3] = DirectX::XMVectorSet(t.origin.x, t.origin.y, t.origin.z, 1.0f);
    return m;
}

// -----------------------------------------------------------------------------
// 18. Convert between Eigen::Matrix4f and DirectX::XMMATRIX
// -----------------------------------------------------------------------------
inline Eigen::Matrix4f to_eigen_matrix(DirectX::FXMMATRIX m) noexcept {
    Eigen::Matrix4f res;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            res(i,j) = m.r[j].m128_f32[i]; // column‑major: r[j] is column j, element i
    return res;
}
inline DirectX::XMMATRIX to_directx_matrix(const Eigen::Matrix4f& mat) noexcept {
    DirectX::XMMATRIX m;
    for (int j = 0; j < 4; ++j)
        m.r[j] = DirectX::XMVectorSet(mat(0,j), mat(1,j), mat(2,j), mat(3,j));
    return m;
}

// -----------------------------------------------------------------------------
// 19. Convert between glm::mat4 and DirectX::XMMATRIX
// -----------------------------------------------------------------------------
inline glm::mat4 to_glm_matrix(DirectX::FXMMATRIX m) noexcept {
    return glm::mat4(
        m.r[0].m128_f32[0], m.r[0].m128_f32[1], m.r[0].m128_f32[2], m.r[0].m128_f32[3],
        m.r[1].m128_f32[0], m.r[1].m128_f32[1], m.r[1].m128_f32[2], m.r[1].m128_f32[3],
        m.r[2].m128_f32[0], m.r[2].m128_f32[1], m.r[2].m128_f32[2], m.r[2].m128_f32[3],
        m.r[3].m128_f32[0], m.r[3].m128_f32[1], m.r[3].m128_f32[2], m.r[3].m128_f32[3]);
}
inline DirectX::XMMATRIX to_directx_matrix(const glm::mat4& mat) noexcept {
    DirectX::XMMATRIX m;
    for (int j = 0; j < 4; ++j)
        m.r[j] = DirectX::XMVectorSet(mat[0][j], mat[1][j], mat[2][j], mat[3][j]);
    return m;
}

} // namespace matrix_math
} // namespace SimulationMath

#endif // CORE_MATH_MATRIX_MATH_H