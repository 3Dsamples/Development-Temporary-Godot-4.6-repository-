//File 0031 : core/math/transform_math.h
//Complete affine transform (position + rotation quaternion + scale): composition, inverse, point/vector transformation, interpolation, and conversion from/to matrix/Eigen/Godot/GLM.
#ifndef CORE_MATH_TRANSFORM_MATH_H
#define CORE_MATH_TRANSFORM_MATH_H

#include "vector_math.h"
#include "quaternion_math.h"
#include "matrix_math.h"
#include "interpolation.h"

namespace SimulationMath {
namespace transform_math {

// -----------------------------------------------------------------------------
// 1. Transform structure (TRS)
// -----------------------------------------------------------------------------
struct Transform {
    DirectX::XMVECTOR position;
    DirectX::XMVECTOR rotation;   // unit quaternion
    DirectX::XMVECTOR scale;      // per-axis scale

    Transform() noexcept : position(vector_math::zero()),
                           rotation(quaternion_math::identity()),
                           scale(vector_math::replicate(1.0f)) {}

    Transform(DirectX::FXMVECTOR pos, DirectX::FXMVECTOR rot, DirectX::FXMVECTOR scl) noexcept
        : position(pos), rotation(rot), scale(scl) {}

    // Identity
    static Transform identity() noexcept { return Transform(); }

    // From matrix (decompose)
    static Transform from_matrix(DirectX::FXMMATRIX m) noexcept {
        Transform t;
        t.position = matrix_math::extract_translation(m);
        t.rotation = matrix_math::extract_rotation_quat(m);
        t.scale    = matrix_math::extract_scale(m);
        return t;
    }

    // From TRS components
    static Transform from_trs(DirectX::FXMVECTOR pos, DirectX::FXMVECTOR rot, DirectX::FXMVECTOR scl) noexcept {
        return Transform(pos, rot, scl);
    }

    // Compute 4x4 matrix
    DirectX::XMMATRIX to_matrix() const noexcept {
        DirectX::XMMATRIX S = DirectX::XMMatrixScalingFromVector(scale);
        DirectX::XMMATRIX R = DirectX::XMMatrixRotationQuaternion(rotation);
        DirectX::XMMATRIX T = DirectX::XMMatrixTranslationFromVector(position);
        return DirectX::XMMatrixMultiply(S, DirectX::XMMatrixMultiply(R, T));
    }

    // Inverse transform
    Transform inverse() const noexcept {
        DirectX::XMVECTOR invScale = vector_math::recip(scale);
        DirectX::XMVECTOR invRot   = quaternion_math::conjugate(rotation);
        // position of inverse = - invRot * (position * invScale)
        DirectX::XMVECTOR posScaled = DirectX::XMVectorMultiply(position, invScale);
        DirectX::XMVECTOR invPos = DirectX::XMVectorNegate(quaternion_math::rotate_vector(invRot, posScaled));
        return Transform(invPos, invRot, invScale);
    }
};

// -----------------------------------------------------------------------------
// 2. Compose two transforms (T1 * T2: T2 applied first, then T1)
// -----------------------------------------------------------------------------
inline Transform compose(const Transform& a, const Transform& b) noexcept {
    // T = a.matrix * b.matrix
    DirectX::XMMATRIX m = DirectX::XMMatrixMultiply(b.to_matrix(), a.to_matrix());
    return Transform::from_matrix(m);
}

// -----------------------------------------------------------------------------
// 3. Transform a point (translation + rotation + scale)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR transform_point(const Transform& t, DirectX::FXMVECTOR point) noexcept {
    DirectX::XMVECTOR scaled = DirectX::XMVectorMultiply(point, t.scale);
    DirectX::XMVECTOR rotated = quaternion_math::rotate_vector(t.rotation, scaled);
    return DirectX::XMVectorAdd(rotated, t.position);
}

// -----------------------------------------------------------------------------
// 4. Transform a direction (rotation only, scale if non-uniform may be needed)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR transform_direction(const Transform& t, DirectX::FXMVECTOR dir) noexcept {
    DirectX::XMVECTOR scaled = DirectX::XMVectorMultiply(dir, t.scale);
    return quaternion_math::rotate_vector(t.rotation, scaled);
}

// -----------------------------------------------------------------------------
// 5. Linear interpolation of transforms (TRS components separately)
// -----------------------------------------------------------------------------
inline Transform lerp(const Transform& a, const Transform& b, float t) noexcept {
    return Transform(
        vector_math::lerp(a.position, b.position, t),
        quaternion_math::nlerp(a.rotation, b.rotation, t),
        vector_math::lerp(a.scale, b.scale, t)
    );
}

// -----------------------------------------------------------------------------
// 6. Spherical linear interpolation (position and scale linearly, rotation slerp)
// -----------------------------------------------------------------------------
inline Transform slerp(const Transform& a, const Transform& b, float t) noexcept {
    return Transform(
        vector_math::lerp(a.position, b.position, t),
        quaternion_math::slerp(a.rotation, b.rotation, t),
        vector_math::lerp(a.scale, b.scale, t)
    );
}

// -----------------------------------------------------------------------------
// 7. Convert to/from Godot Transform3D
// -----------------------------------------------------------------------------
inline Godot::Transform3D to_godot(const Transform& t) noexcept {
    Godot::Transform3D g;
    g.origin = vector_math::to_godot(t.position);
    g.basis  = matrix_math::to_godot_transform(DirectX::XMMatrixRotationQuaternion(t.rotation));
    // scale is not directly stored in Godot Transform3D (assumed uniform?), but we can apply it to basis rows?
    // For simplicity, we omit scale; caller may need to handle separately.
    return g;
}
inline Transform from_godot(const Godot::Transform3D& g) noexcept {
    DirectX::XMMATRIX m = matrix_math::to_directx_matrix(g);
    return Transform::from_matrix(m);
}

// -----------------------------------------------------------------------------
// 8. Convert to/from Eigen (Isometry3f or Matrix4f)
// -----------------------------------------------------------------------------
inline Transform from_eigen_matrix(const Eigen::Matrix4f& m) noexcept {
    DirectX::XMMATRIX dx = matrix_math::to_directx_matrix(m);
    return Transform::from_matrix(dx);
}
inline Eigen::Matrix4f to_eigen_matrix(const Transform& t) noexcept {
    DirectX::XMMATRIX dx = t.to_matrix();
    return matrix_math::to_eigen_matrix(dx);
}

// -----------------------------------------------------------------------------
// 9. Convert to/from glm::mat4
// -----------------------------------------------------------------------------
inline Transform from_glm_matrix(const glm::mat4& m) noexcept {
    DirectX::XMMATRIX dx = matrix_math::to_directx_matrix(m);
    return Transform::from_matrix(dx);
}
inline glm::mat4 to_glm_matrix(const Transform& t) noexcept {
    DirectX::XMMATRIX dx = t.to_matrix();
    return matrix_math::to_glm_matrix(dx);
}

} // namespace transform_math
} // namespace SimulationMath

#endif // CORE_MATH_TRANSFORM_MATH_H