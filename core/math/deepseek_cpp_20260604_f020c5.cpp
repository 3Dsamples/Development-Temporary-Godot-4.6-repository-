// system name : onetbb-warp
// File 0036 : core/math/transforms.h
// Description : Unified 2D/3D transform stack, TRS decomposition, polar decomposition, interpolation.

#ifndef __TBB_WARP_CORE_MATH_TRANSFORMS_H
#define __TBB_WARP_CORE_MATH_TRANSFORMS_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/matrix3.h"
#include "core/math/matrix4.h"
#include "core/math/quaternion.h"
#include "core/math/dual_quaternion.h"
#include <cmath>
#include <type_traits>
#include <array>
#include <vector>
#include <algorithm>
#include <functional>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Transform 3D (position, rotation, scale)
// ============================================================

template<typename T>
struct transform3d {
    vector3<T>    translation;
    quaternion<T> rotation;
    vector3<T>    scale;

    constexpr transform3d() noexcept
        : translation(T(0)), rotation(), scale(T(1)) {}
    constexpr transform3d(const vector3<T>& t, const quaternion<T>& r, const vector3<T>& s) noexcept
        : translation(t), rotation(r), scale(s) {}

    static constexpr transform3d identity() noexcept { return transform3d(); }

    // Conversion to matrix4
    matrix4<T> to_matrix() const noexcept {
        matrix3<T> R = matrix3<T>(rotation);
        matrix4<T> m(R);
        // Apply scale
        m(0,0) *= scale.x; m(0,1) *= scale.y; m(0,2) *= scale.z;
        m(1,0) *= scale.x; m(1,1) *= scale.y; m(1,2) *= scale.z;
        m(2,0) *= scale.x; m(2,1) *= scale.y; m(2,2) *= scale.z;
        m(0,3) = translation.x;
        m(1,3) = translation.y;
        m(2,3) = translation.z;
        return m;
    }

    // Compose two transforms: this * other (first apply other, then this)
    transform3d operator*(const transform3d& other) const noexcept {
        vector3<T> new_translation = translation + rotate(rotation, other.translation * scale);
        quaternion<T> new_rotation = rotation * other.rotation;
        vector3<T> new_scale = scale * other.scale; // component‑wise, not correct for non‑uniform, but common for simplicity
        // More correct: new_scale = scale * other.scale (if uniform), but here we keep component‑wise.
        return transform3d(new_translation, new_rotation, new_scale);
    }

    // Inverse
    transform3d inverse() const noexcept {
        quaternion<T> inv_rot = inverse(rotation);
        vector3<T> inv_scale(T(1)/scale.x, T(1)/scale.y, T(1)/scale.z);
        vector3<T> inv_trans = rotate(inv_rot, -translation) * inv_scale;
        return transform3d(inv_trans, inv_rot, inv_scale);
    }

    // Transform a point
    vector3<T> transform_point(const vector3<T>& p) const noexcept {
        return translation + rotate(rotation, p * scale);
    }

    // Transform a direction (no translation)
    vector3<T> transform_vector(const vector3<T>& v) const noexcept {
        return rotate(rotation, v * scale);
    }
};

// ============================================================
// TRS decomposition from matrix4
// ============================================================

template<typename T>
transform3d<T> decompose_trs(const matrix4<T>& m) noexcept {
    vector3<T> t(m(0,3), m(1,3), m(2,3));
    matrix3<T> R(m(0,0), m(0,1), m(0,2),
                 m(1,0), m(1,1), m(1,2),
                 m(2,0), m(2,1), m(2,2));
    vector3<T> s(length(R.col[0]), length(R.col[1]), length(R.col[2]));
    if (s.x > T(1e-12)) R.col[0] = R.col[0] / s.x;
    if (s.y > T(1e-12)) R.col[1] = R.col[1] / s.y;
    if (s.z > T(1e-12)) R.col[2] = R.col[2] / s.z;
    if (determinant(R) < T(0)) {
        R.col[2] = -R.col[2];
        s.z = -s.z;
    }
    quaternion<T> rot(R);
    return transform3d<T>(t, rot, s);
}

// ============================================================
// Polar decomposition of 3x3 matrix: M = R * S (or S * R)
// ============================================================

template<typename T>
void polar_decompose(const matrix3<T>& M, matrix3<T>& R, matrix3<T>& S) noexcept {
    const int max_iter = 30;
    R = M;
    for (int iter = 0; iter < max_iter; ++iter) {
        matrix3<T> R_inv_t = transpose(inverse(R));
        matrix3<T> next = (R + R_inv_t) * T(0.5);
        if (norm(next - R) < T(1e-6)) break;
        R = next;
    }
    S = transpose(R) * M;
}

// ============================================================
// Interpolation of transforms
// ============================================================

template<typename T>
transform3d<T> lerp_transform(const transform3d<T>& a, const transform3d<T>& b, T t) noexcept {
    return transform3d<T>(
        lerp(a.translation, b.translation, t),
        slerp(a.rotation, b.rotation, t),
        lerp(a.scale, b.scale, t)
    );
}

template<typename T>
transform3d<T> nlerp_transform(const transform3d<T>& a, const transform3d<T>& b, T t) noexcept {
    return transform3d<T>(
        lerp(a.translation, b.translation, t),
        nlerp(a.rotation, b.rotation, t),
        lerp(a.scale, b.scale, t)
    );
}

// ============================================================
// Transform using dual quaternion (rigid + uniform scale)
// ============================================================

template<typename T>
dual_quaternion<T> transform_to_dual_quaternion(const transform3d<T>& tf) noexcept {
    // Dual quaternion can represent rigid motion (rotation + translation) but not scale.
    // We assume scale is uniform and use only rotation and translation.
    return from_rotation_translation(tf.rotation, tf.translation);
}

template<typename T>
transform3d<T> dual_quaternion_to_transform(const dual_quaternion<T>& dq) noexcept {
    quaternion<T> rot;
    vector3<T> t;
    decompose(dq, rot, t);
    return transform3d<T>(t, rot, vector3<T>(T(1)));
}

// ============================================================
// Hierarchical transform (bone tree)
// ============================================================

template<typename T>
struct bone_transform {
    transform3d<T> local;
    transform3d<T> global;
    int parent_index = -1;
};

template<typename T>
void update_global_transforms(std::vector<bone_transform<T>>& bones) noexcept {
    for (std::size_t i = 0; i < bones.size(); ++i) {
        if (bones[i].parent_index < 0) {
            bones[i].global = bones[i].local;
        } else {
            const auto& parent = bones[bones[i].parent_index];
            bones[i].global = parent.global * bones[i].local;
        }
    }
}

// ============================================================
// Transform from/to column‑major float array (for GPU / API)
// ============================================================

template<typename T>
void transform_to_array(const transform3d<T>& tf, T* out) noexcept {
    matrix4<T> m = tf.to_matrix();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            out[j * 4 + i] = m(i, j); // column‑major
}

template<typename T>
transform3d<T> array_to_transform(const T* in) noexcept {
    matrix4<T> m;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            m(i, j) = in[j * 4 + i];
    return decompose_trs(m);
}

// ============================================================
// Transform 2D (position, rotation angle, scale)
// ============================================================

template<typename T>
struct transform2d {
    vector2<T> translation;
    T rotation;         // angle in radians
    vector2<T> scale;

    constexpr transform2d() noexcept : translation(T(0)), rotation(T(0)), scale(T(1)) {}
    constexpr transform2d(const vector2<T>& t, T r, const vector2<T>& s) noexcept
        : translation(t), rotation(r), scale(s) {}

    static constexpr transform2d identity() noexcept { return transform2d(); }

    // To 3x3 matrix (homogeneous)
    void to_matrix(T out[9]) const noexcept {
        T c = std::cos(rotation), s = std::sin(rotation);
        out[0] = c * scale.x; out[1] = s * scale.x; out[2] = T(0);
        out[3] = -s * scale.y; out[4] = c * scale.y; out[5] = T(0);
        out[6] = translation.x; out[7] = translation.y; out[8] = T(1);
    }

    transform2d operator*(const transform2d& other) const noexcept {
        T c = std::cos(rotation), s = std::sin(rotation);
        vector2<T> new_trans;
        new_trans.x = translation.x + c * other.translation.x * scale.x - s * other.translation.y * scale.y;
        new_trans.y = translation.y + s * other.translation.x * scale.x + c * other.translation.y * scale.y;
        return transform2d(new_trans, rotation + other.rotation, scale * other.scale);
    }

    transform2d inverse() const noexcept {
        T c = std::cos(-rotation), s = std::sin(-rotation);
        vector2<T> inv_scale(T(1)/scale.x, T(1)/scale.y);
        vector2<T> inv_trans;
        inv_trans.x = -(c * translation.x + s * translation.y) * inv_scale.x;
        inv_trans.y = -(-s * translation.x + c * translation.y) * inv_scale.y;
        return transform2d(inv_trans, -rotation, inv_scale);
    }
};

// ============================================================
// 2D Transform interpolation
// ============================================================

template<typename T>
transform2d<T> lerp_transform(const transform2d<T>& a, const transform2d<T>& b, T t) noexcept {
    return transform2d<T>(
        lerp(a.translation, b.translation, t),
        lerp(a.rotation, b.rotation, t), // shortest‑path not handled here; use quaternion in 3D
        lerp(a.scale, b.scale, t)
    );
}

// ============================================================
// Look‑At transform (3D)
// ============================================================

template<typename T>
transform3d<T> look_at_transform(const vector3<T>& eye, const vector3<T>& center, const vector3<T>& up) noexcept {
    vector3<T> f = normalize(center - eye);
    vector3<T> s = normalize(cross(f, up));
    vector3<T> u = cross(s, f);
    matrix3<T> R;
    R(0,0)=s.x; R(0,1)=s.y; R(0,2)=s.z;
    R(1,0)=u.x; R(1,1)=u.y; R(1,2)=u.z;
    R(2,0)=-f.x;R(2,1)=-f.y;R(2,2)=-f.z;
    quaternion<T> rot(R);
    return transform3d<T>(eye, rot, vector3<T>(T(1)));
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_TRANSFORMS_H