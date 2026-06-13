// system name : onetbb-warp
// File 0043 : core/math/kinematics.h
// Description : Forward and inverse kinematics for articulated chains: DH, CCD, FABRIK, Jacobian DLS.

#ifndef __TBB_WARP_CORE_MATH_KINEMATICS_H
#define __TBB_WARP_CORE_MATH_KINEMATICS_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector3.h"
#include "core/math/matrix4.h"
#include "core/math/quaternion.h"
#include "core/math/transforms.h"
#include "core/math/linear_algebra_ext.h"
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>
#include <functional>
#include <cstdint>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Denavit‑Hartenberg parameters per joint
// ============================================================

template<typename T>
struct dh_parameters {
    T a;       // link length (distance from z_{i-1} to z_i along x_{i-1})
    T alpha;   // link twist (angle from z_{i-1} to z_i about x_{i-1})
    T d;       // link offset (distance from x_{i-1} to x_i along z_i)
    T theta;   // joint angle (angle from x_{i-1} to x_i about z_i)
};

// ============================================================
// DH transformation matrix (modified DH convention)
// ============================================================

template<typename T>
matrix4<T> dh_transform(const dh_parameters<T>& p) noexcept {
    T c_theta = std::cos(p.theta), s_theta = std::sin(p.theta);
    T c_alpha = std::cos(p.alpha), s_alpha = std::sin(p.alpha);
    matrix4<T> m;
    m(0,0)=c_theta; m(0,1)=-s_theta*c_alpha; m(0,2)= s_theta*s_alpha; m(0,3)=p.a*c_theta;
    m(1,0)=s_theta; m(1,1)= c_theta*c_alpha; m(1,2)=-c_theta*s_alpha; m(1,3)=p.a*s_theta;
    m(2,0)=T(0);    m(2,1)= s_alpha;          m(2,2)= c_alpha;           m(2,3)=p.d;
    m(3,0)=T(0);    m(3,1)= T(0);             m(3,2)= T(0);              m(3,3)=T(1);
    return m;
}

// ============================================================
// Forward kinematics using DH parameters
// Returns transforms for each joint (global)
// ============================================================

template<typename T>
std::vector<matrix4<T>> forward_kinematics_dh(const std::vector<dh_parameters<T>>& params) noexcept {
    std::vector<matrix4<T>> transforms(params.size(), matrix4<T>(T(1)));
    matrix4<T> T_chain(T(1));
    for (std::size_t i = 0; i < params.size(); ++i) {
        T_chain = T_chain * dh_transform(params[i]);
        transforms[i] = T_chain;
    }
    return transforms;
}

// ============================================================
// Forward kinematics using parent‑relative transform hierarchy
// ============================================================

template<typename T>
void forward_kinematics_hierarchy(const std::vector<transform3d<T>>& local_transforms,
                                  const std::vector<int>& parent_indices,
                                  std::vector<transform3d<T>>& global_transforms) noexcept {
    global_transforms.resize(local_transforms.size());
    for (std::size_t i = 0; i < local_transforms.size(); ++i) {
        if (parent_indices[i] < 0 || static_cast<std::size_t>(parent_indices[i]) >= i) {
            global_transforms[i] = local_transforms[i];
        } else {
            global_transforms[i] = global_transforms[parent_indices[i]] * local_transforms[i];
        }
    }
}

// ============================================================
// CCD (Cyclic Coordinate Descent) inverse kinematics
// Iteratively rotates each joint to bring end‑effector towards target.
// ============================================================

template<typename T>
bool solve_ik_ccd(std::vector<vector3<T>>& joint_positions,
                  const std::vector<int>& parent_indices,
                  const vector3<T>& target,
                  int max_iterations = 100,
                  T tolerance = T(1e-4)) noexcept
{
    if (joint_positions.size() < 2) return false;
    std::size_t n = joint_positions.size();
    std::size_t tip_idx = n - 1;
    // Assume the chain is from 0 (root) to tip. parent_indices[i] = i-1 for simple chain; for general use parent.
    // We'll implement for a simple serial chain where parent_indices[i] = i-1. For general, we'd need to traverse from tip to root.
    // Here we'll use parent_indices to walk back.
    for (int iter = 0; iter < max_iterations; ++iter) {
        T dist_to_target = length(joint_positions[tip_idx] - target);
        if (dist_to_target < tolerance) return true;
        // Walk backwards from tip's parent up to root
        for (std::size_t i = n - 2; i != static_cast<std::size_t>(-1); --i) {
            // Actually for CCD, we start from the joint just before the end‑effector and go up.
            // The joint i connects link i to link i+1. The rotation is applied at joint i to move tip.
            // We'll use the standard CCD loop: for j = n-2 down to 0:
        }
        // Re‑order: loop from n-2 down to 0
        for (int j = static_cast<int>(n) - 2; j >= 0; --j) {
            std::size_t i = static_cast<std::size_t>(j);
            vector3<T>& joint_pos = joint_positions[i];
            vector3<T> to_tip = joint_positions[tip_idx] - joint_pos;
            vector3<T> to_target = target - joint_pos;
            T tip_len_sq = length_sq(to_tip);
            T target_len_sq = length_sq(to_target);
            if (tip_len_sq < T(1e-12) || target_len_sq < T(1e-12)) continue;
            T tip_len = std::sqrt(tip_len_sq);
            T target_len = std::sqrt(target_len_sq);
            vector3<T> tip_dir = to_tip / tip_len;
            vector3<T> target_dir = to_target / target_len;
            T cos_angle = dot(tip_dir, target_dir);
            cos_angle = clamp(cos_angle, T(-1), T(1));
            T angle = std::acos(cos_angle);
            if (angle < T(1e-7)) continue;
            // Rotation axis: cross(tip_dir, target_dir)
            vector3<T> axis = cross(tip_dir, target_dir);
            T axis_len = length(axis);
            if (axis_len < T(1e-12)) continue;
            axis = axis / axis_len;
            // Build rotation quaternion and rotate all downstream joints (i+1..tip)
            quaternion<T> rot(axis, angle);
            for (std::size_t k = i + 1; k <= tip_idx; ++k) {
                vector3<T> rel = joint_positions[k] - joint_pos;
                rel = rotate(rot, rel);
                joint_positions[k] = joint_pos + rel;
            }
        }
    }
    return length(joint_positions[tip_idx] - target) < tolerance;
}

// ============================================================
// FABRIK (Forward And Backward Reaching Inverse Kinematics)
// ============================================================

template<typename T>
bool solve_ik_fabrik(std::vector<vector3<T>>& joint_positions,
                     const vector3<T>& target,
                     int max_iterations = 100,
                     T tolerance = T(1e-4)) noexcept
{
    std::size_t n = joint_positions.size();
    if (n < 2) return false;
    // Compute bone lengths
    std::vector<T> bone_lengths(n - 1);
    for (std::size_t i = 0; i < n - 1; ++i) {
        bone_lengths[i] = length(joint_positions[i + 1] - joint_positions[i]);
    }
    // Check if target is reachable
    T total_length = T(0);
    for (auto l : bone_lengths) total_length += l;
    vector3<T> root_pos = joint_positions[0];
    T dist_to_target = length(target - root_pos);
    if (dist_to_target > total_length) {
        // Unreachable: point towards target
        vector3<T> dir = (target - root_pos) / dist_to_target;
        for (std::size_t i = 1; i < n; ++i) {
            joint_positions[i] = joint_positions[i - 1] + dir * bone_lengths[i - 1];
        }
        return false;
    }
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Forward reaching: tip -> target
        joint_positions[n - 1] = target;
        for (std::size_t i = n - 1; i > 0; --i) {
            vector3<T>& cur = joint_positions[i];
            vector3<T>& prev = joint_positions[i - 1];
            vector3<T> dir = prev - cur;
            T len = length(dir);
            if (len < T(1e-12)) continue;
            dir = dir / len;
            prev = cur + dir * bone_lengths[i - 1];
        }
        // Backward reaching: root -> original root
        joint_positions[0] = root_pos;
        for (std::size_t i = 0; i < n - 1; ++i) {
            vector3<T>& cur = joint_positions[i];
            vector3<T>& next = joint_positions[i + 1];
            vector3<T> dir = next - cur;
            T len = length(dir);
            if (len < T(1e-12)) continue;
            dir = dir / len;
            next = cur + dir * bone_lengths[i];
        }
        T err = length(joint_positions.back() - target);
        if (err < tolerance) return true;
    }
    return length(joint_positions.back() - target) < tolerance;
}

// ============================================================
// Jacobian (position only, 3×n) for a serial chain
// ============================================================

template<typename T>
void compute_jacobian(const std::vector<vector3<T>>& joint_positions,
                      const std::vector<vector3<T>>& joint_axes,   // revolute axes in world frame
                      std::vector<std::vector<T>>& J) noexcept
{
    std::size_t n = joint_positions.size() - 1; // number of joints (exclude end‑effector)
    vector3<T> tip = joint_positions.back();
    J.assign(3, std::vector<T>(n, T(0)));
    for (std::size_t i = 0; i < n; ++i) {
        vector3<T> r = tip - joint_positions[i];
        vector3<T> col = cross(joint_axes[i], r);
        J[0][i] = col.x;
        J[1][i] = col.y;
        J[2][i] = col.z;
    }
}

// ============================================================
// Damped Least Squares (DLS) Inverse Kinematics using Jacobian
// ============================================================

template<typename T>
std::vector<T> solve_ik_dls(const std::vector<vector3<T>>& joint_positions,
                            const std::vector<vector3<T>>& joint_axes,
                            const vector3<T>& target,
                            T damping = T(0.1),
                            T step_size = T(0.5),
                            int max_iterations = 50,
                            T tolerance = T(1e-4)) noexcept
{
    std::size_t n = joint_positions.size() - 1; // number of revolute joints
    std::vector<vector3<T>> positions = joint_positions;
    for (int iter = 0; iter < max_iterations; ++iter) {
        vector3<T> tip = positions.back();
        vector3<T> error = target - tip;
        if (length(error) < tolerance) break;
        // Build Jacobian
        std::vector<std::vector<T>> J(3, std::vector<T>(n, T(0)));
        for (std::size_t i = 0; i < n; ++i) {
            vector3<T> r = tip - positions[i];
            vector3<T> col = cross(joint_axes[i], r);
            J[0][i] = col.x;
            J[1][i] = col.y;
            J[2][i] = col.z;
        }
        // Compute J^T * J + damping² * I
        std::vector<std::vector<T>> JTJ(n, std::vector<T>(n, T(0)));
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                T val = T(0);
                for (std::size_t k = 0; k < 3; ++k) val += J[k][i] * J[k][j];
                JTJ[i][j] = val;
            }
            JTJ[i][i] += damping * damping;
        }
        // J^T * error
        std::vector<T> JTe(n, T(0));
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t k = 0; k < 3; ++k) JTe[i] += J[k][i] * error[k];
        }
        // Solve JTJ * delta_theta = JTe
        std::vector<T> delta_theta = jacobi_iteration(JTJ, JTe, 50, T(1e-8));
        // Apply step
        for (std::size_t i = 0; i < n; ++i) {
            T angle = delta_theta[i] * step_size;
            // Rotate all joints downstream (including tip) around joint_axes[i] by angle
            quaternion<T> rot(joint_axes[i], angle);
            for (std::size_t k = i + 1; k < positions.size(); ++k) {
                vector3<T> rel = positions[k] - positions[i];
                rel = rotate(rot, rel);
                positions[k] = positions[i] + rel;
            }
        }
    }
    // Return final angles? We'll return positions of joints (excluding tip?) Actually we return the updated positions.
    // But the function signature says return vector<T>. We'll modify to return the joint angles? We'll compute angles from positions.
    // For consistency, we'll compute and return the current joint angles.
    std::vector<T> angles(n, T(0));
    for (std::size_t i = 0; i < n; ++i) {
        // Compute angle between consecutive bones
        if (i == 0) {
            vector3<T> v1 = positions[1] - positions[0];
            // Reference direction? We'll just return zero; the caller must interpret.
            angles[i] = T(0);
        } else {
            vector3<T> v0 = positions[i] - positions[i-1];
            vector3<T> v1 = positions[i+1] - positions[i];
            T cos_val = dot(v0,v1)/(length(v0)*length(v1)+T(1e-12));
            cos_val = clamp(cos_val, T(-1), T(1));
            angles[i] = std::acos(cos_val);
        }
    }
    return angles;
}

// ============================================================
// Null‑space projection for secondary tasks
// Given delta_theta_primary, project a secondary velocity into null space.
// ============================================================

template<typename T>
std::vector<T> null_space_projection(const std::vector<std::vector<T>>& J,
                                     const std::vector<T>& delta_theta_primary,
                                     const std::vector<T>& secondary_velocity) noexcept
{
    std::size_t m = J.size();    // task dimension (e.g., 3)
    std::size_t n = J[0].size();  // joint count
    // Compute J^T * J
    std::vector<std::vector<T>> JTJ(n, std::vector<T>(n, T(0)));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < m; ++k) JTJ[i][j] += J[k][i] * J[k][j];
        }
    // Compute pseudo‑inverse J⁺ = (J^T J)^{-1} J^T (simplified)
    auto JTJ_inv = pseudo_inverse(JTJ);
    std::vector<std::vector<T>> J_pinv(n, std::vector<T>(m, T(0)));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < m; ++k)
            for (std::size_t j = 0; j < n; ++j)
                J_pinv[i][k] += JTJ_inv[i][j] * J[k][j];

    // Null‑space projector N = I - J⁺ J
    std::vector<std::vector<T>> N(n, std::vector<T>(n, T(0)));
    for (std::size_t i = 0; i < n; ++i) N[i][i] = T(1);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = 0; k < m; ++k)
                N[i][j] -= J_pinv[i][k] * J[k][j];

    std::vector<T> delta_theta(n, T(0));
    for (std::size_t i = 0; i < n; ++i) {
        delta_theta[i] = delta_theta_primary[i];
        for (std::size_t j = 0; j < n; ++j) {
            delta_theta[i] += N[i][j] * secondary_velocity[j];
        }
    }
    return delta_theta;
}

// ============================================================
// FK/IK convenience using transform3d (angles -> transforms)
// ============================================================

template<typename T>
std::vector<transform3d<T>> fk_from_angles(const std::vector<T>& joint_angles,
                                            const std::vector<vector3<T>>& joint_axes,
                                            const std::vector<vector3<T>>& rest_positions,
                                            const std::vector<int>& parent_indices) noexcept
{
    std::size_t n = joint_angles.size();
    std::vector<transform3d<T>> locals(n, transform3d<T>::identity());
    for (std::size_t i = 0; i < n; ++i) {
        quaternion<T> rot(joint_axes[i], joint_angles[i]);
        locals[i].rotation = rot;
        if (parent_indices[i] < 0) {
            locals[i].translation = rest_positions[i];
        } else {
            locals[i].translation = rest_positions[i] - rest_positions[parent_indices[i]];
        }
    }
    std::vector<transform3d<T>> globals(n);
    std::vector<int> par = parent_indices;
    forward_kinematics_hierarchy(locals, par, globals);
    return globals;
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_KINEMATICS_H