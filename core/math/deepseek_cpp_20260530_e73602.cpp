// File 0032 : core/math/kinematics.h
// Forward and inverse kinematics for articulated chains using CCD, FABRIK, and Jacobian methods.

#pragma once

#include "vec3.h"
#include "quat.h"
#include "transform3.h"
#include "mat4.h"
#include "constants.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace wp {

// ── Skeletal Chain ──────────────────────────────────────────────────
// A joint chain is defined by a list of local poses relative to a parent.
// The root is at world origin if its local pose is identity, or can be placed arbitrarily.
// For IK we operate directly on world-space joint positions and optionally orientations.

template <typename T>
struct Joint {
    vec3<T>     position;        // world position (updated during FK/IK)
    quat<T>     orientation;     // world orientation
    vec3<T>     local_position;  // position relative to parent, in parent's frame
    quat<T>     local_rotation;  // rotation relative to parent
    int32       parent = -1;     // index of parent joint (-1 = root)
    bool        is_end_effector = false;

    constexpr Joint() noexcept : position(T(0)), orientation(T(0),T(0),T(0),T(1)), local_position(T(0)), local_rotation(T(0),T(0),T(0),T(1)) {}
};

template <typename T>
using JointChain = std::vector<Joint<T>>;

// ── Forward Kinematics ──────────────────────────────────────────────
// Compute world positions and orientations for all joints given local transforms.
template <typename T>
void forward_kinematics(JointChain<T>& chain) {
    for (size_t i = 0; i < chain.size(); ++i) {
        if (chain[i].parent < 0) {
            // root
            chain[i].orientation = chain[i].local_rotation;
            chain[i].position    = chain[i].local_position;
        } else {
            const auto& p = chain[chain[i].parent];
            chain[i].orientation = mul(p.orientation, chain[i].local_rotation);
            chain[i].position    = p.position + rotate(p.orientation, chain[i].local_position);
        }
    }
}

// ── CCD (Cyclic Coordinate Descent) Inverse Kinematics ──────────────
// Iteratively rotates each joint so that the end effector moves toward the target.
// `chain` must have FK already computed. Only joints from root to the end effector parent are modified.
// `effector_index` is the index of the end effector joint (usually the last).
// Returns the number of iterations performed (up to max_iter) and the final error distance.

template <typename T>
struct CCDResult {
    int     iterations = 0;
    T       final_error = T(0);
    bool    converged = false;
};

template <typename T>
CCDResult<T> solve_ik_ccd(JointChain<T>& chain, int32 effector_idx,
                          const vec3<T>& target, int max_iter = 20,
                          T tolerance = epsilon<T>) {
    CCDResult<T> res;
    if (chain.empty() || effector_idx < 0 || effector_idx >= static_cast<int32>(chain.size()))
        return res;

    // Ensure FK is up-to-date
    forward_kinematics(chain);

    for (int iter = 0; iter < max_iter; ++iter) {
        vec3<T> effector_pos = chain[effector_idx].position;
        T error = length(effector_pos - target);
        if (error <= tolerance) {
            res.converged = true;
            res.iterations = iter;
            res.final_error = error;
            return res;
        }

        // Iterate from the parent of the end effector up to the root
        int32 current = chain[effector_idx].parent;
        while (current >= 0) {
            vec3<T> joint_pos = chain[current].position;
            // Direction from joint to effector
            vec3<T> to_effector = normalize(effector_pos - joint_pos);
            // Direction from joint to target
            vec3<T> to_target   = normalize(target - joint_pos);

            // Quaternion that rotates to_effector into to_target
            T cos_theta = dot(to_effector, to_target);
            if (cos_theta < T(1) - epsilon<T>) {
                vec3<T> rot_axis = normalize(cross(to_effector, to_target));
                T angle = std::acos(clamp(cos_theta, T(-1), T(1)));
                quat<T> delta_rot = from_axis_angle(rot_axis, angle);

                // Apply to the joint's local rotation
                chain[current].local_rotation = normalize(mul(delta_rot, chain[current].local_rotation));

                // Recompute FK from this joint downward
                partial_fk(chain, current);
                effector_pos = chain[effector_idx].position;
            }

            // Move to the parent of this joint
            current = chain[current].parent;
        }
    }

    res.iterations = max_iter;
    res.final_error = length(chain[effector_idx].position - target);
    return res;
}

// ── Partial FK (recompute from a given joint downward) ──────────────
template <typename T>
void partial_fk(JointChain<T>& chain, int32 start_idx) {
    for (size_t i = start_idx; i < chain.size(); ++i) {
        if (chain[i].parent < 0) continue;
        // Only recompute if this joint is a descendant of start_idx
        bool is_descendant = false;
        int32 p = static_cast<int32>(i);
        while (p >= 0) {
            if (p == start_idx) { is_descendant = true; break; }
            p = chain[p].parent;
        }
        if (!is_descendant) continue;
        const auto& parent = chain[chain[i].parent];
        chain[i].orientation = mul(parent.orientation, chain[i].local_rotation);
        chain[i].position    = parent.position + rotate(parent.orientation, chain[i].local_position);
    }
}

// ── FABRIK (Forward And Backward Reaching Inverse Kinematics) ───────
// Operates on world-space joint positions directly. Better for long chains.
// `positions` are the world positions of all joints (including root and effector).
// `distances` are the rest distances between consecutive joints (size = n-1).
// `root_idx` and `effector_idx` define the sub-chain to operate on.
// Modifies `positions` in place. Returns the number of iterations and final error.

template <typename T>
struct FABRIKResult {
    int     iterations = 0;
    T       final_error = T(0);
    bool    converged = false;
};

template <typename T>
FABRIKResult<T> solve_ik_fabrik(std::vector<vec3<T>>& positions,
                                const std::vector<T>& distances,
                                int32 root_idx, int32 effector_idx,
                                const vec3<T>& target, int max_iter = 20,
                                T tolerance = epsilon<T>) {
    FABRIKResult<T> res;
    if (positions.empty() || distances.empty()) return res;
    int32 n = effector_idx - root_idx + 1;
    if (n < 2) return res;
    if (distances.size() < static_cast<size_t>(n - 1)) return res;

    // Store the root position (it stays fixed)
    vec3<T> root_pos = positions[root_idx];

    for (int iter = 0; iter < max_iter; ++iter) {
        // ── Forward pass: effector to root ──
        positions[effector_idx] = target;
        for (int32 i = effector_idx - 1; i >= root_idx; --i) {
            vec3<T> dir = positions[i] - positions[i + 1];
            T len = length(dir);
            if (len < tolerance) {
                dir = vec3<T>(T(0), T(1), T(0));
                len = tolerance;
            }
            dir = dir / len;
            positions[i] = positions[i + 1] + dir * distances[i - root_idx];
        }

        // ── Backward pass: root to effector ──
        positions[root_idx] = root_pos;
        for (int32 i = root_idx; i < effector_idx; ++i) {
            vec3<T> dir = positions[i + 1] - positions[i];
            T len = length(dir);
            if (len < tolerance) {
                dir = vec3<T>(T(0), T(1), T(0));
                len = tolerance;
            }
            dir = dir / len;
            positions[i + 1] = positions[i] + dir * distances[i - root_idx];
        }

        // Check convergence
        T error = length(positions[effector_idx] - target);
        if (error <= tolerance) {
            res.converged = true;
            res.iterations = iter + 1;
            res.final_error = error;
            return res;
        }
    }

    res.iterations = max_iter;
    res.final_error = length(positions[effector_idx] - target);
    return res;
}

// ── Build FABRIK data from a joint chain ───────────────────────────
template <typename T>
void extract_fabrik_data(const JointChain<T>& chain,
                         std::vector<vec3<T>>& positions,
                         std::vector<T>& distances) {
    positions.clear();
    distances.clear();
    for (const auto& j : chain) {
        positions.push_back(j.position);
    }
    for (size_t i = 1; i < chain.size(); ++i) {
        if (chain[i].parent >= 0) {
            distances.push_back(length(chain[i].position - chain[chain[i].parent].position));
        }
    }
}

// ── Apply FABRIK result back to joint chain ────────────────────────
template <typename T>
void apply_fabrik_result(JointChain<T>& chain,
                         const std::vector<vec3<T>>& new_positions) {
    for (size_t i = 0; i < chain.size() && i < new_positions.size(); ++i) {
        vec3<T> old_pos = chain[i].position;
        chain[i].position = new_positions[i];

        // Recompute local position relative to parent
        if (chain[i].parent >= 0) {
            const auto& p = chain[chain[i].parent];
            quat<T> inv_parent_rot = conjugate(p.orientation);
            chain[i].local_position = rotate(inv_parent_rot, new_positions[i] - p.position);
        } else {
            chain[i].local_position = new_positions[i];
        }

        // Orientation is harder to recover from positions alone; we leave orientation unchanged.
        // A full implementation would use a pole vector or maintain the orientation plane.
    }
}

// ── Jacobian IK (1‑level, using pseudo‑inverse) ────────────────────
// For a chain of n joints (0..n-1, 0=root, n-1=effector), compute the 3×m Jacobian
// where m is the number of revolute joints (DOFs). This implementation assumes
// each joint has one revolute axis (local axis), and the effector position is the only goal.

template <typename T>
struct JacobianIKResult {
    int     iterations = 0;
    T       final_error = T(0);
    bool    converged = false;
};

template <typename T>
JacobianIKResult<T> solve_ik_jacobian(JointChain<T>& chain, int32 effector_idx,
                                       const vec3<T>& target,
                                       const std::vector<vec3<T>>& joint_axes,   // local rotation axis per joint
                                       int max_iter = 20, T damping = T(0.01),
                                       T tolerance = epsilon<T>) {
    JacobianIKResult<T> res;
    int32 n_joints = static_cast<int32>(chain.size());
    if (n_joints < 2) return res;
    if (joint_axes.size() < static_cast<size_t>(n_joints)) return res;

    forward_kinematics(chain);

    for (int iter = 0; iter < max_iter; ++iter) {
        vec3<T> e_pos = chain[effector_idx].position;
        vec3<T> error = target - e_pos;
        T err_len = length(error);
        if (err_len <= tolerance) {
            res.converged = true;
            res.iterations = iter;
            res.final_error = err_len;
            return res;
        }

        // Build Jacobian (3 rows, n_joints columns) – each column is the linear velocity
        // contribution of joint i to the effector: axis_i × (effector - joint_i)
        std::vector<vec3<T>> jacobian_cols(n_joints);
        for (int32 i = 0; i < n_joints; ++i) {
            if (chain[i].parent < 0 && i > 0) continue; // skip root if not the base
            vec3<T> world_axis = rotate(chain[i].orientation, joint_axes[i]);
            vec3<T> r = e_pos - chain[i].position;
            jacobian_cols[i] = cross(world_axis, r);
        }

        // Compute J * J^T (3×3 symmetric matrix) + damping * I
        mat3<T> JJt(T(0));
        for (int32 i = 0; i < n_joints; ++i) {
            const vec3<T>& col = jacobian_cols[i];
            JJt(0,0) += col.x * col.x; JJt(0,1) += col.x * col.y; JJt(0,2) += col.x * col.z;
            JJt(1,0) += col.y * col.x; JJt(1,1) += col.y * col.y; JJt(1,2) += col.y * col.z;
            JJt(2,0) += col.z * col.x; JJt(2,1) += col.z * col.y; JJt(2,2) += col.z * col.z;
        }
        JJt(0,0) += damping; JJt(1,1) += damping; JJt(2,2) += damping;

        // Solve (J*J^T + damp*I) * lambda = error
        mat3<T> JJt_inv = inverse(JJt);
        vec3<T> lambda = mul(JJt_inv, error);

        // Compute angle change per joint: delta_angle = J^T * lambda
        for (int32 i = 0; i < n_joints; ++i) {
            T delta_angle = dot(jacobian_cols[i], lambda);
            // Clamp to reasonable step
            delta_angle = clamp(delta_angle, -pi<T> / T(4), pi<T> / T(4));
            vec3<T> world_axis = rotate(chain[i].orientation, joint_axes[i]);
            quat<T> delta_rot = from_axis_angle(world_axis, delta_angle);
            chain[i].local_rotation = normalize(mul(delta_rot, chain[i].local_rotation));
        }

        forward_kinematics(chain);
    }

    res.iterations = max_iter;
    res.final_error = length(chain[effector_idx].position - target);
    return res;
}

// ── Joint limit enforcement ─────────────────────────────────────────
template <typename T>
void enforce_joint_limits(JointChain<T>& chain,
                          const std::vector<T>& min_angles,
                          const std::vector<T>& max_angles,
                          const std::vector<vec3<T>>& axes) {
    for (size_t i = 0; i < chain.size() && i < min_angles.size() &&
                         i < max_angles.size() && i < axes.size(); ++i) {
        // Convert local rotation to angle around the given axis
        vec3<T> current_axis;
        T current_angle;
        to_axis_angle(chain[i].local_rotation, current_axis, current_angle);
        // Project onto the joint axis
        T proj = std::abs(dot(current_axis, axes[i]));
        if (proj < T(0.99)) continue; // rotation not around this axis

        T sign = dot(current_axis, axes[i]) > T(0) ? T(1) : T(-1);
        T angle = current_angle * sign;

        if (angle < min_angles[i]) {
            chain[i].local_rotation = from_axis_angle(axes[i], min_angles[i]);
        } else if (angle > max_angles[i]) {
            chain[i].local_rotation = from_axis_angle(axes[i], max_angles[i]);
        }
    }
}

} // namespace wp