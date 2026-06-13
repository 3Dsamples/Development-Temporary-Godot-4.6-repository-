// File 0032 : core/math/kinematics.h
// Forward, CCD, FABRIK, Jacobian inverse kinematics for articulated chains, with joint limits.

#pragma once

#include "vec3.h"
#include "quat.h"
#include "transform3.h"
#include "mat3.h"
#include "constants.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace wp {

template <typename T>
struct Joint {
    vec3<T>     position;              // world position (updated by FK/IK)
    quat<T>     orientation;           // world orientation
    vec3<T>     local_position;        // position relative to parent, in parent's frame
    quat<T>     local_rotation;        // rotation relative to parent
    int32       parent = -1;           // index of parent joint (-1 = root)
    bool        is_end_effector = false;

    constexpr Joint() noexcept : position(T(0)), orientation(T(0),T(0),T(0),T(1)),
                                  local_position(T(0)), local_rotation(T(0),T(0),T(0),T(1)) {}
};

template <typename T>
using JointChain = std::vector<Joint<T>>;

// ── Forward Kinematics ──────────────────────────────────────────────
template <typename T>
void forward_kinematics(JointChain<T>& chain) {
    for (size_t i = 0; i < chain.size(); ++i) {
        if (chain[i].parent < 0) {
            chain[i].orientation = chain[i].local_rotation;
            chain[i].position    = chain[i].local_position;
        } else {
            const auto& p = chain[chain[i].parent];
            chain[i].orientation = mul(p.orientation, chain[i].local_rotation);
            chain[i].position    = p.position + rotate(p.orientation, chain[i].local_position);
        }
    }
}

// ── Helper: recompute FK from a given joint index downward ──────────
template <typename T>
void partial_fk(JointChain<T>& chain, int32 start_idx) {
    for (size_t i = static_cast<size_t>(start_idx); i < chain.size(); ++i) {
        if (chain[i].parent < 0) continue;
        // only recompute if this joint is a descendant of start_idx
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

// ── CCD Inverse Kinematics ──────────────────────────────────────────
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

        // Iterate from parent of effector up to root
        int32 current = chain[effector_idx].parent;
        while (current >= 0) {
            vec3<T> joint_pos = chain[current].position;
            vec3<T> to_effector = normalize(effector_pos - joint_pos);
            vec3<T> to_target   = normalize(target - joint_pos);

            T cos_theta = dot(to_effector, to_target);
            if (cos_theta < T(1) - epsilon<T>) {
                vec3<T> rot_axis = normalize(cross(to_effector, to_target));
                T angle = std::acos(clamp(cos_theta, T(-1), T(1)));
                quat<T> delta_rot = from_axis_angle(rot_axis, angle);

                chain[current].local_rotation = normalize(mul(delta_rot, chain[current].local_rotation));

                partial_fk(chain, current);
                effector_pos = chain[effector_idx].position;
            }
            current = chain[current].parent;
        }
    }

    res.iterations = max_iter;
    res.final_error = length(chain[effector_idx].position - target);
    return res;
}

// ── FABRIK Inverse Kinematics ───────────────────────────────────────
// Operates on world-space positions; distances array stores the rest lengths
// between consecutive joints along the chain.
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
    if (static_cast<size_t>(n - 1) > distances.size()) return res;

    vec3<T> root_pos = positions[root_idx];

    for (int iter = 0; iter < max_iter; ++iter) {
        // Forward reaching: end effector to root
        positions[effector_idx] = target;
        for (int32 i = effector_idx - 1; i >= root_idx; --i) {
            vec3<T> dir = positions[i] - positions[i + 1];
            T len = length(dir);
            if (len < tolerance) { dir = vec3<T>(T(0), T(1), T(0)); len = tolerance; }
            dir = dir / len;
            positions[i] = positions[i + 1] + dir * distances[i - root_idx];
        }

        // Backward reaching: root to end effector
        positions[root_idx] = root_pos;
        for (int32 i = root_idx; i < effector_idx; ++i) {
            vec3<T> dir = positions[i + 1] - positions[i];
            T len = length(dir);
            if (len < tolerance) { dir = vec3<T>(T(0), T(1), T(0)); len = tolerance; }
            dir = dir / len;
            positions[i + 1] = positions[i] + dir * distances[i - root_idx];
        }

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

// ── Extract FABRIK data from a JointChain ──────────────────────────
template <typename T>
void extract_fabrik_data(const JointChain<T>& chain,
                         std::vector<vec3<T>>& positions,
                         std::vector<T>& distances) {
    positions.clear();
    distances.clear();
    for (const auto& j : chain)
        positions.push_back(j.position);
    for (size_t i = 1; i < chain.size(); ++i) {
        if (chain[i].parent >= 0) {
            T d = length(chain[i].position - chain[chain[i].parent].position);
            distances.push_back(d);
        }
    }
}

// ── Apply FABRIK result back to joint chain, recovering orientations ──
template <typename T>
void apply_fabrik_result(JointChain<T>& chain,
                         const std::vector<vec3<T>>& new_positions,
                         const vec3<T>& up_direction = vec3<T>(T(0), T(1), T(0))) {
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

        // Recover orientation from the direction to the next joint (or keep previous if terminal)
        if (i + 1 < chain.size() && chain[i + 1].parent == static_cast<int32>(i)) {
            vec3<T> forward = normalize(chain[i + 1].position - chain[i].position);
            // Build rotation that maps (0,0,1) to forward, keeping up as reference
            quat<T> target_rot = look_at_quat(forward, up_direction);
            if (chain[i].parent >= 0) {
                const auto& p = chain[chain[i].parent];
                quat<T> inv_parent_rot = conjugate(p.orientation);
                chain[i].local_rotation = mul(inv_parent_rot, target_rot);
            } else {
                chain[i].local_rotation = target_rot;
            }
            chain[i].orientation = target_rot; // world orientation, will be updated by FK if needed
        }
    }
    // Final FK pass to ensure consistency
    forward_kinematics(chain);
}

// ── Jacobian-based Inverse Kinematics (pseudo‑inverse with damping) ──
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

        // Build Jacobian columns (3 x n_joints)
        std::vector<vec3<T>> Jcols(n_joints);
        for (int32 i = 0; i < n_joints; ++i) {
            if (i == 0 && chain[i].parent < 0) {
                // root with no parent – we can treat it as a floating base? Ignore for simplicity
                Jcols[i] = vec3<T>(T(0));
                continue;
            }
            vec3<T> world_axis = rotate(chain[i].orientation, joint_axes[i]);
            vec3<T> r = e_pos - chain[i].position;
            Jcols[i] = cross(world_axis, r);
        }

        // Compute JJ^T + damping*I (3x3)
        mat3<T> JJt(T(0));
        for (int32 i = 0; i < n_joints; ++i) {
            const vec3<T>& col = Jcols[i];
            JJt(0,0) += col.x * col.x;
            JJt(0,1) += col.x * col.y;
            JJt(0,2) += col.x * col.z;
            JJt(1,0) += col.y * col.x;
            JJt(1,1) += col.y * col.y;
            JJt(1,2) += col.y * col.z;
            JJt(2,0) += col.z * col.x;
            JJt(2,1) += col.z * col.y;
            JJt(2,2) += col.z * col.z;
        }
        JJt(0,0) += damping;
        JJt(1,1) += damping;
        JJt(2,2) += damping;

        // Solve (JJ^T + damp*I) * lambda = error
        mat3<T> invJJt = inverse(JJt);
        vec3<T> lambda = mul(invJJt, error);

        // Compute angle changes and apply
        for (int32 i = 0; i < n_joints; ++i) {
            T delta_angle = dot(Jcols[i], lambda);
            // Clamp to avoid large jumps
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
// Clamps each joint's local rotation angle around the given axis to [min, max].
template <typename T>
void enforce_joint_limits(JointChain<T>& chain,
                          const std::vector<T>& min_angles,
                          const std::vector<T>& max_angles,
                          const std::vector<vec3<T>>& axes) {
    for (size_t i = 0; i < chain.size() && i < min_angles.size() &&
                         i < max_angles.size() && i < axes.size(); ++i) {
        vec3<T> current_axis;
        T current_angle;
        to_axis_angle(chain[i].local_rotation, current_axis, current_angle);

        // Project onto the limit axis
        T proj = std::abs(dot(current_axis, axes[i]));
        if (proj < T(0.99)) continue; // rotation is not primarily around this axis

        T sign = dot(current_axis, axes[i]) > T(0) ? T(1) : T(-1);
        T angle = current_angle * sign;

        if (angle < min_angles[i])
            chain[i].local_rotation = from_axis_angle(axes[i], min_angles[i]);
        else if (angle > max_angles[i])
            chain[i].local_rotation = from_axis_angle(axes[i], max_angles[i]);
    }
}

} // namespace wp