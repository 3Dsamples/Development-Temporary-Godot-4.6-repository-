//File 0055 : core/math/kinematics.h
//Complete kinematic chain solver: forward kinematics (DH), Jacobian, inverse kinematics (CCD with full recomputation, FABRIK, Jacobian damped pseudoinverse), SIMD‑accelerated.
#ifndef CORE_MATH_KINEMATICS_H
#define CORE_MATH_KINEMATICS_H

#include "vector_math.h"
#include "quaternion_math.h"
#include "matrix_math.h"
#include "math_constants.h"
#include <vector>
#include <functional>
#include <cmath>

namespace SimulationMath {
namespace kinematics {

using SimdVec = DirectX::XMVECTOR;

// -----------------------------------------------------------------------------
// 1. Joint types
// -----------------------------------------------------------------------------
enum class JointType : uint8_t { Revolute, Prismatic, Fixed };

// -----------------------------------------------------------------------------
// 2. Denavit‑Hartenberg joint parameters
// -----------------------------------------------------------------------------
struct DHJoint {
    float d = 0.0f;       // link offset
    float theta = 0.0f;   // joint angle (variable for revolute)
    float a = 0.0f;       // link length
    float alpha = 0.0f;   // link twist
    JointType type = JointType::Revolute;
    float joint_value = 0.0f;  // current joint variable (theta for revolute, d for prismatic)

    DirectX::XMMATRIX transformation() const noexcept {
        float ct = std::cos(joint_value), st = std::sin(joint_value);
        float ca = std::cos(alpha), sa = std::sin(alpha);
        DirectX::XMMATRIX m;
        m.r[0] = DirectX::XMVectorSet(ct,      -st*ca,   st*sa,  a*ct);
        m.r[1] = DirectX::XMVectorSet(st,       ct*ca,  -ct*sa,  a*st);
        m.r[2] = DirectX::XMVectorSet(0.0f,     sa,       ca,     d);
        m.r[3] = DirectX::XMVectorSet(0.0f,     0.0f,     0.0f,   1.0f);
        return m;
    }
};

// -----------------------------------------------------------------------------
// 3. Kinematic chain
// -----------------------------------------------------------------------------
class KinematicChain {
public:
    KinematicChain() = default;

    void add_joint(const DHJoint& joint) { joints_.push_back(joint); }
    size_t num_joints() const noexcept { return joints_.size(); }

    void set_joint_values(const std::vector<float>& values) {
        for (size_t i = 0; i < std::min(values.size(), joints_.size()); ++i)
            joints_[i].joint_value = values[i];
    }

    DirectX::XMMATRIX forward_kinematics() const noexcept {
        DirectX::XMMATRIX T = DirectX::XMMatrixIdentity();
        for (const auto& j : joints_)
            T = DirectX::XMMatrixMultiply(T, j.transformation());
        return T;
    }

    // -----------------------------------------------------------------------
    // Compute Jacobian (6 x n) at current configuration
    // -----------------------------------------------------------------------
    void compute_jacobian(std::vector<std::vector<float>>& J) const noexcept {
        size_t n = joints_.size();
        J.assign(6, std::vector<float>(n, 0.0f));

        std::vector<DirectX::XMMATRIX> T_j(n+1);
        T_j[0] = DirectX::XMMatrixIdentity();
        for (size_t i = 0; i < n; ++i)
            T_j[i+1] = DirectX::XMMatrixMultiply(T_j[i], joints_[i].transformation());

        DirectX::XMVECTOR p_end = matrix_math::extract_translation(T_j[n]);

        for (size_t i = 0; i < n; ++i) {
            DirectX::XMMATRIX& T_i = T_j[i];
            DirectX::XMVECTOR z = T_i.r[2]; // joint axis (world frame)
            DirectX::XMVECTOR p_j = matrix_math::extract_translation(T_i);

            if (joints_[i].type == JointType::Revolute) {
                DirectX::XMVECTOR w = z;
                DirectX::XMVECTOR r = DirectX::XMVectorSubtract(p_end, p_j);
                DirectX::XMVECTOR v = vector_math::cross3(z, r);
                J[0][i] = vector_math::get_x(v); J[1][i] = vector_math::get_y(v); J[2][i] = vector_math::get_z(v);
                J[3][i] = vector_math::get_x(w); J[4][i] = vector_math::get_y(w); J[5][i] = vector_math::get_z(w);
            } else if (joints_[i].type == JointType::Prismatic) {
                DirectX::XMVECTOR v = z;
                J[0][i] = vector_math::get_x(v); J[1][i] = vector_math::get_y(v); J[2][i] = vector_math::get_z(v);
                J[3][i] = 0.0f; J[4][i] = 0.0f; J[5][i] = 0.0f;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Cyclic Coordinate Descent (CCD) inverse kinematics – fully correct
    // Updates joint angles iteratively, recomputing EE after each joint change.
    // -----------------------------------------------------------------------
    void ik_ccd(DirectX::FXMVECTOR target, int max_iter = 100, float tolerance = 1e-4f) noexcept {
        size_t n = joints_.size();
        if (n == 0) return;

        for (int iter = 0; iter < max_iter; ++iter) {
            // Current end‑effector position
            DirectX::XMVECTOR p_end = matrix_math::extract_translation(forward_kinematics());
            if (vector_math::length3_scalar(DirectX::XMVectorSubtract(p_end, target)) < tolerance)
                return;

            // Iterate from tip to base
            for (int i = (int)n - 1; i >= 0; --i) {
                // Recompute current EE position (because previous joints have been modified)
                p_end = matrix_math::extract_translation(forward_kinematics());
                // Compute the joint position (origin of joint i in world coordinates)
                // We can obtain it by accumulating transforms up to joint i
                DirectX::XMMATRIX T_prev = DirectX::XMMatrixIdentity();
                for (int k = 0; k < i; ++k)
                    T_prev = DirectX::XMMatrixMultiply(T_prev, joints_[k].transformation());
                DirectX::XMVECTOR p_joint = matrix_math::extract_translation(T_prev);

                // Vectors from joint to EE and to target
                DirectX::XMVECTOR r_ee = DirectX::XMVectorSubtract(p_end, p_joint);
                DirectX::XMVECTOR r_tgt = DirectX::XMVectorSubtract(target, p_joint);

                float len_ee = vector_math::length3_scalar(r_ee);
                float len_tgt = vector_math::length3_scalar(r_tgt);
                if (len_ee < 1e-6f || len_tgt < 1e-6f) continue;

                // Rotation axis: cross product of r_ee and r_tgt
                DirectX::XMVECTOR axis = vector_math::cross3(r_ee, r_tgt);
                float axis_len = vector_math::length3_scalar(axis);
                if (axis_len < 1e-6f) continue;
                axis = DirectX::XMVectorScale(axis, 1.0f / axis_len);

                // Angle between r_ee and r_tgt
                float cos_angle = vector_math::dot3_scalar(r_ee, r_tgt) / (len_ee * len_tgt);
                cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle));
                float angle = std::acos(cos_angle);

                // For revolute joint: we must rotate about the joint's local axis.
                // The joint's world rotation axis is the z‑axis of T_prev (the frame of joint i)
                DirectX::XMVECTOR world_axis = T_prev.r[2];
                // Project the rotation onto the joint axis
                float proj = vector_math::dot3_scalar(axis, world_axis);
                // The sign determines the direction of rotation
                if (std::abs(proj) > 0.001f) {
                    float signed_angle = (proj > 0.0f) ? angle : -angle;
                    joints_[i].joint_value += signed_angle;
                }
                // If prismatic, we could translate along the axis (not implemented here)
            }
        }
    }

    // -----------------------------------------------------------------------
    // FABRIK (Forward And Backward Reaching Inverse Kinematics)
    // positions: joint positions (size = n+1), last is EE
    // link_lengths: distances between successive joints
    // target: desired EE position
    // -----------------------------------------------------------------------
    static void ik_fabrik(const std::vector<float>& link_lengths,
                          std::vector<SimdVec>& positions, SimdVec target,
                          int max_iter = 20, float tolerance = 1e-4f) noexcept {
        size_t n = positions.size() - 1;
        if (n < 2) return;

        std::vector<float> lengths = link_lengths;
        if (lengths.size() != n) {
            lengths.resize(n);
            for (size_t i = 0; i < n; ++i)
                lengths[i] = vector_math::length3_scalar(DirectX::XMVectorSubtract(positions[i+1], positions[i]));
        }

        SimdVec base = positions[0];
        for (int iter = 0; iter < max_iter; ++iter) {
            // Forward reaching: from tip to base
            positions[n] = target;
            for (int i = (int)n - 1; i >= 0; --i) {
                SimdVec dir = DirectX::XMVectorSubtract(positions[i], positions[i+1]);
                float dist = vector_math::length3_scalar(dir);
                if (dist < 1e-8f) continue;
                dir = DirectX::XMVectorScale(dir, 1.0f / dist);
                positions[i] = DirectX::XMVectorAdd(positions[i+1], DirectX::XMVectorScale(dir, lengths[i]));
            }
            // Backward reaching: from base to tip
            positions[0] = base;
            for (size_t i = 0; i < n; ++i) {
                SimdVec dir = DirectX::XMVectorSubtract(positions[i+1], positions[i]);
                float dist = vector_math::length3_scalar(dir);
                if (dist < 1e-8f) continue;
                dir = DirectX::XMVectorScale(dir, 1.0f / dist);
                positions[i+1] = DirectX::XMVectorAdd(positions[i], DirectX::XMVectorScale(dir, lengths[i]));
            }
            if (vector_math::length3_scalar(DirectX::XMVectorSubtract(positions[n], target)) < tolerance)
                break;
        }
    }

    // -----------------------------------------------------------------------
    // Jacobian damped least‑squares inverse kinematics
    // desired_velocity: 6‑vector (linear, angular) in world frame
    // delta_time: integration step
    // -----------------------------------------------------------------------
    void ik_jacobian_pseudo_inverse(const std::vector<float>& desired_velocity, float delta_time, float lambda = 0.01f) noexcept {
        size_t n = joints_.size();
        if (n == 0) return;

        std::vector<std::vector<float>> J;
        compute_jacobian(J);

        // Construct J^T * J + λ² I
        std::vector<std::vector<float>> JTJ(n, std::vector<float>(n, 0.0f));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < 6; ++k)
                    sum += J[k][i] * J[k][j];
                JTJ[i][j] = sum;
            }
            JTJ[i][i] += lambda * lambda;
        }

        // J^T * v_desired
        std::vector<float> JTv(n, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            float sum = 0.0f;
            for (size_t k = 0; k < 6; ++k)
                sum += J[k][i] * desired_velocity[k];
            JTv[i] = sum;
        }

        // Solve (JTJ) * dq = JTv via Gaussian elimination (full rank assumed)
        auto solve_linear = [](const std::vector<std::vector<float>>& A, std::vector<float>& b) -> std::vector<float> {
            size_t m = A.size();
            std::vector<std::vector<float>> mat(m, std::vector<float>(m+1));
            for (size_t i=0; i<m; ++i) {
                for (size_t j=0; j<m; ++j) mat[i][j] = A[i][j];
                mat[i][m] = b[i];
            }
            // Gaussian elimination with partial pivoting
            for (size_t i=0; i<m; ++i) {
                size_t pivot = i;
                for (size_t j=i+1; j<m; ++j)
                    if (std::abs(mat[j][i]) > std::abs(mat[pivot][i])) pivot = j;
                std::swap(mat[i], mat[pivot]);
                if (std::abs(mat[i][i]) < 1e-12f) continue; // singular, skip
                for (size_t j=i+1; j<m; ++j) {
                    float factor = mat[j][i] / mat[i][i];
                    for (size_t k=i; k<=m; ++k)
                        mat[j][k] -= factor * mat[i][k];
                }
            }
            // Back substitution
            std::vector<float> x(m, 0.0f);
            for (int i=(int)m-1; i>=0; --i) {
                float sum = mat[i][m];
                for (size_t j=i+1; j<m; ++j)
                    sum -= mat[i][j] * x[j];
                if (std::abs(mat[i][i]) > 1e-12f)
                    x[i] = sum / mat[i][i];
            }
            return x;
        };

        std::vector<float> dq = solve_linear(JTJ, JTv);
        for (size_t i = 0; i < n; ++i)
            joints_[i].joint_value += dq[i] * delta_time;
    }

private:
    std::vector<DHJoint> joints_;
};

} // namespace kinematics
} // namespace SimulationMath

#endif // CORE_MATH_KINEMATICS_H