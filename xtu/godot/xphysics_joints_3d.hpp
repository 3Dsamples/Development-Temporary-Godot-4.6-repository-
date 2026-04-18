// include/xtu/godot/xphysics_joints_3d.hpp
// xtensor-unified - Advanced 3D Physics Joints for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XPHYSICS_JOINTS_3D_HPP
#define XTU_GODOT_XPHYSICS_JOINTS_3D_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xnode.hpp"
#include "xtu/godot/xphysics3d.hpp"
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/graphics/xtransform.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace physics {

// #############################################################################
// Forward declarations
// #############################################################################
class Generic6DOFJoint3D;
class ConeTwistJoint3D;
class SliderJoint3D;
class HingeJoint3D;
class JointSolver3D;

// #############################################################################
// Joint solver parameters
// #############################################################################
struct JointSolverParams {
    float erp = 0.8f;           // Error reduction parameter
    float cfm = 0.0001f;        // Constraint force mixing
    int iterations = 8;         // Solver iterations
    float damping = 0.0f;       // Joint damping
    float restitution = 0.0f;   // Bounciness at limits
    float max_force = std::numeric_limits<float>::max();
};

// #############################################################################
// Generic6DOFJoint3D - Six degrees of freedom constraint
// #############################################################################
class Generic6DOFJoint3D : public Joint3D {
    XTU_GODOT_REGISTER_CLASS(Generic6DOFJoint3D, Joint3D)

public:
    enum Param {
        PARAM_LINEAR_LOWER_LIMIT,
        PARAM_LINEAR_UPPER_LIMIT,
        PARAM_LINEAR_LIMIT_SOFTNESS,
        PARAM_LINEAR_RESTITUTION,
        PARAM_LINEAR_DAMPING,
        PARAM_LINEAR_MOTOR_TARGET_VELOCITY,
        PARAM_LINEAR_MOTOR_FORCE_LIMIT,
        PARAM_LINEAR_SPRING_STIFFNESS,
        PARAM_LINEAR_SPRING_DAMPING,
        PARAM_LINEAR_SPRING_EQUILIBRIUM,
        PARAM_ANGULAR_LOWER_LIMIT_X,
        PARAM_ANGULAR_UPPER_LIMIT_X,
        PARAM_ANGULAR_LOWER_LIMIT_Y,
        PARAM_ANGULAR_UPPER_LIMIT_Y,
        PARAM_ANGULAR_LOWER_LIMIT_Z,
        PARAM_ANGULAR_UPPER_LIMIT_Z,
        PARAM_ANGULAR_LIMIT_SOFTNESS,
        PARAM_ANGULAR_DAMPING,
        PARAM_ANGULAR_RESTITUTION,
        PARAM_ANGULAR_ERP,
        PARAM_ANGULAR_MOTOR_TARGET_VELOCITY_X,
        PARAM_ANGULAR_MOTOR_TARGET_VELOCITY_Y,
        PARAM_ANGULAR_MOTOR_TARGET_VELOCITY_Z,
        PARAM_ANGULAR_MOTOR_FORCE_LIMIT_X,
        PARAM_ANGULAR_MOTOR_FORCE_LIMIT_Y,
        PARAM_ANGULAR_MOTOR_FORCE_LIMIT_Z,
        PARAM_ANGULAR_SPRING_STIFFNESS_X,
        PARAM_ANGULAR_SPRING_STIFFNESS_Y,
        PARAM_ANGULAR_SPRING_STIFFNESS_Z,
        PARAM_ANGULAR_SPRING_DAMPING_X,
        PARAM_ANGULAR_SPRING_DAMPING_Y,
        PARAM_ANGULAR_SPRING_DAMPING_Z,
        PARAM_ANGULAR_SPRING_EQUILIBRIUM_X,
        PARAM_ANGULAR_SPRING_EQUILIBRIUM_Y,
        PARAM_ANGULAR_SPRING_EQUILIBRIUM_Z,
        PARAM_MAX
    };

    enum Flag {
        FLAG_ENABLE_LINEAR_LIMIT_X,
        FLAG_ENABLE_LINEAR_LIMIT_Y,
        FLAG_ENABLE_LINEAR_LIMIT_Z,
        FLAG_ENABLE_ANGULAR_LIMIT_X,
        FLAG_ENABLE_ANGULAR_LIMIT_Y,
        FLAG_ENABLE_ANGULAR_LIMIT_Z,
        FLAG_ENABLE_LINEAR_MOTOR_X,
        FLAG_ENABLE_LINEAR_MOTOR_Y,
        FLAG_ENABLE_LINEAR_MOTOR_Z,
        FLAG_ENABLE_ANGULAR_MOTOR_X,
        FLAG_ENABLE_ANGULAR_MOTOR_Y,
        FLAG_ENABLE_ANGULAR_MOTOR_Z,
        FLAG_ENABLE_LINEAR_SPRING_X,
        FLAG_ENABLE_LINEAR_SPRING_Y,
        FLAG_ENABLE_LINEAR_SPRING_Z,
        FLAG_ENABLE_ANGULAR_SPRING_X,
        FLAG_ENABLE_ANGULAR_SPRING_Y,
        FLAG_ENABLE_ANGULAR_SPRING_Z,
        FLAG_MAX
    };

private:
    std::map<Param, float> m_params;
    std::map<Flag, bool> m_flags;
    mat4f m_frame_a;
    mat4f m_frame_b;
    JointSolverParams m_solver_params;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("Generic6DOFJoint3D"); }

    Generic6DOFJoint3D() {
        initialize_defaults();
    }

    void set_param(Param param, float value) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_params[param] = value;
    }

    float get_param(Param param) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_params.find(param);
        return it != m_params.end() ? it->second : 0.0f;
    }

    void set_flag(Flag flag, bool enabled) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_flags[flag] = enabled;
    }

    bool get_flag(Flag flag) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_flags.find(flag);
        return it != m_flags.end() && it->second;
    }

    void set_frame_a(const mat4f& frame) { m_frame_a = frame; }
    mat4f get_frame_a() const { return m_frame_a; }

    void set_frame_b(const mat4f& frame) { m_frame_b = frame; }
    mat4f get_frame_b() const { return m_frame_b; }

    void set_solver_erp(float erp) { m_solver_params.erp = erp; }
    void set_solver_cfm(float cfm) { m_solver_params.cfm = cfm; }
    void set_solver_iterations(int iterations) { m_solver_params.iterations = iterations; }

    // #########################################################################
    // Constraint solving
    // #########################################################################
    void solve_constraints(float dt) {
        PhysicsBody3D* body_a = get_body_a();
        PhysicsBody3D* body_b = get_body_b();
        if (!body_a && !body_b) return;

        mat4f transform_a = body_a ? body_a->get_transform() : mat4f::identity();
        mat4f transform_b = body_b ? body_b->get_transform() : mat4f::identity();

        mat4f world_frame_a = transform_a * m_frame_a;
        mat4f world_frame_b = transform_b * m_frame_b;

        // Compute relative transform
        mat4f relative = world_frame_b.affine_inverse() * world_frame_a;
        vec3f linear_error = relative.get_origin();
        quatf angular_error = quatf::from_matrix(relative);

        // Apply linear limits
        for (int axis = 0; axis < 3; ++axis) {
            Flag limit_flag = static_cast<Flag>(FLAG_ENABLE_LINEAR_LIMIT_X + axis);
            if (get_flag(limit_flag)) {
                float lower = get_param(static_cast<Param>(PARAM_LINEAR_LOWER_LIMIT));
                float upper = get_param(static_cast<Param>(PARAM_LINEAR_UPPER_LIMIT));
                float error = linear_error[axis];
                
                if (error < lower) {
                    float correction = (lower - error) * m_solver_params.erp;
                    apply_linear_correction(axis, correction, body_a, body_b, world_frame_a, world_frame_b);
                } else if (error > upper) {
                    float correction = (upper - error) * m_solver_params.erp;
                    apply_linear_correction(axis, correction, body_a, body_b, world_frame_a, world_frame_b);
                }

                // Linear motor
                Flag motor_flag = static_cast<Flag>(FLAG_ENABLE_LINEAR_MOTOR_X + axis);
                if (get_flag(motor_flag)) {
                    float target_vel = get_param(static_cast<Param>(PARAM_LINEAR_MOTOR_TARGET_VELOCITY));
                    float max_force = get_param(static_cast<Param>(PARAM_LINEAR_MOTOR_FORCE_LIMIT));
                    apply_linear_motor(axis, target_vel, max_force, dt, body_a, body_b, world_frame_a, world_frame_b);
                }

                // Linear spring
                Flag spring_flag = static_cast<Flag>(FLAG_ENABLE_LINEAR_SPRING_X + axis);
                if (get_flag(spring_flag)) {
                    float stiffness = get_param(static_cast<Param>(PARAM_LINEAR_SPRING_STIFFNESS));
                    float damping = get_param(static_cast<Param>(PARAM_LINEAR_SPRING_DAMPING));
                    float equilibrium = get_param(static_cast<Param>(PARAM_LINEAR_SPRING_EQUILIBRIUM));
                    apply_linear_spring(axis, error - equilibrium, stiffness, damping, dt,
                                        body_a, body_b, world_frame_a, world_frame_b);
                }
            }
        }

        // Apply angular limits
        vec3f euler = quat_to_euler(angular_error);
        for (int axis = 0; axis < 3; ++axis) {
            Flag limit_flag = static_cast<Flag>(FLAG_ENABLE_ANGULAR_LIMIT_X + axis);
            if (get_flag(limit_flag)) {
                float lower = get_param(static_cast<Param>(static_cast<int>(PARAM_ANGULAR_LOWER_LIMIT_X) + axis * 2));
                float upper = get_param(static_cast<Param>(static_cast<int>(PARAM_ANGULAR_UPPER_LIMIT_X) + axis * 2));
                float error = euler[axis];
                
                if (error < lower || error > upper) {
                    float softness = get_param(PARAM_ANGULAR_LIMIT_SOFTNESS);
                    float correction = std::clamp(error, lower, upper) - error;
                    correction *= (1.0f - softness) * m_solver_params.erp;
                    apply_angular_correction(axis, correction, body_a, body_b, world_frame_a, world_frame_b);
                }
            }

            // Angular motor
            Flag motor_flag = static_cast<Flag>(FLAG_ENABLE_ANGULAR_MOTOR_X + axis);
            if (get_flag(motor_flag)) {
                float target_vel = get_param(static_cast<Param>(static_cast<int>(PARAM_ANGULAR_MOTOR_TARGET_VELOCITY_X) + axis));
                float max_force = get_param(static_cast<Param>(static_cast<int>(PARAM_ANGULAR_MOTOR_FORCE_LIMIT_X) + axis));
                apply_angular_motor(axis, target_vel, max_force, dt, body_a, body_b, world_frame_a, world_frame_b);
            }
        }
    }

private:
    void initialize_defaults() {
        // Linear limits
        m_params[PARAM_LINEAR_LOWER_LIMIT] = 0.0f;
        m_params[PARAM_LINEAR_UPPER_LIMIT] = 0.0f;
        m_params[PARAM_LINEAR_LIMIT_SOFTNESS] = 0.7f;
        m_params[PARAM_LINEAR_DAMPING] = 0.0f;
        m_params[PARAM_LINEAR_RESTITUTION] = 0.0f;

        // Angular limits
        for (int i = 0; i < 3; ++i) {
            m_params[static_cast<Param>(PARAM_ANGULAR_LOWER_LIMIT_X + i * 2)] = 0.0f;
            m_params[static_cast<Param>(PARAM_ANGULAR_UPPER_LIMIT_X + i * 2)] = 0.0f;
        }
        m_params[PARAM_ANGULAR_LIMIT_SOFTNESS] = 0.5f;
        m_params[PARAM_ANGULAR_DAMPING] = 0.0f;
        m_params[PARAM_ANGULAR_RESTITUTION] = 0.0f;
        m_params[PARAM_ANGULAR_ERP] = 0.2f;

        // Default all limits disabled
        for (int i = 0; i < FLAG_MAX; ++i) {
            m_flags[static_cast<Flag>(i)] = false;
        }
    }

    vec3f quat_to_euler(const quatf& q) const {
        vec3f euler;
        float sinr_cosp = 2.0f * (q.w() * q.x() + q.y() * q.z());
        float cosr_cosp = 1.0f - 2.0f * (q.x() * q.x() + q.y() * q.y());
        euler.x() = std::atan2(sinr_cosp, cosr_cosp);

        float sinp = 2.0f * (q.w() * q.y() - q.z() * q.x());
        if (std::abs(sinp) >= 1.0f) {
            euler.y() = std::copysign(M_PI_2, sinp);
        } else {
            euler.y() = std::asin(sinp);
        }

        float siny_cosp = 2.0f * (q.w() * q.z() + q.x() * q.y());
        float cosy_cosp = 1.0f - 2.0f * (q.y() * q.y() + q.z() * q.z());
        euler.z() = std::atan2(siny_cosp, cosy_cosp);

        return euler;
    }

    void apply_linear_correction(int axis, float correction,
                                  PhysicsBody3D* body_a, PhysicsBody3D* body_b,
                                  const mat4f& frame_a, const mat4f& frame_b) {
        vec3f direction = frame_a.get_axis(axis);
        vec3f impulse = direction * correction;

        if (body_a) body_a->apply_impulse(impulse, frame_a.get_origin());
        if (body_b) body_b->apply_impulse(-impulse, frame_b.get_origin());
    }

    void apply_linear_motor(int axis, float target_vel, float max_force, float dt,
                            PhysicsBody3D* body_a, PhysicsBody3D* body_b,
                            const mat4f& frame_a, const mat4f& frame_b) {
        vec3f direction = frame_a.get_axis(axis);
        vec3f vel_a = body_a ? body_a->get_linear_velocity() : vec3f(0);
        vec3f vel_b = body_b ? body_b->get_linear_velocity() : vec3f(0);
        float relative_vel = dot(vel_a - vel_b, direction);
        float vel_error = target_vel - relative_vel;

        float impulse_mag = vel_error / dt;
        impulse_mag = std::clamp(impulse_mag, -max_force, max_force);

        vec3f impulse = direction * impulse_mag;
        if (body_a) body_a->apply_impulse(impulse, frame_a.get_origin());
        if (body_b) body_b->apply_impulse(-impulse, frame_b.get_origin());
    }

    void apply_linear_spring(int axis, float displacement, float stiffness, float damping, float dt,
                             PhysicsBody3D* body_a, PhysicsBody3D* body_b,
                             const mat4f& frame_a, const mat4f& frame_b) {
        vec3f direction = frame_a.get_axis(axis);
        vec3f vel_a = body_a ? body_a->get_linear_velocity() : vec3f(0);
        vec3f vel_b = body_b ? body_b->get_linear_velocity() : vec3f(0);
        float relative_vel = dot(vel_a - vel_b, direction);

        float force = -stiffness * displacement - damping * relative_vel;
        vec3f impulse = direction * force * dt;

        if (body_a) body_a->apply_impulse(impulse, frame_a.get_origin());
        if (body_b) body_b->apply_impulse(-impulse, frame_b.get_origin());
    }

    void apply_angular_correction(int axis, float correction,
                                   PhysicsBody3D* body_a, PhysicsBody3D* body_b,
                                   const mat4f& frame_a, const mat4f& frame_b) {
        vec3f axis_vec = frame_a.get_axis(axis);
        vec3f torque = axis_vec * correction;

        if (body_a) body_a->apply_torque_impulse(torque);
        if (body_b) body_b->apply_torque_impulse(-torque);
    }

    void apply_angular_motor(int axis, float target_vel, float max_force, float dt,
                             PhysicsBody3D* body_a, PhysicsBody3D* body_b,
                             const mat4f& frame_a, const mat4f& frame_b) {
        vec3f axis_vec = frame_a.get_axis(axis);
        vec3f ang_vel_a = body_a ? body_a->get_angular_velocity() : vec3f(0);
        vec3f ang_vel_b = body_b ? body_b->get_angular_velocity() : vec3f(0);
        float relative_vel = dot(ang_vel_a - ang_vel_b, axis_vec);
        float vel_error = target_vel - relative_vel;

        float torque_mag = vel_error / dt;
        torque_mag = std::clamp(torque_mag, -max_force, max_force);

        vec3f torque = axis_vec * torque_mag;
        if (body_a) body_a->apply_torque_impulse(torque);
        if (body_b) body_b->apply_torque_impulse(-torque);
    }

    PhysicsBody3D* get_body_a() const {
        Node* node = get_node_or_null(m_node_a);
        return dynamic_cast<PhysicsBody3D*>(node);
    }

    PhysicsBody3D* get_body_b() const {
        Node* node = get_node_or_null(m_node_b);
        return dynamic_cast<PhysicsBody3D*>(node);
    }
};

// #############################################################################
// ConeTwistJoint3D - Cone and twist constraint
// #############################################################################
class ConeTwistJoint3D : public Joint3D {
    XTU_GODOT_REGISTER_CLASS(ConeTwistJoint3D, Joint3D)

private:
    float m_swing_span = 45.0f;      // Cone angle in degrees
    float m_twist_span = 180.0f;     // Twist angle in degrees
    float m_softness = 0.8f;
    float m_bias = 0.3f;
    float m_relaxation = 1.0f;
    float m_swing_motor_target_velocity = 0.0f;
    float m_twist_motor_target_velocity = 0.0f;
    float m_swing_motor_max_impulse = 1.0f;
    float m_twist_motor_max_impulse = 1.0f;
    bool m_swing_motor_enabled = false;
    bool m_twist_motor_enabled = false;
    JointSolverParams m_solver_params;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("ConeTwistJoint3D"); }

    void set_swing_span(float span) { m_swing_span = std::clamp(span, 0.0f, 180.0f); }
    float get_swing_span() const { return m_swing_span; }

    void set_twist_span(float span) { m_twist_span = std::clamp(span, 0.0f, 180.0f); }
    float get_twist_span() const { return m_twist_span; }

    void set_softness(float softness) { m_softness = std::clamp(softness, 0.0f, 1.0f); }
    float get_softness() const { return m_softness; }

    void set_bias(float bias) { m_bias = std::clamp(bias, 0.0f, 1.0f); }
    float get_bias() const { return m_bias; }

    void set_relaxation(float relaxation) { m_relaxation = std::max(0.0f, relaxation); }
    float get_relaxation() const { return m_relaxation; }

    void set_swing_motor_enabled(bool enabled) { m_swing_motor_enabled = enabled; }
    bool is_swing_motor_enabled() const { return m_swing_motor_enabled; }

    void set_twist_motor_enabled(bool enabled) { m_twist_motor_enabled = enabled; }
    bool is_twist_motor_enabled() const { return m_twist_motor_enabled; }

    void set_swing_motor_target_velocity(float vel) { m_swing_motor_target_velocity = vel; }
    float get_swing_motor_target_velocity() const { return m_swing_motor_target_velocity; }

    void set_twist_motor_target_velocity(float vel) { m_twist_motor_target_velocity = vel; }
    float get_twist_motor_target_velocity() const { return m_twist_motor_target_velocity; }

    void solve_constraints(float dt) {
        PhysicsBody3D* body_a = get_body_a();
        PhysicsBody3D* body_b = get_body_b();
        if (!body_a || !body_b) return;

        mat4f transform_a = body_a->get_transform();
        mat4f transform_b = body_b->get_transform();

        vec3f anchor_a = transform_a.xform(get_position());
        vec3f anchor_b = transform_b.xform(get_position());

        // Swing-twist decomposition
        quatf q_a = quatf::from_matrix(transform_a);
        quatf q_b = quatf::from_matrix(transform_b);
        quatf q_rel = q_a * q_b.inverse();

        vec3f twist_axis(0, 0, 1);
        vec3f swing_axis;
        float swing_angle, twist_angle;
        decompose_swing_twist(q_rel, twist_axis, swing_axis, swing_angle, twist_angle);

        // Apply twist limit
        float twist_limit_rad = m_twist_span * M_PI / 180.0f;
        if (std::abs(twist_angle) > twist_limit_rad) {
            float correction = (twist_angle > 0 ? twist_limit_rad : -twist_limit_rad) - twist_angle;
            vec3f torque = twist_axis * correction * m_solver_params.erp;
            body_a->apply_torque_impulse(torque);
            body_b->apply_torque_impulse(-torque);
        }

        // Apply swing limit
        float swing_limit_rad = m_swing_span * M_PI / 180.0f;
        if (swing_angle > swing_limit_rad) {
            float correction = swing_limit_rad - swing_angle;
            vec3f torque = swing_axis * correction * m_solver_params.erp;
            body_a->apply_torque_impulse(torque);
            body_b->apply_torque_impulse(-torque);
        }

        // Apply motors
        if (m_twist_motor_enabled) {
            vec3f ang_vel_a = body_a->get_angular_velocity();
            vec3f ang_vel_b = body_b->get_angular_velocity();
            float relative_vel = dot(ang_vel_a - ang_vel_b, twist_axis);
            float vel_error = m_twist_motor_target_velocity - relative_vel;
            float torque_mag = std::clamp(vel_error / dt, -m_twist_motor_max_impulse, m_twist_motor_max_impulse);
            vec3f torque = twist_axis * torque_mag;
            body_a->apply_torque_impulse(torque);
            body_b->apply_torque_impulse(-torque);
        }
    }

private:
    void decompose_swing_twist(const quatf& q, const vec3f& twist_axis,
                                vec3f& swing_axis, float& swing_angle, float& twist_angle) {
        vec3f qv(q.x(), q.y(), q.z());
        float proj = dot(qv, twist_axis);
        vec3f twist_qv = twist_axis * proj;
        vec3f swing_qv = qv - twist_qv;

        twist_angle = 2.0f * std::atan2(proj, q.w());
        swing_angle = 2.0f * std::atan2(swing_qv.length(), q.w());

        if (swing_qv.length_sq() > 1e-6f) {
            swing_axis = swing_qv.normalized();
        } else {
            swing_axis = vec3f(1, 0, 0);
        }
    }

    PhysicsBody3D* get_body_a() const {
        Node* node = get_node_or_null(m_node_a);
        return dynamic_cast<PhysicsBody3D*>(node);
    }

    PhysicsBody3D* get_body_b() const {
        Node* node = get_node_or_null(m_node_b);
        return dynamic_cast<PhysicsBody3D*>(node);
    }
};

// #############################################################################
// JointSolver3D - Iterative constraint solver
// #############################################################################
class JointSolver3D : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(JointSolver3D, RefCounted)

private:
    std::vector<Joint3D*> m_joints;
    int m_iterations = 8;
    float m_erp = 0.8f;
    float m_cfm = 0.0001f;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("JointSolver3D"); }

    void add_joint(Joint3D* joint) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_joints.push_back(joint);
    }

    void remove_joint(Joint3D* joint) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = std::find(m_joints.begin(), m_joints.end(), joint);
        if (it != m_joints.end()) {
            m_joints.erase(it);
        }
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_joints.clear();
    }

    void solve(float dt) {
        for (int iter = 0; iter < m_iterations; ++iter) {
            parallel::parallel_for(0, m_joints.size(), [&](size_t i) {
                if (auto* g6dof = dynamic_cast<Generic6DOFJoint3D*>(m_joints[i])) {
                    g6dof->solve_constraints(dt);
                } else if (auto* cone = dynamic_cast<ConeTwistJoint3D*>(m_joints[i])) {
                    cone->solve_constraints(dt);
                }
            });
        }
    }

    void set_iterations(int iterations) { m_iterations = iterations; }
    void set_erp(float erp) { m_erp = erp; }
    void set_cfm(float cfm) { m_cfm = cfm; }
};

} // namespace physics

// Bring into main namespace
using physics::Generic6DOFJoint3D;
using physics::ConeTwistJoint3D;
using physics::JointSolver3D;
using physics::JointSolverParams;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XPHYSICS_JOINTS_3D_HPP