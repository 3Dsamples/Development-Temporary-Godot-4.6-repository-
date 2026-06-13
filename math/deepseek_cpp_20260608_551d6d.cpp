// File 345: modules/wicked/src/joints/wicked_generic_6dof_joint.h
// Wicked Generic 6DOF Joint – constrains up to six axes (three linear,
// three angular) with independent limits, motors, springs, and dampers.
// Useful for vehicle suspensions, robotic arms, and complex linkages.
// The solve method is fully inline for maximum solver performance.

#ifndef WICKED_JOINTS_GENERIC_6DOF_JOINT_H
#define WICKED_JOINTS_GENERIC_6DOF_JOINT_H

#include "wicked_joint.h"
#include "../bodies/wicked_body.h"

namespace wicked {

class WickedGeneric6DOFJoint : public WickedJoint {
    GDCLASS(WickedGeneric6DOFJoint, WickedJoint);

public:
    // Axis description for a single degree of freedom.
    struct AxisParams {
        bool limited = false;                  // whether limits are active
        real_t lower_limit = -1.0;             // minimum value (m for linear, rad for angular)
        real_t upper_limit =  1.0;             // maximum value
        bool motor_enabled = false;            // drive to a target velocity
        real_t motor_target_velocity = 0.0;    // target velocity (m/s or rad/s)
        real_t motor_max_force = INFINITY;     // max force (N) or torque (Nm)
        bool spring_enabled = false;           // spring‑damper attached to this axis
        real_t spring_stiffness = 100.0;       // N/m or Nm/rad
        real_t spring_damper = 10.0;           // Ns/m or Nms/rad
    };

    WickedGeneric6DOFJoint() { joint_type = JointType::GENERIC_6DOF; }

    // Set the local frames of body A and body B.  The constraint operates in
    // the space of body A; body B must match.  Default: identity.
    void set_frame_a(const mat4 &p_frame) { frame_a = p_frame; }
    mat4 get_frame_a() const { return frame_a; }
    void set_frame_b(const mat4 &p_frame) { frame_b = p_frame; }
    mat4 get_frame_b() const { return frame_b; }

    // Access individual axis parameters.  Linear: 0=X, 1=Y, 2=Z.
    // Angular: 3=rotX, 4=rotY, 5=rotZ.
    AxisParams &get_axis(int p_index) {
        ERR_FAIL_INDEX_V(p_index, 6, axes[0]);
        return axes[p_index];
    }
    const AxisParams &get_axis(int p_index) const {
        ERR_FAIL_INDEX_V(p_index, 6, axes[0]);
        return axes[p_index];
    }

    // Convenience to set all parameters for one axis.
    void set_axis_params(int p_axis, bool p_limited,
                         real_t p_lower, real_t p_upper,
                         bool p_motor, real_t p_target_vel, real_t p_max_force,
                         bool p_spring, real_t p_stiffness, real_t p_damper) {
        ERR_FAIL_INDEX(p_axis, 6);
        AxisParams &ax = axes[p_axis];
        ax.limited = p_limited;
        ax.lower_limit = MIN(p_lower, p_upper);
        ax.upper_limit = MAX(p_lower, p_upper);
        ax.motor_enabled = p_motor;
        ax.motor_target_velocity = p_target_vel;
        ax.motor_max_force = MAX(p_max_force, 0.0);
        ax.spring_enabled = p_spring;
        ax.spring_stiffness = MAX(p_stiffness, 0.0);
        ax.spring_damper = MAX(p_damper, 0.0);
    }

    // Solver entry – solves all 6 DOFs inline.
    virtual void solve(WickedBody *a, WickedBody *b, real_t dt) override {
        if (!enabled || !a || !b) return;
        if (a->get_type() == BodyType::STATIC && b->get_type() == BodyType::STATIC) return;

        // Build world frames
        mat4 worldFrameA = a->get_transform() * frame_a;
        mat4 worldFrameB = b->get_transform() * frame_b;

        // Solve each linear axis (0,1,2)
        for (int i = 0; i < 3; ++i) {
            solve_linear_axis(i, a, b, worldFrameA, worldFrameB, dt);
        }
        // Solve each angular axis (3,4,5)
        for (int i = 3; i < 6; ++i) {
            solve_angular_axis(i, a, b, worldFrameA, worldFrameB, dt);
        }
    }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_frame_a", "frame"), &WickedGeneric6DOFJoint::set_frame_a);
        ClassDB::bind_method(D_METHOD("get_frame_a"), &WickedGeneric6DOFJoint::get_frame_a);
        ClassDB::bind_method(D_METHOD("set_frame_b", "frame"), &WickedGeneric6DOFJoint::set_frame_b);
        ClassDB::bind_method(D_METHOD("get_frame_b"), &WickedGeneric6DOFJoint::get_frame_b);
        ClassDB::bind_method(D_METHOD("set_axis_params", "axis", "limited", "lower", "upper",
            "motor", "target_vel", "max_force", "spring", "stiffness", "damper"),
            &WickedGeneric6DOFJoint::set_axis_params);
        ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "frame_a"), "set_frame_a", "get_frame_a");
        ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "frame_b"), "set_frame_b", "get_frame_b");
    }

private:
    mat4 frame_a;
    mat4 frame_b;
    AxisParams axes[6]; // 0‑2 linear, 3‑5 angular

    // Solve a single linear axis constraint.
    void solve_linear_axis(int p_axis, WickedBody *a, WickedBody *b,
                           const mat4 &worldA, const mat4 &worldB, real_t dt) {
        const AxisParams &ax = axes[p_axis];

        // World axis direction (X, Y, or Z of frame A)
        vec3 worldAxis = worldA.basis.get_column(p_axis).normalized();
        // Anchor points: origins of the two frames
        vec3 anchorA = worldA.origin;
        vec3 anchorB = worldB.origin;

        // Current relative position along the axis
        vec3 relPos = anchorB - anchorA;
        real_t current = relPos.dot(worldAxis);

        // Velocities at anchor points
        vec3 velA = a->get_linear_velocity() + a->get_angular_velocity().cross(anchorA - a->get_position());
        vec3 velB = b->get_linear_velocity() + b->get_angular_velocity().cross(anchorB - b->get_position());
        real_t relVel = (velB - velA).dot(worldAxis);

        // Effective inverse mass along this axis (including rotational contributions)
        real_t invMA = a->get_inverse_mass();
        real_t invMB = b->get_inverse_mass();
        vec3 rA = anchorA - a->get_position();
        vec3 rB = anchorB - b->get_position();
        real_t invEff = invMA + invMB +
            worldAxis.dot(a->get_inverse_inertia_world().xform(rA.cross(worldAxis)).cross(rA)) +
            worldAxis.dot(b->get_inverse_inertia_world().xform(rB.cross(worldAxis)).cross(rB));

        if (invEff < CMP_EPSILON) return;

        // Limit error
        real_t limitError = 0.0;
        if (ax.limited) {
            if (current < ax.lower_limit) limitError = ax.lower_limit - current;
            else if (current > ax.upper_limit) limitError = ax.upper_limit - current;
        }

        // Motor target velocity
        real_t motorTarget = ax.motor_enabled ? ax.motor_target_velocity : 0.0f;

        // Spring‑damper force
        real_t springForce = 0.0;
        real_t damperForce = 0.0;
        if (ax.spring_enabled) {
            springForce = -ax.spring_stiffness * current; // returns to zero displacement
            damperForce = -ax.spring_damper * relVel;
        }

        // Desired velocity: motor + limit correction + spring/damper
        real_t targetVel = motorTarget;
        if (Math::abs(limitError) > CMP_EPSILON) {
            targetVel += limitError * 0.5f / dt; // erp 0.5
        }
        real_t springAccel = (springForce + damperForce) * invEff;
        real_t desiredAccel = (targetVel - relVel) / dt + springAccel;

        real_t lambda = desiredAccel * dt / invEff;

        // Clamp motor force
        if (ax.motor_enabled && ax.motor_max_force > 0.0) {
            real_t maxImpulse = ax.motor_max_force * dt;
            lambda = CLAMP(lambda, -maxImpulse, maxImpulse);
        }

        vec3 impulse = worldAxis * lambda;
        if (invMA > 0.0) a->apply_impulse( impulse, anchorA);
        if (invMB > 0.0) b->apply_impulse(-impulse, anchorB);
    }

    // Solve a single angular axis constraint (rotation around local X, Y, Z).
    void solve_angular_axis(int p_axis, WickedBody *a, WickedBody *b,
                            const mat4 &worldA, const mat4 &worldB, real_t dt) {
        const AxisParams &ax = axes[p_axis];
        int angIdx = p_axis - 3; // 0,1,2

        // World axis around which rotation is constrained
        vec3 worldAxis = worldA.basis.get_column(angIdx).normalized();

        // Relative rotation error
        mat3 RA = worldA.basis;
        mat3 RB = worldB.basis;
        mat3 Rerr = RB * RA.transposed();
        quat qerr(Rerr);
        vec3 rotAxis;
        real_t rotAngle;
        qerr.get_axis_angle(rotAxis, rotAngle);

        // Project error onto the desired angular axis
        real_t angleError = rotAngle * (rotAxis.dot(worldAxis)); // signed
        angleError = CLAMP(angleError, -Math_PI * 0.5f, Math_PI * 0.5f);

        // Relative angular velocity around the axis
        vec3 omegaA = a->get_angular_velocity();
        vec3 omegaB = b->get_angular_velocity();
        real_t relOmega = (omegaB - omegaA).dot(worldAxis);

        // Effective inverse inertia
        vec3 invIA = a->get_inverse_inertia_world().xform(worldAxis);
        vec3 invIB = b->get_inverse_inertia_world().xform(worldAxis);
        real_t invEff = invIA.dot(worldAxis) + invIB.dot(worldAxis);

        if (invEff < CMP_EPSILON) return;

        // Limit error
        real_t limitError = 0.0;
        if (ax.limited) {
            if (angleError < ax.lower_limit) limitError = ax.lower_limit - angleError;
            else if (angleError > ax.upper_limit) limitError = ax.upper_limit - angleError;
        }

        // Motor target velocity
        real_t motorTarget = ax.motor_enabled ? ax.motor_target_velocity : 0.0f;

        // Spring‑damper torque
        real_t springTorque = 0.0;
        real_t damperTorque = 0.0;
        if (ax.spring_enabled) {
            springTorque = -ax.spring_stiffness * angleError;
            damperTorque = -ax.spring_damper * relOmega;
        }

        // Desired angular acceleration
        real_t targetAccel = motorTarget;
        if (Math::abs(limitError) > CMP_EPSILON) {
            targetAccel += limitError * 0.5f / dt;
        }
        real_t springAccel = (springTorque + damperTorque) * invEff;
        real_t desiredAccel = (targetAccel - relOmega) / dt + springAccel;

        real_t lambda = desiredAccel * dt / invEff;

        // Clamp motor torque
        if (ax.motor_enabled && ax.motor_max_force > 0.0) {
            real_t maxImpulse = ax.motor_max_force * dt;
            lambda = CLAMP(lambda, -maxImpulse, maxImpulse);
        }

        vec3 angularImpulse = worldAxis * lambda;
        if (a->get_inverse_mass() > 0.0) a->apply_impulse(vec3(),  angularImpulse);
        if (b->get_inverse_mass() > 0.0) b->apply_impulse(vec3(), -angularImpulse);
    }
};

} // namespace wicked

#endif // WICKED_JOINTS_GENERIC_6DOF_JOINT_H