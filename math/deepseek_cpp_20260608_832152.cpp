// File 344: modules/wicked/src/joints/wicked_cone_twist_joint.h
// Wicked ConeTwist Joint – constrains rotation between two bodies with a cone
// limit (elliptic) and twist limits. The pivot is a ball joint, rotation is
// limited by swing (within a cone) and twist around the axis.  Motor can drive
// twist.  The solver enforces these via Baumgarte-correction impulses.

#ifndef WICKED_JOINTS_CONE_TWIST_JOINT_H
#define WICKED_JOINTS_CONE_TWIST_JOINT_H

#include "wicked_joint.h"
#include "../bodies/wicked_body.h"

namespace wicked {

class WickedConeTwistJoint : public WickedJoint {
    GDCLASS(WickedConeTwistJoint, WickedJoint);

public:
    WickedConeTwistJoint() { joint_type = JointType::CONE_TWIST; }

    // Pivot point (ball joint) in local frame of A.
    void set_pivot(const vec3 &p_pivot) { pivot_a = p_pivot; }
    vec3 get_pivot() const { return pivot_a; }

    // The twist axis (in local frame of A). Usually Z or Y.
    void set_twist_axis(const vec3 &p_axis) { twist_axis = p_axis.normalized(); }
    vec3 get_twist_axis() const { return twist_axis; }

    // Swing span (half cone angle) for the primary and secondary axes.
    void set_swing_span(real_t p_swing1, real_t p_swing2) {
        swing_span1 = CLAMP(p_swing1, 0.001f, Math_PI);
        swing_span2 = CLAMP(p_swing2, 0.001f, Math_PI);
    }
    real_t get_swing_span1() const { return swing_span1; }
    real_t get_swing_span2() const { return swing_span2; }

    // Twist limits (min/max around twist axis).
    void set_twist_span(real_t p_min, real_t p_max) {
        twist_min = MIN(p_min, p_max);
        twist_max = MAX(p_min, p_max);
    }
    real_t get_twist_min() const { return twist_min; }
    real_t get_twist_max() const { return twist_max; }

    // Motor driving twist.
    void set_motor_enabled(bool p_enable) { motor_enabled = p_enable; }
    bool is_motor_enabled() const { return motor_enabled; }
    void set_motor_target_velocity(real_t p_omega) { motor_target_vel = p_omega; }
    real_t get_motor_target_velocity() const { return motor_target_vel; }
    void set_motor_max_torque(real_t p_torque) { motor_max_torque = MAX(p_torque, 0.0); }
    real_t get_motor_max_torque() const { return motor_max_torque; }

    // Solver entry (fully inline).
    virtual void solve(WickedBody *a, WickedBody *b, real_t dt) override {
        if (!enabled || !a || !b) return;
        if (a->get_type() == BodyType::STATIC && b->get_type() == BodyType::STATIC) return;

        const mat4 &xA = a->get_transform();
        const mat4 &xB = b->get_transform();

        // ---- Ball joint: keep pivots coincident ----
        vec3 worldPivotA = xA.xform(pivot_a);
        vec3 worldPivotB = xB.xform(pivot_a);
        vec3 posError = worldPivotB - worldPivotA;
        real_t invMassA = a->get_inverse_mass();
        real_t invMassB = b->get_inverse_mass();
        real_t invMassSum = invMassA + invMassB;
        if (invMassSum > CMP_EPSILON) {
            real_t erp = 0.2f;
            vec3 correction = posError * (erp / dt);
            vec3 impulse = correction / invMassSum;
            if (invMassA > 0.0) a->apply_impulse( impulse, worldPivotA);
            if (invMassB > 0.0) b->apply_impulse(-impulse, worldPivotB);
        }

        // ---- Frame definitions ----
        // Body A's twist axis in world space.
        vec3 worldTwistAxisA = xA.basis.xform(twist_axis).normalized();
        // The secondary axis for body A (perpendicular to twist)
        vec3 worldAxis2A = (Math::abs(worldTwistAxisA.x) < 0.999f) ?
            worldTwistAxisA.cross(vec3(1,0,0)).normalized() :
            worldTwistAxisA.cross(vec3(0,1,0)).normalized();
        // The third axis.
        vec3 worldAxis3A = worldTwistAxisA.cross(worldAxis2A).normalized();

        // Same for body B.
        vec3 worldTwistAxisB = xB.basis.xform(twist_axis).normalized();
        vec3 worldAxis2B = (Math::abs(worldTwistAxisB.x) < 0.999f) ?
            worldTwistAxisB.cross(vec3(1,0,0)).normalized() :
            worldTwistAxisB.cross(vec3(0,1,0)).normalized();
        vec3 worldAxis3B = worldTwistAxisB.cross(worldAxis2B).normalized();

        // ---- Swing limits (elliptic cone) ----
        // Compute the direction of B's twist axis projected onto A's perpendicular plane.
        vec3 bTwistProj = worldTwistAxisB - worldTwistAxisA * worldTwistAxisB.dot(worldTwistAxisA);
        real_t projLen = bTwistProj.length();
        if (projLen > CMP_EPSILON) {
            bTwistProj /= projLen;
            // Decompose projection into A's frame axes (2 and 3)
            real_t comp2 = bTwistProj.dot(worldAxis2A);
            real_t comp3 = bTwistProj.dot(worldAxis3A);
            // Distance from origin in the elliptic parameter space
            real_t ellipse_dist = Math::sqrt( (comp2*comp2)/(swing_span1*swing_span1) +
                                              (comp3*comp3)/(swing_span2*swing_span2) );
            if (ellipse_dist > 1.0) {
                // The twist axis of B is outside the allowed cone.
                // Compute the correction direction in 3D space (toward the cone edge).
                // We'll find the corrected direction on the ellipse.
                // Simplify: scale the components back to the ellipse surface.
                real_t scale = 1.0 / ellipse_dist;
                vec3 desiredProj = bTwistProj * scale; // this is on the ellipse, but we need to push B's axis.
                // The error in projection space.
                vec3 projError = (bTwistProj - desiredProj) * swing_span1; // approximate
                // Apply angular impulse to rotate body B's twist axis towards the correct direction.
                vec3 rotAxis = worldTwistAxisA.cross(projError);
                real_t rotAxisLen = rotAxis.length();
                if (rotAxisLen > CMP_EPSILON) {
                    rotAxis /= rotAxisLen;
                    real_t error = Math::acos(CLAMP(worldTwistAxisB.dot(worldTwistAxisA), -1.0, 1.0)) -
                                  Math::acos(CLAMP(desiredProj.dot(worldTwistAxisA), -1.0, 1.0)); // approximate
                    real_t maxError = 0.2f;
                    error = CLAMP(error, -maxError, maxError);
                    vec3 invIA = a->get_inverse_inertia_world().xform(rotAxis);
                    vec3 invIB = b->get_inverse_inertia_world().xform(rotAxis);
                    real_t invEff = invIA.dot(rotAxis) + invIB.dot(rotAxis);
                    if (invEff > CMP_EPSILON) {
                        real_t angularSpeed = error * 0.5f / dt;
                        vec3 angularImpulse = rotAxis * (angularSpeed / invEff);
                        if (invMassA > 0.0) a->apply_impulse(vec3(),  angularImpulse);
                        if (invMassB > 0.0) b->apply_impulse(vec3(), -angularImpulse);
                    }
                }
            }
        }

        // ---- Twist limit ----
        // Compute the signed twist angle between the two bodies around the twist axis.
        // Use the secondary axes.
        vec3 refA = worldAxis2A;
        vec3 refB = worldAxis2B - worldTwistAxisA * worldAxis2B.dot(worldTwistAxisA);
        real_t refBLen = refB.length();
        if (refBLen > CMP_EPSILON) {
            refB /= refBLen;
            vec3 crossRef = refB.cross(refA);
            real_t dotRef   = refB.dot(refA);
            real_t twistAngle = Math::atan2(crossRef.dot(worldTwistAxisA), dotRef);

            real_t lower = twist_min;
            real_t upper = twist_max;
            real_t limitError = 0.0f;
            if (twistAngle < lower) limitError = lower - twistAngle;
            else if (twistAngle > upper) limitError = upper - twistAngle;

            if (Math::abs(limitError) > CMP_EPSILON) {
                vec3 rotAxis = worldTwistAxisA;
                vec3 invIA = a->get_inverse_inertia_world().xform(rotAxis);
                vec3 invIB = b->get_inverse_inertia_world().xform(rotAxis);
                real_t invEff = invIA.dot(rotAxis) + invIB.dot(rotAxis);
                if (invEff > CMP_EPSILON) {
                    real_t angularSpeed = limitError * 0.5f / dt;
                    vec3 angularImpulse = rotAxis * (angularSpeed / invEff);
                    if (invMassA > 0.0) a->apply_impulse(vec3(),  angularImpulse);
                    if (invMassB > 0.0) b->apply_impulse(vec3(), -angularImpulse);
                }
            }
        }

        // ---- Motor ----
        if (motor_enabled) {
            vec3 omegaA = a->get_angular_velocity();
            vec3 omegaB = b->get_angular_velocity();
            real_t currentOmega = (omegaB - omegaA).dot(worldTwistAxisA);
            real_t omegaError = motor_target_vel - currentOmega;
            vec3 invIA = a->get_inverse_inertia_world().xform(worldTwistAxisA);
            vec3 invIB = b->get_inverse_inertia_world().xform(worldTwistAxisA);
            real_t invEff = invIA.dot(worldTwistAxisA) + invIB.dot(worldTwistAxisA);
            if (invEff > CMP_EPSILON) {
                real_t motorAccel = omegaError / dt;
                real_t motorTorque = motorAccel / invEff;
                motorTorque = CLAMP(motorTorque, -motor_max_torque, motor_max_torque);
                vec3 motorImpulse = worldTwistAxisA * motorTorque;
                if (invMassA > 0.0) a->apply_impulse(vec3(), -motorImpulse);
                if (invMassB > 0.0) b->apply_impulse(vec3(),  motorImpulse);
            }
        }
    }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_pivot", "pivot"), &WickedConeTwistJoint::set_pivot);
        ClassDB::bind_method(D_METHOD("get_pivot"), &WickedConeTwistJoint::get_pivot);
        ClassDB::bind_method(D_METHOD("set_twist_axis", "axis"), &WickedConeTwistJoint::set_twist_axis);
        ClassDB::bind_method(D_METHOD("get_twist_axis"), &WickedConeTwistJoint::get_twist_axis);
        ClassDB::bind_method(D_METHOD("set_swing_span", "span1", "span2"), &WickedConeTwistJoint::set_swing_span);
        ClassDB::bind_method(D_METHOD("get_swing_span1"), &WickedConeTwistJoint::get_swing_span1);
        ClassDB::bind_method(D_METHOD("get_swing_span2"), &WickedConeTwistJoint::get_swing_span2);
        ClassDB::bind_method(D_METHOD("set_twist_span", "min", "max"), &WickedConeTwistJoint::set_twist_span);
        ClassDB::bind_method(D_METHOD("get_twist_min"), &WickedConeTwistJoint::get_twist_min);
        ClassDB::bind_method(D_METHOD("get_twist_max"), &WickedConeTwistJoint::get_twist_max);
        ClassDB::bind_method(D_METHOD("set_motor_enabled", "enabled"), &WickedConeTwistJoint::set_motor_enabled);
        ClassDB::bind_method(D_METHOD("is_motor_enabled"), &WickedConeTwistJoint::is_motor_enabled);
        ClassDB::bind_method(D_METHOD("set_motor_target_velocity", "omega"), &WickedConeTwistJoint::set_motor_target_velocity);
        ClassDB::bind_method(D_METHOD("get_motor_target_velocity"), &WickedConeTwistJoint::get_motor_target_velocity);
        ClassDB::bind_method(D_METHOD("set_motor_max_torque", "torque"), &WickedConeTwistJoint::set_motor_max_torque);
        ClassDB::bind_method(D_METHOD("get_motor_max_torque"), &WickedConeTwistJoint::get_motor_max_torque);

        ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "pivot"), "set_pivot", "get_pivot");
        ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "twist_axis"), "set_twist_axis", "get_twist_axis");
        ADD_PROPERTY(PropertyInfo(Variant::BOOL, "motor_enabled"), "set_motor_enabled", "is_motor_enabled");
    }

private:
    vec3 pivot_a = vec3();
    vec3 twist_axis = vec3(0, 1, 0);
    real_t swing_span1 = Math_PI * 0.25f;
    real_t swing_span2 = Math_PI * 0.25f;
    real_t twist_min = -Math_PI;
    real_t twist_max =  Math_PI;
    bool motor_enabled = false;
    real_t motor_target_vel = 0.0;
    real_t motor_max_torque = 100.0;
};

} // namespace wicked

#endif // WICKED_JOINTS_CONE_TWIST_JOINT_H