// File 341: modules/wicked/src/joints/wicked_hinge_joint.h
// Wicked Hinge Joint – constrains two bodies to rotate around a common axis,
// with optional angle limits and motor. All constraint solving is performed
// inline to avoid call overhead in the solver hot loop.

#ifndef WICKED_JOINTS_HINGE_JOINT_H
#define WICKED_JOINTS_HINGE_JOINT_H

#include "wicked_joint.h"
#include "../bodies/wicked_body.h"

namespace wicked {

class WickedHingeJoint : public WickedJoint {
    GDCLASS(WickedHingeJoint, WickedJoint);

public:
    WickedHingeJoint() { joint_type = JointType::HINGE; }

    // Set the hinge axis in the local frame of body A.
    void set_axis(const vec3 &p_axis) { axis_a = p_axis.normalized(); }
    vec3 get_axis() const { return axis_a; }

    // Set the pivot point in the local frame of body A.
    void set_pivot(const vec3 &p_pivot) { pivot_a = p_pivot; }
    vec3 get_pivot() const { return pivot_a; }

    // --- Angle limits (radians) ---
    void set_limit_enabled(bool p_enable) { limit_enabled = p_enable; }
    bool is_limit_enabled() const { return limit_enabled; }
    void set_limit_angle(real_t p_min, real_t p_max) {
        min_angle = MIN(p_min, p_max);
        max_angle = MAX(p_min, p_max);
    }
    real_t get_min_angle() const { return min_angle; }
    real_t get_max_angle() const { return max_angle; }

    // --- Motor ---
    void set_motor_enabled(bool p_enable) { motor_enabled = p_enable; }
    bool is_motor_enabled() const { return motor_enabled; }
    void set_motor_target_velocity(real_t p_omega) { motor_target_vel = p_omega; }
    real_t get_motor_target_velocity() const { return motor_target_vel; }
    void set_motor_max_torque(real_t p_torque) { motor_max_torque = MAX(p_torque, 0.0); }
    real_t get_motor_max_torque() const { return motor_max_torque; }

    // Solver entry – fully inline for maximum performance.
    virtual void solve(WickedBody *a, WickedBody *b, real_t dt) override {
        if (!enabled || !a || !b) return;
        if (a->get_type() == BodyType::STATIC && b->get_type() == BodyType::STATIC) return;

        const mat4 &xA = a->get_transform();
        const mat4 &xB = b->get_transform();

        // World pivot on each body (symmetric local pivot)
        vec3 worldAnchorA = xA.xform(pivot_a);
        vec3 worldAnchorB = xB.xform(pivot_a);

        // World hinge axis (from body A)
        vec3 worldAxisA = xA.basis.xform(axis_a).normalized();
        vec3 worldAxisB = xB.basis.xform(axis_a).normalized();

        real_t invMassA = a->get_inverse_mass();
        real_t invMassB = b->get_inverse_mass();
        real_t invMassSum = invMassA + invMassB;

        // ---- Pivot constraint: keep anchors coincident ----
        vec3 posError = worldAnchorB - worldAnchorA;
        if (invMassSum > CMP_EPSILON) {
            real_t erp = 0.2f;
            vec3 correction = posError * (erp / dt);
            vec3 impulse = correction / invMassSum;
            if (invMassA > 0.0) a->apply_impulse( impulse, worldAnchorA);
            if (invMassB > 0.0) b->apply_impulse(-impulse, worldAnchorB);
        }

        // ---- Axis alignment: enforce worldAxisB parallel to worldAxisA ----
        vec3 crossAxes = worldAxisB.cross(worldAxisA);
        real_t crossLen = crossAxes.length();
        if (crossLen > CMP_EPSILON) {
            vec3 rotAxis = crossAxes / crossLen;
            real_t rotAngle = Math::asin(crossLen);   // small angle
            rotAngle = CLAMP(rotAngle, -0.5f, 0.5f);

            vec3 invIA = a->get_inverse_inertia_world().xform(rotAxis);
            vec3 invIB = b->get_inverse_inertia_world().xform(rotAxis);
            real_t invEffInertia = invIA.dot(rotAxis) + invIB.dot(rotAxis);
            if (invEffInertia > CMP_EPSILON) {
                real_t angularSpeed = rotAngle * 0.5f / dt;
                vec3 angularImpulse = rotAxis * (angularSpeed / invEffInertia);
                if (invMassA > 0.0) a->apply_impulse(vec3(),  angularImpulse);
                if (invMassB > 0.0) b->apply_impulse(vec3(), -angularImpulse);
            }
        }

        // ---- Limit enforcement ----
        if (limit_enabled) {
            // Compute a reference direction in body A perpendicular to the hinge axis.
            vec3 refDirA = (Math::abs(worldAxisA.x) < 0.999f) ?
                worldAxisA.cross(vec3(1, 0, 0)).normalized() :
                worldAxisA.cross(vec3(0, 1, 0)).normalized();

            // The same direction expressed in body B's frame, after removing the axial component.
            vec3 refDirB = xB.basis.get_column(0).normalized(); // local X of B
            refDirB = (refDirB - worldAxisB * refDirB.dot(worldAxisB)).normalized();

            // Signed angle from refDirA to refDirB around the hinge axis.
            vec3 crossRefs = refDirB.cross(refDirA);
            real_t dotRefs = refDirB.dot(refDirA);
            real_t currentAngle = Math::atan2(crossRefs.dot(worldAxisA), dotRefs);

            real_t lower = min_angle;
            real_t upper = max_angle;
            real_t limitError = 0.0f;
            if (currentAngle < lower) limitError = lower - currentAngle;
            else if (currentAngle > upper) limitError = upper - currentAngle;

            if (Math::abs(limitError) > CMP_EPSILON) {
                vec3 rotAxis = worldAxisA;
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
            real_t currentOmega = (omegaB - omegaA).dot(worldAxisA);
            real_t omegaError = motor_target_vel - currentOmega;

            vec3 invIA = a->get_inverse_inertia_world().xform(worldAxisA);
            vec3 invIB = b->get_inverse_inertia_world().xform(worldAxisA);
            real_t invEff = invIA.dot(worldAxisA) + invIB.dot(worldAxisA);
            if (invEff > CMP_EPSILON) {
                real_t motorAccel = omegaError / dt;
                real_t motorTorque = motorAccel / invEff;
                motorTorque = CLAMP(motorTorque, -motor_max_torque, motor_max_torque);
                vec3 motorImpulse = worldAxisA * motorTorque;
                if (invMassA > 0.0) a->apply_impulse(vec3(), -motorImpulse);
                if (invMassB > 0.0) b->apply_impulse(vec3(),  motorImpulse);
            }
        }

        // ---- Breakable check (inherited) ----
        if (breakable) {
            // Break if the applied impulse exceeds limits – implemented in the base class
            // through applied forces/torques; we skip details here.
        }
    }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_axis", "axis"), &WickedHingeJoint::set_axis);
        ClassDB::bind_method(D_METHOD("get_axis"), &WickedHingeJoint::get_axis);
        ClassDB::bind_method(D_METHOD("set_pivot", "pivot"), &WickedHingeJoint::set_pivot);
        ClassDB::bind_method(D_METHOD("get_pivot"), &WickedHingeJoint::get_pivot);
        ClassDB::bind_method(D_METHOD("set_limit_enabled", "enabled"), &WickedHingeJoint::set_limit_enabled);
        ClassDB::bind_method(D_METHOD("is_limit_enabled"), &WickedHingeJoint::is_limit_enabled);
        ClassDB::bind_method(D_METHOD("set_limit_angle", "min", "max"), &WickedHingeJoint::set_limit_angle);
        ClassDB::bind_method(D_METHOD("get_min_angle"), &WickedHingeJoint::get_min_angle);
        ClassDB::bind_method(D_METHOD("get_max_angle"), &WickedHingeJoint::get_max_angle);
        ClassDB::bind_method(D_METHOD("set_motor_enabled", "enabled"), &WickedHingeJoint::set_motor_enabled);
        ClassDB::bind_method(D_METHOD("is_motor_enabled"), &WickedHingeJoint::is_motor_enabled);
        ClassDB::bind_method(D_METHOD("set_motor_target_velocity", "omega"), &WickedHingeJoint::set_motor_target_velocity);
        ClassDB::bind_method(D_METHOD("get_motor_target_velocity"), &WickedHingeJoint::get_motor_target_velocity);
        ClassDB::bind_method(D_METHOD("set_motor_max_torque", "torque"), &WickedHingeJoint::set_motor_max_torque);
        ClassDB::bind_method(D_METHOD("get_motor_max_torque"), &WickedHingeJoint::get_motor_max_torque);

        ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "axis"), "set_axis", "get_axis");
        ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "pivot"), "set_pivot", "get_pivot");
        ADD_PROPERTY(PropertyInfo(Variant::BOOL, "limit_enabled"), "set_limit_enabled", "is_limit_enabled");
        ADD_PROPERTY(PropertyInfo(Variant::BOOL, "motor_enabled"), "set_motor_enabled", "is_motor_enabled");
    }

private:
    vec3 axis_a = vec3(1, 0, 0);
    vec3 pivot_a = vec3();
    bool limit_enabled = false;
    real_t min_angle = -Math_PI;
    real_t max_angle =  Math_PI;
    bool motor_enabled = false;
    real_t motor_target_vel = 0.0;
    real_t motor_max_torque = 100.0;
};

} // namespace wicked

#endif // WICKED_JOINTS_HINGE_JOINT_H