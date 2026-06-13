// File 343: modules/wicked/src/joints/wicked_fixed_joint.h
// Wicked Fixed Joint – welds two bodies together, eliminating all relative
// translational and rotational degrees of freedom.  Supports breakable force/
// torque thresholds.  The solve method is fully inline for maximum solver
// throughput.

#ifndef WICKED_JOINTS_FIXED_JOINT_H
#define WICKED_JOINTS_FIXED_JOINT_H

#include "wicked_joint.h"
#include "../bodies/wicked_body.h"

namespace wicked {

class WickedFixedJoint : public WickedJoint {
    GDCLASS(WickedFixedJoint, WickedJoint);

public:
    WickedFixedJoint() { joint_type = JointType::FIXED; }

    // Set the relative transform from body A to body B that the joint enforces.
    void set_relative_transform(const mat4 &p_rel) { relative_xform = p_rel; }
    mat4 get_relative_transform() const { return relative_xform; }

    // Breakable: if the force or torque on the joint exceeds these limits,
    // the joint is automatically disabled.
    void set_breakable(bool p_enable) { breakable = p_enable; }
    bool is_breakable() const { return breakable; }
    void set_break_force(real_t p_force) { break_force = MAX(p_force, 0.0); }
    real_t get_break_force() const { return break_force; }
    void set_break_torque(real_t p_torque) { break_torque = MAX(p_torque, 0.0); }
    real_t get_break_torque() const { return break_torque; }

    // --- Solver entry (fully inline) ---
    virtual void solve(WickedBody *a, WickedBody *b, real_t dt) override {
        if (!enabled || !a || !b) return;
        if (a->get_type() == BodyType::STATIC && b->get_type() == BodyType::STATIC) return;

        const mat4 &xA = a->get_transform();
        const mat4 &xB = b->get_transform();

        // Desired world transform of B: A * relative_xform
        mat4 targetB = xA * relative_xform;

        // Position error (translation)
        vec3 posError = targetB.origin - xB.origin;

        // Rotation error: targetB.basis * currentB.basis^T
        mat3 rotErrorMat = targetB.basis * xB.basis.transposed();
        quat rotErrorQuat(rotErrorMat);
        vec3 rotAxis;
        real_t rotAngle;
        rotErrorQuat.get_axis_angle(rotAxis, rotAngle);

        real_t invMassA = a->get_inverse_mass();
        real_t invMassB = b->get_inverse_mass();
        real_t invMassSum = invMassA + invMassB;

        // Accumulate applied force/torque for breakable detection
        vec3 totalForce;
        vec3 totalTorque;

        // ---- Translation correction ----
        if (invMassSum > CMP_EPSILON) {
            real_t erp = 0.2f;
            vec3 correction = posError * (erp / dt);
            vec3 impulse = correction / invMassSum;
            if (invMassA > 0.0) a->apply_impulse( impulse, xA.origin);
            if (invMassB > 0.0) b->apply_impulse(-impulse, xB.origin);
            totalForce = impulse / dt;
        }

        // ---- Rotation correction ----
        if (Math::abs(rotAngle) > CMP_EPSILON) {
            vec3 invIA = a->get_inverse_inertia_world().xform(rotAxis);
            vec3 invIB = b->get_inverse_inertia_world().xform(rotAxis);
            real_t invEffInertia = invIA.dot(rotAxis) + invIB.dot(rotAxis);
            if (invEffInertia > CMP_EPSILON) {
                real_t angularSpeed = rotAngle * 0.5f / dt;
                vec3 angularImpulse = rotAxis * (angularSpeed / invEffInertia);
                if (invMassA > 0.0) a->apply_impulse(vec3(),  angularImpulse);
                if (invMassB > 0.0) b->apply_impulse(vec3(), -angularImpulse);
                totalTorque = angularImpulse / dt;
            }
        }

        // ---- Breakable check ----
        if (breakable) {
            real_t forceMag = totalForce.length();
            real_t torqueMag = totalTorque.length();
            if (forceMag > break_force || torqueMag > break_torque) {
                enabled = false;  // joint breaks
            }
        }
    }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_relative_transform", "rel"), &WickedFixedJoint::set_relative_transform);
        ClassDB::bind_method(D_METHOD("get_relative_transform"), &WickedFixedJoint::get_relative_transform);
        ClassDB::bind_method(D_METHOD("set_breakable", "enabled"), &WickedFixedJoint::set_breakable);
        ClassDB::bind_method(D_METHOD("is_breakable"), &WickedFixedJoint::is_breakable);
        ClassDB::bind_method(D_METHOD("set_break_force", "force"), &WickedFixedJoint::set_break_force);
        ClassDB::bind_method(D_METHOD("get_break_force"), &WickedFixedJoint::get_break_force);
        ClassDB::bind_method(D_METHOD("set_break_torque", "torque"), &WickedFixedJoint::set_break_torque);
        ClassDB::bind_method(D_METHOD("get_break_torque"), &WickedFixedJoint::get_break_torque);

        ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "relative_transform"), "set_relative_transform", "get_relative_transform");
        ADD_PROPERTY(PropertyInfo(Variant::BOOL, "breakable"), "set_breakable", "is_breakable");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "break_force"), "set_break_force", "get_break_force");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "break_torque"), "set_break_torque", "get_break_torque");
    }

private:
    mat4 relative_xform;
    bool breakable = false;
    real_t break_force = INFINITY;
    real_t break_torque = INFINITY;
};

} // namespace wicked

#endif // WICKED_JOINTS_FIXED_JOINT_H