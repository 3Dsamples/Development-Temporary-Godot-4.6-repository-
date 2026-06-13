// File 282: modules/vienna/src/joints/vienna_fixed_joint.h
// Vienna Fixed Joint – welds two bodies together with zero relative motion.
// Enforces both translational and rotational constraints using position‑level
// Baumgarte correction and velocity‑level damping.  Supports breakable limits.

#ifndef VIENNA_JOINTS_FIXED_JOINT_H
#define VIENNA_JOINTS_FIXED_JOINT_H

#include "vienna_joint.h"

namespace vienna {

class ViennaFixedJoint : public ViennaJoint {
	GDCLASS(ViennaFixedJoint, ViennaJoint);

public:
	ViennaFixedJoint() { joint_type = JointType::FIXED; }

	// Set the relative transform from body A to body B that the joint enforces.
	void set_relative_transform(const mat4 &p_rel) { relative_xform = p_rel; }
	mat4 get_relative_transform() const { return relative_xform; }

	// Breakable forces: if the force or torque on the joint exceeds these limits,
	// the joint is automatically disabled.
	void set_breakable_enabled(bool p_enable) { breakable = p_enable; }
	bool is_breakable() const { return breakable; }

	void set_break_force(real_t p_force) { break_force = MAX(p_force, 0.0); }
	real_t get_break_force() const { return break_force; }

	void set_break_torque(real_t p_torque) { break_torque = MAX(p_torque, 0.0); }
	real_t get_break_torque() const { return break_torque; }

	virtual void solve(ViennaBody *a, ViennaBody *b, real_t dt) override;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_relative_transform", "rel"), &ViennaFixedJoint::set_relative_transform);
		ClassDB::bind_method(D_METHOD("get_relative_transform"), &ViennaFixedJoint::get_relative_transform);
		ClassDB::bind_method(D_METHOD("set_breakable_enabled", "enabled"), &ViennaFixedJoint::set_breakable_enabled);
		ClassDB::bind_method(D_METHOD("is_breakable"), &ViennaFixedJoint::is_breakable);
		ClassDB::bind_method(D_METHOD("set_break_force", "force"), &ViennaFixedJoint::set_break_force);
		ClassDB::bind_method(D_METHOD("get_break_force"), &ViennaFixedJoint::get_break_force);
		ClassDB::bind_method(D_METHOD("set_break_torque", "torque"), &ViennaFixedJoint::set_break_torque);
		ClassDB::bind_method(D_METHOD("get_break_torque"), &ViennaFixedJoint::get_break_torque);

		ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "relative_transform"), "set_relative_transform", "get_relative_transform");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "breakable"), "set_breakable_enabled", "is_breakable");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "break_force"), "set_break_force", "get_break_force");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "break_torque"), "set_break_torque", "get_break_torque");
	}

private:
	mat4 relative_xform;
	bool breakable = false;
	real_t break_force = INFINITY;
	real_t break_torque = INFINITY;
};

// Implementation of solve
inline void ViennaFixedJoint::solve(ViennaBody *a, ViennaBody *b, real_t dt) {
	if (!enabled || !a || !b) return;
	if (a->get_type() == BodyType::STATIC && b->get_type() == BodyType::STATIC) return;

	const mat4 &xA = a->get_transform();
	const mat4 &xB = b->get_transform();

	// Desired relative transform: B = A * relative_xform
	mat4 targetB = xA * relative_xform;

	// Position error
	vec3 posError = targetB.origin - xB.origin;

	// Rotational error: targetB.basis * currentB.basis^T
	mat3 rotErrorMat = targetB.basis * xB.basis.transposed();
	quat rotErrorQuat(rotErrorMat);
	vec3 rotAxis;
	real_t rotAngle;
	rotErrorQuat.get_axis_angle(rotAxis, rotAngle);

	real_t invMassA = a->get_inverse_mass();
	real_t invMassB = b->get_inverse_mass();
	real_t invMassSum = invMassA + invMassB;

	// Track total force and torque for breakability check
	vec3 totalForce(0, 0, 0);
	vec3 totalTorque(0, 0, 0);

	// ---- POSITION CORRECTION (translation) ----
	if (invMassSum > CMP_EPSILON) {
		real_t erp = 0.2f;
		vec3 correction = posError * (erp / dt);
		vec3 impulse = correction / invMassSum;
		if (invMassA > 0.0) a->apply_impulse( impulse, xA.origin);
		if (invMassB > 0.0) b->apply_impulse(-impulse, xB.origin);
		totalForce += impulse / dt;  // force = impulse/dt
	}

	// ---- ANGULAR CORRECTION ----
	if (Math::abs(rotAngle) > CMP_EPSILON) {
		vec3 invIA = a->get_inverse_inertia_world().xform(rotAxis);
		vec3 invIB = b->get_inverse_inertia_world().xform(rotAxis);
		real_t invEffInertia = invIA.dot(rotAxis) + invIB.dot(rotAxis);
		if (invEffInertia > CMP_EPSILON) {
			real_t angularSpeed = rotAngle * 0.5f / dt;
			vec3 angularImpulse = rotAxis * (angularSpeed / invEffInertia);
			if (invMassA > 0.0) a->apply_impulse(vec3(),  angularImpulse);
			if (invMassB > 0.0) b->apply_impulse(vec3(), -angularImpulse);
			totalTorque += angularImpulse / dt;
		}
	}

	// ---- BREAKABLE CHECK ----
	if (breakable) {
		real_t forceMag = totalForce.length();
		real_t torqueMag = totalTorque.length();
		if (forceMag > break_force || torqueMag > break_torque) {
			enabled = false;  // joint breaks
		}
	}
}

} // namespace vienna

#endif // VIENNA_JOINTS_FIXED_JOINT_H