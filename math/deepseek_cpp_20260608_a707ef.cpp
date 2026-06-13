// File 213: modules/newton/src/joints/newton_fixed_joint.h
// Fixed joint: welds two bodies together, removing all relative translational
// and rotational degrees of freedom.  Enforces a constant relative transform.
// Supports breakable force/torque thresholds.

#ifndef NEWTON_JOINTS_FIXED_JOINT_H
#define NEWTON_JOINTS_FIXED_JOINT_H

#include "newton_joint.h"

namespace newton {

class NewtonFixedJoint : public NewtonJoint {
	GDCLASS(NewtonFixedJoint, NewtonJoint);

public:
	NewtonFixedJoint() {
		joint_type = JointType::FIXED_DISTANCE;
	}

	// Set the relative transform from body A to body B that the joint enforces.
	void set_relative_transform(const mat4 &p_rel) { relative_xform = p_rel; }
	mat4 get_relative_transform() const { return relative_xform; }

	// Breakable: if the force or torque on the joint exceeds these limits,
	// the joint is automatically disabled (enabled = false).
	void set_breakable_enabled(bool p_enable) { breakable = p_enable; }
	bool is_breakable() const { return breakable; }

	void set_break_force(real_t p_force) { break_force = MAX(p_force, 0.0); }
	real_t get_break_force() const { return break_force; }

	void set_break_torque(real_t p_torque) { break_torque = MAX(p_torque, 0.0); }
	real_t get_break_torque() const { return break_torque; }

	virtual void solve(NewtonBody *a, NewtonBody *b, real_t dt) override;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_relative_transform", "rel"), &NewtonFixedJoint::set_relative_transform);
		ClassDB::bind_method(D_METHOD("get_relative_transform"), &NewtonFixedJoint::get_relative_transform);
		ClassDB::bind_method(D_METHOD("set_breakable_enabled", "enabled"), &NewtonFixedJoint::set_breakable_enabled);
		ClassDB::bind_method(D_METHOD("is_breakable"), &NewtonFixedJoint::is_breakable);
		ClassDB::bind_method(D_METHOD("set_break_force", "force"), &NewtonFixedJoint::set_break_force);
		ClassDB::bind_method(D_METHOD("get_break_force"), &NewtonFixedJoint::get_break_force);
		ClassDB::bind_method(D_METHOD("set_break_torque", "torque"), &NewtonFixedJoint::set_break_torque);
		ClassDB::bind_method(D_METHOD("get_break_torque"), &NewtonFixedJoint::get_break_torque);

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

} // namespace newton

#endif // NEWTON_JOINTS_FIXED_JOINT_H