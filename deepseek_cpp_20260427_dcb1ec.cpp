// File 228: modules/newton/src/joints/newton_gear_joint.h
// Gear joint: constrains the rotation angles of two hinge joints (or two
// rigid bodies around their own axes) to maintain a fixed ratio.  Typically
// used to simulate gears, cogwheels, or differentials.  Supports a transfer
// ratio and optional limits on the relative angle.

#ifndef NEWTON_JOINTS_GEAR_JOINT_H
#define NEWTON_JOINTS_GEAR_JOINT_H

#include "newton_joint.h"

namespace newton {

class NewtonGearJoint : public NewtonJoint {
	GDCLASS(NewtonGearJoint, NewtonJoint);

public:
	NewtonGearJoint() { joint_type = JointType::CUSTOM; } // could be a dedicated GEAR type

	// Set the axis of rotation for body A (in local frame of body A).
	void set_axis_a(const vec3 &p_axis) { axis_a = p_axis.normalized(); }
	vec3 get_axis_a() const { return axis_a; }

	// Set the axis of rotation for body B (in local frame of body B).
	void set_axis_b(const vec3 &p_axis) { axis_b = p_axis.normalized(); }
	vec3 get_axis_b() const { return axis_b; }

	// Set the gear ratio: angleB = ratio * angleA.  Default 1.0.
	void set_ratio(real_t p_ratio) { ratio = p_ratio; }
	real_t get_ratio() const { return ratio; }

	// Optional limits on the relative angle (in radians, with respect to the initial offset).
	// Limits are on the accumulated angleB - ratio*angleA.
	void set_limit_enabled(bool p_enable) { limit_enabled = p_enable; }
	bool is_limit_enabled() const { return limit_enabled; }

	void set_limit_angle(real_t p_min, real_t p_max) {
		limit_min = MIN(p_min, p_max);
		limit_max = MAX(p_min, p_max);
	}
	real_t get_limit_min() const { return limit_min; }
	real_t get_limit_max() const { return limit_max; }

	virtual void solve(NewtonBody *a, NewtonBody *b, real_t dt) override;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_axis_a", "axis"), &NewtonGearJoint::set_axis_a);
		ClassDB::bind_method(D_METHOD("get_axis_a"), &NewtonGearJoint::get_axis_a);
		ClassDB::bind_method(D_METHOD("set_axis_b", "axis"), &NewtonGearJoint::set_axis_b);
		ClassDB::bind_method(D_METHOD("get_axis_b"), &NewtonGearJoint::get_axis_b);
		ClassDB::bind_method(D_METHOD("set_ratio", "ratio"), &NewtonGearJoint::set_ratio);
		ClassDB::bind_method(D_METHOD("get_ratio"), &NewtonGearJoint::get_ratio);
		ClassDB::bind_method(D_METHOD("set_limit_enabled", "enabled"), &NewtonGearJoint::set_limit_enabled);
		ClassDB::bind_method(D_METHOD("is_limit_enabled"), &NewtonGearJoint::is_limit_enabled);
		ClassDB::bind_method(D_METHOD("set_limit_angle", "min", "max"), &NewtonGearJoint::set_limit_angle);
		ClassDB::bind_method(D_METHOD("get_limit_min"), &NewtonGearJoint::get_limit_min);
		ClassDB::bind_method(D_METHOD("get_limit_max"), &NewtonGearJoint::get_limit_max);

		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "axis_a"), "set_axis_a", "get_axis_a");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "axis_b"), "set_axis_b", "get_axis_b");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ratio"), "set_ratio", "get_ratio");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "limit_enabled"), "set_limit_enabled", "is_limit_enabled");
	}

private:
	vec3 axis_a = vec3(1, 0, 0);
	vec3 axis_b = vec3(1, 0, 0);
	real_t ratio = 1.0;
	bool limit_enabled = false;
	real_t limit_min = -Math_PI;
	real_t limit_max =  Math_PI;
	// Accumulated relative angle for tracking limits.
	real_t accumulated_angle = 0.0;
};

} // namespace newton

#endif // NEWTON_JOINTS_GEAR_JOINT_H