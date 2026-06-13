// File 190: modules/newton/src/joints/newton_universal_joint.h
// Universal joint – constrains two rotation axes, leaving one rotational
// degree of freedom (the twisting axis).  Also supports swing limits
// and motors on each axis.

#ifndef NEWTON_JOINTS_UNIVERSAL_H
#define NEWTON_JOINTS_UNIVERSAL_H

#include "newton_joint.h"

namespace newton {

class NewtonUniversalJoint : public NewtonJoint {
	GDCLASS(NewtonUniversalJoint, NewtonJoint);

public:
	NewtonUniversalJoint() {
		joint_type = JointType::UNIVERSAL;
	}

	// Set the two axes in the local frame of body A.
	void set_axis_a(const vec3 &p_axis) { axis_a1 = p_axis.normalized(); }
	vec3 get_axis_a() const { return axis_a1; }

	void set_axis_b(const vec3 &p_axis) { axis_a2 = p_axis.normalized(); }
	vec3 get_axis_b() const { return axis_a2; }

	// Set the pivot in body A's local frame.
	void set_pivot(const vec3 &p_pivot) { pivot_a = p_pivot; }
	vec3 get_pivot() const { return pivot_a; }

	// Limits for the swing (angle around axis_a2 after projecting out axis_a1)
	void set_limit_enabled(bool p_enable) { limit_enabled = p_enable; }
	bool is_limit_enabled() const { return limit_enabled; }

	void set_swing_limit_angle(real_t p_min, real_t p_max) {
		swing_min = CLAMP(p_min, -Math_PI, Math_PI);
		swing_max = CLAMP(p_max, -Math_PI, Math_PI);
	}
	real_t get_swing_min() const { return swing_min; }
	real_t get_swing_max() const { return swing_max; }

	// Motor on the primary axis (axis_a1)
	void set_motor_enabled(bool p_enable) { motor_enabled = p_enable; }
	bool is_motor_enabled() const { return motor_enabled; }

	void set_motor_target_velocity(real_t p_omega) { motor_target_vel = p_omega; }
	real_t get_motor_target_velocity() const { return motor_target_vel; }

	void set_motor_max_torque(real_t p_torque) { motor_max_torque = MAX(p_torque, 0.0); }
	real_t get_motor_max_torque() const { return motor_max_torque; }

	// Solve the universal joint constraint.
	virtual void solve(real_t dt) override;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_axis_a", "axis"), &NewtonUniversalJoint::set_axis_a);
		ClassDB::bind_method(D_METHOD("get_axis_a"), &NewtonUniversalJoint::get_axis_a);
		ClassDB::bind_method(D_METHOD("set_axis_b", "axis"), &NewtonUniversalJoint::set_axis_b);
		ClassDB::bind_method(D_METHOD("get_axis_b"), &NewtonUniversalJoint::get_axis_b);
		ClassDB::bind_method(D_METHOD("set_pivot", "pivot"), &NewtonUniversalJoint::set_pivot);
		ClassDB::bind_method(D_METHOD("get_pivot"), &NewtonUniversalJoint::get_pivot);
		ClassDB::bind_method(D_METHOD("set_limit_enabled", "enabled"), &NewtonUniversalJoint::set_limit_enabled);
		ClassDB::bind_method(D_METHOD("is_limit_enabled"), &NewtonUniversalJoint::is_limit_enabled);
		ClassDB::bind_method(D_METHOD("set_swing_limit_angle", "min", "max"), &NewtonUniversalJoint::set_swing_limit_angle);
		ClassDB::bind_method(D_METHOD("get_swing_min"), &NewtonUniversalJoint::get_swing_min);
		ClassDB::bind_method(D_METHOD("get_swing_max"), &NewtonUniversalJoint::get_swing_max);
		ClassDB::bind_method(D_METHOD("set_motor_enabled", "enabled"), &NewtonUniversalJoint::set_motor_enabled);
		ClassDB::bind_method(D_METHOD("is_motor_enabled"), &NewtonUniversalJoint::is_motor_enabled);
		ClassDB::bind_method(D_METHOD("set_motor_target_velocity", "omega"), &NewtonUniversalJoint::set_motor_target_velocity);
		ClassDB::bind_method(D_METHOD("get_motor_target_velocity"), &NewtonUniversalJoint::get_motor_target_velocity);
		ClassDB::bind_method(D_METHOD("set_motor_max_torque", "torque"), &NewtonUniversalJoint::set_motor_max_torque);
		ClassDB::bind_method(D_METHOD("get_motor_max_torque"), &NewtonUniversalJoint::get_motor_max_torque);

		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "axis_a"), "set_axis_a", "get_axis_a");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "axis_b"), "set_axis_b", "get_axis_b");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "pivot"), "set_pivot", "get_pivot");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "limit_enabled"), "set_limit_enabled", "is_limit_enabled");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "motor_enabled"), "set_motor_enabled", "is_motor_enabled");
	}

private:
	vec3 axis_a1 = vec3(1, 0, 0);
	vec3 axis_a2 = vec3(0, 1, 0);
	vec3 pivot_a = vec3();
	bool limit_enabled = false;
	real_t swing_min = -Math_PI * 0.25;
	real_t swing_max =  Math_PI * 0.25;
	bool motor_enabled = false;
	real_t motor_target_vel = 0.0;
	real_t motor_max_torque = 100.0;
};

} // namespace newton

#endif // NEWTON_JOINTS_UNIVERSAL_H