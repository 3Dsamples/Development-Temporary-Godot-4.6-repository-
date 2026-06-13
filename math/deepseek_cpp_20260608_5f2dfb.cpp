// File 188: modules/newton/src/joints/newton_hinge_joint.h
// Hinge joint – restricts two bodies to rotate around a common axis.
// Allows a single degree of freedom (rotation angle) with optional limits
// and a motor. Implements position‑level Baumgarte correction and
// velocity‑level damping.

#ifndef NEWTON_JOINTS_HINGE_H
#define NEWTON_JOINTS_HINGE_H

#include "newton_joint.h"

namespace newton {

class NewtonHingeJoint : public NewtonJoint {
	GDCLASS(NewtonHingeJoint, NewtonJoint);

public:
	NewtonHingeJoint() {
		joint_type = JointType::HINGE;
	}

	// Set the hinge axis in the local frame of body A.
	void set_axis(const vec3 &p_axis) { axis_a = p_axis.normalized(); }
	vec3 get_axis() const { return axis_a; }

	// Set the pivot point in the local frame of body A (anchor).
	void set_pivot(const vec3 &p_pivot) { pivot_a = p_pivot; }
	vec3 get_pivot() const { return pivot_a; }

	// Limits (in radians)
	void set_limit_enabled(bool p_enable) { limit_enabled = p_enable; }
	bool is_limit_enabled() const { return limit_enabled; }

	void set_limit_angle(real_t p_min, real_t p_max) {
		min_angle = MIN(p_min, p_max);
		max_angle = MAX(p_min, p_max);
	}
	real_t get_min_angle() const { return min_angle; }
	real_t get_max_angle() const { return max_angle; }

	// Motor
	void set_motor_enabled(bool p_enable) { motor_enabled = p_enable; }
	bool is_motor_enabled() const { return motor_enabled; }

	void set_motor_target_velocity(real_t p_omega) { motor_target_vel = p_omega; }
	real_t get_motor_target_velocity() const { return motor_target_vel; }

	void set_motor_max_torque(real_t p_torque) { motor_max_torque = MAX(p_torque, 0.0); }
	real_t get_motor_max_torque() const { return motor_max_torque; }

	// Solve the hinge constraint.
	virtual void solve(real_t dt) override;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_axis", "axis"), &NewtonHingeJoint::set_axis);
		ClassDB::bind_method(D_METHOD("get_axis"), &NewtonHingeJoint::get_axis);
		ClassDB::bind_method(D_METHOD("set_pivot", "pivot"), &NewtonHingeJoint::set_pivot);
		ClassDB::bind_method(D_METHOD("get_pivot"), &NewtonHingeJoint::get_pivot);
		ClassDB::bind_method(D_METHOD("set_limit_enabled", "enabled"), &NewtonHingeJoint::set_limit_enabled);
		ClassDB::bind_method(D_METHOD("is_limit_enabled"), &NewtonHingeJoint::is_limit_enabled);
		ClassDB::bind_method(D_METHOD("set_limit_angle", "min", "max"), &NewtonHingeJoint::set_limit_angle);
		ClassDB::bind_method(D_METHOD("get_min_angle"), &NewtonHingeJoint::get_min_angle);
		ClassDB::bind_method(D_METHOD("get_max_angle"), &NewtonHingeJoint::get_max_angle);
		ClassDB::bind_method(D_METHOD("set_motor_enabled", "enabled"), &NewtonHingeJoint::set_motor_enabled);
		ClassDB::bind_method(D_METHOD("is_motor_enabled"), &NewtonHingeJoint::is_motor_enabled);
		ClassDB::bind_method(D_METHOD("set_motor_target_velocity", "omega"), &NewtonHingeJoint::set_motor_target_velocity);
		ClassDB::bind_method(D_METHOD("get_motor_target_velocity"), &NewtonHingeJoint::get_motor_target_velocity);
		ClassDB::bind_method(D_METHOD("set_motor_max_torque", "torque"), &NewtonHingeJoint::set_motor_max_torque);
		ClassDB::bind_method(D_METHOD("get_motor_max_torque"), &NewtonHingeJoint::get_motor_max_torque);

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
	real_t max_angle = Math_PI;
	bool motor_enabled = false;
	real_t motor_target_vel = 0.0;
	real_t motor_max_torque = 100.0;
};

} // namespace newton

#endif // NEWTON_JOINTS_HINGE_H