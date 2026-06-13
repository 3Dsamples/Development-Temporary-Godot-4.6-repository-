// File 189: modules/newton/src/joints/newton_slider_joint.h
// Slider (prismatic) joint – allows translation along a single axis.
// Optional limits and linear motor. Solves position‑level error and
// velocity‑level damping. Inherits from NewtonJoint.

#ifndef NEWTON_JOINTS_SLIDER_H
#define NEWTON_JOINTS_SLIDER_H

#include "newton_joint.h"

namespace newton {

class NewtonSliderJoint : public NewtonJoint {
	GDCLASS(NewtonSliderJoint, NewtonJoint);

public:
	NewtonSliderJoint() {
		joint_type = JointType::SLIDER;
	}

	// Set the slider axis in the local frame of body A.
	void set_axis(const vec3 &p_axis) { axis_a = p_axis.normalized(); }
	vec3 get_axis() const { return axis_a; }

	// Set the anchor point in body A's local frame.
	void set_anchor(const vec3 &p_anchor) { anchor_a = p_anchor; }
	vec3 get_anchor() const { return anchor_a; }

	// Translation limits along the axis (relative to initial offset).
	void set_limit_enabled(bool p_enable) { limit_enabled = p_enable; }
	bool is_limit_enabled() const { return limit_enabled; }

	void set_limit_range(real_t p_min, real_t p_max) {
		min_limit = MIN(p_min, p_max);
		max_limit = MAX(p_min, p_max);
	}
	real_t get_min_limit() const { return min_limit; }
	real_t get_max_limit() const { return max_limit; }

	// Linear motor.
	void set_motor_enabled(bool p_enable) { motor_enabled = p_enable; }
	bool is_motor_enabled() const { return motor_enabled; }

	void set_motor_target_velocity(real_t p_vel) { motor_target_vel = p_vel; }
	real_t get_motor_target_velocity() const { return motor_target_vel; }

	void set_motor_max_force(real_t p_force) { motor_max_force = MAX(p_force, 0.0); }
	real_t get_motor_max_force() const { return motor_max_force; }

	// Solve the slider constraint.
	virtual void solve(real_t dt) override;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_axis", "axis"), &NewtonSliderJoint::set_axis);
		ClassDB::bind_method(D_METHOD("get_axis"), &NewtonSliderJoint::get_axis);
		ClassDB::bind_method(D_METHOD("set_anchor", "anchor"), &NewtonSliderJoint::set_anchor);
		ClassDB::bind_method(D_METHOD("get_anchor"), &NewtonSliderJoint::get_anchor);
		ClassDB::bind_method(D_METHOD("set_limit_enabled", "enabled"), &NewtonSliderJoint::set_limit_enabled);
		ClassDB::bind_method(D_METHOD("is_limit_enabled"), &NewtonSliderJoint::is_limit_enabled);
		ClassDB::bind_method(D_METHOD("set_limit_range", "min", "max"), &NewtonSliderJoint::set_limit_range);
		ClassDB::bind_method(D_METHOD("get_min_limit"), &NewtonSliderJoint::get_min_limit);
		ClassDB::bind_method(D_METHOD("get_max_limit"), &NewtonSliderJoint::get_max_limit);
		ClassDB::bind_method(D_METHOD("set_motor_enabled", "enabled"), &NewtonSliderJoint::set_motor_enabled);
		ClassDB::bind_method(D_METHOD("is_motor_enabled"), &NewtonSliderJoint::is_motor_enabled);
		ClassDB::bind_method(D_METHOD("set_motor_target_velocity", "vel"), &NewtonSliderJoint::set_motor_target_velocity);
		ClassDB::bind_method(D_METHOD("get_motor_target_velocity"), &NewtonSliderJoint::get_motor_target_velocity);
		ClassDB::bind_method(D_METHOD("set_motor_max_force", "force"), &NewtonSliderJoint::set_motor_max_force);
		ClassDB::bind_method(D_METHOD("get_motor_max_force"), &NewtonSliderJoint::get_motor_max_force);

		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "axis"), "set_axis", "get_axis");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "anchor"), "set_anchor", "get_anchor");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "limit_enabled"), "set_limit_enabled", "is_limit_enabled");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "motor_enabled"), "set_motor_enabled", "is_motor_enabled");
	}

private:
	vec3 axis_a = vec3(1, 0, 0);
	vec3 anchor_a = vec3();
	bool limit_enabled = false;
	real_t min_limit = -1.0;
	real_t max_limit = 1.0;
	bool motor_enabled = false;
	real_t motor_target_vel = 0.0;
	real_t motor_max_force = 100.0;
};

} // namespace newton

#endif // NEWTON_JOINTS_SLIDER_H