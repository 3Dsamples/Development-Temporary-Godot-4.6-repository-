// File 208: modules/newton/src/joints/newton_ball_joint.h
// Ball-and-socket joint: constrains translation of two bodies at a common
// pivot point, allowing free relative rotation.  Supports cone and twist
// limits and an optional motor.

#ifndef NEWTON_JOINTS_BALL_JOINT_H
#define NEWTON_JOINTS_BALL_JOINT_H

#include "newton_joint.h"

namespace newton {

class NewtonBallJoint : public NewtonJoint {
	GDCLASS(NewtonBallJoint, NewtonJoint);

public:
	NewtonBallJoint() { joint_type = JointType::BALL; }

	// Pivot point in the local frame of body A (same as body B).
	void set_pivot(const vec3 &p_pivot) { pivot_a = p_pivot; }
	vec3 get_pivot() const { return pivot_a; }

	// Cone limit: maximum angle between the two body axes from the pivot.
	void set_cone_limit_enabled(bool p_enable) { cone_limit_enabled = p_enable; }
	bool is_cone_limit_enabled() const { return cone_limit_enabled; }

	void set_cone_angle(real_t p_radians) { cone_angle = CLAMP(p_radians, 0.0, Math_PI); }
	real_t get_cone_angle() const { return cone_angle; }

	// Twist limit: min and max angles around the cone axis.
	void set_twist_limit_enabled(bool p_enable) { twist_limit_enabled = p_enable; }
	bool is_twist_limit_enabled() const { return twist_limit_enabled; }

	void set_twist_angle(real_t p_min, real_t p_max) {
		twist_min = MIN(p_min, p_max);
		twist_max = MAX(p_min, p_max);
	}
	real_t get_twist_min() const { return twist_min; }
	real_t get_twist_max() const { return twist_max; }

	// Motor (drives relative angular velocity around the cone axis).
	void set_motor_enabled(bool p_enable) { motor_enabled = p_enable; }
	bool is_motor_enabled() const { return motor_enabled; }

	void set_motor_target_velocity(real_t p_omega) { motor_target_vel = p_omega; }
	real_t get_motor_target_velocity() const { return motor_target_vel; }

	void set_motor_max_torque(real_t p_torque) { motor_max_torque = MAX(p_torque, 0.0); }
	real_t get_motor_max_torque() const { return motor_max_torque; }

	virtual void solve(NewtonBody *a, NewtonBody *b, real_t dt) override;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_pivot", "pivot"), &NewtonBallJoint::set_pivot);
		ClassDB::bind_method(D_METHOD("get_pivot"), &NewtonBallJoint::get_pivot);
		ClassDB::bind_method(D_METHOD("set_cone_limit_enabled", "enabled"), &NewtonBallJoint::set_cone_limit_enabled);
		ClassDB::bind_method(D_METHOD("is_cone_limit_enabled"), &NewtonBallJoint::is_cone_limit_enabled);
		ClassDB::bind_method(D_METHOD("set_cone_angle", "angle"), &NewtonBallJoint::set_cone_angle);
		ClassDB::bind_method(D_METHOD("get_cone_angle"), &NewtonBallJoint::get_cone_angle);
		ClassDB::bind_method(D_METHOD("set_twist_limit_enabled", "enabled"), &NewtonBallJoint::set_twist_limit_enabled);
		ClassDB::bind_method(D_METHOD("is_twist_limit_enabled"), &NewtonBallJoint::is_twist_limit_enabled);
		ClassDB::bind_method(D_METHOD("set_twist_angle", "min", "max"), &NewtonBallJoint::set_twist_angle);
		ClassDB::bind_method(D_METHOD("get_twist_min"), &NewtonBallJoint::get_twist_min);
		ClassDB::bind_method(D_METHOD("get_twist_max"), &NewtonBallJoint::get_twist_max);
		ClassDB::bind_method(D_METHOD("set_motor_enabled", "enabled"), &NewtonBallJoint::set_motor_enabled);
		ClassDB::bind_method(D_METHOD("is_motor_enabled"), &NewtonBallJoint::is_motor_enabled);
		ClassDB::bind_method(D_METHOD("set_motor_target_velocity", "omega"), &NewtonBallJoint::set_motor_target_velocity);
		ClassDB::bind_method(D_METHOD("get_motor_target_velocity"), &NewtonBallJoint::get_motor_target_velocity);
		ClassDB::bind_method(D_METHOD("set_motor_max_torque", "torque"), &NewtonBallJoint::set_motor_max_torque);
		ClassDB::bind_method(D_METHOD("get_motor_max_torque"), &NewtonBallJoint::get_motor_max_torque);

		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "pivot"), "set_pivot", "get_pivot");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cone_limit_enabled"), "set_cone_limit_enabled", "is_cone_limit_enabled");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cone_angle", PROPERTY_HINT_RANGE, "0,180,0.1"), "set_cone_angle", "get_cone_angle");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "twist_limit_enabled"), "set_twist_limit_enabled", "is_twist_limit_enabled");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "motor_enabled"), "set_motor_enabled", "is_motor_enabled");
	}

private:
	vec3 pivot_a = vec3();
	bool cone_limit_enabled = false;
	real_t cone_angle = Math_PI * 0.5;
	bool twist_limit_enabled = false;
	real_t twist_min = -Math_PI;
	real_t twist_max =  Math_PI;
	bool motor_enabled = false;
	real_t motor_target_vel = 0.0;
	real_t motor_max_torque = 100.0;
};

} // namespace newton

#endif // NEWTON_JOINTS_BALL_JOINT_H