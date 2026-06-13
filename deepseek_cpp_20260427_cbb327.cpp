// File 250: modules/newton/src/joints/newton_d6_joint.h
// Newton D6 Joint (6‑DOF): a fully configurable joint that can lock or limit
// up to three linear and three angular degrees of freedom between two bodies.
// Each axis can have independent limits, motors, spring‑damper, and erp.

#ifndef NEWTON_JOINTS_D6_JOINT_H
#define NEWTON_JOINTS_D6_JOINT_H

#include "newton_joint.h"

namespace newton {

class NewtonD6Joint : public NewtonJoint {
	GDCLASS(NewtonD6Joint, NewtonJoint);

public:
	// Single degree‑of‑freedom parameters.
	struct AxisParams {
		bool limited = false;          // whether limits are active
		real_t lower_limit = -1.0;     // minimum value (translation in m, rotation in rad)
		real_t upper_limit = 1.0;      // maximum value
		bool motor_enabled = false;    // drive to a target velocity
		real_t motor_target_velocity = 0.0;   // target velocity (m/s or rad/s)
		real_t motor_max_force = 0.0;         // max force/torque (N or Nm)
		real_t motor_max_acceleration = INFINITY; // maximal acceleration
		bool spring_enabled = false;   // spring‑damper attached to the axis
		real_t spring_stiffness = 100.0;         // N/m or Nm/rad
		real_t spring_damper = 10.0;            // Ns/m or Nms/rad
	};

	NewtonD6Joint() { joint_type = JointType::CUSTOM; }

	// Set the local frames of body A and body B (the constraint operates in
	// the space of body A, and body B must match). By default, both frames
	// are identity.
	void set_frame_a(const mat4 &p_frame) { frame_a = p_frame; }
	mat4 get_frame_a() const { return frame_a; }
	void set_frame_b(const mat4 &p_frame) { frame_b = p_frame; }
	mat4 get_frame_b() const { return frame_b; }

	// Access individual axis parameters.  Linear axes: 0=X, 1=Y, 2=Z.
	// Angular axes: 3=rotX, 4=rotY, 5=rotZ.
	AxisParams &get_axis(int p_axis) {
		ERR_FAIL_INDEX_V(p_axis, 6, axes[0]);
		return axes[p_axis];
	}
	const AxisParams &get_axis(int p_axis) const { return axes[p_axis]; }

	void set_axis_params(int p_axis, bool p_limited,
						 real_t p_lower, real_t p_upper,
						 bool p_motor, real_t p_target_vel, real_t p_max_force,
						 bool p_spring, real_t p_stiffness, real_t p_damper) {
		ERR_FAIL_INDEX(p_axis, 6);
		AxisParams &a = axes[p_axis];
		a.limited = p_limited;
		a.lower_limit = MIN(p_lower, p_upper);
		a.upper_limit = MAX(p_lower, p_upper);
		a.motor_enabled = p_motor;
		a.motor_target_velocity = p_target_vel;
		a.motor_max_force = MAX(p_max_force, 0.0);
		a.spring_enabled = p_spring;
		a.spring_stiffness = MAX(p_stiffness, 0.0);
		a.spring_damper = MAX(p_damper, 0.0);
	}

	virtual void solve(NewtonBody *a, NewtonBody *b, real_t dt) override;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_frame_a", "frame"), &NewtonD6Joint::set_frame_a);
		ClassDB::bind_method(D_METHOD("get_frame_a"), &NewtonD6Joint::get_frame_a);
		ClassDB::bind_method(D_METHOD("set_frame_b", "frame"), &NewtonD6Joint::set_frame_b);
		ClassDB::bind_method(D_METHOD("get_frame_b"), &NewtonD6Joint::get_frame_b);
		ClassDB::bind_method(D_METHOD("set_axis_params", "axis", "limited", "lower", "upper",
			"motor", "target_vel", "max_force", "spring", "stiffness", "damper"),
			&NewtonD6Joint::set_axis_params);
		ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "frame_a"), "set_frame_a", "get_frame_a");
		ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "frame_b"), "set_frame_b", "get_frame_b");
	}

private:
	// Internal: solve a single linear axis constraint.
	void solve_linear_axis(int p_axis, NewtonBody *a, NewtonBody *b,
						   const mat4 &xA, const mat4 &xB, real_t dt);
	// Internal: solve a single angular axis constraint.
	void solve_angular_axis(int p_axis, NewtonBody *a, NewtonBody *b,
							const mat4 &xA, const mat4 &xB, real_t dt);

	mat4 frame_a; // local to body A
	mat4 frame_b; // local to body B
	AxisParams axes[6]; // 0-2 linear, 3-5 angular
};

} // namespace newton

#endif // NEWTON_JOINTS_D6_JOINT_H