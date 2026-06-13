// File 191: modules/newton/src/joints/newton_corkscrew_joint.h
// Corkscrew joint – couples a slider and a hinge: translation along an axis
// is linked to rotation around that axis (like a screw).  Supports limits
// on the translation and a motor.

#ifndef NEWTON_JOINTS_CORKSCREW_H
#define NEWTON_JOINTS_CORKSCREW_H

#include "newton_joint.h"

namespace newton {

class NewtonCorkscrewJoint : public NewtonJoint {
	GDCLASS(NewtonCorkscrewJoint, NewtonJoint);

public:
	NewtonCorkscrewJoint() {
		joint_type = JointType::CORKSCREW;
	}

	// Set the common axis in the local frame of body A.
	void set_axis(const vec3 &p_axis) { axis_a = p_axis.normalized(); }
	vec3 get_axis() const { return axis_a; }

	// Set the pivot in body A's local frame.
	void set_pivot(const vec3 &p_pivot) { pivot_a = p_pivot; }
	vec3 get_pivot() const { return pivot_a; }

	// Pitch: translation per full revolution (2π radians).  Positive means
	// translation along +axis for a positive rotation.
	void set_pitch(real_t p_pitch) { pitch = p_pitch; }
	real_t get_pitch() const { return pitch; }

	// Translation limits (relative to initial anchor offset).
	void set_limit_enabled(bool p_enable) { limit_enabled = p_enable; }
	bool is_limit_enabled() const { return limit_enabled; }

	void set_translation_limit(real_t p_min, real_t p_max) {
		trans_min = MIN(p_min, p_max);
		trans_max = MAX(p_min, p_max);
	}
	real_t get_translation_min() const { return trans_min; }
	real_t get_translation_max() const { return trans_max; }

	// Motor on the screw (drives rotation).
	void set_motor_enabled(bool p_enable) { motor_enabled = p_enable; }
	bool is_motor_enabled() const { return motor_enabled; }

	void set_motor_target_velocity(real_t p_omega) { motor_target_vel = p_omega; }
	real_t get_motor_target_velocity() const { return motor_target_vel; }

	void set_motor_max_force(real_t p_force) { motor_max_force = MAX(p_force, 0.0); }
	real_t get_motor_max_force() const { return motor_max_force; }

	void set_motor_max_torque(real_t p_torque) { motor_max_torque = MAX(p_torque, 0.0); }
	real_t get_motor_max_torque() const { return motor_max_torque; }

	// Solve the corkscrew constraint.
	virtual void solve(real_t dt) override;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_axis", "axis"), &NewtonCorkscrewJoint::set_axis);
		ClassDB::bind_method(D_METHOD("get_axis"), &NewtonCorkscrewJoint::get_axis);
		ClassDB::bind_method(D_METHOD("set_pivot", "pivot"), &NewtonCorkscrewJoint::set_pivot);
		ClassDB::bind_method(D_METHOD("get_pivot"), &NewtonCorkscrewJoint::get_pivot);
		ClassDB::bind_method(D_METHOD("set_pitch", "pitch"), &NewtonCorkscrewJoint::set_pitch);
		ClassDB::bind_method(D_METHOD("get_pitch"), &NewtonCorkscrewJoint::get_pitch);
		ClassDB::bind_method(D_METHOD("set_limit_enabled", "enabled"), &NewtonCorkscrewJoint::set_limit_enabled);
		ClassDB::bind_method(D_METHOD("is_limit_enabled"), &NewtonCorkscrewJoint::is_limit_enabled);
		ClassDB::bind_method(D_METHOD("set_translation_limit", "min", "max"), &NewtonCorkscrewJoint::set_translation_limit);
		ClassDB::bind_method(D_METHOD("get_translation_min"), &NewtonCorkscrewJoint::get_translation_min);
		ClassDB::bind_method(D_METHOD("get_translation_max"), &NewtonCorkscrewJoint::get_translation_max);
		ClassDB::bind_method(D_METHOD("set_motor_enabled", "enabled"), &NewtonCorkscrewJoint::set_motor_enabled);
		ClassDB::bind_method(D_METHOD("is_motor_enabled"), &NewtonCorkscrewJoint::is_motor_enabled);
		ClassDB::bind_method(D_METHOD("set_motor_target_velocity", "omega"), &NewtonCorkscrewJoint::set_motor_target_velocity);
		ClassDB::bind_method(D_METHOD("get_motor_target_velocity"), &NewtonCorkscrewJoint::get_motor_target_velocity);
		ClassDB::bind_method(D_METHOD("set_motor_max_force", "force"), &NewtonCorkscrewJoint::set_motor_max_force);
		ClassDB::bind_method(D_METHOD("get_motor_max_force"), &NewtonCorkscrewJoint::get_motor_max_force);
		ClassDB::bind_method(D_METHOD("set_motor_max_torque", "torque"), &NewtonCorkscrewJoint::set_motor_max_torque);
		ClassDB::bind_method(D_METHOD("get_motor_max_torque"), &NewtonCorkscrewJoint::get_motor_max_torque);

		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "axis"), "set_axis", "get_axis");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "pivot"), "set_pivot", "get_pivot");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "pitch"), "set_pitch", "get_pitch");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "limit_enabled"), "set_limit_enabled", "is_limit_enabled");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "motor_enabled"), "set_motor_enabled", "is_motor_enabled");
	}

private:
	vec3 axis_a   = vec3(1, 0, 0);
	vec3 pivot_a  = vec3();
	real_t pitch = 0.01;          // translation per radian
	bool limit_enabled = false;
	real_t trans_min = -0.5;
	real_t trans_max = 0.5;
	bool motor_enabled = false;
	real_t motor_target_vel = 0.0;
	real_t motor_max_force = 100.0;
	real_t motor_max_torque = 100.0;
};

} // namespace newton

#endif // NEWTON_JOINTS_CORKSCREW_H