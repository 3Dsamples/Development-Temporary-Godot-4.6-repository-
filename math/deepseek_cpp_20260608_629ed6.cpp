// File 192: modules/newton/src/joints/newton_kinematic_controller.h
// Kinematic controller – drives a dynamic body towards a target transform
// using spring‑damper forces on position and rotation.  Optionally respects
// joint limits.  Can be used for ragdolls driven by animation.

#ifndef NEWTON_JOINTS_KINEMATIC_CONTROLLER_H
#define NEWTON_JOINTS_KINEMATIC_CONTROLLER_H

#include "newton_joint.h"

namespace newton {

class NewtonKinematicController : public NewtonJoint {
	GDCLASS(NewtonKinematicController, NewtonJoint);

public:
	NewtonKinematicController() {
		joint_type = JointType::KINEMATIC;
	}

	// Set the target transform for the body (world space).
	void set_target_transform(const mat4 &p_target) { target_xform = p_target; }
	mat4 get_target_transform() const { return target_xform; }

	// Set maximum linear force the controller may apply.
	void set_max_linear_force(real_t p_force) { max_force = MAX(p_force, 0.0); }
	real_t get_max_linear_force() const { return max_force; }

	// Set maximum angular torque the controller may apply.
	void set_max_angular_torque(real_t p_torque) { max_torque = MAX(p_torque, 0.0); }
	real_t get_max_angular_torque() const { return max_torque; }

	// Set proportional gain for position error.
	void set_position_gain(real_t p_kp) { pos_kp = MAX(p_kp, 0.0); }
	real_t get_position_gain() const { return pos_kp; }

	// Set derivative gain for velocity error.
	void set_velocity_gain(real_t p_kd) { pos_kd = MAX(p_kd, 0.0); }
	real_t get_velocity_gain() const { return pos_kd; }

	// Set proportional gain for rotation error (torque scale).
	void set_rotation_gain(real_t p_kp) { rot_kp = MAX(p_kp, 0.0); }
	real_t get_rotation_gain() const { return rot_kp; }

	// Set derivative gain for angular velocity error.
	void set_angular_gain(real_t p_kd) { rot_kd = MAX(p_kd, 0.0); }
	real_t get_angular_gain() const { return rot_kd; }

	// Solve is called each substep – apply spring forces.
	virtual void solve(real_t dt) override;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_target_transform", "target"), &NewtonKinematicController::set_target_transform);
		ClassDB::bind_method(D_METHOD("get_target_transform"), &NewtonKinematicController::get_target_transform);
		ClassDB::bind_method(D_METHOD("set_max_linear_force", "force"), &NewtonKinematicController::set_max_linear_force);
		ClassDB::bind_method(D_METHOD("get_max_linear_force"), &NewtonKinematicController::get_max_linear_force);
		ClassDB::bind_method(D_METHOD("set_max_angular_torque", "torque"), &NewtonKinematicController::set_max_angular_torque);
		ClassDB::bind_method(D_METHOD("get_max_angular_torque"), &NewtonKinematicController::get_max_angular_torque);
		ClassDB::bind_method(D_METHOD("set_position_gain", "kp"), &NewtonKinematicController::set_position_gain);
		ClassDB::bind_method(D_METHOD("get_position_gain"), &NewtonKinematicController::get_position_gain);
		ClassDB::bind_method(D_METHOD("set_velocity_gain", "kd"), &NewtonKinematicController::set_velocity_gain);
		ClassDB::bind_method(D_METHOD("get_velocity_gain"), &NewtonKinematicController::get_velocity_gain);
		ClassDB::bind_method(D_METHOD("set_rotation_gain", "kp"), &NewtonKinematicController::set_rotation_gain);
		ClassDB::bind_method(D_METHOD("get_rotation_gain"), &NewtonKinematicController::get_rotation_gain);
		ClassDB::bind_method(D_METHOD("set_angular_gain", "kd"), &NewtonKinematicController::set_angular_gain);
		ClassDB::bind_method(D_METHOD("get_angular_gain"), &NewtonKinematicController::get_angular_gain);

		ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "target_transform"), "set_target_transform", "get_target_transform");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_linear_force"), "set_max_linear_force", "get_max_linear_force");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_angular_torque"), "set_max_angular_torque", "get_max_angular_torque");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "position_gain"), "set_position_gain", "get_position_gain");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "velocity_gain"), "set_velocity_gain", "get_velocity_gain");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rotation_gain"), "set_rotation_gain", "get_rotation_gain");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_gain"), "set_angular_gain", "get_angular_gain");
	}

private:
	mat4 target_xform;
	real_t max_force = 10000.0;
	real_t max_torque = 1000.0;
	real_t pos_kp = 500.0;
	real_t pos_kd = 50.0;
	real_t rot_kp = 1000.0;
	real_t rot_kd = 100.0;
};

} // namespace newton

#endif // NEWTON_JOINTS_KINEMATIC_CONTROLLER_H