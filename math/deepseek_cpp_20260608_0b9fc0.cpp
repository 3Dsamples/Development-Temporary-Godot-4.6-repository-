// File 171: modules/genesis/src/entities/tool_entity.cpp
// Tool entity implementation: control mode logic, PID force computation,
// and property bindings.

#include "tool_entity.h"

#include "../core/genesis_types.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"

namespace genesis {

void ToolEntity::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_control_mode", "mode"), &ToolEntity::set_control_mode);
	ClassDB::bind_method(D_METHOD("get_control_mode"), &ToolEntity::get_control_mode);
	ClassDB::bind_method(D_METHOD("set_target_position", "pos"), &ToolEntity::set_target_position);
	ClassDB::bind_method(D_METHOD("get_target_position"), &ToolEntity::get_target_position);
	ClassDB::bind_method(D_METHOD("set_target_rotation", "basis"), &ToolEntity::set_target_rotation);
	ClassDB::bind_method(D_METHOD("get_target_rotation"), &ToolEntity::get_target_rotation);
	ClassDB::bind_method(D_METHOD("set_target_velocity", "vel"), &ToolEntity::set_target_velocity);
	ClassDB::bind_method(D_METHOD("get_target_velocity"), &ToolEntity::get_target_velocity);
	ClassDB::bind_method(D_METHOD("set_target_angular_velocity", "omega"), &ToolEntity::set_target_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_target_angular_velocity"), &ToolEntity::get_target_angular_velocity);
	ClassDB::bind_method(D_METHOD("set_pid_kp", "kp"), &ToolEntity::set_pid_kp);
	ClassDB::bind_method(D_METHOD("get_pid_kp"), &ToolEntity::get_pid_kp);
	ClassDB::bind_method(D_METHOD("set_pid_kd", "kd"), &ToolEntity::set_pid_kd);
	ClassDB::bind_method(D_METHOD("get_pid_kd"), &ToolEntity::get_pid_kd);
	ClassDB::bind_method(D_METHOD("set_pid_ki", "ki"), &ToolEntity::set_pid_ki);
	ClassDB::bind_method(D_METHOD("get_pid_ki"), &ToolEntity::get_pid_ki);
	ClassDB::bind_method(D_METHOD("set_max_force", "force"), &ToolEntity::set_max_force);
	ClassDB::bind_method(D_METHOD("get_max_force"), &ToolEntity::get_max_force);
	ClassDB::bind_method(D_METHOD("set_max_torque", "torque"), &ToolEntity::set_max_torque);
	ClassDB::bind_method(D_METHOD("get_max_torque"), &ToolEntity::get_max_torque);
	ClassDB::bind_method(D_METHOD("compute_control_forces", "dt"), &ToolEntity::compute_control_forces);

	BIND_ENUM_CONSTANT(POSITION_CONTROL);
	BIND_ENUM_CONSTANT(VELOCITY_CONTROL);
	BIND_ENUM_CONSTANT(FORCE_TORQUE);
	BIND_ENUM_CONSTANT(PID_TARGET);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "control_mode", PROPERTY_HINT_ENUM, "Position,Velocity,ForceTorque,PID"), "set_control_mode", "get_control_mode");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "target_position"), "set_target_position", "get_target_position");
	ADD_PROPERTY(PropertyInfo(Variant::BASIS, "target_rotation"), "set_target_rotation", "get_target_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "target_velocity"), "set_target_velocity", "get_target_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "target_angular_velocity"), "set_target_angular_velocity", "get_target_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "pid_kp", PROPERTY_HINT_RANGE, "0,10000,0.1"), "set_pid_kp", "get_pid_kp");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "pid_kd", PROPERTY_HINT_RANGE, "0,10000,0.1"), "set_pid_kd", "get_pid_kd");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "pid_ki", PROPERTY_HINT_RANGE, "0,1000,0.01"), "set_pid_ki", "get_pid_ki");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_force", PROPERTY_HINT_RANGE, "0,1e10,1"), "set_max_force", "get_max_force");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_torque", PROPERTY_HINT_RANGE, "0,1e10,1"), "set_max_torque", "get_max_torque");
}

void ToolEntity::compute_control_forces(real_t dt) {
	switch (control_mode) {
		case POSITION_CONTROL: {
			// Simple spring-damper to target position
			Vector3 pos_error = target_position - transform.origin;
			Vector3 vel_error = target_velocity - linear_velocity;
			Vector3 force = pid_kp * pos_error + pid_kd * vel_error;
			real_t f_len = force.length();
			if (f_len > max_force && max_force > 0.0f) force = force / f_len * max_force;
			apply_force(force);

			// Angular spring to target rotation
			Quaternion q_cur(transform.basis);
			Quaternion q_tar(target_rotation);
			Quaternion q_diff = q_tar * q_cur.inverse();
			Vector3 axis; real_t angle;
			q_diff.get_axis_angle(axis, angle);
			Vector3 torque = axis * angle * pid_kp;
			Vector3 angvel_error = target_angular_velocity - angular_velocity;
			torque += angvel_error * pid_kd;
			real_t t_len = torque.length();
			if (t_len > max_torque && max_torque > 0.0f) torque = torque / t_len * max_torque;
			apply_force(Vector3(), torque);
			break;
		}
		case PID_TARGET: {
			// PID with integral accumulation
			Vector3 pos_error = target_position - transform.origin;
			integral_error_pos += pos_error * dt;
			Vector3 force = pid_kp * pos_error + pid_ki * integral_error_pos + pid_kd * (-linear_velocity);
			real_t f_len = force.length();
			if (f_len > max_force && max_force > 0.0f) force = force / f_len * max_force;
			apply_force(force);

			Quaternion q_cur(transform.basis);
			Quaternion q_tar(target_rotation);
			Quaternion q_diff = q_tar * q_cur.inverse();
			Vector3 axis; real_t angle;
			q_diff.get_axis_angle(axis, angle);
			integral_error_rot += axis * angle * dt;
			Vector3 torque = pid_kp * axis * angle + pid_ki * integral_error_rot + pid_kd * (-angular_velocity);
			real_t t_len = torque.length();
			if (t_len > max_torque && max_torque > 0.0f) torque = torque / t_len * max_torque;
			apply_force(Vector3(), torque);
			break;
		}
		default: break;
	}
}

} // namespace genesis