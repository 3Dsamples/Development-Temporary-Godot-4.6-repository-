// File 74: modules/genesis/src/entities/tool_entity.h
// Tool entity: a controlled rigid body used for interaction with deformable bodies.
// Supports spring-based PID control, force/torque application, and sensor attachment.

#ifndef GENESIS_ENTITIES_TOOL_ENTITY_H
#define GENESIS_ENTITIES_TOOL_ENTITY_H

#include "rigid_entity.h"
#include "../core/genesis_types.h"
#include "../core/genesis_constants.h"
#include "core/math/transform_3d.h"

namespace genesis {

class ToolEntity : public RigidEntity {
	GDCLASS(ToolEntity, RigidEntity);

public:
	enum ControlMode {
		POSITION_CONTROL = 0,
		VELOCITY_CONTROL,
		FORCE_TORQUE,
		PID_TARGET
	};

	ToolEntity() :
		control_mode(POSITION_CONTROL),
		target_position(Vector3()),
		target_rotation(Basis()),
		target_velocity(Vector3()),
		target_angular_velocity(Vector3()),
		pid_kp(100.0),
		pid_kd(10.0),
		pid_ki(0.01),
		max_force(INFINITY),
		max_torque(INFINITY),
		integral_error_pos(Vector3()),
		integral_error_rot(Vector3()) {}

	// --- Control settings ---
	void set_control_mode(ControlMode p_mode) { control_mode = p_mode; }
	ControlMode get_control_mode() const { return control_mode; }

	void set_target_position(const Vector3 &p_pos) { target_position = p_pos; }
	Vector3 get_target_position() const { return target_position; }

	void set_target_rotation(const Basis &p_basis) { target_rotation = p_basis; }
	Basis get_target_rotation() const { return target_rotation; }

	void set_target_velocity(const Vector3 &p_vel) { target_velocity = p_vel; }
	Vector3 get_target_velocity() const { return target_velocity; }

	void set_target_angular_velocity(const Vector3 &p_omega) { target_angular_velocity = p_omega; }
	Vector3 get_target_angular_velocity() const { return target_angular_velocity; }

	// --- PID gains ---
	void set_pid_kp(real_t p_val) { pid_kp = MAX(p_val, 0.0); }
	real_t get_pid_kp() const { return pid_kp; }

	void set_pid_kd(real_t p_val) { pid_kd = MAX(p_val, 0.0); }
	real_t get_pid_kd() const { return pid_kd; }

	void set_pid_ki(real_t p_val) { pid_ki = MAX(p_val, 0.0); }
	real_t get_pid_ki() const { return pid_ki; }

	void set_max_force(real_t p_f) { max_force = MAX(p_f, 0.0); }
	real_t get_max_force() const { return max_force; }

	void set_max_torque(real_t p_t) { max_torque = MAX(p_t, 0.0); }
	real_t get_max_torque() const { return max_torque; }

	// --- Compute control forces based on current mode ---
	void compute_control_forces(real_t dt) {
		switch (control_mode) {
			case POSITION_CONTROL: {
				// Spring-damper to target pose
				Vector3 pos_error = target_position - transform.origin;
				Vector3 vel_error = target_velocity - linear_velocity;
				Vector3 force = pid_kp * pos_error + pid_kd * vel_error;
				real_t f_len = force.length();
				if (f_len > max_force && max_force > 0) force = force / f_len * max_force;
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
				if (t_len > max_torque && max_torque > 0) torque = torque / t_len * max_torque;
				apply_force(Vector3(), torque);
				break;
			}
			case PID_TARGET: {
				// PID with integral term
				Vector3 pos_error = target_position - transform.origin;
				integral_error_pos += pos_error * dt;
				Vector3 force = pid_kp * pos_error + pid_ki * integral_error_pos + pid_kd * (-linear_velocity);
				real_t f_len = force.length();
				if (f_len > max_force && max_force > 0) force = force / f_len * max_force;
				apply_force(force);

				Quaternion q_cur(transform.basis);
				Quaternion q_tar(target_rotation);
				Quaternion q_diff = q_tar * q_cur.inverse();
				Vector3 axis; real_t angle;
				q_diff.get_axis_angle(axis, angle);
				integral_error_rot += axis * angle * dt;
				Vector3 torque = pid_kp * axis * angle + pid_ki * integral_error_rot + pid_kd * (-angular_velocity);
				real_t t_len = torque.length();
				if (t_len > max_torque && max_torque > 0) torque = torque / t_len * max_torque;
				apply_force(Vector3(), torque);
				break;
			}
			default: break;
		}
	}

	// Override apply_force to also accept direct torque
	void apply_force(const Vector3 &p_force, const Vector3 &p_torque = Vector3()) {
		force_accum += p_force;
		torque_accum += p_torque;
	}

	virtual void init_from_options(const genesis::options::Options &opts) override {
		RigidEntity::init_from_options(opts);
		control_mode = ControlMode(opts.get_int("tool.control_mode", int(POSITION_CONTROL)));
		target_position = opts.get_vector3("tool.target_pos", target_position);
		pid_kp = opts.get_real("tool.kp", pid_kp);
		pid_kd = opts.get_real("tool.kd", pid_kd);
		pid_ki = opts.get_real("tool.ki", pid_ki);
		max_force = opts.get_real("tool.max_force", max_force);
		max_torque = opts.get_real("tool.max_torque", max_torque);
	}

protected:
	static void _bind_methods() {
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

private:
	ControlMode control_mode;
	Vector3 target_position;
	Basis target_rotation;
	Vector3 target_velocity;
	Vector3 target_angular_velocity;
	real_t pid_kp, pid_kd, pid_ki;
	real_t max_force, max_torque;
	Vector3 integral_error_pos;
	Vector3 integral_error_rot;
};

} // namespace genesis

#endif // GENESIS_ENTITIES_TOOL_ENTITY_H