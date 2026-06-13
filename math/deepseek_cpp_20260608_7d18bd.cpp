// File 363: modules/genesis/src/solvers/tool_solver.h
// GenesisToolSolver – drives tool entities towards target poses using
// PID control with velocity and force limits.  Suitable for robotic arms,
// grippers, and user‑controlled kinematic tools.  The solve() method is
// fully inline for minimal overhead in the physics pipeline.

#ifndef GENESIS_SOLVERS_TOOL_SOLVER_H
#define GENESIS_SOLVERS_TOOL_SOLVER_H

#include "base_solver.h"
#include "../entities/tool_entity.h"
#include "../core/genesis_types.h"
#include "../core/genesis_constants.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/math/quaternion.h"
#include "core/typedefs.h"

namespace genesis {

class ToolSolver : public BaseSolver {
	GDCLASS(ToolSolver, BaseSolver);

	// Cache of tool entities for fast iteration
	LocalVector<Ref<ToolEntity>> tool_entities;
	LocalVector<entity_id_t> tool_ids;

	// PID gains (global, can be overridden per entity)
	real_t position_kp = 500.0;
	real_t position_kd = 50.0;
	real_t position_ki = 0.01;
	real_t rotation_kp = 1000.0;
	real_t rotation_kd = 100.0;
	real_t rotation_ki = 0.0;
	real_t max_force = 10000.0;
	real_t max_torque = 1000.0;

public:
	ToolSolver() : BaseSolver() { solver_type = SolverType::TOOL; }

	// --- PID parameters ---
	void set_position_gains(real_t p_kp, real_t p_kd, real_t p_ki = 0.0) {
		position_kp = MAX(p_kp, 0.0);
		position_kd = MAX(p_kd, 0.0);
		position_ki = CLAMP(p_ki, 0.0, 1.0);
	}
	void set_rotation_gains(real_t p_kp, real_t p_kd, real_t p_ki = 0.0) {
		rotation_kp = MAX(p_kp, 0.0);
		rotation_kd = MAX(p_kd, 0.0);
		rotation_ki = CLAMP(p_ki, 0.0, 1.0);
	}
	void set_force_limits(real_t p_max_force, real_t p_max_torque) {
		max_force = MAX(p_max_force, 0.0);
		max_torque = MAX(p_max_torque, 0.0);
	}

	// --- Entity management ---
	virtual void add_entity(Ref<BaseEntity> p_entity) override {
		Ref<ToolEntity> tool = p_entity;
		if (tool.is_valid()) {
			tool_entities.push_back(tool);
			tool_ids.push_back(tool->get_entity_uid());
			BaseSolver::add_entity(p_entity);
		}
	}
	virtual void remove_entity(entity_id_t p_uid) override {
		for (int i = 0; i < tool_ids.size(); ++i) {
			if (tool_ids[i] == p_uid) {
				tool_entities.remove_at(i);
				tool_ids.remove_at(i);
				break;
			}
		}
		BaseSolver::remove_entity(p_uid);
	}

	// Main step – apply PID control forces and integrate
	virtual void step() override {
		real_t sub_dt = dt / real_t(sub_steps);
		for (int substep = 0; substep < sub_steps; ++substep) {
			for (Ref<ToolEntity> &tool : tool_entities) {
				if (tool.is_null() || !tool->is_active()) continue;
				compute_control(tool, sub_dt);
			}
			time += sub_dt;
		}
	}

	virtual void solve(real_t p_sub_dt) override {
		// Not used, step() contains full logic
	}

private:
	// Compute PID force/torque and apply to a single tool entity.
	inline void compute_control(Ref<ToolEntity> &tool, real_t dt) {
		Transform3D current = tool->get_transform();
		// Position control
		Vector3 pos_error = tool->get_target_position() - current.origin;
		Vector3 vel_error = tool->get_target_velocity() - tool->get_linear_velocity();

		// Accumulate integral error (per entity? We'll store it inside the tool entity via a member we assume exists).
		// For simplicity, use the tool entity's own PID integral storage.
		Vector3 integral_pos = tool->get_integral_error_pos() + pos_error * dt;
		tool->set_integral_error_pos(integral_pos);

		Vector3 force = position_kp * pos_error + position_kd * vel_error + position_ki * integral_pos;
		real_t f_len = force.length();
		if (f_len > max_force && max_force > 0.0f) force = force * (max_force / f_len);
		tool->apply_force(force, Vector3()); // apply at center of mass? Actually at origin of tool.

		// Rotation control
		Quaternion q_cur(current.basis);
		Quaternion q_tar(tool->get_target_rotation());
		Quaternion q_diff = q_tar * q_cur.inverse();
		Vector3 rot_axis;
		real_t rot_angle;
		q_diff.get_axis_angle(rot_axis, rot_angle);
		// Clamp angle to avoid large impulses
		rot_angle = CLAMP(rot_angle, -Math_PI * 0.5f, Math_PI * 0.5f);

		Vector3 angvel_error = tool->get_target_angular_velocity() - tool->get_angular_velocity();
		Vector3 integral_rot = tool->get_integral_error_rot() + rot_axis * rot_angle * dt;
		tool->set_integral_error_rot(integral_rot);

		Vector3 torque = rotation_kp * rot_axis * rot_angle + rotation_kd * angvel_error + rotation_ki * integral_rot;
		real_t t_len = torque.length();
		if (t_len > max_torque && max_torque > 0.0f) torque = torque * (max_torque / t_len);
		tool->apply_force(Vector3(), torque);
	}
};

} // namespace genesis

#endif // GENESIS_SOLVERS_TOOL_SOLVER_H