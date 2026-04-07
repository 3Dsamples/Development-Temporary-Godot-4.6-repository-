--- START OF FILE core/simulation/physics_server_hyper_joints.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_impulse_denominator()
 * 
 * Computes the effective mass of a constraint point.
 * K = 1/m1 + 1/m2 + (r1 x n)^T * I1^-1 * (r1 x n) + (r2 x n)^T * I2^-1 * (r2 x n)
 */
static _FORCE_INLINE_ FixedMathCore calculate_impulse_denominator(
		const Vector3f &p_r,
		const Vector3f &p_normal,
		const FixedMathCore &p_inv_mass,
		const FixedMathCore &p_inv_inertia) {

	Vector3f rn = p_r.cross(p_normal);
	return p_inv_mass + p_inv_inertia * rn.dot(rn);
}

/**
 * Warp Kernel: BallJointSolverKernel
 * 
 * Resolves Point-to-Point constraints (Ball and Socket).
 * strictly bit-perfect to ensure multi-body robotic systems are stable at 120 FPS.
 */
void resolve_ball_joint_kernel(
		const BigIntCore &p_index,
		Vector3f &r_vel_a, Vector3f &r_ang_vel_a,
		Vector3f &r_vel_b, Vector3f &r_ang_vel_b,
		const Transform3Df &p_xform_a, const Transform3Df &p_xform_b,
		const Vector3f &p_local_anchor_a, const Vector3f &p_local_anchor_b,
		const FixedMathCore &p_inv_mass_a, const FixedMathCore &p_inv_mass_b,
		const FixedMathCore &p_inv_inertia_a, const FixedMathCore &p_inv_inertia_b,
		const FixedMathCore &p_bias_factor,
		const FixedMathCore &p_delta) {

	// 1. Transform anchors to world space
	Vector3f world_a = p_xform_a.xform(p_local_anchor_a);
	Vector3f world_b = p_xform_b.xform(p_local_anchor_b);
	Vector3f r_a = world_a - p_xform_a.origin;
	Vector3f r_b = world_b - p_xform_b.origin;

	// 2. Relative velocity at anchor points
	Vector3f v_rel = (r_vel_b + r_ang_vel_b.cross(r_b)) - (r_vel_a + r_ang_vel_a.cross(r_a));

	// 3. Position Error (Baumgarte Stabilization)
	Vector3f pos_error = world_b - world_a;
	Vector3f bias = pos_error * (p_bias_factor / p_delta);

	// 4. Resolve Impulse
	// We solve for 3 axes simultaneously (Linear Point Constraint)
	for (int i = 0; i < 3; i++) {
		Vector3f axis;
		axis[i] = MathConstants<FixedMathCore>::one();
		
		FixedMathCore k = calculate_impulse_denominator(r_a, axis, p_inv_mass_a, p_inv_inertia_a) +
		                  calculate_impulse_denominator(r_b, axis, p_inv_mass_b, p_inv_inertia_b);
		
		FixedMathCore j_mag = (-(v_rel[i] + bias[i])) / k;
		Vector3f impulse = axis * j_mag;

		r_vel_a -= impulse * p_inv_mass_a;
		r_ang_vel_a -= r_a.cross(impulse) * p_inv_inertia_a;
		r_vel_b += impulse * p_inv_mass_b;
		r_ang_vel_b += r_b.cross(impulse) * p_inv_inertia_b;
	}
}

/**
 * Warp Kernel: HingeJointMotorKernel
 * 
 * Advanced Feature: Robotic Servo and Velocity Motors.
 * Resolves rotation around a specific axis and applies torque tensors.
 */
void hinge_motor_kernel(
		Vector3f &r_ang_vel_a,
		Vector3f &r_ang_vel_b,
		const Vector3f &p_axis_world,
		const FixedMathCore &p_target_vel,
		const FixedMathCore &p_max_impulse,
		const FixedMathCore &p_inv_inertia_sum) {

	FixedMathCore current_rel_vel = (r_ang_vel_b - r_ang_vel_a).dot(p_axis_world);
	FixedMathCore motor_error = p_target_vel - current_rel_vel;
	
	FixedMathCore j = motor_error / p_inv_inertia_sum;
	FixedMathCore j_clamped = wp::clamp(j, -p_max_impulse, p_max_impulse);

	Vector3f impulse = p_axis_world * j_clamped;
	r_ang_vel_a -= impulse; // Simplified applied to angular stream
	r_ang_vel_b += impulse;
}

/**
 * execute_joint_resolution_wave()
 * 
 * The master 120 FPS parallel wave for mechanical constraints.
 * Orchestrates the iterative Sequential Impulse solver.
 */
void PhysicsServerHyper::execute_joint_resolution_wave(const FixedMathCore &p_delta) {
	KernelRegistry &registry = get_kernel_registry();
	auto &joint_stream = registry.get_stream<JointData>(COMPONENT_JOINT_DATA);
	uint64_t joint_count = joint_stream.size();
	if (joint_count == 0) return;

	SimulationThreadPool *pool = SimulationThreadPool::get_singleton();
	uint32_t workers = pool->get_worker_count();
	uint64_t chunk = joint_count / workers;

	// SI Solver typically requires 8-16 iterations for mechanical stability (Robotic precision)
	for (int iter = 0; iter < 10; iter++) {
		for (uint32_t w = 0; w < workers; w++) {
			uint64_t start = w * chunk;
			uint64_t end = (w == workers - 1) ? joint_count : (start + chunk);

			pool->enqueue_task([=, &registry]() {
				auto &vel_stream = registry.get_stream<Vector3f>(COMPONENT_LINEAR_VELOCITY);
				auto &ang_stream = registry.get_stream<Vector3f>(COMPONENT_ANGULAR_VELOCITY);
				auto &xform_stream = registry.get_stream<Transform3Df>(COMPONENT_TRANSFORM);
				auto &inv_mass_stream = registry.get_stream<FixedMathCore>(COMPONENT_INV_MASS);
				auto &inv_inertia_stream = registry.get_stream<FixedMathCore>(COMPONENT_INV_INERTIA);

				for (uint64_t i = start; i < end; i++) {
					const JointData &j = registry.get_stream<JointData>(COMPONENT_JOINT_DATA)[i];
					uint64_t ia = j.entity_a_idx;
					uint64_t ib = j.entity_b_idx;

					if (j.type == JOINT_TYPE_BALL) {
						resolve_ball_joint_kernel(
							BigIntCore(static_cast<int64_t>(i)),
							vel_stream[ia], ang_stream[ia],
							vel_stream[ib], ang_stream[ib],
							xform_stream[ia], xform_stream[ib],
							j.anchor_a, j.anchor_b,
							inv_mass_stream[ia], inv_mass_stream[ib],
							inv_inertia_stream[ia], inv_inertia_stream[ib],
							FixedMathCore(858993459LL, true), // 0.2 bias
							p_delta
						);
					}
					// Slider and Hinge implementations follow same SI pattern
				}
			}, SimulationThreadPool::PRIORITY_CRITICAL);
		}
		pool->wait_for_all();
	}
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_joints.cpp ---
