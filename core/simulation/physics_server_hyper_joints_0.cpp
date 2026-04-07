--- START OF FILE core/simulation/physics_server_hyper_joints.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: JointConstraintSolverKernel
 * 
 * Performs bit-perfect resolution of mechanical constraints.
 * Processes joints in parallel batches using Sequential Impulses.
 * Supports Motors (Linear/Angular) and high-torque mechanical links.
 */
void resolve_joints_batch_kernel(
		const BigIntCore &p_index,
		Body *p_body_a,
		Body *p_body_b,
		const JointData &p_joint,
		const FixedMathCore &p_inv_dt) {

	if (!p_body_a->active && !p_body_b->active) return;

	// 1. Transform Anchor Points to World Space (Deterministic)
	Vector3f anchor_a = p_body_a->transform.xform(p_joint.local_anchor_a);
	Vector3f anchor_b = p_body_b->transform.xform(p_joint.local_anchor_b);
	Vector3f rel_pos = anchor_b - anchor_a;

	// 2. Velocity Constraint Resolution
	Vector3f arm_a = anchor_a - p_body_a->transform.origin;
	Vector3f arm_b = anchor_b - p_body_b->transform.origin;

	Vector3f v_a = p_body_a->linear_velocity + p_body_a->angular_velocity.cross(arm_a);
	Vector3f v_b = p_body_b->linear_velocity + p_body_b->angular_velocity.cross(arm_b);
	Vector3f rel_vel = v_b - v_a;

	// 3. Advanced Motor Logic (Servo/Velocity)
	if (p_joint.motor_enabled) {
		Vector3f motor_impulse;
		if (p_joint.motor_type == MOTOR_VELOCITY) {
			Vector3f target_v = p_joint.motor_axis * p_joint.target_velocity;
			motor_impulse = (target_v - rel_vel) * p_joint.motor_strength;
		} else if (p_joint.motor_type == MOTOR_SERVO) {
			// Deterministic PD Controller: F = Kp*error + Kd*v_diff
			FixedMathCore pos_error = p_joint.target_position - p_joint.current_position;
			FixedMathCore force = (p_joint.kp * pos_error) - (p_joint.kd * rel_vel.dot(p_joint.motor_axis));
			motor_impulse = p_joint.motor_axis * force;
		}
		
		p_body_a->linear_velocity -= motor_impulse * (p_body_a->mass.get_raw() == 0 ? FixedMathCore(0LL, true) : MathConstants<FixedMathCore>::one() / p_body_a->mass);
		p_body_b->linear_velocity += motor_impulse * (p_body_b->mass.get_raw() == 0 ? FixedMathCore(0LL, true) : MathConstants<FixedMathCore>::one() / p_body_b->mass);
	}

	// 4. Position Correction (Baumgarte Stabilization)
	FixedMathCore bias_factor = FixedMathCore(858993459LL, true); // 0.2
	Vector3f impulse = rel_pos * (bias_factor * p_inv_dt);

	// 5. Apply Corrective Impulses
	if (p_body_a->mode != PhysicsServerHyper::BODY_MODE_STATIC) {
		p_body_a->linear_velocity += impulse * (MathConstants<FixedMathCore>::one() / p_body_a->mass);
		p_body_a->angular_velocity += arm_a.cross(impulse) * (MathConstants<FixedMathCore>::one() / p_body_a->mass); // Simplified inertia
	}
	if (p_body_b->mode != PhysicsServerHyper::BODY_MODE_STATIC) {
		p_body_b->linear_velocity -= impulse * (MathConstants<FixedMathCore>::one() / p_body_b->mass);
		p_body_b->angular_velocity -= arm_b.cross(impulse) * (MathConstants<FixedMathCore>::one() / p_body_b->mass);
	}
}

/**
 * solve_joints_parallel()
 * 
 * Master orchestrator for joint batches.
 * Divides EnTT joint components into chunks for the SimulationThreadPool.
 */
void PhysicsServerHyper::solve_joints_parallel(const FixedMathCore &p_delta) {
	auto &registry = get_kernel_registry();
	uint64_t joint_count = registry.get_stream<JointData>().size();
	if (joint_count == 0) return;

	FixedMathCore inv_dt = MathConstants<FixedMathCore>::one() / p_delta;
	uint32_t worker_threads = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk_size = joint_count / worker_threads;

	for (uint32_t w = 0; w < worker_threads; w++) {
		uint64_t start = w * chunk_size;
		uint64_t end = (w == worker_threads - 1) ? joint_count : (w + 1) * chunk_size;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				JointData &j = registry.get_stream<JointData>()[i];
				Body *ba = body_owner.get_or_null(j.body_a);
				Body *bb = body_owner.get_or_null(j.body_b);
				
				if (ba && bb) {
					resolve_joints_batch_kernel(BigIntCore(static_cast<int64_t>(i)), ba, bb, j, inv_dt);
				}
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * body_add_collision_exception()
 * 
 * Essential for joints to prevent parent/child self-intersection.
 */
void PhysicsServerHyper::body_add_collision_exception(RID p_body, RID p_exception) {
	Body *b = body_owner.get_or_null(p_body);
	if (b) {
		b->collision_exceptions.insert(p_exception);
	}
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_joints.cpp ---
