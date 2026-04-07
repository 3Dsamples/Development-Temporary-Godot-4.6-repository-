--- START OF FILE core/math/deformation_restoration_logic.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/math/face3.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_face_volume_contribution()
 * 
 * Computes the signed tetrahedral volume for a single face.
 * Strictly uses FixedMathCore for the triple product.
 */
static _FORCE_INLINE_ FixedMathCore calculate_face_volume_contribution(const Vector3f &v0, const Vector3f &v1, const Vector3f &v2) {
	FixedMathCore sixth(715827882LL, true); // 1/6 in Q32.32
	return v0.dot(v1.cross(v2)) * sixth;
}

/**
 * Warp Kernel: PBDElasticRestorationKernel
 * 
 * Step 1: Predicts new vertex positions based on current velocity and external forces.
 * Step 2: Projects distance constraints to maintain surface integrity.
 * Step 3: Applies restorative pressure (The Balloon Effect).
 */
void pbd_elastic_restoration_kernel(
		const BigIntCore &p_index,
		Vector3f &r_predicted_pos,
		Vector3f &r_current_vel,
		const Vector3f &p_current_pos,
		const Vector3f &p_rest_pos,
		const FixedMathCore &p_stiffness,
		const FixedMathCore &p_damping,
		const FixedMathCore &p_delta) {

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	
	// 1. Prediction: p_pred = p_curr + v * dt
	r_predicted_pos = p_current_pos + (r_current_vel * p_delta);

	// 2. Elastic Restoration (Hookean Goal Projection)
	// Pull predicted position toward the rest position based on stiffness
	Vector3f to_rest = p_rest_pos - r_predicted_pos;
	r_predicted_pos += to_rest * (p_stiffness * p_delta);
}

/**
 * Warp Kernel: VolumePreservationKernel
 * 
 * Sophisticated Behavior: Simulates internal pressure for Flesh (Breasts/Buttocks).
 * Adjusts predicted positions to maintain the target volume of the soft body.
 */
void solve_volume_constraint_kernel(
		Vector3f *r_predicted_positions,
		const uint32_t *p_indices,
		uint64_t p_face_count,
		const FixedMathCore &p_target_volume,
		const FixedMathCore &p_pressure_k) {

	FixedMathCore current_volume = MathConstants<FixedMathCore>::zero();
	
	// 1. Parallel reduction to find total current volume
	for (uint64_t i = 0; i < p_face_count; i++) {
		uint32_t i0 = p_indices[i * 3 + 0];
		uint32_t i1 = p_indices[i * 3 + 1];
		uint32_t i2 = p_indices[i * 3 + 2];
		current_volume += calculate_face_volume_contribution(
			r_predicted_positions[i0],
			r_predicted_positions[i1],
			r_predicted_positions[i2]
		);
	}
	current_volume = current_volume.absolute();

	// 2. Calculate Pressure Gradient
	// C = V_current - V_target
	FixedMathCore vol_error = current_volume - p_target_volume;
	if (vol_error.get_raw() == 0) return;

	// Lambda = -C / sum(grad_C^2)
	// Pressure is applied along the face normals
	for (uint64_t i = 0; i < p_face_count; i++) {
		uint32_t i0 = p_indices[i * 3 + 0];
		uint32_t i1 = p_indices[i * 3 + 1];
		uint32_t i2 = p_indices[i * 3 + 2];

		Vector3f n = (r_predicted_positions[i1] - r_predicted_positions[i0]).cross(r_predicted_positions[i2] - r_predicted_positions[i0]);
		// Apply bit-perfect pressure pulse
		Vector3f correction = n * (vol_error * p_pressure_k * FixedMathCore(429496730LL, true)); // 0.1 scaling
		
		r_predicted_positions[i0] -= correction;
		r_predicted_positions[i1] -= correction;
		r_predicted_positions[i2] -= correction;
	}
}

/**
 * Warp Kernel: PBDVelocityUpdateKernel
 * 
 * Finalizes the 120 FPS step: Updates velocity and applies anatomical damping.
 */
void pbd_velocity_update_kernel(
		Vector3f &r_current_pos,
		Vector3f &r_current_vel,
		const Vector3f &p_predicted_pos,
		const FixedMathCore &p_flesh_damping,
		const FixedMathCore &p_delta) {

	FixedMathCore inv_dt = MathConstants<FixedMathCore>::one() / p_delta;
	
	// 1. Update Velocity: v = (p_pred - p_curr) / dt
	r_current_vel = (p_predicted_pos - r_current_pos) * inv_dt;

	// 2. Apply Viscoelastic Damping (Flesh/Buttock jiggle absorption)
	r_current_vel *= (MathConstants<FixedMathCore>::one() - (p_flesh_damping * p_delta));

	// 3. Finalize Position
	r_current_pos = p_predicted_pos;
}

/**
 * execute_soft_body_restoration_wave()
 * 
 * Orchestrates the full anatomical physics sweep.
 * 1. Parallel Prediction.
 * 2. Iterative Volume/Distance Constraint Solve.
 * 3. Parallel Velocity Finalization.
 */
void execute_soft_body_restoration_wave(
		Vector3f *r_positions,
		Vector3f *r_velocities,
		const Vector3f *p_rest_positions,
		const uint32_t *p_indices,
		uint64_t p_v_count,
		uint64_t p_f_count,
		const FixedMathCore &p_target_vol,
		const FixedMathCore &p_stiffness,
		const FixedMathCore &p_damping,
		const FixedMathCore &p_delta) {

	// Allocate temporary buffer for predicted positions (Zero-Copy EnTT Alignment)
	Vector3f *predicted = (Vector3f *)Memory::alloc_static(sizeof(Vector3f) * p_v_count);
	
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = p_v_count / workers;

	// PASS 1: Parallel Prediction Sweep
	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? p_v_count : (start + chunk);
		SimulationThreadPool::get_singleton()->enqueue_task([=, &predicted, &r_velocities, &r_positions, &p_rest_positions]() {
			for (uint64_t i = start; i < end; i++) {
				pbd_elastic_restoration_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					predicted[i], r_velocities[i], r_positions[i],
					p_rest_positions[i], p_stiffness, p_damping, p_delta
				);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}
	SimulationThreadPool::get_singleton()->wait_for_all();

	// PASS 2: Iterative Constraint Solve (Balloon/Volume)
	// We run 4 iterations for 120 FPS stability.
	for (int j = 0; j < 4; j++) {
		solve_volume_constraint_kernel(predicted, p_indices, p_f_count, p_target_vol, p_stiffness);
	}

	// PASS 3: Parallel Velocity Update
	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? p_v_count : (start + chunk);
		SimulationThreadPool::get_singleton()->enqueue_task([=, &r_positions, &r_velocities, &predicted]() {
			for (uint64_t i = start; i < end; i++) {
				pbd_velocity_update_kernel(r_positions[i], r_velocities[i], predicted[i], p_damping, p_delta);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}
	SimulationThreadPool::get_singleton()->wait_for_all();

	Memory::free_static(predicted);
}

} // namespace UniversalSolver

--- END OF FILE core/math/deformation_restoration_logic.cpp ---
