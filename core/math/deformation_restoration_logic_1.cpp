--- START OF FILE core/math/deformation_restoration_logic.cpp ---

#include "core/math/dynamic_mesh.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_tetrahedral_volume()
 * Computes the signed volume of a single tetrahedron formed by origin and a face.
 * Used for deterministic volume preservation.
 */
static _FORCE_INLINE_ FixedMathCore calculate_tetrahedral_volume(const Vector3f &v0, const Vector3f &v1, const Vector3f &v2) {
	FixedMathCore sixth(715827882LL, true); // 1/6
	return v0.dot(v1.cross(v2)) * sixth;
}

/**
 * Warp Kernel: PBD_PositionPrediction
 * Step 1: Integrates external forces and predicts the next unconstrained position.
 */
void pbd_predict_kernel(
		const BigIntCore &p_index,
		Vector3f &r_predicted_pos,
		Vector3f &r_velocity,
		const Vector3f &p_current_pos,
		const Vector3f &p_external_accel,
		const FixedMathCore &p_delta) {

	r_velocity += p_external_accel * p_delta;
	r_predicted_pos = p_current_pos + (r_velocity * p_delta);
}

/**
 * Warp Kernel: VolumeConstraintKernel
 * Step 2: Projects vertices to satisfy volume preservation (Balloon Effect).
 * Computes the global volume and applies a pressure-based correction tensor.
 */
void pbd_volume_solve_kernel(
		Vector3f *r_predicted_positions,
		const uint32_t *p_indices,
		uint64_t p_face_count,
		const FixedMathCore &p_target_volume,
		const FixedMathCore &p_stiffness,
		const FixedMathCore &p_delta) {

	FixedMathCore current_volume = MathConstants<FixedMathCore>::zero();
	
	// 1. Accumulate signed volumes (Parallel Reduction logic applied here)
	for (uint64_t i = 0; i < p_face_count; i++) {
		current_volume += calculate_tetrahedral_volume(
			r_predicted_positions[p_indices[i * 3 + 0]],
			r_predicted_positions[p_indices[i * 3 + 1]],
			r_predicted_positions[p_indices[i * 3 + 2]]
		);
	}
	current_volume = current_volume.absolute();

	// 2. Resolve volume error
	FixedMathCore vol_error = current_volume - p_target_volume;
	if (wp::abs(vol_error) < FixedMathCore(4294LL, true)) return;

	// 3. Apply Correction (Outward Normal Pressure)
	FixedMathCore pressure = -vol_error * p_stiffness;
	for (uint64_t i = 0; i < p_face_count; i++) {
		uint32_t i0 = p_indices[i * 3 + 0];
		uint32_t i1 = p_indices[i * 3 + 1];
		uint32_t i2 = p_indices[i * 3 + 2];

		Vector3f v0 = r_predicted_positions[i0];
		Vector3f v1 = r_predicted_positions[i1];
		Vector3f v2 = r_predicted_positions[i2];

		// Gradient of volume is the face normal
		Vector3f normal = (v1 - v0).cross(v2 - v0);
		Vector3f correction = normal * (pressure * p_delta);

		r_predicted_positions[i0] += correction;
		r_predicted_positions[i1] += correction;
		r_predicted_positions[i2] += correction;
	}
}

/**
 * Warp Kernel: PBD_VelocityFinalize
 * Step 3: Resolves final velocities and applies anatomical damping.
 */
void pbd_finalize_kernel(
		const BigIntCore &p_index,
		Vector3f &r_current_pos,
		Vector3f &r_current_vel,
		const Vector3f &p_predicted_pos,
		const FixedMathCore &p_flesh_damping,
		const FixedMathCore &p_delta) {

	FixedMathCore inv_dt = MathConstants<FixedMathCore>::one() / p_delta;
	
	// Update velocity based on position change: v = (p_new - p_old) / dt
	r_current_vel = (p_predicted_pos - r_current_pos) * inv_dt;

	// Apply anatomical damping (Viscoelastic response)
	FixedMathCore damp_factor = MathConstants<FixedMathCore>::one() - (p_flesh_damping * p_delta);
	r_current_vel *= wp::max(MathConstants<FixedMathCore>::zero(), damp_factor);

	// Commit final bit-perfect position
	r_current_pos = p_predicted_pos;
}

/**
 * execute_soft_body_restoration_wave()
 * 
 * Master orchestrator for anatomical physics (Breasts, Buttocks, Flesh).
 * 1. Parallel Position Prediction.
 * 2. Iterative Volume Constraint Solver (4 passes for 120 FPS stability).
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

	// Allocate temporary prediction buffer
	Vector3f *predicted = (Vector3f *)Memory::alloc_static(sizeof(Vector3f) * p_v_count);
	
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = p_v_count / workers;

	// Pass 1: Parallel Prediction Sweep
	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? p_v_count : (start + chunk);
		SimulationThreadPool::get_singleton()->enqueue_task([=, &predicted, &r_velocities, &r_positions]() {
			for (uint64_t i = start; i < end; i++) {
				pbd_predict_kernel(BigIntCore(static_cast<int64_t>(i)), predicted[i], r_velocities[i], r_positions[i], Vector3f_ZERO, p_delta);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}
	SimulationThreadPool::get_singleton()->wait_for_all();

	// Pass 2: Iterative Volume Constraint
	for (int iter = 0; iter < 4; iter++) {
		pbd_volume_solve_kernel(predicted, p_indices, p_f_count, p_target_vol, p_stiffness, p_delta);
	}

	// Pass 3: Finalize and Damp
	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? p_v_count : (start + chunk);
		SimulationThreadPool::get_singleton()->enqueue_task([=, &r_positions, &r_velocities, &predicted]() {
			for (uint64_t i = start; i < end; i++) {
				pbd_finalize_kernel(BigIntCore(static_cast<int64_t>(i)), r_positions[i], r_velocities[i], predicted[i], p_damping, p_delta);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}
	SimulationThreadPool::get_singleton()->wait_for_all();

	Memory::free_static(predicted);
}

} // namespace UniversalSolver

--- END OF FILE core/math/deformation_restoration_logic.cpp ---
