--- START OF FILE core/math/soft_body_volume_solver.cpp ---

#include "core/math/face3.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_signed_tetra_volume()
 * 
 * Static helper to compute the signed volume of a tetrahedron formed by the origin 
 * and three vertices of a face. strictly uses FixedMathCore for bit-perfection.
 * V = (1/6) * (v0 . (v1 x v2))
 */
static _FORCE_INLINE_ FixedMathCore calculate_signed_tetra_volume(
		const Vector3f &p_v0, 
		const Vector3f &p_v1, 
		const Vector3f &p_v2) {
	
	FixedMathCore sixth(715827882LL, true); // 0.1666666667 in Q32.32
	Vector3f cross_prod = p_v1.cross(p_v2);
	FixedMathCore dot_prod = p_v0.dot(cross_prod);
	return dot_prod * sixth;
}

/**
 * Warp Kernel: VolumeCalculationKernel
 * 
 * Parallel reduction kernel that sums the signed volumes of a batch of triangles.
 * Part of the iterative PBD (Position-Based Dynamics) volume constraint.
 */
void volume_calculation_kernel(
		const Face3f *p_faces,
		uint64_t p_start,
		uint64_t p_end,
		FixedMathCore &r_partial_volume) {

	FixedMathCore acc = MathConstants<FixedMathCore>::zero();
	for (uint64_t i = p_start; i < p_end; i++) {
		const Face3f &f = p_faces[i];
		acc += calculate_signed_tetra_volume(f.vertex[0], f.vertex[1], f.vertex[2]);
	}
	r_partial_volume = acc;
}

/**
 * Warp Kernel: VolumeConstraintProjectionKernel
 * 
 * Applies the PBD correction to vertices based on the volume error.
 * delta_p = -(V_current - V_target) * (grad_V / sum(|grad_V|^2))
 * Strictly deterministic to ensure flesh and balloon restoration is identical across nodes.
 */
void volume_constraint_projection_kernel(
		const BigIntCore &p_index,
		Vector3f &r_predicted_pos,
		const Vector3f &p_normal,
		const FixedMathCore &p_volume_error,
		const FixedMathCore &p_inv_total_grad_sq,
		const FixedMathCore &p_pressure_stiffness) {

	if (p_volume_error.get_raw() == 0) return;

	// The gradient of volume with respect to a vertex position is the area-weighted normal.
	// Correction is applied along the outward normal to inflate/deflate the body.
	FixedMathCore lambda = -p_volume_error * p_inv_total_grad_sq * p_pressure_stiffness;
	Vector3f correction = p_normal * lambda;

	r_predicted_pos += correction;
}

/**
 * execute_volume_preservation_pass()
 * 
 * The master 120 FPS orchestrator for soft-body volume.
 * 1. Parallel Volume Summation (Reduction).
 * 2. Parallel Gradient Normal Accumulation.
 * 3. Deterministic Constraint Projection.
 */
void execute_volume_preservation_pass(
		KernelRegistry &p_registry,
		const FixedMathCore &p_target_volume,
		const FixedMathCore &p_stiffness,
		const FixedMathCore &p_delta) {

	auto &face_stream = p_registry.get_stream<Face3f>(COMPONENT_GEOMETRY);
	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_PREDICTED_POS);
	auto &norm_stream = p_registry.get_stream<Vector3f>(COMPONENT_NORMAL);

	uint64_t f_count = face_stream.size();
	uint64_t v_count = pos_stream.size();
	if (f_count == 0 || v_count == 0) return;

	SimulationThreadPool *pool = SimulationThreadPool::get_singleton();
	uint32_t workers = pool->get_worker_count();
	
	// --- Phase 1: Parallel Volume Reduction ---
	Vector<FixedMathCore> partial_volumes;
	partial_volumes.resize(workers);
	uint64_t f_chunk = f_count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * f_chunk;
		uint64_t end = (w == workers - 1) ? f_count : (w + 1) * f_chunk;
		pool->enqueue_task([=, &face_stream, &partial_volumes]() {
			volume_calculation_kernel(face_stream.get_base_ptr(), start, end, partial_volumes.ptrw()[w]);
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}
	pool->wait_for_all();

	FixedMathCore current_volume = MathConstants<FixedMathCore>::zero();
	for (uint32_t w = 0; w < workers; w++) {
		current_volume += partial_volumes[w];
	}
	current_volume = current_volume.absolute();

	// --- Phase 2: Resolve Correction Tensors ---
	FixedMathCore vol_error = current_volume - p_target_volume;
	if (Math::abs(vol_error) < FixedMathCore(42949LL, true)) return; // Epsilon exit

	// Calculate sum of squared gradients for PBD denominator
	// (Simplified as normalized pressure distribution for 120 FPS performance)
	FixedMathCore inv_grad_sum = MathConstants<FixedMathCore>::one() / FixedMathCore(static_cast<int64_t>(v_count), false);

	uint64_t v_chunk = v_count / workers;
	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * v_chunk;
		uint64_t end = (w == workers - 1) ? v_count : (w + 1) * v_chunk;

		pool->enqueue_task([=, &pos_stream, &norm_stream]() {
			for (uint64_t i = start; i < end; i++) {
				volume_constraint_projection_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					pos_stream[i],
					norm_stream[i],
					vol_error,
					inv_grad_sum,
					p_stiffness
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}
	pool->wait_for_all();
}

/**
 * apply_anatomical_jiggle_physics()
 * 
 * Specialized Sophisticated Behavior:
 * Adds inertia-based jiggle to flesh components (breast/buttock).
 * strictly uses bit-perfect FixedMath damping to prevent energy gain.
 */
void apply_anatomical_jiggle_physics(
		Vector3f *r_velocities,
		const Vector3f *p_accelerations,
		uint64_t p_count,
		const FixedMathCore &p_flesh_viscosity,
		const FixedMathCore &p_delta) {

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore damping = one - (p_flesh_viscosity * p_delta);

	for (uint64_t i = 0; i < p_count; i++) {
		// v = (v + a*dt) * damping
		r_velocities[i] = (r_velocities[i] + p_accelerations[i] * p_delta) * damping;
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/soft_body_volume_solver.cpp ---
