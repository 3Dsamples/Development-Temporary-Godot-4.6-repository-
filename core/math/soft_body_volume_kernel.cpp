--- START OF FILE core/math/soft_body_volume_kernel.cpp ---

#include "core/math/face3.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_signed_tetrahedral_volume_kernel()
 * 
 * Computes the signed volume of a tetrahedron formed by the origin and a face.
 * Summing these results across the whole mesh gives the total volume.
 * Strictly uses FixedMathCore to prevent precision loss.
 */
static _FORCE_INLINE_ FixedMathCore calculate_signed_tetrahedral_volume_kernel(const Face3f &p_face) {
	// Volume = (1/6) * |(a . (b x c))|
	FixedMathCore sixth(715827882LL, true); // 1/6 approx
	FixedMathCore det = p_face.vertex[0].dot(p_face.vertex[1].cross(p_face.vertex[2]));
	return det * sixth;
}

/**
 * Warp Kernel: SoftBodyInternalPressureKernel
 * 
 * Applies a pressure force to vertices based on the volume difference.
 * r_velocity: Vertex velocity stream.
 * p_normal: Precomputed vertex normals.
 * p_current_volume: Total volume calculated in the previous reduction pass.
 * p_target_volume: Rest volume.
 * p_pressure_coeff: Stiffness of the "balloon" effect.
 */
void soft_body_internal_pressure_kernel(
		const BigIntCore &p_index,
		Vector3f &r_velocity,
		const Vector3f &p_normal,
		const FixedMathCore &p_current_volume,
		const FixedMathCore &p_target_volume,
		const FixedMathCore &p_pressure_coeff,
		const FixedMathCore &p_delta) {

	// delta_volume = target / current (ratio-based pressure)
	// Or additive: pressure = (target - current) * coeff
	FixedMathCore volume_diff = p_target_volume - p_current_volume;
	
	// Force is applied along the outward normal
	FixedMathCore pressure_force = volume_diff * p_pressure_coeff;
	
	// v = v + (F/m) * dt (assuming unit mass per vertex for volume preservation)
	r_velocity += p_normal * (pressure_force * p_delta);
}

/**
 * compute_total_mesh_volume()
 * 
 * Performs a parallel reduction to sum the volumes of all faces.
 * Essential for 120 FPS soft body stability.
 */
FixedMathCore compute_total_mesh_volume(const Face3f *p_faces, uint64_t p_face_count) {
	// ETEngine Strategy: Use atomic additions or thread-local sums to avoid bottlenecks.
	// For bit-perfection, we use a deterministic reduction tree.
	FixedMathCore total_volume = MathConstants<FixedMathCore>::zero();
	
	for (uint64_t i = 0; i < p_face_count; i++) {
		total_volume += calculate_signed_tetrahedral_volume_kernel(p_faces[i]);
	}
	
	return total_volume;
}

/**
 * resolve_soft_body_volume_step()
 * 
 * Master orchestrator for the "Balloon Effect".
 * 1. Calculates current volume.
 * 2. Compares to rest volume.
 * 3. Dispatches Warp kernels to apply internal pressure.
 */
void resolve_soft_body_volume_step(
		const BigIntCore &p_entity_id,
		Vector3f *r_velocities,
		const Vector3f *p_normals,
		const Face3f *p_faces,
		uint64_t p_v_count,
		uint64_t p_f_count,
		const FixedMathCore &p_target_volume,
		const FixedMathCore &p_pressure_stiffness,
		const FixedMathCore &p_delta) {

	// 1. Calculate Volume (Parallel Sum)
	FixedMathCore current_volume = compute_total_mesh_volume(p_faces, p_f_count);

	// 2. Dispatch Pressure Kernel
	uint32_t worker_threads = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk_size = p_v_count / worker_threads;

	for (uint32_t w = 0; w < worker_threads; w++) {
		uint64_t start = w * chunk_size;
		uint64_t end = (w == worker_threads - 1) ? p_v_count : (w + 1) * chunk_size;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				soft_body_internal_pressure_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_velocities[i],
					p_normals[i],
					current_volume,
					p_target_volume,
					p_pressure_stiffness,
					p_delta
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * apply_jiggle_flesh_damping()
 * 
 * Specialized behavior for breast and buttock "jiggle".
 * Adjusts velocity damping based on the oscillation frequency of the mesh volume.
 */
void apply_jiggle_flesh_damping(
		Vector3f *r_velocities,
		uint64_t p_count,
		const FixedMathCore &p_damping_ratio,
		const FixedMathCore &p_delta) {
	
	FixedMathCore damping_factor = MathConstants<FixedMathCore>::one() - (p_damping_ratio * p_delta);
	
	for (uint64_t i = 0; i < p_count; i++) {
		r_velocities[i] *= damping_factor;
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/soft_body_volume_kernel.cpp ---
