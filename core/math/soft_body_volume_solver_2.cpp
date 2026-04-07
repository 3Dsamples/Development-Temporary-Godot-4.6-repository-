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
 * calculate_signed_tetrahedron_volume()
 * 
 * Computes the signed volume of a tetrahedron formed by the origin and a mesh face.
 * V = (1/6) * (v0 . (v1 x v2))
 * strictly uses FixedMathCore to ensure zero-drift volume tracking.
 */
static _FORCE_INLINE_ FixedMathCore calculate_signed_tetrahedron_volume(
		const Vector3f &p_v0, 
		const Vector3f &p_v1, 
		const Vector3f &p_v2) {
	
	FixedMathCore sixth(715827882LL, true); // 0.1666666667 in Q32.32
	Vector3f cross_prod = p_v1.cross(p_v2);
	FixedMathCore dot_prod = p_v0.dot(cross_prod);
	
	return dot_prod * sixth;
}

/**
 * Warp Kernel: VolumePartialSumKernel
 * 
 * Performs a deterministic accumulation of face volumes.
 * Partitioned for parallel execution in the SimulationThreadPool.
 */
void volume_partial_sum_kernel(
		const Face3f *p_faces,
		uint64_t p_start,
		uint64_t p_end,
		BigIntCore &r_partial_volume_raw) {

	BigIntCore accumulator(0LL);
	for (uint64_t i = p_start; i < p_end; i++) {
		const Face3f &f = p_faces[i];
		FixedMathCore vol = calculate_signed_tetrahedron_volume(f.vertex[0], f.vertex[1], f.vertex[2]);
		// Accumulate raw bits to prevent 64-bit overflow in massive meshes
		accumulator += BigIntCore(vol.get_raw());
	}
	r_partial_volume_raw = accumulator;
}

/**
 * Warp Kernel: PBD_VolumePressureProjectionKernel
 * 
 * Applies a Position-Based Dynamics (PBD) correction to maintain rest volume.
 * 1. Computes the volume gradient (Face Normal).
 * 2. Projects vertices along the normal based on the global volume error.
 * 3. strictly deterministic to ensure identical "Balloon" behavior on all nodes.
 */
void pbd_volume_pressure_projection_kernel(
		const BigIntCore &p_index,
		Vector3f &r_predicted_pos,
		const Vector3f &p_vertex_normal,
		const FixedMathCore &p_volume_error,
		const FixedMathCore &p_inv_mass,
		const FixedMathCore &p_stiffness,
		const FixedMathCore &p_delta) {

	if (p_volume_error.get_raw() == 0 || p_inv_mass.get_raw() == 0) return;

	// Pressure acceleration mag = error * stiffness
	// In PBD, we move the position directly to satisfy the constraint
	FixedMathCore lambda = -p_volume_error * p_stiffness;
	
	// Correction is applied along the outward normal (Gradient of Volume)
	Vector3f correction = p_vertex_normal * (lambda * p_inv_mass * p_delta);
	r_predicted_pos += correction;
}

/**
 * execute_volume_preservation_pass()
 * 
 * Absolute implementation of the 120 FPS volume-stabilization wave.
 * 1. Parallel Reduction: Sums all face volumes into a single BigIntCore.
 * 2. Error Calculation: Compares current volume to Rest Volume.
 * 3. Parallel Projection: Applies pressure pulses to EnTT vertex streams.
 */
void execute_volume_preservation_pass(
		KernelRegistry &p_registry,
		const FixedMathCore &p_target_volume,
		const FixedMathCore &p_pressure_stiffness,
		const FixedMathCore &p_delta) {

	auto &face_stream = p_registry.get_stream<Face3f>(COMPONENT_GEOMETRY);
	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_PREDICTED_POS);
	auto &norm_stream = p_registry.get_stream<Vector3f>(COMPONENT_NORMAL);
	auto &inv_mass_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_INV_MASS);

	uint64_t f_count = face_stream.size();
	uint64_t v_count = pos_stream.size();
	if (f_count == 0 || v_count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	
	// --- PHASE 1: Parallel Volume Reduction ---
	Vector<BigIntCore> partial_sums;
	partial_sums.resize(workers);
	uint64_t f_chunk = f_count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * f_chunk;
		uint64_t end = (w == workers - 1) ? f_count : (start + f_chunk);
		SimulationThreadPool::get_singleton()->enqueue_task([=, &face_stream, &partial_sums]() {
			volume_partial_sum_kernel(face_stream.get_base_ptr(), start, end, partial_sums.ptrw()[w]);
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}
	SimulationThreadPool::get_singleton()->wait_for_all();

	// Merge partial BigInt sums into total volume
	BigIntCore total_vol_raw(0LL);
	for (uint32_t w = 0; w < workers; w++) {
		total_vol_raw += partial_sums[w];
	}
	
	// Convert BigInt bits back to FixedMathCore volume
	FixedMathCore current_volume(static_cast<int64_t>(total_vol_raw.operator int64_t()), true);
	current_volume = current_volume.absolute();

	// --- PHASE 2: Pressure Resolve ---
	FixedMathCore volume_error = current_volume - p_target_volume;
	
	// Fast exit if within bit-perfect epsilon
	if (wp::abs(volume_error) < FixedMathCore(42949LL, true)) return;

	uint64_t v_chunk = v_count / workers;
	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * v_chunk;
		uint64_t end = (w == workers - 1) ? v_count : (start + v_chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &pos_stream, &norm_stream, &inv_mass_stream]() {
			for (uint64_t i = start; i < end; i++) {
				pbd_volume_pressure_projection_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					pos_stream[i],
					norm_stream[i],
					volume_error,
					inv_mass_stream[i],
					p_pressure_stiffness,
					p_delta
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}
	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * apply_thermal_buoyancy_kernel()
 * 
 * Sophisticated Real-Time Behavior:
 * Adjusts the internal pressure of a soft body based on its average temperature.
 * Simulates thermal expansion/contraction of air-filled volumes (Balloons/Lungs).
 */
void apply_thermal_buoyancy_kernel(
		FixedMathCore &r_target_volume,
		const FixedMathCore &p_avg_temperature,
		const FixedMathCore &p_base_temperature,
		const FixedMathCore &p_expansion_coeff) {
	
	// Charles's Law: V1/T1 = V2/T2 -> V2 = V1 * (T2 / T1)
	FixedMathCore temp_ratio = p_avg_temperature / p_base_temperature;
	r_target_volume *= temp_ratio;
}

} // namespace UniversalSolver

--- END OF FILE core/math/soft_body_volume_solver.cpp ---
