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
 * Computes the signed volume of a tetrahedron formed by the world origin 
 * and a single mesh face. Summing these results across a manifold 
 * yields the total bit-perfect volume.
 */
static _FORCE_INLINE_ FixedMathCore calculate_signed_tetrahedron_volume(
		const Vector3f &p_v0, 
		const Vector3f &p_v1, 
		const Vector3f &p_v2) {
	
	// Volume = (1/6) * dot(v0, cross(v1, v2))
	FixedMathCore sixth(715827882LL, true); // 0.1666666667
	FixedMathCore det = p_v0.dot(p_v1.cross(p_v2));
	return det * sixth;
}

/**
 * Warp Kernel: VolumeReductionKernel
 * 
 * Performs a parallel accumulation of face volumes into a local buffer.
 * Optimized for EnTT Face SoA streams.
 */
void volume_reduction_kernel(
		const Face3f *p_faces,
		uint64_t p_start,
		uint64_t p_end,
		FixedMathCore &r_partial_volume) {

	FixedMathCore acc = MathConstants<FixedMathCore>::zero();
	for (uint64_t i = p_start; i < p_end; i++) {
		const Face3f &f = p_faces[i];
		acc += calculate_signed_tetrahedron_volume(f.vertex[0], f.vertex[1], f.vertex[2]);
	}
	r_partial_volume = acc;
}

/**
 * Warp Kernel: InternalPressureKernel
 * 
 * Applies an outward pressure force to vertices based on the 
 * deviation from the target rest volume.
 * F = (V_rest - V_current) * k_pressure * Normal
 */
void internal_pressure_kernel(
		const BigIntCore &p_index,
		Vector3f &r_velocity,
		const Vector3f &p_normal,
		const FixedMathCore &p_volume_delta,
		const FixedMathCore &p_pressure_k,
		const FixedMathCore &p_delta) {

	if (p_volume_delta.get_raw() == 0) return;

	// Calculate pressure-induced acceleration
	FixedMathCore accel_mag = p_volume_delta * p_pressure_k;
	
	// v = v + a * dt (Along outward surface normal)
	r_velocity += p_normal * (accel_mag * p_delta);
}

/**
 * execute_volume_preservation_wave()
 * 
 * Master orchestrator for the "Balloon" and "Flesh" physical behaviors.
 * 1. Parallel reduction to find global volume in bit-perfect FixedMath.
 * 2. Parallel sweep to apply internal pressure to all EnTT vertices.
 * 3. 120 FPS Synchronization barrier.
 */
void execute_volume_preservation_wave(
		KernelRegistry &p_registry,
		const FixedMathCore &p_target_volume,
		const FixedMathCore &p_stiffness,
		const FixedMathCore &p_delta) {

	auto &face_stream = p_registry.get_stream<Face3f>(COMPONENT_GEOMETRY);
	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &vel_stream = p_registry.get_stream<Vector3f>(COMPONENT_VELOCITY);
	auto &norm_stream = p_registry.get_stream<Vector3f>(COMPONENT_NORMAL);

	uint64_t f_count = face_stream.size();
	uint64_t v_count = pos_stream.size();
	if (f_count == 0 || v_count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	
	// --- PHASE 1: Parallel Volume Reduction ---
	Vector<FixedMathCore> partial_volumes;
	partial_volumes.resize(workers);
	
	uint64_t f_chunk = f_count / workers;
	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * f_chunk;
		uint64_t end = (w == workers - 1) ? f_count : (w + 1) * f_chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &face_stream, &partial_volumes]() {
			volume_reduction_kernel(face_stream.get_base_ptr(), start, end, partial_volumes.ptrw()[w]);
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}
	SimulationThreadPool::get_singleton()->wait_for_all();

	FixedMathCore current_volume = MathConstants<FixedMathCore>::zero();
	for (uint32_t w = 0; w < workers; w++) {
		current_volume += partial_volumes[w];
	}
	current_volume = current_volume.absolute();

	// --- PHASE 2: Pressure Application ---
	FixedMathCore vol_error = p_target_volume - current_volume;
	
	uint64_t v_chunk = v_count / workers;
	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * v_chunk;
		uint64_t end = (w == workers - 1) ? v_count : (w + 1) * v_chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &vel_stream, &norm_stream]() {
			for (uint64_t i = start; i < end; i++) {
				internal_pressure_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					vel_stream[i],
					norm_stream[i],
					vol_error,
					p_stiffness,
					p_delta
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}
	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * apply_flesh_jiggle_damping()
 * 
 * Specialized behavior for breast/buttock simulation.
 * Applies a frequency-dependent damping to the volume restoration
 * to simulate the internal friction of biological tissues.
 */
void apply_flesh_jiggle_damping(
		Vector3f *r_velocities,
		uint64_t p_count,
		const FixedMathCore &p_viscosity,
		const FixedMathCore &p_delta) {

	FixedMathCore damping_factor = MathConstants<FixedMathCore>::one() - (p_viscosity * p_delta);
	damping_factor = wp::max(MathConstants<FixedMathCore>::zero(), damping_factor);

	for (uint64_t i = 0; i < p_count; i++) {
		r_velocities[i] *= damping_factor;
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/soft_body_volume_solver.cpp ---
