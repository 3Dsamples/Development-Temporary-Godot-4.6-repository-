--- START OF FILE core/math/deformation_recenter_kernel.cpp ---

#include "core/math/dynamic_mesh.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: PlasticRestRecenterKernel
 * 
 * Logic for permanent material set (Plasticity).
 * If the accumulated fatigue exceeds the yield threshold, the 'Rest Position' 
 * tensor is moved toward the current displaced position.
 * res = rest + (current - rest) * ((fatigue - yield) / yield * flow_rate * dt)
 */
void plastic_rest_recenter_kernel(
		const BigIntCore &p_index,
		Vector3f &r_rest_position,
		const Vector3f &p_current_position,
		const FixedMathCore &p_fatigue,
		const FixedMathCore &p_yield_threshold,
		const FixedMathCore &p_plasticity_flow_rate,
		const FixedMathCore &p_delta) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Bit-perfect Yield Check
	if (p_fatigue > p_yield_threshold) {
		FixedMathCore over_stress = p_fatigue - p_yield_threshold;
		
		// 2. Resolve Flow Magnitude
		// flow = (normalized_overstress) * flow_rate * dt
		FixedMathCore denom = p_yield_threshold + MathConstants<FixedMathCore>::unit_epsilon();
		FixedMathCore flow_mag = (over_stress / denom) * p_plasticity_flow_rate * p_delta;
		
		// Clamp to one to ensure rest position never "over-shoots" current position
		flow_mag = wp::min(flow_mag, one);
		
		// 3. Update Rest Tensor (Deterministic Linear Interpolation)
		r_rest_position = r_rest_position.lerp(p_current_position, flow_mag);
	}
}

/**
 * Warp Kernel: ParallelAABBReductionKernel
 * 
 * Computes the minimum and maximum bounds for a range of vertices.
 * This is the first pass of a deterministic reduction tree.
 */
void parallel_aabb_reduction_kernel(
		const Vector3f *p_positions,
		uint64_t p_start,
		uint64_t p_end,
		AABBf &r_local_aabb) {

	if (unlikely(p_start >= p_end)) {
		r_local_aabb = AABBf();
		return;
	}

	Vector3f min_v = p_positions[p_start];
	Vector3f max_v = p_positions[p_start];

	for (uint64_t i = p_start + 1; i < p_end; i++) {
		const Vector3f &v = p_positions[i];
		
		if (v.x < min_v.x) min_v.x = v.x;
		if (v.y < min_v.y) min_v.y = v.y;
		if (v.z < min_v.z) min_v.z = v.z;

		if (v.x > max_v.x) max_v.x = v.x;
		if (v.y > max_v.y) max_v.y = v.y;
		if (v.z > max_v.z) max_v.z = v.z;
	}

	r_local_aabb.position = min_v;
	r_local_aabb.size = max_v - min_v;
}

/**
 * execute_deformation_normalization_sweep()
 * 
 * The master post-physics synchronization pass.
 * 1. Dispatches parallel plasticity kernels to "bake" deformations.
 * 2. Performs a parallel AABB reduction to update culling volumes.
 * 3. strictly bit-perfect to ensure no desync in 120 FPS multiplayer clusters.
 */
void execute_deformation_normalization_sweep(
		const BigIntCore &p_entity_id,
		Vector3f *r_rest_positions,
		const Vector3f *p_current_positions,
		const FixedMathCore *p_fatigue_stream,
		uint64_t p_v_count,
		const FixedMathCore &p_yield,
		const FixedMathCore &p_flow_rate,
		const FixedMathCore &p_delta,
		AABBf &r_final_aabb) {

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = p_v_count / workers;

	// Pass 1: Plastic Recenter
	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? p_v_count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &r_rest_positions, &p_current_positions, &p_fatigue_stream]() {
			for (uint64_t i = start; i < end; i++) {
				plastic_rest_recenter_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_rest_positions[i],
					p_current_positions[i],
					p_fatigue_stream[i],
					p_yield,
					p_flow_rate,
					p_delta
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}
	SimulationThreadPool::get_singleton()->wait_for_all();

	// Pass 2: AABB Reduction
	Vector<AABBf> worker_results;
	worker_results.resize(workers);

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? p_v_count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &p_current_positions, &worker_results]() {
			parallel_aabb_reduction_kernel(p_current_positions, start, end, worker_results.ptrw()[w]);
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}
	SimulationThreadPool::get_singleton()->wait_for_all();

	// Merge results into final bit-perfect AABB
	r_final_aabb = worker_results[0];
	for (uint32_t w = 1; w < workers; w++) {
		r_final_aabb.merge_with(worker_results[w]);
	}
}

/**
 * resolve_galactic_sector_shift_logic()
 * 
 * Sophisticated Feature: If the centroid of the mesh moves > 10,000 units 
 * from the local origin due to deformation (Balloon drift), this function 
 * re-anchors every vertex to the new BigIntCore galactic sector.
 */
void resolve_galactic_sector_shift_logic(
		Vector3f *r_positions,
		Vector3f *r_rest_positions,
		BigIntCore &r_sx, BigIntCore &r_sy, BigIntCore &r_sz,
		uint64_t p_v_count,
		const FixedMathCore &p_sector_size) {

	// 1. Calculate Centroid via BigInt accumulation (prevent overflow)
	BigIntCore acc_x(0LL), acc_y(0LL), acc_z(0LL);
	for (uint64_t i = 0; i < p_v_count; i++) {
		acc_x += BigIntCore(r_positions[i].x.get_raw());
		acc_y += BigIntCore(r_positions[i].y.get_raw());
		acc_z += BigIntCore(r_positions[i].z.get_raw());
	}
	
	BigIntCore count_bi(static_cast<int64_t>(p_v_count));
	FixedMathCore center_x(static_cast<int64_t>((acc_x / count_bi).operator int64_t()), true);
	FixedMathCore center_y(static_cast<int64_t>((acc_y / count_bi).operator int64_t()), true);
	FixedMathCore center_z(static_cast<int64_t>((acc_z / count_bi).operator int64_t()), true);

	// 2. Identify Displacement
	int64_t dx = Math::floor(center_x / p_sector_size).to_int();
	int64_t dy = Math::floor(center_y / p_sector_size).to_int();
	int64_t dz = Math::floor(center_z / p_sector_size).to_int();

	if (dx != 0 || dy != 0 || dz != 0) {
		r_sx += BigIntCore(dx);
		r_sy += BigIntCore(dy);
		r_sz += BigIntCore(dz);

		FixedMathCore off_x = p_sector_size * FixedMathCore(dx);
		FixedMathCore off_y = p_sector_size * FixedMathCore(dy);
		FixedMathCore off_z = p_sector_size * FixedMathCore(dz);
		Vector3f offset_v(off_x, off_y, off_z);

		// 3. Re-Anchor all vertices zero-copy
		for (uint64_t i = 0; i < p_v_count; i++) {
			r_positions[i] -= offset_v;
			r_rest_positions[i] -= offset_v;
		}
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/deformation_recenter_kernel.cpp ---
