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
 * When a material exceeds its yield threshold, the deformation becomes permanent.
 * This kernel updates the 'Rest Position' of the vertex to match its current 
 * displaced position, effectively 'baking' the dent, poke, or stretch.
 */
void plastic_rest_recenter_kernel(
		const BigIntCore &p_index,
		Vector3f &r_rest_position,
		const Vector3f &p_current_position,
		const FixedMathCore &p_fatigue,
		const FixedMathCore &p_yield_threshold,
		const FixedMathCore &p_plasticity_flow_rate,
		const FixedMathCore &p_delta) {

	// If fatigue is high, the material 'flows' into a new rest shape
	// Strictly deterministic comparison avoiding FPU involvement
	if (p_fatigue > p_yield_threshold) {
		FixedMathCore flow = (p_fatigue - p_yield_threshold) * p_plasticity_flow_rate * p_delta;
		
		// Clamp flow to prevent overshoot using bit-perfect intrinsics
		FixedMathCore one = MathConstants<FixedMathCore>::one();
		flow = wp::min(flow, one);
		
		// Interpolate rest position toward current deformed position
		// rest = rest + (current - rest) * flow
		r_rest_position = r_rest_position.lerp(p_current_position, flow);
	}
}

/**
 * Warp Kernel: ParallelAABBReductionKernel
 * 
 * Calculates the local bounding box of a deformed vertex stream.
 * Uses a SIMD-friendly min/max reduction pattern.
 */
void parallel_aabb_reduction_kernel(
		const Vector3f *p_positions,
		uint64_t p_start,
		uint64_t p_end,
		AABBf &r_local_aabb) {

	if (unlikely(p_start >= p_end)) return;

	Vector3f min_v = p_positions[p_start];
	Vector3f max_v = p_positions[p_start];

	for (uint64_t i = p_start + 1; i < p_end; i++) {
		const Vector3f &pos = p_positions[i];
		
		if (pos.x < min_v.x) min_v.x = pos.x;
		if (pos.y < min_v.y) min_v.y = pos.y;
		if (pos.z < min_v.z) min_v.z = pos.z;

		if (pos.x > max_v.x) max_v.x = pos.x;
		if (pos.y > max_v.y) max_v.y = pos.y;
		if (pos.z > max_v.z) max_v.z = pos.z;
	}

	r_local_aabb.position = min_v;
	r_local_aabb.size = max_v - min_v;
}

/**
 * execute_deformation_normalization_sweep()
 * 
 * Master orchestrator for normalizing deformed EnTT components.
 * 1. Updates rest positions (Plasticity).
 * 2. Recalculates AABB (Culling).
 * 3. Checks for Galactic Sector overflow of mesh clusters.
 */
void execute_deformation_normalization_sweep(
		const BigIntCore &p_entity_id,
		Vector3f *r_rest_positions,
		const Vector3f *p_current_positions,
		const FixedMathCore *p_fatigue_stream,
		uint64_t p_vertex_count,
		const FixedMathCore &p_yield,
		const FixedMathCore &p_delta) {

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = p_vertex_count / workers;

	// 1. Parallel Plastic Bake Pass
	// Updates the 'memory' of the material shape based on permanent stress
	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? p_vertex_count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				plastic_rest_recenter_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_rest_positions[i],
					p_current_positions[i],
					p_fatigue_stream[i],
					p_yield,
					FixedMathCore(214748364LL, true), // 0.05 flow rate
					p_delta
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();

	// 2. Reduction Pass: Update AABB for culling
	// Since vertices have moved, the bounding volume must be bit-perfectly resized
	AABBf final_aabb;
	parallel_aabb_reduction_kernel(p_current_positions, 0, p_vertex_count, final_aabb);
}

/**
 * calculate_mesh_centroid_bi()
 * 
 * Returns the average position of all vertices using BigIntCore accumulation 
 * to prevent 64-bit overflow during the summation of millions of FixedMath points.
 */
Vector3f calculate_mesh_centroid_bi(const Vector3f *p_positions, uint64_t p_count) {
	if (unlikely(p_count == 0)) return Vector3f();
	
	BigIntCore acc_x(0LL), acc_y(0LL), acc_z(0LL);
	
	for (uint64_t i = 0; i < p_count; i++) {
		acc_x += BigIntCore(p_positions[i].x.get_raw());
		acc_y += BigIntCore(p_positions[i].y.get_raw());
		acc_z += BigIntCore(p_positions[i].z.get_raw());
	}
	
	BigIntCore count_bi(static_cast<int64_t>(p_count));
	return Vector3f(
		FixedMathCore(static_cast<int64_t>((acc_x / count_bi).operator int64_t()), true),
		FixedMathCore(static_cast<int64_t>((acc_y / count_bi).operator int64_t()), true),
		FixedMathCore(static_cast<int64_t>((acc_z / count_bi).operator int64_t()), true)
	);
}

/**
 * resolve_extreme_deformation_drift()
 * 
 * If a mesh is "Pulled" so far that its centroid crosses a sector boundary,
 * this function re-anchors every vertex to the new BigIntCore sector.
 */
void resolve_extreme_deformation_drift(
		Vector3f *r_positions, 
		Vector3f *r_rest_positions,
		BigIntCore &r_sx, BigIntCore &r_sy, BigIntCore &r_sz,
		uint64_t p_count,
		const FixedMathCore &p_sector_size) {

	Vector3f centroid = calculate_mesh_centroid_bi(r_positions, p_count);
	
	int64_t dx = Math::floor(centroid.x / p_sector_size).to_int();
	int64_t dy = Math::floor(centroid.y / p_sector_size).to_int();
	int64_t dz = Math::floor(centroid.z / p_sector_size).to_int();

	if (dx != 0 || dy != 0 || dz != 0) {
		r_sx += BigIntCore(dx);
		r_sy += BigIntCore(dy);
		r_sz += BigIntCore(dz);

		FixedMathCore off_x = p_sector_size * FixedMathCore(dx);
		FixedMathCore off_y = p_sector_size * FixedMathCore(dy);
		FixedMathCore off_z = p_sector_size * FixedMathCore(dz);
		Vector3f offset(off_x, off_y, off_z);

		for (uint64_t i = 0; i < p_count; i++) {
			r_positions[i] -= offset;
			r_rest_positions[i] -= offset;
		}
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/deformation_recenter_kernel.cpp ---
