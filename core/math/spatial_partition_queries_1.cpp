--- START OF FILE core/math/spatial_partition_queries.cpp ---

#include "core/math/spatial_partition.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"
#include <algorithm>

namespace UniversalSolver {

/**
 * morton_encode_part()
 * 
 * Helper for bit-interleaving. Spreads 21 bits of an integer into 64 bits
 * with two-zero gaps between each bit.
 */
static _FORCE_INLINE_ uint64_t morton_encode_part(uint32_t v) {
	uint64_t x = v & 0x1fffff;
	x = (x | x << 32) & 0x1f00000000ffffULL;
	x = (x | x << 16) & 0x1f0000ff0000ffULL;
	x = (x | x << 8)  & 0x100f00f00f00f00fULL;
	x = (x | x << 4)  & 0x10c30c30c30c30c3ULL;
	x = (x | x << 2)  & 0x1249249249249249ULL;
	return x;
}

/**
 * morton_encode_3d()
 * 
 * Generates a 64-bit Z-order curve index for a 3D coordinate.
 * Transforms 3D spatial locality into 1D memory locality for EnTT cache-optimization.
 */
static _FORCE_INLINE_ uint64_t morton_encode_3d(uint32_t x, uint32_t y, uint32_t z) {
	return morton_encode_part(x) | (morton_encode_part(y) << 1) | (morton_encode_part(z) << 2);
}

/**
 * Warp Kernel: LinearOctreeSearchKernel
 * 
 * Performs a parallel radius search over a sorted Morton-encoded stream.
 * 1. Computes the Morton range for the query AABB.
 * 2. Uses bit-perfect binary search to find the entry point in the EnTT stream.
 * 3. Sweeps the curve and performs bit-identical distance checks.
 */
void linear_octree_search_kernel(
		const uint64_t *p_sorted_morton_codes,
		const BigIntCore *p_entity_handles,
		const Vector3f *p_positions,
		uint64_t p_count,
		const Vector3f &p_query_pos,
		const FixedMathCore &p_radius,
		const FixedMathCore &p_cell_size,
		LocalVector<BigIntCore> &r_results) {

	FixedMathCore r2 = p_radius * p_radius;
	Vector3f min_p = p_query_pos - Vector3f(p_radius, p_radius, p_radius);
	Vector3f max_p = p_query_pos + Vector3f(p_radius, p_radius, p_radius);

	// Convert coordinates to grid-space integers for Morton encoding
	uint32_t x_min = static_cast<uint32_t>(Math::floor(min_p.x / p_cell_size).to_int());
	uint32_t y_min = static_cast<uint32_t>(Math::floor(min_p.y / p_cell_size).to_int());
	uint32_t z_min = static_cast<uint32_t>(Math::floor(min_p.z / p_cell_size).to_int());

	uint32_t x_max = static_cast<uint32_t>(Math::floor(max_p.x / p_cell_size).to_int());
	uint32_t y_max = static_cast<uint32_t>(Math::floor(max_p.y / p_cell_size).to_int());
	uint32_t z_max = static_cast<uint32_t>(Math::floor(max_p.z / p_cell_size).to_int());

	uint64_t m_start = morton_encode_3d(x_min, y_min, z_min);
	uint64_t m_end = morton_encode_3d(x_max, y_max, z_max);

	// Find starting index in O(log N)
	const uint64_t *ptr = std::lower_bound(p_sorted_morton_codes, p_sorted_morton_codes + p_count, m_start);
	uint64_t start_idx = std::distance(p_sorted_morton_codes, ptr);

	for (uint64_t i = start_idx; i < p_count; i++) {
		uint64_t code = p_sorted_morton_codes[i];
		if (code > m_end) break;

		// Verify spatial bit-perfection (Morton range is a bounding approximation)
		Vector3f diff = p_positions[i] - p_query_pos;
		if (diff.length_squared() <= r2) {
			r_results.push_back(p_entity_handles[i]);
		}
	}
}

/**
 * execute_robotic_perception_sweep()
 * 
 * Orchestrates a massive batch of spatial queries for robotic AI and machine sensors.
 * Distributes query-load across the SimulationThreadPool to maintain 120 FPS.
 */
void execute_robotic_perception_sweep(
		KernelRegistry &p_registry,
		const Vector<Vector3f> &p_query_positions,
		const FixedMathCore &p_query_radius,
		const FixedMathCore &p_cell_size,
		Vector<LocalVector<BigIntCore>> &r_all_results) {

	auto &morton_stream = p_registry.get_stream<uint64_t>(COMPONENT_MORTON_CODE);
	auto &handle_stream = p_registry.get_stream<BigIntCore>(COMPONENT_ENTITY_ID);
	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);

	uint64_t total_entities = morton_stream.size();
	uint64_t total_queries = static_cast<uint64_t>(p_query_positions.size());
	r_all_results.resize(total_queries);

	if (total_entities == 0 || total_queries == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total_queries / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t q_start = w * chunk;
		uint64_t q_end = (w == workers - 1) ? total_queries : (q_start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &morton_stream, &handle_stream, &pos_stream, &p_query_positions, &r_all_results]() {
			for (uint64_t q = q_start; q < q_end; q++) {
				linear_octree_search_kernel(
					morton_stream.get_base_ptr(),
					handle_stream.get_base_ptr(),
					pos_stream.get_base_ptr(),
					total_entities,
					p_query_positions[q],
					p_query_radius,
					p_cell_size,
					r_all_results.ptrw()[q]
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * calculate_nearest_neighbor_deterministic()
 * 
 * Sophisticated robotic feature: find the single nearest entity.
 * Uses bit-perfect Z-order distance heuristics to prune search branches.
 */
BigIntCore calculate_nearest_neighbor_deterministic(
		const Vector3f &p_origin,
		const uint64_t *p_codes,
		const BigIntCore *p_handles,
		const Vector3f *p_positions,
		uint64_t p_count,
		const FixedMathCore &p_cell_size) {

	uint32_t x = static_cast<uint32_t>(Math::floor(p_origin.x / p_cell_size).to_int());
	uint32_t y = static_cast<uint32_t>(Math::floor(p_origin.y / p_cell_size).to_int());
	uint32_t z = static_cast<uint32_t>(Math::floor(p_origin.z / p_cell_size).to_int());
	uint64_t target_code = morton_encode_3d(x, y, z);

	const uint64_t *ptr = std::lower_bound(p_codes, p_codes + p_count, target_code);
	uint64_t base_idx = std::distance(p_codes, ptr);

	FixedMathCore min_d2 = FixedMathCore(2147483647LL, false);
	BigIntCore best_handle(-1LL);

	// Search local radius on curve (bidirectional)
	uint64_t search_radius = 16;
	uint64_t start = (base_idx > search_radius) ? base_idx - search_radius : 0;
	uint64_t end = wp::min(base_idx + search_radius, p_count);

	for (uint64_t i = start; i < end; i++) {
		FixedMathCore d2 = (p_positions[i] - p_origin).length_squared();
		if (d2 < min_d2) {
			min_d2 = d2;
			best_handle = p_handles[i];
		}
	}

	return best_handle;
}

} // namespace UniversalSolver

--- END OF FILE core/math/spatial_partition_queries.cpp ---
