--- START OF FILE core/math/shannon_entropy_logic.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_cell_entropy_kernel()
 * 
 * Computes Shannon Entropy (H) for a single WFC voxel.
 * H = log(sum(weights)) - (sum(weight * log(weight)) / sum(weights))
 * strictly avoids FPU drift by using deterministic FixedMathCore logarithms.
 */
static _FORCE_INLINE_ FixedMathCore calculate_cell_entropy_kernel(
		const bool *p_possibilities,
		const FixedMathCore *p_tile_weights,
		uint32_t p_tile_count) {

	FixedMathCore sum_w = MathConstants<FixedMathCore>::zero();
	FixedMathCore sum_w_log_w = MathConstants<FixedMathCore>::zero();
	FixedMathCore epsilon(4294LL, true); // 0.000001 safety for log stability
	uint32_t possible_count = 0;

	// Process tile possibilities in a SIMD-friendly loop
	for (uint32_t i = 0; i < p_tile_count; i++) {
		if (p_possibilities[i]) {
			FixedMathCore weight = p_tile_weights[i];
			sum_w += weight;
			
			// h_part = w * ln(w) using software-defined fixed-point log
			FixedMathCore log_w = Math::log(weight + epsilon);
			sum_w_log_w += weight * log_w;
			possible_count++;
		}
	}

	// Logic: If only 1 tile remains, entropy is zero (Finalized).
	// If 0 remain, it's a contradiction (Handled by solver state).
	if (possible_count <= 1 || sum_w.get_raw() == 0) {
		return MathConstants<FixedMathCore>::zero();
	}

	// H = log(W_total) - (W_log_sum / W_total)
	return Math::log(sum_w) - (sum_w_log_w / sum_w);
}

/**
 * execute_entropy_batch_sweep()
 * 
 * Master Warp kernel for updating the entropy buffer of an EnTT WFC registry.
 * Processes millions of cells in parallel to maintain 120 FPS.
 * 
 * r_entropy_buffer: SoA stream of calculated entropy tensors.
 * p_possibilities_matrix: Flattened bitmask stream [CellCount * TileCount].
 */
void execute_entropy_batch_sweep(
		FixedMathCore *r_entropy_buffer,
		const bool *p_possibilities_matrix,
		const FixedMathCore *p_tile_weights,
		uint32_t p_tile_count,
		uint64_t p_total_cells) {

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = p_total_cells / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? p_total_cells : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				const bool *cell_bits = &p_possibilities_matrix[i * p_tile_count];
				r_entropy_buffer[i] = calculate_cell_entropy_kernel(cell_bits, p_tile_weights, p_tile_count);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	// Block until all threads complete the entropy resolve
	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * find_min_entropy_reduction()
 * 
 * Scans the entropy stream using BigIntCore indices to identify the 
 * next focal point for the Observation Phase.
 * Targeted cell: Smallest non-zero entropy (the most "certain" uncertain choice).
 */
BigIntCore find_min_entropy_reduction(
		const FixedMathCore *p_entropy_stream,
		const bool *p_collapsed_mask,
		const BigIntCore &p_total_cells) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_total_cells.to_string()));
	FixedMathCore min_h(2147483647LL, false); // Initialize with deterministic "Infinity"
	BigIntCore best_idx(-1LL);

	for (uint64_t i = 0; i < total; i++) {
		// Skip finalized/collapsed cells
		if (p_collapsed_mask[i]) continue;

		FixedMathCore h = p_entropy_stream[i];
		// Bit-perfect comparison: find the cell closest to collapse
		if (h.get_raw() > 0 && h < min_h) {
			min_h = h;
			best_idx = BigIntCore(static_cast<int64_t>(i));
		}
	}

	return best_idx;
}

/**
 * Sophisticated Feature: inject_physics_to_wfc_constraints()
 * 
 * Real-Time Behavior: Allows the physical world to influence the procedural generation.
 * If a body occupies a WFC voxel, this kernel removes "Empty" or "Void" 
 * tile possibilities from that cell's bitmask.
 */
void inject_physics_to_wfc_constraints(
		bool *r_possibilities,
		const uint32_t *p_tiles_requiring_void,
		uint32_t p_void_tile_count,
		uint32_t p_total_tiles,
		bool p_is_occupied_by_physics) {

	if (!p_is_occupied_by_physics) return;

	for (uint32_t i = 0; i < p_void_tile_count; i++) {
		uint32_t tile_id = p_tiles_requiring_void[i];
		if (tile_id < p_total_tiles) {
			// Physical obstruction forces a bit-perfect elimination of void tiles
			r_possibilities[tile_id] = false;
		}
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/shannon_entropy_logic.cpp ---
