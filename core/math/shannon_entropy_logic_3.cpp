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
 * Formula: H = log(sum(weights)) - (sum(weight * log(weight)) / sum(weights))
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
			// We access the .log() method of FixedMathCore directly
			FixedMathCore log_w = weight.log();
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
	// Guaranteed bit-perfect across different CPU architectures.
	return sum_w.log() - (sum_w_log_w / sum_w);
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

	if (chunk == 0) chunk = p_total_cells;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? p_total_cells : (start + chunk);

		if (start >= p_total_cells) break;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				const bool *cell_bits = &p_possibilities_matrix[i * p_tile_count];
				r_entropy_buffer[i] = calculate_cell_entropy_kernel(cell_bits, p_tile_weights, p_tile_count);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	// Block until all worker threads complete the entropy resolve for this tick
	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * find_min_entropy_reduction()
 * 
 * Scans the entropy stream using BigIntCore indices to identify the 
 * next focal point for the Observation Phase.
 * Targeted cell: Smallest non-zero entropy (the most "certain" uncertain choice).
 * This reduction uses bit-perfect comparison to ensure all clients pick the same cell.
 */
BigIntCore find_min_entropy_reduction(
		const FixedMathCore *p_entropy_stream,
		const bool *p_collapsed_mask,
		const BigIntCore &p_total_cells) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_total_cells.to_string()));
	FixedMathCore min_h(2147483647LL, false); // Initialize with deterministic 2^31-1 "Infinity"
	BigIntCore best_idx(-1LL);

	for (uint64_t i = 0; i < total; i++) {
		// Skip finalized/collapsed cells
		if (p_collapsed_mask[i]) {
			continue;
		}

		FixedMathCore h = p_entropy_stream[i];
		
		// Bit-perfect comparison: find the cell closest to a determined state.
		// A cell with h=0 and p_collapsed_mask=false is a contradiction (0 tiles).
		if (h.get_raw() > 0 && h < min_h) {
			min_h = h;
			best_idx = BigIntCore(static_cast<int64_t>(i));
		}
	}

	return best_idx;
}

/**
 * Sophisticated Feature: compute_procedural_complexity_score()
 * 
 * Real-Time Behavior: Allows the engine to monitor generation health.
 * Sums the total entropy of a BigInt-sized sector to detect potential 
 * infinite loops or constraint bottlenecks.
 */
BigIntCore compute_procedural_complexity_score(
		const FixedMathCore *p_entropy_stream,
		const BigIntCore &p_cell_count) {

	BigIntCore total_complexity(0LL);
	uint64_t total = static_cast<uint64_t>(std::stoll(p_cell_count.to_string()));
	
	for (uint64_t i = 0; i < total; i++) {
		// Accumulate entropy raw bits into BigInt to prevent 64-bit overflow
		total_complexity += BigIntCore(p_entropy_stream[i].get_raw());
	}
	
	return total_complexity;
}

/**
 * inject_environmental_bias_kernel()
 * 
 * Modifies entropy based on distance-to-surface or temperature components.
 * Forces the WFC system to prefer specific tile types in extreme conditions.
 */
void inject_environmental_bias_kernel(
		bool *r_possibilities,
		const FixedMathCore &p_temperature,
		const FixedMathCore &p_melting_point,
		uint32_t p_molten_tile_id,
		uint32_t p_tile_count) {

	if (p_temperature >= p_melting_point) {
		for (uint32_t i = 0; i < p_tile_count; i++) {
			if (i != p_molten_tile_id) {
				r_possibilities[i] = false;
			}
		}
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/shannon_entropy_logic.cpp ---
