--- START OF FILE core/math/shannon_entropy_logic.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_cell_entropy_kernel()
 * 
 * Computes Shannon Entropy (H) for a single voxel cell.
 * H = log(sum(weights)) - (sum(weight * log(weight)) / sum(weights))
 * strictly avoids FPU drift by using deterministic FixedMathCore logarithms.
 */
static _FORCE_INLINE_ FixedMathCore calculate_cell_entropy_kernel(
		const bool *p_possibilities,
		const FixedMathCore *p_tile_weights,
		uint32_t p_tile_count) {

	FixedMathCore sum_w = MathConstants<FixedMathCore>::zero();
	FixedMathCore sum_w_log_w = MathConstants<FixedMathCore>::zero();
	FixedMathCore epsilon(4294LL, true); // 0.000001 safety for log
	uint32_t possible_count = 0;

	for (uint32_t i = 0; i < p_tile_count; i++) {
		if (p_possibilities[i]) {
			FixedMathCore weight = p_tile_weights[i];
			sum_w += weight;
			// h_part = w * ln(w)
			sum_w_log_w += weight * Math::log(weight + epsilon);
			possible_count++;
		}
	}

	// Logic: If only 1 tile remains, entropy is zero (Determined state).
	// If 0 remain, the state is a contradiction.
	if (possible_count <= 1 || sum_w.get_raw() == 0) {
		return MathConstants<FixedMathCore>::zero();
	}

	// Final H resolution: log(W_total) - (W_log_sum / W_total)
	return Math::log(sum_w) - (sum_w_log_w / sum_w);
}

/**
 * execute_entropy_batch_sweep()
 * 
 * Master Warp kernel for updating the entropy buffer of an EnTT WFC grid.
 * Processes millions of cells in parallel to maintain 120 FPS stability.
 * 
 * r_entropy_buffer: SoA stream of calculated entropy.
 * p_possibilities_matrix: Flattened bitmask [CellCount * TileCount].
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
		uint64_t end = (w == workers - 1) ? p_total_cells : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				const bool *cell_bits = &p_possibilities_matrix[i * p_tile_count];
				r_entropy_buffer[i] = calculate_cell_entropy_kernel(cell_bits, p_tile_weights, p_tile_count);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * find_min_entropy_reduction()
 * 
 * Scans the entropy stream using BigIntCore indices.
 * Finds the index of the most "constrained" cell (lowest non-zero entropy).
 * Used for the Observation Phase of the procedural solver.
 */
BigIntCore find_min_entropy_reduction(
		const FixedMathCore *p_entropy_stream,
		const bool *p_collapsed_mask,
		const BigIntCore &p_total_cells) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_total_cells.to_string()));
	FixedMathCore min_h(2147483647LL, false); // Infinity
	BigIntCore best_idx(-1LL);

	for (uint64_t i = 0; i < total; i++) {
		// Skip cells that are already finalized
		if (p_collapsed_mask[i]) continue;

		FixedMathCore h = p_entropy_stream[i];
		// Target the cell with the least uncertainty (closest to zero)
		if (h.get_raw() > 0 && h < min_h) {
			min_h = h;
			best_idx = BigIntCore(static_cast<int64_t>(i));
		}
	}

	return best_idx;
}

/**
 * Sophisticated Feature: EntropyDensityClustering
 * 
 * Aggregates entropy values into BigIntCore sectors to identify 
 * "Unstable Zones" in galactic procedural generation.
 */
void calculate_sector_entropy_density(
		const FixedMathCore *p_entropy_stream,
		const BigIntCore &p_cell_count,
		BigIntCore &r_density_sum) {

	BigIntCore acc(0LL);
	uint64_t total = static_cast<uint64_t>(std::stoll(p_cell_count.to_string()));
	
	for (uint64_t i = 0; i < total; i++) {
		acc += BigIntCore(p_entropy_stream[i].get_raw());
	}
	r_density_sum = acc;
}

} // namespace UniversalSolver

--- END OF FILE core/math/shannon_entropy_logic.cpp ---
