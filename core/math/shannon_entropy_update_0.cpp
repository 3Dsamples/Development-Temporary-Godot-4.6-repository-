--- START OF FILE core/math/shannon_entropy_update.cpp ---

#include "core/math/wfc_solver_logic.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: ShannonEntropyKernel
 * 
 * Recalculates the entropy of a WFC cell based on its current possibility bitmask.
 * Uses the formula: H = log(sum(weights)) - (sum(weights * log(weights)) / sum(weights)).
 * 
 * r_entropy: The entropy component in the EnTT registry.
 * p_possibilities: Pointer to the bitmask for this cell.
 * p_weights: Global weights for each tile type.
 * p_tile_count: Total number of possible tiles.
 */
void shannon_entropy_kernel(
		const BigIntCore &p_index,
		FixedMathCore &r_entropy,
		const bool *p_possibilities,
		const FixedMathCore *p_weights,
		uint32_t p_tile_count) {

	FixedMathCore sum_w = MathConstants<FixedMathCore>::zero();
	FixedMathCore sum_w_log_w = MathConstants<FixedMathCore>::zero();
	uint32_t valid_count = 0;

	// SIMD-friendly loop over tile possibilities
	for (uint32_t i = 0; i < p_tile_count; ++i) {
		if (p_possibilities[i]) {
			FixedMathCore w = p_weights[i];
			sum_w += w;
			// Use deterministic fixed-point logarithm
			sum_w_log_w += w * Math::log(w + FixedMathCore(4294LL, true)); // 1e-6 epsilon
			valid_count++;
		}
	}

	// Logic: If only one tile is possible, entropy is exactly zero (Collapsed)
	if (valid_count <= 1 || sum_w.get_raw() == 0) {
		r_entropy = MathConstants<FixedMathCore>::zero();
		return;
	}

	// Final H calculation
	r_entropy = Math::log(sum_w) - (sum_w_log_w / sum_w);
}

/**
 * execute_parallel_shannon_update()
 * 
 * Master sweep to refresh the entropy state of the entire grid.
 * Operates directly on the EnTT SoA buffers for Zero-Copy performance.
 */
void execute_parallel_shannon_update(
		FixedMathCore *r_entropy_stream,
		const bool *p_possibilities_stream,
		const FixedMathCore *p_tile_weights,
		uint32_t p_tile_count,
		const BigIntCore &p_cell_count) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_cell_count.to_string()));
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total / workers;

	if (chunk < 64) {
		// Single-threaded path for small grids
		for (uint64_t i = 0; i < total; ++i) {
			shannon_entropy_kernel(
				BigIntCore(static_cast<int64_t>(i)),
				r_entropy_stream[i],
				&p_possibilities_stream[i * p_tile_count],
				p_tile_weights,
				p_tile_count
			);
		}
		return;
	}

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? total : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				shannon_entropy_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_entropy_stream[i],
					&p_possibilities_stream[i * p_tile_count],
					p_tile_weights,
					p_tile_count
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * find_minimum_entropy_cell()
 * 
 * Post-update reduction to identify the next 'Observation' point.
 * Returns the BigIntCore index of the cell with the lowest non-zero entropy.
 */
BigIntCore find_minimum_entropy_cell(
		const FixedMathCore *p_entropy_stream,
		const bool *p_collapsed_stream,
		const BigIntCore &p_count) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_count.to_string()));
	FixedMathCore min_h(2147483647LL, false); // "Infinity"
	BigIntCore best_idx(-1LL);

	for (uint64_t i = 0; i < total; i++) {
		if (p_collapsed_stream[i]) continue;

		FixedMathCore h = p_entropy_stream[i];
		if (h > MathConstants<FixedMathCore>::zero() && h < min_h) {
			min_h = h;
			best_idx = BigIntCore(static_cast<int64_t>(i));
		}
	}

	return best_idx;
}

} // namespace UniversalSolver

--- END OF FILE core/math/shannon_entropy_update.cpp ---
