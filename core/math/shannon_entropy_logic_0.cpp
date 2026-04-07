--- START OF FILE core/math/shannon_entropy_logic.cpp ---

#include "core/math/wfc_solver_logic.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * compute_entropy_batch_kernel()
 * 
 * A Warp-style parallel kernel that calculates the Shannon Entropy for every cell in a WFC grid.
 * Formula: H = log(sum(weights)) - (sum(weight * log(weight)) / sum(weights))
 * 
 * p_possibilities_matrix: SoA stream indicating which tiles are still valid [CellCount * TileCount].
 * p_tile_weights: Constant stream of relative frequency weights for each tile type.
 * r_entropy_buffer: Output stream for calculated entropy values.
 * p_tile_count: Number of unique tiles in the tileset.
 * p_start / p_end: Range of cells processed by this specific worker thread.
 */
void compute_entropy_batch_kernel(
		const bool *p_possibilities_matrix,
		const FixedMathCore *p_tile_weights,
		FixedMathCore *r_entropy_buffer,
		uint32_t p_tile_count,
		uint64_t p_start,
		uint64_t p_end) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	// Epsilon to prevent log(0)
	FixedMathCore epsilon(4294LL, true); 

	for (uint64_t i = p_start; i < p_end; i++) {
		const bool *cell_possibilities = &p_possibilities_matrix[i * p_tile_count];
		
		FixedMathCore sum_weights = zero;
		FixedMathCore sum_weight_log_weights = zero;
		uint32_t possible_tile_count = 0;

		for (uint32_t t = 0; t < p_tile_count; t++) {
			if (cell_possibilities[t]) {
				FixedMathCore w = p_tile_weights[t];
				sum_weights += w;
				
				// H_part = w * log(w)
				// Math::log is our deterministic software-defined fixed-point logarithm
				sum_weight_log_weights += w * Math::log(w + epsilon);
				possible_tile_count++;
			}
		}

		// If 0 or 1 possibility remains, entropy is zero (fully collapsed or impossible)
		if (possible_tile_count <= 1 || sum_weights.get_raw() == 0) {
			r_entropy_buffer[i] = zero;
		} else {
			// H = log(W_total) - (Sum(w * log(w)) / W_total)
			r_entropy_buffer[i] = Math::log(sum_weights) - (sum_weight_log_weights / sum_weights);
		}
	}
}

/**
 * execute_parallel_entropy_update()
 * 
 * Orchestrates the parallel sweep of entropy calculations across the EnTT registry.
 * Uses BigIntCore to support voxel grids with trillions of cells.
 */
void execute_parallel_entropy_update(
		const BigIntCore &p_total_cells,
		uint32_t p_tile_count,
		const bool *p_possibilities_matrix,
		const FixedMathCore *p_tile_weights,
		FixedMathCore *r_entropy_buffer) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_total_cells.to_string()));
	uint32_t worker_threads = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk_size = total / worker_threads;

	if (chunk_size < 32) {
		// Single-threaded fallback for small local grids
		compute_entropy_batch_kernel(p_possibilities_matrix, p_tile_weights, r_entropy_buffer, p_tile_count, 0, total);
		return;
	}

	for (uint32_t w = 0; w < worker_threads; w++) {
		uint64_t start = w * chunk_size;
		uint64_t end = (w == worker_threads - 1) ? total : (w + 1) * chunk_size;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			compute_entropy_batch_kernel(
				p_possibilities_matrix,
				p_tile_weights,
				r_entropy_buffer,
				p_tile_count,
				start,
				end
			);
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	// Finalize synchronization for bit-perfect result availability
	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/shannon_entropy_logic.cpp ---
