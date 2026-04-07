--- START OF FILE core/math/wfc_solver_observation.cpp ---

#include "core/math/wfc_solver_logic.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: ComputeEntropyBatchKernel
 * 
 * Parallel execution sweep to calculate the Shannon Entropy of every cell.
 * H = log(Sum(Weights)) - (Sum(Weight * log(Weight)) / Sum(Weights))
 * 
 * p_possibilities_matrix: Flattened SoA bitmask [CellCount * TileCount].
 * p_tile_weights: Frequency weights for each tile in the tileset.
 * r_entropy_buffer: Output stream for deterministic decision-making.
 */
void compute_entropy_batch_kernel(
		const bool *p_possibilities_matrix,
		const FixedMathCore *p_tile_weights,
		FixedMathCore *r_entropy_buffer,
		uint32_t p_tile_count,
		uint64_t p_start,
		uint64_t p_end) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore eps(4294LL, true); // 0.000001 epsilon to prevent log(0)

	for (uint64_t i = p_start; i < p_end; i++) {
		const bool *cell_bits = &p_possibilities_matrix[i * p_tile_count];
		
		FixedMathCore sum_w = zero;
		FixedMathCore sum_wlogw = zero;
		uint32_t active_options = 0;

		for (uint32_t t = 0; t < p_tile_count; t++) {
			if (cell_bits[t]) {
				FixedMathCore w = p_tile_weights[t];
				sum_w += w;
				// H_sum = w * ln(w)
				sum_wlogw += w * Math::log(w + eps);
				active_options++;
			}
		}

		// If a cell has 0 or 1 options, it has no entropy (collapsed or impossible)
		if (active_options <= 1 || sum_w.get_raw() == 0) {
			r_entropy_buffer[i] = zero;
		} else {
			// Shannon Entropy calculation in bit-perfect FixedMath
			r_entropy_buffer[i] = Math::log(sum_w) - (sum_wlogw / sum_w);
		}
	}
}

/**
 * find_min_entropy_reduction()
 * 
 * Performs a deterministic reduction to find the next cell for "Observation".
 * Scans the entropy buffer for the smallest value > 0.
 * Uses BigIntCore to support grids exceeding 2^32 cells.
 */
BigIntCore find_min_entropy_reduction(
		const FixedMathCore *p_entropy_buffer,
		const bool *p_collapsed_mask,
		const BigIntCore &p_total_cells) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_total_cells.to_string()));
	FixedMathCore min_h(2147483647LL, false); // Initialize with "Infinity"
	BigIntCore best_idx(-1LL);

	for (uint64_t i = 0; i < total; i++) {
		// Skip finalized cells
		if (p_collapsed_mask[i]) continue;

		FixedMathCore h = p_entropy_buffer[i];
		// Smallest non-zero entropy is the most "constrained" choice
		if (h.get_raw() > 0 && h < min_h) {
			min_h = h;
			best_idx = BigIntCore(static_cast<int64_t>(i));
		}
	}

	return best_idx;
}

/**
 * Warp Kernel: ObserveAndCollapseKernel
 * 
 * Collapses a superposed cell into a single deterministic state.
 * 1. Takes the lowest-entropy cell index.
 * 2. Uses a bit-perfect PCG random value to select a tile based on weights.
 * 3. Updates the EnTT component stream for the next propagation wave.
 */
void observe_and_collapse_kernel(
		bool *r_possibilities,
		bool &r_is_collapsed,
		int32_t &r_final_tile_id,
		const FixedMathCore *p_weights,
		uint32_t p_tile_count,
		const FixedMathCore &p_random_val) {

	FixedMathCore total_weight = MathConstants<FixedMathCore>::zero();
	for (uint32_t i = 0; i < p_tile_count; i++) {
		if (r_possibilities[i]) {
			total_weight += p_weights[i];
		}
	}

	// Normalized selection threshold
	FixedMathCore threshold = p_random_val * total_weight;
	FixedMathCore cumulative_w = MathConstants<FixedMathCore>::zero();
	int32_t selected_id = -1;

	for (uint32_t i = 0; i < p_tile_count; i++) {
		if (r_possibilities[i]) {
			cumulative_w += p_weights[i];
			if (cumulative_w >= threshold) {
				selected_id = static_cast<int32_t>(i);
				break;
			}
		}
	}

	// If selection fails due to zero weight, pick the first valid option
	if (selected_id == -1) {
		for (uint32_t i = 0; i < p_tile_count; i++) {
			if (r_possibilities[i]) {
				selected_id = i;
				break;
			}
		}
	}

	// Finalize State in SoA
	r_final_tile_id = selected_id;
	r_is_collapsed = true;
	for (uint32_t i = 0; i < p_tile_count; i++) {
		r_possibilities[i] = (i == static_cast<uint32_t>(selected_id));
	}
}

/**
 * Sophisticated Interaction: PhysicsConstraintInjection
 * 
 * Advanced WFC Feature: Modifies cell possibilities based on real-time 
 * physical interactions (e.g. an explosion removes 'solid' tile options).
 */
void inject_volumetric_physics_constraints(
		bool *r_cell_possibilities,
		const uint32_t *p_forbidden_ids,
		uint32_t p_forbidden_count,
		uint32_t p_tile_count) {

	for (uint32_t i = 0; i < p_forbidden_count; i++) {
		uint32_t tile_idx = p_forbidden_ids[i];
		if (tile_idx < p_tile_count) {
			r_cell_possibilities[tile_idx] = false;
		}
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/wfc_solver_observation.cpp ---
