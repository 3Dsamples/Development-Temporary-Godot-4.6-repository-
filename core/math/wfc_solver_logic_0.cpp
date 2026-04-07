--- START OF FILE core/math/wfc_solver_logic.cpp ---

#include "core/math/wfc_solver_logic.h"
#include "core/math/math_funcs.h"
#include "core/os/memory.h"

/**
 * propagate_neighbor_constraints()
 * 
 * The core computational bottleneck of WFC.
 * This implementation optimizes the constraint check by using a pre-calculated 
 * adjacency matrix. It determines which tiles in a neighbor cell are still 
 * compatible with the remaining possibilities in the current cell.
 * 
 * Performance: O(TileCount^2) bitwise operations per neighbor.
 * Hardware Symmetry: This logic is designed to be easily vectorized for 
 * SIMD/Warp execution paths.
 */
bool WFCSolverLogic::propagate_neighbor_constraints(
		bool *r_neighbor_possibilities,
		const bool *p_current_possibilities,
		const bool *p_adjacency_matrix, 
		uint32_t p_tile_count) {

	bool changed = false;

	// For every possible tile 'T_neighbor' in the neighbor cell...
	for (uint32_t j = 0; j < p_tile_count; j++) {
		if (!r_neighbor_possibilities[j]) {
			continue; // Already ruled out
		}

		bool is_still_possible = false;

		// Check if there is ANY possible tile 'T_current' in the current cell 
		// that allows 'T_neighbor' to exist next to it.
		for (uint32_t i = 0; i < p_tile_count; i++) {
			if (p_current_possibilities[i]) {
				// Access the adjacency matrix: [CurrentTile][NeighborTile]
				if (p_adjacency_matrix[i * p_tile_count + j]) {
					is_still_possible = true;
					break; 
				}
			}
		}

		// If no tile in the current cell supports this neighbor possibility, eliminate it.
		if (!is_still_possible) {
			r_neighbor_possibilities[j] = false;
			changed = true;
		}
	}

	return changed;
}

/**
 * solve_entropy_minima_search()
 * 
 * Scans a batch of EnTT entities for the cell with the lowest non-zero entropy.
 * Uses BigIntCore for indexing to support massive voxel volumes.
 * Zero-Copy: Operates directly on the entropy buffer provided by the registry.
 */
BigIntCore solve_entropy_minima_search(const FixedMathCore *p_entropy_buffer, const bool *p_collapsed_buffer, const BigIntCore &p_total_cells) {
	BigIntCore best_index = BigIntCore(-1LL);
	FixedMathCore min_entropy = FixedMathCore(2147483647LL, false); // Initialize with max possible
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();

	uint64_t total = static_cast<uint64_t>(std::stoll(p_total_cells.to_string()));

	for (uint64_t i = 0; i < total; i++) {
		if (p_collapsed_buffer[i]) {
			continue;
		}

		FixedMathCore h = p_entropy_buffer[i];
		if (h > zero && h < min_entropy) {
			min_entropy = h;
			best_index = BigIntCore(static_cast<int64_t>(i));
		}
	}

	return best_index;
}

--- END OF FILE core/math/wfc_solver_logic.cpp ---
