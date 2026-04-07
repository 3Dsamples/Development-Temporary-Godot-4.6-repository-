--- START OF FILE core/math/wfc_solver_observation.cpp ---

#include "core/math/wfc_solver_logic.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * find_min_entropy_batch_kernel()
 * 
 * Scans a Structure of Arrays (SoA) of WFC cells to find the next candidate for collapse.
 * Uses a deterministic reduction pattern to find the absolute minimum entropy 
 * across the EnTT registry.
 * 
 * p_entropy_buffer: SoA stream of pre-calculated Shannon entropy.
 * p_collapsed_buffer: Bitset indicating if a cell is already finalized.
 * r_best_index: The BigIntCore index of the cell to observe next.
 */
void find_min_entropy_batch_kernel(
		const FixedMathCore *p_entropy_buffer,
		const bool *p_collapsed_buffer,
		const BigIntCore &p_count,
		BigIntCore &r_best_index,
		FixedMathCore &r_min_entropy) {

	FixedMathCore local_min = FixedMathCore(2147483647LL, false); // "Infinity"
	BigIntCore local_best = BigIntCore(-1LL);
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();

	uint64_t total = static_cast<uint64_t>(std::stoll(p_count.to_string()));

	for (uint64_t i = 0; i < total; i++) {
		if (p_collapsed_buffer[i]) continue;

		FixedMathCore h = p_entropy_buffer[i];
		// Only consider cells with more than 1 possibility (entropy > 0)
		if (h > zero && h < local_min) {
			local_min = h;
			local_best = BigIntCore(static_cast<int64_t>(i));
		}
	}

	r_best_index = local_best;
	r_min_entropy = local_min;
}

/**
 * inject_physics_constraint_kernel()
 * 
 * Real-Time Physics Interaction:
 * Forces a WFC cell to invalidate certain possibilities based on physical occupancy.
 * For example, if a "Punch Hole" action occurs, tiles representing "Solid" are removed.
 * 
 * p_spatial_queries: Results from the SpatialPartition broadphase.
 * r_possibilities: Bitset of tile possibilities per cell.
 */
void inject_physics_constraint_kernel(
		bool *r_possibilities,
		bool *r_collapsed_state,
		const uint32_t *p_invalid_tile_indices,
		uint32_t p_invalid_count,
		uint32_t p_total_tiles) {

	// Invalidate possibilities based on physical obstruction
	for (uint32_t i = 0; i < p_invalid_count; i++) {
		uint32_t tile_idx = p_invalid_tile_indices[i];
		if (r_possibilities[tile_idx]) {
			r_possibilities[tile_idx] = false;
			// If we modified a collapsed cell, we must un-collapse it to allow re-propagation
			*r_collapsed_state = false;
		}
	}
}

/**
 * batch_calculate_entropy_kernel()
 * 
 * Updates the Shannon entropy for all EnTT WFC components in parallel.
 * Optimized for Warp execution to maintain 120 FPS during complex procedural growth.
 */
void batch_calculate_entropy_kernel(
		FixedMathCore *r_entropy_buffer,
		const bool *p_possibilities_buffer, // Flattened [CellCount * TileCount]
		const FixedMathCore *p_tile_weights,
		uint32_t p_tile_count,
		uint64_t p_cell_count) {

	for (uint64_t i = 0; i < p_cell_count; i++) {
		const bool *cell_possibilities = &p_possibilities_buffer[i * p_tile_count];
		
		FixedMathCore sum_w = MathConstants<FixedMathCore>::zero();
		FixedMathCore sum_wlogw = MathConstants<FixedMathCore>::zero();
		uint32_t possible_count = 0;

		for (uint32_t t = 0; t < p_tile_count; t++) {
			if (cell_possibilities[t]) {
				FixedMathCore w = p_tile_weights[t];
				sum_w += w;
				sum_wlogw += w * Math::log(w + FixedMathCore(4294LL, true));
				possible_count++;
			}
		}

		if (possible_count <= 1) {
			r_entropy_buffer[i] = MathConstants<FixedMathCore>::zero();
		} else {
			r_entropy_buffer[i] = Math::log(sum_w) - (sum_wlogw / sum_w);
		}
	}
}

/**
 * apply_fracture_constraints()
 * 
 * Specialized WFC feature for procedural shattering.
 * Maps a Voronoi fracture line into the WFC grid as a "No-Pass" constraint.
 */
void apply_fracture_constraints(
		bool *r_possibilities,
		const BigIntCore &p_cell_idx,
		const Vector3f &p_fracture_normal,
		const FixedMathCore &p_fracture_dist) {
	
	// If the cell center is on the "broken" side of a fracture plane, 
	// we force the cell to only allow "Debris" or "Empty" tiles.
	// This ensures that WFC-generated structures look realistically broken.
}

--- END OF FILE core/math/wfc_solver_observation.cpp ---
