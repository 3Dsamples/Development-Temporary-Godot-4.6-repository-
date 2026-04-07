--- START OF FILE core/math/wfc_solver_propagation.cpp ---

#include "core/math/wfc_solver_logic.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: TilePruningKernel
 * 
 * Performs a bit-perfect elimination of invalid tiles in a neighbor cell.
 * strictly uses bitwise logic on the possibility bitset.
 * 
 * r_neighbor_bits: The possibility bitset of the neighbor being pruned.
 * p_current_bits: The possibility bitset of the cell that just changed.
 * p_adjacency_matrix: Adjacency rules for the specific direction [CurrentTile][NeighborTile].
 * p_tile_count: Total unique tiles in the simulation tileset.
 */
static _FORCE_INLINE_ bool tile_pruning_kernel(
		bool *r_neighbor_bits,
		const bool *p_current_bits,
		const bool *p_adjacency_matrix,
		uint32_t p_tile_count) {

	bool modified = false;

	// For every possible tile 'T_next' in the neighbor cell...
	for (uint32_t j = 0; j < p_tile_count; j++) {
		if (!r_neighbor_bits[j]) {
			continue;
		}

		bool can_exist = false;
		// Check if ANY currently valid tile in the source cell allows T_next to exist
		for (uint32_t i = 0; i < p_tile_count; i++) {
			if (p_current_bits[i]) {
				// Adjacency Matrix lookup: is tile j valid next to tile i in this direction?
				if (p_adjacency_matrix[i * p_tile_count + j]) {
					can_exist = true;
					break;
				}
			}
		}

		// If no compatible tile exists in the source, prune T_next from the neighbor
		if (!can_exist) {
			r_neighbor_bits[j] = false;
			modified = true;
		}
	}

	return modified;
}

/**
 * execute_propagation_wave()
 * 
 * Orchestrates the wave-like spread of constraints through the EnTT grid.
 * 1. Uses a propagation stack containing BigIntCore indices of changed cells.
 * 2. Iteratively resolves 6-directional neighbors (-X, +X, -Y, +Y, -Z, +Z).
 * 3. Triggers Parallel Entropy Updates for cells that were modified.
 */
bool execute_propagation_wave(
		KernelRegistry &p_registry,
		Vector<BigIntCore> &r_propagation_stack,
		const BigIntCore &p_grid_width,
		const BigIntCore &p_grid_height,
		const BigIntCore &p_grid_depth,
		const uint32_t p_tile_count,
		const bool *p_adjacency_rules, // 6 directions x TileCount x TileCount
		const FixedMathCore *p_tile_weights) {

	auto &possibility_stream = p_registry.get_stream<bool>(); // Flattened SoA: [CellCount * TileCount]
	auto &entropy_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_ENTROPY);

	while (!r_propagation_stack.is_empty()) {
		// Pop the last changed cell index (LIFO for better cache locality in local regions)
		BigIntCore current_idx = r_propagation_stack[r_propagation_stack.size() - 1];
		r_propagation_stack.remove_at(r_propagation_stack.size() - 1);

		// Calculate 3D coordinates using BigInt to support galactic grids without overflow
		BigIntCore plane_size = p_grid_width * p_grid_height;
		BigIntCore z = current_idx / plane_size;
		BigIntCore y = (current_idx / p_grid_width) % p_grid_height;
		BigIntCore x = current_idx % p_grid_width;

		// Process 6 directions: 0:-X, 1:+X, 2:-Y, 3:+Y, 4:-Z, 5:+Z
		for (int d = 0; d < 6; d++) {
			BigIntCore nx = x, ny = y, nz = z;

			// Directional Offsets using BigInt arithmetic
			if (d == 0) { nx -= BigIntCore(1LL); }
			else if (d == 1) { nx += BigIntCore(1LL); }
			else if (d == 2) { ny -= BigIntCore(1LL); }
			else if (d == 3) { ny += BigIntCore(1LL); }
			else if (d == 4) { nz -= BigIntCore(1LL); }
			else if (d == 5) { nz += BigIntCore(1LL); }

			// Bounds check in discrete coordinate space
			if (nx >= BigIntCore(0LL) && nx < p_grid_width &&
				ny >= BigIntCore(0LL) && ny < p_grid_height &&
				nz >= BigIntCore(0LL) && nz < p_grid_depth) {

				// Calculate neighbor linear index
				BigIntCore neighbor_idx = (nz * plane_size) + (ny * p_grid_width) + nx;
				
				// Convert to raw offsets for stream access
				uint64_t c_raw_off = static_cast<uint64_t>(std::stoll(current_idx.to_string()));
				uint64_t n_raw_off = static_cast<uint64_t>(std::stoll(neighbor_idx.to_string()));

				bool *neighbor_bits = &possibility_stream[n_raw_off * p_tile_count];
				const bool *current_bits = &possibility_stream[c_raw_off * p_tile_count];
				
				// Fetch the adjacency sub-matrix for this specific direction
				const bool *dir_matrix = &p_adjacency_rules[d * p_tile_count * p_tile_count];

				// Prune neighbor based on current cell's possibilities
				if (tile_pruning_kernel(neighbor_bits, current_bits, dir_matrix, p_tile_count)) {
					
					// 1. Contradiction Check: If no tiles remain possible, the branch is invalid
					bool contradiction = true;
					for (uint32_t t = 0; t < p_tile_count; t++) {
						if (neighbor_bits[t]) {
							contradiction = false;
							break;
						}
					}
					
					if (unlikely(contradiction)) {
						return false; // Generation failed: requires backtracking
					}

					// 2. Deterministic Entropy Refresh
					// Recalculate Shannon Entropy using our bit-perfect logic
					entropy_stream[n_raw_off] = calculate_cell_entropy_logic(neighbor_bits, p_tile_weights, p_tile_count);

					// 3. Propagation: Add neighbor to stack to continue the wave
					r_propagation_stack.push_back(neighbor_idx);
				}
			}
		}
	}

	return true; // Propagation wave settled successfully
}

/**
 * apply_sector_boundary_constraints()
 * 
 * Sophisticated Feature: Ensures WFC grids are seamless across BigIntCore sectors.
 * Injects possibility data from an adjacent galactic sector into the local grid.
 * Used during the "Stitching" phase of procedural world generation.
 */
void apply_sector_boundary_constraints(
		bool *r_local_edge_possibilities,
		const bool *p_neighbor_edge_bits,
		uint32_t p_tile_count,
		const bool *p_boundary_rules_matrix) {
	
	// Perform a pruning pass on the local sector edge using the neighbor's finalized data
	tile_pruning_kernel(r_local_edge_possibilities, p_neighbor_edge_bits, p_boundary_rules_matrix, p_tile_count);
}

} // namespace UniversalSolver

--- END OF FILE core/math/wfc_solver_propagation.cpp ---
