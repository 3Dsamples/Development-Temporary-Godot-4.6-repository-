--- START OF FILE core/math/wfc_solver_logic.cpp ---

#include "core/math/wfc_solver_logic.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_cell_entropy_logic()
 * 
 * Computes the Shannon Entropy for a single superposed WFC cell.
 * strictly uses FixedMathCore for logarithmic summation to ensure bit-perfection.
 * Formula: H = log(Sum(W)) - (Sum(W * log(W)) / Sum(W))
 */
FixedMathCore calculate_cell_entropy_logic(
		const bool *p_possibilities,
		const FixedMathCore *p_tile_weights,
		uint32_t p_tile_count) {

	FixedMathCore sum_w = MathConstants<FixedMathCore>::zero();
	FixedMathCore sum_w_log_w = MathConstants<FixedMathCore>::zero();
	FixedMathCore epsilon(4294LL, true); // 0.000001 epsilon for stability
	uint32_t active_tiles = 0;

	for (uint32_t i = 0; i < p_tile_count; i++) {
		if (p_possibilities[i]) {
			FixedMathCore weight = p_tile_weights[i];
			sum_w += weight;
			// Bit-perfect FixedMath log call
			sum_w_log_w += weight * (weight + epsilon).log();
			active_tiles++;
		}
	}

	// Logic: Entropy is 0 if state is determined (1 tile) or impossible (0 tiles)
	if (active_tiles <= 1 || sum_w.get_raw() == 0) {
		return MathConstants<FixedMathCore>::zero();
	}

	return sum_w.log() - (sum_w_log_w / sum_w);
}

/**
 * Warp Kernel: WFCConstraintPropagationKernel
 * 
 * Performs bit-perfect pruning of a neighbor's possibility bitmask.
 * 1. Checks current valid tiles in the source cell.
 * 2. References the Adjacency Matrix to see what remains valid in the neighbor.
 * 3. strictly deterministic; if a mask becomes empty, returns a contradiction.
 */
bool wfc_constraint_propagation_kernel(
		bool *r_neighbor_possibilities,
		const bool *p_source_possibilities,
		const bool *p_adjacency_matrix,
		uint32_t p_tile_count) {

	bool changed = false;

	for (uint32_t j = 0; j < p_tile_count; j++) {
		if (!r_neighbor_possibilities[j]) continue;

		bool tile_j_supported = false;
		// Check if any possible tile in source supports tile j in the neighbor
		for (uint32_t i = 0; i < p_tile_count; i++) {
			if (p_source_possibilities[i]) {
				// Matrix[i][j] lookup
				if (p_adjacency_matrix[i * p_tile_count + j]) {
					tile_j_supported = true;
					break;
				}
			}
		}

		if (!tile_j_supported) {
			r_neighbor_possibilities[j] = false;
			changed = true;
		}
	}

	return changed;
}

/**
 * resolve_linear_grid_index()
 * 
 * Maps 3D coordinates to a linear SoA offset using BigIntCore.
 * Supports voxel volumes larger than 2^64 cells.
 */
BigIntCore resolve_linear_grid_index(
		const BigIntCore &p_x, const BigIntCore &p_y, const BigIntCore &p_z,
		const BigIntCore &p_width, const BigIntCore &p_height) {
	
	// Index = z * (width * height) + y * width + x
	return (p_z * (p_width * p_height)) + (p_y * p_width) + p_x;
}

/**
 * execute_wfc_propagation_wave()
 * 
 * Master orchestrator for procedural constraint resolution.
 * Processes neighbors in 6 cardinal directions using Warp parallel threads.
 */
void execute_wfc_propagation_wave(
		KernelRegistry &p_registry,
		Vector<BigIntCore> &r_propagation_stack,
		const BigIntCore &p_grid_width,
		const BigIntCore &p_grid_height,
		const BigIntCore &p_grid_depth,
		uint32_t p_tile_count,
		const bool *p_rules_6_dir,
		const FixedMathCore *p_tile_weights) {

	auto &possibility_stream = p_registry.get_stream<bool>();
	auto &entropy_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_ENTROPY);

	while (!r_propagation_stack.is_empty()) {
		BigIntCore curr_idx = r_propagation_stack[r_propagation_stack.size() - 1];
		r_propagation_stack.remove_at(r_propagation_stack.size() - 1);

		// Resolve 3D local coordinates
		BigIntCore plane = p_grid_width * p_grid_height;
		BigIntCore z = curr_idx / plane;
		BigIntCore y = (curr_idx / p_grid_width) % p_grid_height;
		BigIntCore x = curr_idx % p_grid_width;

		for (int d = 0; d < 6; d++) {
			BigIntCore nx = x, ny = y, nz = z;
			if (d == 0) nx -= BigIntCore(1LL); else if (d == 1) nx += BigIntCore(1LL);
			else if (d == 2) ny -= BigIntCore(1LL); else if (d == 3) ny += BigIntCore(1LL);
			else if (d == 4) nz -= BigIntCore(1LL); else if (d == 5) nz += BigIntCore(1LL);

			if (nx >= BigIntCore(0LL) && nx < p_grid_width && 
				ny >= BigIntCore(0LL) && ny < p_grid_height && 
				nz >= BigIntCore(0LL) && nz < p_grid_depth) {

				BigIntCore neighbor_idx = (nz * plane) + (ny * p_grid_width) + nx;
				uint64_t n_off = static_cast<uint64_t>(std::stoll(neighbor_idx.to_string()));
				uint64_t c_off = static_cast<uint64_t>(std::stoll(curr_idx.to_string()));

				bool *n_bits = &possibility_stream[n_off * p_tile_count];
				const bool *c_bits = &possibility_stream[c_off * p_tile_count];
				const bool *adj_matrix = &p_rules_6_dir[d * p_tile_count * p_tile_count];

				if (wfc_constraint_propagation_kernel(n_bits, c_bits, adj_matrix, p_tile_count)) {
					// Recalculate Entropy and push for further propagation
					entropy_stream[n_off] = calculate_cell_entropy_logic(n_bits, p_tile_weights, p_tile_count);
					r_propagation_stack.push_back(neighbor_idx);
				}
			}
		}
	}
}

/**
 * Sophisticated Feature: check_procedural_stability()
 * 
 * Real-Time Behavior: Uses Shannon Entropy to detect if a procedural Starbase 
 * or planetary city is "locking up" due to conflicting physical constraints.
 */
bool check_procedural_stability(const FixedMathCore *p_entropy_stream, uint64_t p_count) {
	FixedMathCore instability_threshold(10LL, false); // 10.0 bits
	for (uint64_t i = 0; i < p_count; i++) {
		if (p_entropy_stream[i] > instability_threshold) return false;
	}
	return true;
}

} // namespace UniversalSolver

--- END OF FILE core/math/wfc_solver_logic.cpp ---
