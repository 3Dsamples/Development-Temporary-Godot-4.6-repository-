--- START OF FILE core/math/wfc_solver_logic.h ---

#ifndef WFC_SOLVER_LOGIC_H
#define WFC_SOLVER_LOGIC_H

#include "core/typedefs.h"
#include "core/templates/vector.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * WFCSolverLogic
 * 
 * Static math kernels for the Wave Function Collapse algorithm.
 * Designed for high-frequency procedural generation sweeps.
 * All transcendental operations (log) are performed via bit-perfect FixedMathCore.
 */
class WFCSolverLogic {
public:
	// ------------------------------------------------------------------------
	// Entropy & Observation Kernels
	// ------------------------------------------------------------------------

	/**
	 * calculate_shannon_entropy()
	 * 
	 * Computes the entropy of a cell based on the weights of remaining possibilities.
	 * H = log(sum(weights)) - (sum(weight * log(weight)) / sum(weights))
	 * Hardware-agnostic and deterministic to prevent procedural desync.
	 */
	static _FORCE_INLINE_ FixedMathCore calculate_shannon_entropy(const bool *p_possibilities, const FixedMathCore *p_weights, uint32_t p_tile_count) {
		FixedMathCore sum_weights = MathConstants<FixedMathCore>::zero();
		FixedMathCore sum_weight_log_weights = MathConstants<FixedMathCore>::zero();
		uint32_t count = 0;

		for (uint32_t i = 0; i < p_tile_count; i++) {
			if (p_possibilities[i]) {
				FixedStore w = p_weights[i];
				sum_weights += w;
				// Bit-perfect log calculation via software-defined kernel
				sum_weight_log_weights += w * Math::log(w + FixedMathCore(42949LL, true)); // Small epsilon
				count++;
			}
		}

		if (count <= 1) {
			return MathConstants<FixedMathCore>::zero();
		}

		return Math::log(sum_weights) - (sum_weight_log_weights / sum_weights);
	}

	/**
	 * pick_random_tile()
	 * 
	 * Selects a tile ID from a distribution of weights.
	 * Uses bit-perfect mapping from a FixedMathCore random value.
	 */
	static int64_t pick_random_tile(const bool *p_possibilities, const FixedMathCore *p_weights, uint32_t p_tile_count, const FixedMathCore &p_random_val) {
		FixedMathCore total_weight = MathConstants<FixedMathCore>::zero();
		for (uint32_t i = 0; i < p_tile_count; i++) {
			if (p_possibilities[i]) {
				total_weight += p_weights[i];
			}
		}

		FixedMathCore threshold = p_random_val * total_weight;
		FixedMathCore current_sum = MathConstants<FixedMathCore>::zero();

		for (uint32_t i = 0; i < p_tile_count; i++) {
			if (p_possibilities[i]) {
				current_sum += p_weights[i];
				if (current_sum >= threshold) {
					return static_cast<int64_t>(i);
				}
			}
		}

		return -1;
	}

	// ------------------------------------------------------------------------
	// Constraint Propagation Kernels
	// ------------------------------------------------------------------------

	/**
	 * propagate_neighbor_constraints()
	 * 
	 * Updates the possibility bitset of a neighbor cell based on valid adjacency rules.
	 * Returns true if the state was modified, triggering further propagation.
	 */
	static bool propagate_neighbor_constraints(
			bool *r_neighbor_possibilities,
			const bool *p_current_possibilities,
			const bool *p_adjacency_matrix, // [CurrentTile][NeighborTile]
			uint32_t p_tile_count);

	// ------------------------------------------------------------------------
	// Galactic Scale Indexing
	// ------------------------------------------------------------------------

	/**
	 * get_grid_index()
	 * 
	 * Maps 3D grid coordinates to a linear index using BigIntCore to avoid 
	 * 32-bit integer overflow in massive voxel volumes.
	 */
	static _FORCE_INLINE_ BigIntCore get_grid_index(const BigIntCore &p_x, const BigIntCore &p_y, const BigIntCore &p_z, const BigIntCore &p_width, const BigIntCore &p_height) {
		return (p_z * p_width * p_height) + (p_y * p_width) + p_x;
	}
};

#endif // WFC_SOLVER_LOGIC_H

--- END OF FILE core/math/wfc_solver_logic.h ---
