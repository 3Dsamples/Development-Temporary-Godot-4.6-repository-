--- START OF FILE core/math/wfc_solver_observation_kernel.cpp ---

#include "core/math/wfc_solver_logic.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/math/random_pcg.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * observe_and_collapse_kernel()
 * 
 * Performs the deterministic "Collapse" of a WFC cell.
 * 1. Accumulates the weights of all remaining valid tiles.
 * 2. Uses a bit-perfect random value to select a single tile.
 * 3. Updates the EnTT bitmask to leave only the selected tile active.
 * 4. strictly avoids FPU involvement to ensure cross-node procedural identity.
 */
void observe_and_collapse_kernel(
		const BigIntCore &p_index,
		bool *r_possibilities,
		bool &r_collapsed_flag,
		int32_t &r_final_tile_id,
		const FixedMathCore *p_tile_weights,
		uint32_t p_tile_count,
		const FixedMathCore &p_random_val) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Calculate the total weight of valid possibilities
	FixedMathCore weight_sum = zero;
	for (uint32_t i = 0; i < p_tile_count; i++) {
		if (r_possibilities[i]) {
			weight_sum += p_tile_weights[i];
		}
	}

	// 2. Deterministic Selection
	// random_threshold = [0..1] * total_weight
	FixedMathCore threshold = p_random_val * weight_sum;
	FixedMathCore current_acc = zero;
	int32_t selected_id = -1;

	for (uint32_t i = 0; i < p_tile_count; i++) {
		if (r_possibilities[i]) {
			current_acc += p_tile_weights[i];
			if (current_acc >= threshold) {
				selected_id = static_cast<int32_t>(i);
				break;
			}
		}
	}

	// 3. Contradiction Fallback
	// If rounding or zero-sum logic fails, pick the first available possibility
	if (unlikely(selected_id == -1)) {
		for (uint32_t i = 0; i < p_tile_count; i++) {
			if (r_possibilities[i]) {
				selected_id = static_cast<int32_t>(i);
				break;
			}
		}
	}

	// 4. Update Cell State in SoA stream
	r_final_tile_id = selected_id;
	r_collapsed_flag = true;

	// Reset all possibilities except the selected one
	for (uint32_t i = 0; i < p_tile_count; i++) {
		r_possibilities[i] = (i == static_cast<uint32_t>(selected_id));
	}
}

/**
 * execute_observation_wave()
 * 
 * Orchestrates the transition from a superposition to a determined state.
 * 1. Takes the cell index with the lowest entropy (Observation Priority).
 * 2. Generates a bit-perfect random value using the entity's BigIntCore seed.
 * 3. Launches the collapse kernel.
 */
void execute_observation_wave(
		KernelRegistry &p_registry,
		const BigIntCore &p_target_cell_idx,
		const FixedMathCore *p_tile_weights,
		uint32_t p_tile_count,
		const BigIntCore &p_global_seed) {

	auto &possibility_stream = p_registry.get_stream<bool>(); // Flattened SoA
	auto &collapsed_stream = p_registry.get_stream<bool>(COMPONENT_IS_COLLAPSED);
	auto &final_id_stream = p_registry.get_stream<int32_t>(COMPONENT_TILE_ID);

	uint64_t raw_idx = static_cast<uint64_t>(std::stoll(p_target_cell_idx.to_string()));

	// Deterministic Entropy Source
	// We use the cell index and global seed to ensure the 'roll' is identical on all nodes.
	RandomPCG pcg;
	pcg.seed_big(p_global_seed);
	pcg.seed(pcg.rand64() ^ p_target_cell_idx.hash());
	
	FixedMathCore roll = pcg.randf();

	// Invoke the collapse kernel for the specific target
	observe_and_collapse_kernel(
		p_target_cell_idx,
		&possibility_stream[raw_idx * p_tile_count],
		collapsed_stream[raw_idx],
		final_id_stream[raw_idx],
		p_tile_weights,
		p_tile_count,
		roll
	);
}

/**
 * Sophisticated Behavior: ForceCollapseVolume()
 * 
 * Real-Time Feature: Forces a specific volume of the WFC grid to collapse instantly.
 * Used for localized events like spawning a spaceship docking-port or an explosion crater.
 */
void force_collapse_volume(
		bool *r_possibilities,
		bool *r_collapsed_flags,
		int32_t *r_tile_ids,
		uint32_t p_forced_tile_id,
		uint64_t p_cell_offset,
		uint32_t p_tile_count) {
	
	// Force the specific ID
	r_tile_ids[p_cell_offset] = static_cast<int32_t>(p_forced_tile_id);
	r_collapsed_flags[p_cell_offset] = true;

	bool *cell_mask = &r_possibilities[p_cell_offset * p_tile_count];
	for (uint32_t i = 0; i < p_tile_count; i++) {
		cell_mask[i] = (i == p_forced_tile_id);
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/wfc_solver_observation_kernel.cpp ---
