--- START OF FILE core/core_constants.cpp ---

#include "core/core_constants.h"
#include "core/object/class_db.h"

/**
 * CoreConstants::bind_global_constants
 * 
 * Registers the Universal Solver's specific enums into the Godot Global Scope.
 * This allows GDScript to access TIER_DETERMINISTIC or WARP_TARGET_CPU_SIMD
 * directly, ensuring the simulation logic is readable and high-performance.
 */
void CoreConstants::bind_global_constants() {

	// Simulation Tiers (Precision Logic)
	ClassDB::bind_integer_constant("@GlobalScope", "SimulationTier", "TIER_STANDARD", TIER_STANDARD);
	ClassDB::bind_integer_constant("@GlobalScope", "SimulationTier", "TIER_DETERMINISTIC", TIER_DETERMINISTIC);
	ClassDB::bind_integer_constant("@GlobalScope", "SimulationTier", "TIER_MACRO_ECONOMY", TIER_MACRO_ECONOMY);

	// Warp Execution Targets (Hardware Agnostic Dispatch)
	ClassDB::bind_integer_constant("@GlobalScope", "WarpExecutionTarget", "WARP_TARGET_CPU_SERIAL", WARP_TARGET_CPU_SERIAL);
	ClassDB::bind_integer_constant("@GlobalScope", "WarpExecutionTarget", "WARP_TARGET_CPU_SIMD", WARP_TARGET_CPU_SIMD);
	ClassDB::bind_integer_constant("@GlobalScope", "WarpExecutionTarget", "WARP_TARGET_GPU_COMPUTE", WARP_TARGET_GPU_COMPUTE);
	ClassDB::bind_integer_constant("@GlobalScope", "WarpExecutionTarget", "WARP_TARGET_GPU_CUDA", WARP_TARGET_GPU_CUDA);

	// BigNumber Notation Styles (UI/UX Formatting)
	ClassDB::bind_integer_constant("@GlobalScope", "BigNumberNotation", "BIGNUM_NOTATION_SCIENTIFIC", BIGNUM_NOTATION_SCIENTIFIC);
	ClassDB::bind_integer_constant("@GlobalScope", "BigNumberNotation", "BIGNUM_NOTATION_AA", BIGNUM_NOTATION_AA);
	ClassDB::bind_integer_constant("@GlobalScope", "BigNumberNotation", "BIGNUM_NOTATION_METRIC_SYMBOL", BIGNUM_NOTATION_METRIC_SYMBOL);
	ClassDB::bind_integer_constant("@GlobalScope", "BigNumberNotation", "BIGNUM_NOTATION_METRIC_NAME", BIGNUM_NOTATION_METRIC_NAME);
}

--- END OF FILE core/core_constants.cpp ---
