--- START OF FILE core/core_constants.h ---

#ifndef CORE_CONSTANTS_H
#define CORE_CONSTANTS_H

#include "core/object/class_db.h"
#include "core/variant/variant.h"

/**
 * SimulationTier
 * 
 * Defines the precision level of the Universal Solver.
 * TIER_STANDARD: Uses standard hardware FPU logic.
 * TIER_DETERMINISTIC: Uses FixedMathCore for bit-perfect synchronization.
 * TIER_MACRO_ECONOMY: Uses BigIntCore for infinite-scale integers.
 */
enum SimulationTier {
	TIER_STANDARD,
	TIER_DETERMINISTIC,
	TIER_MACRO_ECONOMY
};

/**
 * WarpExecutionTarget
 * 
 * Execution hints for Warp-style mathematical kernels.
 * Allows the math pipeline to choose the most efficient hardware path
 * based on the size of the EnTT component batch.
 */
enum WarpExecutionTarget {
	WARP_TARGET_CPU_SERIAL,
	WARP_TARGET_CPU_SIMD,
	WARP_TARGET_GPU_COMPUTE,
	WARP_TARGET_GPU_CUDA
};

/**
 * BigNumberNotation
 * 
 * Global formatting standards for arbitrary-precision integers
 * displayed within the Godot UI.
 */
enum BigNumberNotation {
	BIGNUM_NOTATION_SCIENTIFIC,
	BIGNUM_NOTATION_AA,
	BIGNUM_NOTATION_METRIC_SYMBOL,
	BIGNUM_NOTATION_METRIC_NAME
};

/**
 * CoreConstants
 * 
 * Logic to register global simulation constants into Godot's @GlobalScope.
 */
class CoreConstants {
public:
	static void bind_global_constants();
};

// Register enums into the Variant system for cross-language compatibility
VARIANT_ENUM_CAST(SimulationTier);
VARIANT_ENUM_CAST(WarpExecutionTarget);
VARIANT_ENUM_CAST(BigNumberNotation);

#endif // CORE_CONSTANTS_H

--- END OF FILE core/core_constants.h ---
