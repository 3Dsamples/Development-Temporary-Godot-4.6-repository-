--- START OF FILE core/math/transform_2d.cpp ---

#include "core/math/transform_2d.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the machine code for the simulation tiers:
 * - Transform2Df: Bit-perfect 2D physics, UI, and robotic pathing (FixedMathCore).
 * - Transform2Db: Discrete macro-grid transformations for infinite map sectors (BigIntCore).
 */
template struct Transform2D<FixedMathCore>;
template struct Transform2D<BigIntCore>;

// ============================================================================
// Global Deterministic Constants (FixedMathCore Q32.32)
// ============================================================================

/**
 * Static Constants for Transform2Df
 * 
 * Uses raw bit injection to ensure that the identity matrix is available
 * without runtime conversion or floating-point jitter. Aligned to 32 bytes
 * to maximize SIMD throughput in Warp kernels.
 */

const Transform2Df Transform2Df_IDENTITY = Transform2Df(
	FixedMathCore(FixedMathCore::ONE_RAW, true), FixedMathCore(0LL, true),
	FixedMathCore(0LL, true),                    FixedMathCore(FixedMathCore::ONE_RAW, true),
	FixedMathCore(0LL, true),                    FixedMathCore(0LL, true)
);

const Transform2Df Transform2Df_FLIP_X = Transform2Df(
	FixedMathCore(-FixedMathCore::ONE_RAW, true), FixedMathCore(0LL, true),
	FixedMathCore(0LL, true),                     FixedMathCore(FixedMathCore::ONE_RAW, true),
	FixedMathCore(0LL, true),                     FixedMathCore(0LL, true)
);

const Transform2Df_FLIP_Y = Transform2Df(
	FixedMathCore(FixedMathCore::ONE_RAW, true), FixedMathCore(0LL, true),
	FixedMathCore(0LL, true),                    FixedMathCore(-FixedMathCore::ONE_RAW, true),
	FixedMathCore(0LL, true),                    FixedMathCore(0LL, true)
);

// ============================================================================
// Global Deterministic Constants (BigIntCore)
// ============================================================================

/**
 * Static Constants for Transform2Db
 * 
 * Used for discrete 2D coordinate space transitions where every 
 * integer unit represents a massive galactic sector or planetary region.
 */

const Transform2Db Transform2Db_IDENTITY = Transform2Db(
	BigIntCore(1LL), BigIntCore(0LL),
	BigIntCore(0LL), BigIntCore(1LL),
	BigIntCore(0LL), BigIntCore(0LL)
);

/**
 * Coherence Validation:
 * 
 * Because Transform2D is ET_ALIGN_32, EnTT registries of Transform2Df components 
 * are packed into CPU cache lines with zero padding gaps. This allows 
 * Warp-Style Parallel Kernels to perform batch multiplications for 
 * millions of 2D entities simultaneously while maintaining bit-perfect 
 * integrity, preventing the "jitter" associated with standard 64-bit 
 * floating point drift in large coordinate systems.
 */

--- END OF FILE core/math/transform_2d.cpp ---
