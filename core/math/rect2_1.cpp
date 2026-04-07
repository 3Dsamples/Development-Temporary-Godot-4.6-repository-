--- START OF FILE core/math/rect2.cpp ---

#include "core/math/rect2.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the machine code for the simulation tiers:
 * - Rect2f: Bit-perfect 2D volumes for physics, UI, and sensor-logic (FixedMathCore).
 * - Rect2b: Discrete macro-grid bounds for infinite map sectors (BigIntCore).
 */
template struct Rect2<FixedMathCore>;
template struct Rect2<BigIntCore>;

// ============================================================================
// Global Deterministic Constants (FixedMathCore Q32.32)
// ============================================================================

/**
 * Static Constants for Rect2f
 * 
 * Uses raw bit pattern injection (FixedMathCore(raw, true)) to ensure these 
 * constants are immutable and available instantly for high-frequency Warp 
 * kernels without any floating-point calculation drift.
 */

// Zero-Sized Rect at Origin (0,0,0,0)
const Rect2f Rect2f_ZERO = Rect2f(
	FixedMathCore(0LL, true), 
	FixedMathCore(0LL, true), 
	FixedMathCore(0LL, true), 
	FixedMathCore(0LL, true)
);

// Unit Rect starting at Origin (0,0,1,1)
const Rect2f Rect2f_UNIT = Rect2f(
	FixedMathCore(0LL, true), 
	FixedMathCore(0LL, true), 
	FixedMathCore(FixedMathCore::ONE_RAW, true), 
	FixedMathCore(FixedMathCore::ONE_RAW, true)
);

// ============================================================================
// Global Deterministic Constants (BigIntCore)
// ============================================================================

/**
 * Static Constants for Rect2b
 * 
 * Used for coordinate space transitions where discrete steps represent 
 * transitions between massive planetary map sectors.
 */

const Rect2b Rect2b_ZERO = Rect2b(
	BigIntCore(0LL), 
	BigIntCore(0LL), 
	BigIntCore(0LL), 
	BigIntCore(0LL)
);

/**
 * Performance & Consistency Note:
 * 
 * Because Rect2 is ET_ALIGN_32, EnTT registries containing Rect2f components 
 * are packed optimally for modern CPU cache lines. This allows Warp Kernels 
 * to perform millions of visibility tests per frame during the 120 FPS 
 * synchronization wave. By strictly using FixedMathCore, we guarantee 
 * that UI occlusion and 2D collision broadphase results are bit-identical 
 * across every simulation node in the galaxy.
 */

--- END OF FILE core/math/rect2.cpp ---
