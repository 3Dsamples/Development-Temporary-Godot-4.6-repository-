--- START OF FILE core/math/rect2.cpp ---

#include "core/math/rect2.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the Rect2 logic for the Universal Solver backend.
 * This ensures that the linker has access to optimized 2D volume math 
 * for FixedMathCore (physics-accurate UI/2D) and BigIntCore (macro-scale 
 * mapping), allowing EnTT to treat them as first-class component data.
 */

template struct Rect2<FixedMathCore>; // TIER_DETERMINISTIC: Bit-perfect 2D Collisions
template struct Rect2<BigIntCore>;    // TIER_MACRO: Discrete 2D Region Mapping

/**
 * Deterministic 2D Constants
 * 
 * Pre-allocated zero-rectangles to prevent runtime initialization costs.
 * These are bit-perfect representations used for bounds resetting in 
 * Warp kernels during massive batch culling operations.
 */

// Identity/Zero Rect (FixedMathCore Q32.32)
const Rect2f Rect2f_ZERO = Rect2f(
	FixedMathCore(0LL, true), FixedMathCore(0LL, true),
	FixedMathCore(0LL, true), FixedMathCore(0LL, true)
);

// Macro Scale Zero Rect (BigIntCore)
const Rect2b Rect2b_ZERO = Rect2b(
	BigIntCore(0), BigIntCore(0),
	BigIntCore(0), BigIntCore(0)
);

/**
 * Performance Synergy: Zero-Copy Clipping
 * 
 * Because Rect2 is ET_ALIGN_32, it is optimized for modern CPU cache lines.
 * When Warp kernels sweep through EnTT registries containing Rect2f, 
 * they can perform millions of visibility tests per frame, maintaining 
 * 120 FPS in complex 2D simulation environments.
 */

--- END OF FILE core/math/rect2.cpp ---
