--- START OF FILE core/math/transform_2d.cpp ---

#include "core/math/transform_2d.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the 2D transform logic for the Universal Solver backend.
 * These symbols enable the engine to link specialized Transform2D 
 * operations for FixedMathCore (deterministic physics) and 
 * BigIntCore (massive-scale 2D grids), facilitating zero-copy 
 * data streaming within EnTT registries.
 */

template struct Transform2D<FixedMathCore>; // TIER_DETERMINISTIC: Bit-perfect 2D Physics
template struct Transform2D<BigIntCore>;    // TIER_MACRO: Discrete 2D Sector Mapping

/**
 * Deterministic 2D Identity Constants
 * 
 * Pre-allocated identities using raw bit-assignment to bypass 
 * runtime parsing. Used to initialize simulation buffers for 
 * sprites, colliders, and UI elements.
 */

// Identity Transform (FixedMathCore Q32.32)
const Transform2Df Transform2Df_IDENTITY = Transform2Df(
	FixedMathCore(1LL, false), FixedMathCore(0LL, true),
	FixedMathCore(0LL, true),  FixedMathCore(1LL, false),
	FixedMathCore(0LL, true),  FixedMathCore(0LL, true)
);

// Macro Scale Identity (BigIntCore)
const Transform2Db Transform2Db_IDENTITY = Transform2Db(
	BigIntCore(1), BigIntCore(0),
	BigIntCore(0), BigIntCore(1),
	BigIntCore(0), BigIntCore(0)
);

/**
 * Warp Integration: Batch Projection
 * 
 * Because Transform2D is ET_ALIGN_32, it occupies exactly two cache lines 
 * (when accounting for padding) or can be packed tightly in EnTT SoA pools. 
 * Warp kernels utilize these compiled symbols to perform millions of 
 * 2D coordinate projections per frame for massive-scale simulations.
 */

--- END OF FILE core/math/transform_2d.cpp ---
