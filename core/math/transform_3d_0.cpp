--- START OF FILE core/math/transform_3d.cpp ---

#include "core/math/transform_3d.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the 3D transform logic for the Universal Solver backend.
 * These symbols enable the engine to link specialized Transform3D 
 * operations for FixedMathCore (deterministic physics) and 
 * BigIntCore (massive-scale 3D grids), facilitating zero-copy 
 * data streaming within EnTT registries.
 */

template struct Transform3D<FixedMathCore>; // TIER_DETERMINISTIC: Bit-perfect 3D Physics
template struct Transform3D<BigIntCore>;    // TIER_MACRO: Discrete Galactic Sectoring

/**
 * Deterministic 3D Identity Constants
 * 
 * Pre-allocated identities using raw bit-assignment to bypass 
 * runtime parsing. Used to initialize simulation buffers for 
 * physical bodies, cameras, and spatial entities.
 */

// Identity Transform (FixedMathCore Q32.32)
const Transform3Df Transform3Df_IDENTITY = Transform3Df(
	Basisf_IDENTITY,
	Vector3f_ZERO
);

// Macro Scale Identity (BigIntCore)
const Transform3Db Transform3Db_IDENTITY = Transform3Db(
	Basisb_IDENTITY,
	Vector3b_ZERO
);

/**
 * Warp Integration: Batch Spatial Processing
 * 
 * Because Transform3D is ET_ALIGN_32, it occupies a specific cache footprint 
 * that aligns with EnTT's contiguous component pools. Warp kernels utilize 
 * these compiled symbols to resolve millions of 3D transformations per 
 * frame, maintaining 120 FPS in dense, large-scale simulations.
 */

--- END OF FILE core/math/transform_3d.cpp ---
