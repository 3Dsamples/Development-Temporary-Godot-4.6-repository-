--- START OF FILE core/math/transform_3d.cpp ---

#include "core/math/transform_3d.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the machine code for the simulation tiers:
 * - Transform3Df: Bit-perfect 3D physical transformations for collisions and robotics (FixedMathCore).
 * - Transform3Db: Discrete macro-transformations for galactic sector anchors (BigIntCore).
 */
template struct Transform3D<FixedMathCore>;
template struct Transform3D<BigIntCore>;

// ============================================================================
// Global Deterministic Constants (FixedMathCore Q32.32)
// ============================================================================

/**
 * Static Constants for Transform3Df
 * 
 * Uses raw bit injection (FixedMathCore(raw, true)) to bypass runtime 
 * parsing and FPU conversion. This guarantees that every Warp lane starts 
 * with an identical bit-mask for the identity transform.
 */

// Identity Transform (Identity Basis, Zero Origin)
const Transform3Df Transform3Df_IDENTITY = Transform3Df(
	Basisf_IDENTITY,
	Vector3f_ZERO
);

// ============================================================================
// Global Deterministic Constants (BigIntCore)
// ============================================================================

/**
 * Static Constants for Transform3Db
 * 
 * Used for coordinate space transitions where discrete steps represent 
 * transitions between astronomical sectors.
 */

const Transform3Db Transform3Db_IDENTITY = Transform3Db(
	Basisb_IDENTITY,
	Vector3b_ZERO
);

/**
 * Scale-Aware Behavioral Implementation:
 * 
 * Because Transform3D is ET_ALIGN_32, EnTT registries of Transform3Df 
 * components are packed into CPU cache lines without padding fragmentation. 
 * When the GalacticOriginShifter identifies a high-speed spaceship 
 * exceeding the precision threshold, it uses these pre-compiled 
 * symbols to perform a parallel zero-copy shift of all nearby 
 * entity transforms in a single 120 FPS update wave.
 * 
 * This ensures that even at relativistic velocities, the spatial hierarchy 
 * remains bit-perfect and free from the 'floating-point jitter' found 
 * in standard 64-bit engine architectures.
 */

--- END OF FILE core/math/transform_3d.cpp ---
