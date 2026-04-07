--- START OF FILE core/math/plane.cpp ---

#include "core/math/plane.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the machine code for the simulation tiers:
 * - Planef: Bit-perfect 3D clipping, CCD narrow-phase, and frustum planes (FixedMathCore).
 * - Planeb: Discrete macro-boundary mapping for galactic sector culling (BigIntCore).
 */
template struct Plane<FixedMathCore>;
template struct Plane<BigIntCore>;

// ============================================================================
// Global Deterministic Constants (FixedMathCore Q32.32)
// ============================================================================

/**
 * Static Constants for Planef
 * 
 * These use raw bit injection (FixedMathCore(raw, true)) to define axis-aligned 
 * planes without any FPU involvement. These are crucial for 120 FPS 
 * Warp kernels that perform batch clipping of deformable meshes.
 */

// Plane XY: Normal (0, 0, 1), D = 0
const Planef Planef_XY = Planef(
	FixedMathCore(0LL, true), 
	FixedMathCore(0LL, true), 
	FixedMathCore(FixedMathCore::ONE_RAW, true), 
	FixedMathCore(0LL, true)
);

// Plane XZ: Normal (0, 1, 0), D = 0
const Planef Planef_XZ = Planef(
	FixedMathCore(0LL, true), 
	FixedMathCore(FixedMathCore::ONE_RAW, true), 
	FixedMathCore(0LL, true), 
	FixedMathCore(0LL, true)
);

// Plane YZ: Normal (1, 0, 0), D = 0
const Planef Planef_YZ = Planef(
	FixedMathCore(FixedMathCore::ONE_RAW, true), 
	FixedMathCore(0LL, true), 
	FixedMathCore(0LL, true), 
	FixedMathCore(0LL, true)
);

// ============================================================================
// Global Deterministic Constants (BigIntCore)
// ============================================================================

/**
 * Static Constants for Planeb
 * 
 * Used for high-level spatial partitioning where boundaries are 
 * defined in discrete galactic units.
 */

const Planeb Planeb_XY = Planeb(BigIntCore(0LL), BigIntCore(0LL), BigIntCore(1LL), BigIntCore(0LL));
const Planeb Planeb_XZ = Planeb(BigIntCore(0LL), BigIntCore(1LL), BigIntCore(0LL), BigIntCore(0LL));
const Planeb Planeb_YZ = Planeb(BigIntCore(1LL), BigIntCore(0LL), BigIntCore(0LL), BigIntCore(0LL));

/**
 * Architectural Coherence Note:
 * 
 * By maintaining ET_ALIGN_32 in the header and providing these compiled 
 * symbols, the EnTT registry can store Plane components in contiguous SoA 
 * blocks. Warp-Style Parallel Kernels can then execute 'intersects_segment' 
 * (Continuous Collision Detection) across millions of planes per frame 
 * with zero-copy efficiency.
 * 
 * Using FixedMathCore ensures that the 'den' (denominator) calculation in 
 * ray-plane intersections is bit-identical across all clients, preventing 
 * 'Quantum Tunneling' desyncs in relativistic spaceship combat at 120 FPS.
 */

--- END OF FILE core/math/plane.cpp ---
