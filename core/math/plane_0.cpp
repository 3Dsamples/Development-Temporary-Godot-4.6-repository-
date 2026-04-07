--- START OF FILE core/math/plane.cpp ---

#include "core/math/plane.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the Plane logic for the Universal Solver backend.
 * These symbols are essential for EnTT to manage plane-based components 
 * (like portals, triggers, or clipping volumes) in contiguous memory, 
 * allowing Warp kernels to perform bit-perfect CCD checks at 120 FPS.
 */

template struct Plane<FixedMathCore>; // TIER_DETERMINISTIC: Bit-perfect CCD & Culling
template struct Plane<BigIntCore>;    // TIER_MACRO: Discrete Sector Boundaries

/**
 * Deterministic Plane Constants
 * 
 * Pre-allocated axis-aligned planes. These utilize the deterministic 
 * raw-bit constructors to ensure they are available instantly for 
 * Warp kernel initialization without any runtime parsing or FPU drift.
 */

// Identity Planes (FixedMathCore Q32.32)
const Planef Planef_XY = Planef(FixedMathCore(0LL, true),  FixedMathCore(0LL, true),  FixedMathCore(1LL, false), FixedMathCore(0LL, true));
const Planef Planef_XZ = Planef(FixedMathCore(0LL, true),  FixedMathCore(1LL, false), FixedMathCore(0LL, true),  FixedMathCore(0LL, true));
const Planef Planef_YZ = Planef(FixedMathCore(1LL, false), FixedMathCore(0LL, true),  FixedMathCore(0LL, true),  FixedMathCore(0LL, true));

// Macro Scale Sector Planes (BigIntCore)
const Planeb Planeb_XY = Planeb(BigIntCore(0), BigIntCore(0), BigIntCore(1), BigIntCore(0));
const Planeb Planeb_XZ = Planeb(BigIntCore(0), BigIntCore(1), BigIntCore(0), BigIntCore(0));
const Planeb Planeb_YZ = Planeb(BigIntCore(1), BigIntCore(0), BigIntCore(0), BigIntCore(1));

/**
 * Universal Solver: CCD Integrity
 * 
 * Because we use FixedMathCore, the 'den' (denominator) calculation in 
 * intersections is bit-identical across all clients. This guarantees that 
 * high-speed projectiles detected via intersects_segment will trigger 
 * the same collision callback on every machine in a multi-scale simulation.
 */

--- END OF FILE core/math/plane.cpp ---
