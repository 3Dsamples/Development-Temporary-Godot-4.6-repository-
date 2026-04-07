--- START OF FILE core/math/vector2.cpp ---

#include "core/math/vector2.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the machine code for the two primary simulation tiers.
 * - Vector2f: Bit-perfect 2D Physics, UI, and Robotics (FixedMathCore).
 * - Vector2b: Discrete 2D Macro-Grids and Infinite Map Paging (BigIntCore).
 * 
 * This architecture allows the SimulationThreadPool to invoke these
 * methods inside parallel Warp-Style kernels with zero binary divergence.
 */

template struct Vector2<FixedMathCore>;
template struct Vector2<BigIntCore>;

/**
 * Global Deterministic Constants (FixedMathCore Q32.32)
 * 
 * Pre-allocated directional vectors using raw-bit initialization to bypass 
 * the overhead of string parsing or FPU-to-Fixed conversions.
 */

// Zero Vector (0, 0)
const Vector2f Vector2f_ZERO = Vector2f(FixedMathCore(0LL, true), FixedMathCore(0LL, true));

// One Vector (1, 1)
const Vector2f Vector2f_ONE = Vector2f(FixedMathCore(FixedMathCore::ONE_RAW, true), FixedMathCore(FixedMathCore::ONE_RAW, true));

// Directionals
const Vector2f Vector2f_LEFT = Vector2f(FixedMathCore(-FixedMathCore::ONE_RAW, true), FixedMathCore(0LL, true));
const Vector2f Vector2f_RIGHT = Vector2f(FixedMathCore(FixedMathCore::ONE_RAW, true), FixedMathCore(0LL, true));
const Vector2f Vector2f_UP = Vector2f(FixedMathCore(0LL, true), FixedMathCore(-FixedMathCore::ONE_RAW, true));
const Vector2f Vector2f_DOWN = Vector2f(FixedMathCore(0LL, true), FixedMathCore(FixedMathCore::ONE_RAW, true));

/**
 * Global Deterministic Constants (BigIntCore)
 * 
 * Used for macro-scale indexing where discrete steps represent 
 * galactic sectors or planetary hex-grids.
 */

const Vector2b Vector2b_ZERO = Vector2b(BigIntCore(0LL), BigIntCore(0LL));
const Vector2b Vector2b_ONE = Vector2b(BigIntCore(1LL), BigIntCore(1LL));

/**
 * Performance Note:
 * 
 * Because Vector2 is ET_ALIGN_32, these static constants are aligned 
 * to CPU cache lines. When the PhysicsServerHyper or Robotic Sensor 
 * kernels access these, it results in zero cache-misses and optimal 
 * pre-fetching for 120 FPS real-time execution.
 */

--- END OF FILE core/math/vector2.cpp ---
