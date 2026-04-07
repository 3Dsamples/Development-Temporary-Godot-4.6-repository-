--- START OF FILE core/math/vector2.cpp ---

#include "core/math/vector2.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * We generate the compiled machine code for the two primary simulation types.
 * This allows the linker to resolve 2D math calls across all engine modules,
 * ensuring that Warp kernels and EnTT registries have direct access to
 * zero-copy 2D vector logic.
 */

template struct Vector2<FixedMathCore>;  // TIER_DETERMINISTIC: Bit-perfect 2D Physics
template struct Vector2<BigIntCore>;    // TIER_MACRO: Discrete 2D Galactic Grids

/**
 * Global Constants
 * 
 * Pre-defined vectors for common directions. These utilize the deterministic
 * constructors to ensure constant-time availability without runtime conversion.
 */

// Deterministic Physics Constants (FixedMathCore)
const Vector2f Vector2f_ZERO  = Vector2f(FixedMathCore(0LL, true),  FixedMathCore(0LL, true));
const Vector2f Vector2f_ONE   = Vector2f(FixedMathCore(1LL, false), FixedMathCore(1LL, false));
const Vector2f Vector2f_LEFT  = Vector2f(FixedMathCore(-1LL, false), FixedMathCore(0LL, true));
const Vector2f Vector2f_RIGHT = Vector2f(FixedMathCore(1LL, false), FixedMathCore(0LL, true));
const Vector2f Vector2f_UP    = Vector2f(FixedMathCore(0LL, true),  FixedMathCore(-1LL, false));
const Vector2f Vector2f_DOWN  = Vector2f(FixedMathCore(0LL, true),  FixedMathCore(1LL, false));

// Macro Grid Constants (BigIntCore)
const Vector2b Vector2b_ZERO  = Vector2b(BigIntCore(0), BigIntCore(0));
const Vector2b Vector2b_ONE   = Vector2b(BigIntCore(1), BigIntCore(1));

/**
 * High-Performance Geometry Implementation
 * 
 * Non-inline implementations for 2D geometry would reside here if the
 * algorithms were too heavy for header inlining. For the Universal Solver,
 * the majority of the logic is inlined in the header to allow Warp kernels
 * to perform aggressive SIMD optimization during EnTT batch sweeps.
 */

--- END OF FILE core/math/vector2.cpp ---
