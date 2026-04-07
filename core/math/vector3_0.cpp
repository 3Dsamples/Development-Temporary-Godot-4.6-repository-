--- START OF FILE core/math/vector3.cpp ---

#include "core/math/vector3.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * We generate the machine code for the two core simulation types. 
 * This architecture mirrors the NVIDIA Warp "Warp-C" backend by ensuring 
 * that the 3D math logic is compiled into high-performance symbols that 
 * EnTT Sparse Sets can map directly to hardware registers.
 */

template struct Vector3<FixedMathCore>;  // TIER_DETERMINISTIC: Bit-perfect Physics & Collision
template struct Vector3<BigIntCore>;    // TIER_MACRO: Discrete Galactic Coordinate Systems

/**
 * Deterministic Directional Constants
 * 
 * Pre-computed bit-perfect vectors. By using the 'true' flag in the FixedMathCore
 * constructor, we bypass the string-parser for maximum initialization speed.
 */

// Deterministic Constants (FixedMathCore Q32.32)
const Vector3f Vector3f_ZERO    = Vector3f(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true));
const Vector3f Vector3f_ONE     = Vector3f(FixedMathCore(1LL, false), FixedMathCore(1LL, false), FixedMathCore(1LL, false));
const Vector3f Vector3f_UP      = Vector3f(FixedMathCore(0LL, true), FixedMathCore(1LL, false), FixedMathCore(0LL, true));
const Vector3f Vector3f_DOWN    = Vector3f(FixedMathCore(0LL, true), FixedMathCore(-1LL, false), FixedMathCore(0LL, true));
const Vector3f Vector3f_LEFT    = Vector3f(FixedMathCore(-1LL, false), FixedMathCore(0LL, true), FixedMathCore(0LL, true));
const Vector3f Vector3f_RIGHT   = Vector3f(FixedMathCore(1LL, false), FixedMathCore(0LL, true), FixedMathCore(0LL, true));
const Vector3f Vector3f_FORWARD = Vector3f(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(-1LL, false));
const Vector3f Vector3f_BACK    = Vector3f(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(1LL, false));

// Macro Scale Constants (BigIntCore)
const Vector3b Vector3b_ZERO    = Vector3b(BigIntCore(0), BigIntCore(0), BigIntCore(0));
const Vector3b Vector3b_ONE     = Vector3b(BigIntCore(1), BigIntCore(1), BigIntCore(1));

/**
 * Universal Solver: Batch Optimization
 * 
 * In this implementation, Vector3 logic is designed to be called by 
 * Warp Kernels. When EnTT provides a pointer to a contiguous block of 
 * Vector3f, the compiler can use the ET_ALIGN_32 hint to apply 
 * auto-vectorization to these operations.
 */

--- END OF FILE core/math/vector3.cpp ---
