--- START OF FILE core/math/vector3.cpp ---

#include "core/math/vector3.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the machine code for the simulation tiers:
 * - FixedMathCore: Deterministic physics, collisions, and local kinematics.
 * - BigIntCore: Discrete galactic sector positioning and macro-scale indexing.
 */
template struct Vector3<FixedMathCore>;
template struct Vector3<BigIntCore>;

// ============================================================================
// Global Deterministic Constants (FixedMathCore Q32.32)
// ============================================================================

// Standard Zero and Identity
const Vector3f Vector3f_ZERO    = Vector3f(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true));
const Vector3f Vector3f_ONE     = Vector3f(FixedMathCore(FixedMathCore::ONE_RAW, true), FixedMathCore(FixedMathCore::ONE_RAW, true), FixedMathCore(FixedMathCore::ONE_RAW, true));

// Cardinal Directions (Godot Coordinate System: Y-Up, -Z-Forward)
const Vector3f Vector3f_UP      = Vector3f(FixedMathCore(0LL, true), FixedMathCore(FixedMathCore::ONE_RAW, true), FixedMathCore(0LL, true));
const Vector3f Vector3f_DOWN    = Vector3f(FixedMathCore(0LL, true), FixedMathCore(-FixedMathCore::ONE_RAW, true), FixedMathCore(0LL, true));
const Vector3f Vector3f_LEFT    = Vector3f(FixedMathCore(-FixedMathCore::ONE_RAW, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true));
const Vector3f Vector3f_RIGHT   = Vector3f(FixedMathCore(FixedMathCore::ONE_RAW, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true));
const Vector3f Vector3f_FORWARD = Vector3f(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(-FixedMathCore::ONE_RAW, true));
const Vector3f Vector3f_BACK    = Vector3f(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(FixedMathCore::ONE_RAW, true));

// ============================================================================
// Global Deterministic Constants (BigIntCore)
// ============================================================================

const Vector3b Vector3b_ZERO    = Vector3b(BigIntCore(0LL), BigIntCore(0LL), BigIntCore(0LL));
const Vector3b Vector3b_ONE     = Vector3b(BigIntCore(1LL), BigIntCore(1LL), BigIntCore(1LL));

/**
 * Warp Integration Optimization:
 * 
 * By defining these constants with raw-bit constructors, we ensure that
 * background worker threads in the SimulationThreadPool can access 
 * zero-copy identity vectors without any runtime parsing or FPU 
 * conversion overhead, which is vital for maintaining 120 FPS.
 */

--- END OF FILE core/math/vector3.cpp ---
