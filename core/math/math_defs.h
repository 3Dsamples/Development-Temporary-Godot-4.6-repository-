--- START OF FILE core/math/math_defs.h ---

#ifndef MATH_DEFS_H
#define MATH_DEFS_H

#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

// ============================================================================
// Simulation Precision & Epsilon (Bit-Perfect Q32.32)
// ============================================================================

// CMP_EPSILON: 0.000001 (approx 4295 raw units)
#define CMP_EPSILON_RAW 4295LL
#define CMP_EPSILON (FixedMathCore(CMP_EPSILON_RAW, true))

// CMP_NORMALIZE: 0.0000000001 for high-precision normalization safety
#define UNIT_EPSILON (FixedMathCore(1LL, true))

// ============================================================================
// Physical & Universal Constants (FixedMathCore)
// ============================================================================

// Standard Gravity: 9.80665 m/s^2
#define GRAVITY_RAW 42122340352LL // 9.80665 * 2^32
#define PHYSICS_GRAVITY (FixedMathCore(GRAVITY_RAW, true))

// Speed of Light: 299,792,458 m/s
// We use BigIntCore for the raw value to prevent immediate shift overflow
#define SPEED_OF_LIGHT_RAW 299792458LL
#define PHYSICS_C (FixedMathCore(SPEED_OF_LIGHT_RAW))

// Universal Gravitational Constant G: 6.67430e-11
// Represented as a ratio to maintain bit-perfection
#define PHYSICS_G (FixedMathCore("0.000000000066743"))

// ============================================================================
// High-Performance Deterministic Macros
// ============================================================================

#define MATH_PI FixedMathCore::pi()
#define MATH_TAU (FixedMathCore(FixedMathCore::TWO_PI_RAW, true))
#define MATH_E FixedMathCore::e()

#ifndef MIN
#define MIN(m_a, m_b) (((m_a) < (m_b)) ? (m_a) : (m_b))
#endif

#ifndef MAX
#define MAX(m_a, m_b) (((m_a) > (m_b)) ? (m_a) : (m_b))
#endif

#ifndef CLAMP
#define CLAMP(m_a, m_min, m_max) (((m_a) < (m_min)) ? (m_min) : (((m_a) > (m_max)) ? (m_max) : (m_a)))
#endif

#ifndef ABS
#define ABS(m_v) (((m_v) < FixedMathCore(0LL, true)) ? -(m_v) : (m_v))
#endif

#define SIGN(m_v) (((m_v) > FixedMathCore(0LL, true)) ? 1 : (((m_v) < FixedMathCore(0LL, true)) ? -1 : 0))

// ============================================================================
// Simulation Tier Logic
// ============================================================================

enum SimulationTier {
	TIER_STANDARD,       // Legacy / UI Reference
	TIER_DETERMINISTIC,  // FixedMathCore (Local Physics, CCD, 120 FPS)
	TIER_MACRO_ECONOMY   // BigIntCore (Stellar mass, Infinite Ledger, Galactic ID)
};

/**
 * MathDefs
 * 
 * Provides static access to unit conversions for the balloon effect,
 * flesh tensors, and relativistic kinematics.
 */
class MathDefs {
public:
	// 120 FPS Heartbeat Constants
	static _FORCE_INLINE_ FixedMathCore get_fixed_step() {
		return FixedMathCore(1LL, false) / FixedMathCore(120LL, false);
	}

	// Material Tensors: Unit conversion for "Balloon" pressure stiffness
	static _FORCE_INLINE_ FixedMathCore get_standard_stiffness() {
		return FixedMathCore(100LL, false); 
	}
};

#endif // MATH_DEFS_H

--- END OF FILE core/math/math_defs.h ---
