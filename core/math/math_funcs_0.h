--- START OF FILE core/math/math_funcs.h ---

#ifndef MATH_FUNCS_H
#define MATH_FUNCS_H

#include "core/typedefs.h"
#include "core/error/error_macros.h"
#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

// ============================================================================
// Core Math Macros (Branchless SIMD-Friendly)
// ============================================================================

#define MIN(m_a, m_b) (((m_a) < (m_b)) ? (m_a) : (m_b))
#define MAX(m_a, m_b) (((m_a) > (m_b)) ? (m_a) : (m_b))
#define CLAMP(m_a, m_min, m_max) (((m_a) < (m_min)) ? (m_min) : (((m_a) > (m_max)) ? (m_max) : (m_a)))
#define ABS(m_v) (((m_v) < 0) ? -(m_v) : (m_v))
#define SIGN(m_v) (((m_v) > 0) ? 1 : (((m_v) < 0) ? -1 : 0))

/**
 * Math Class
 * 
 * The central mathematical authority for the Universal Solver.
 * Replaces standard C math with bit-perfect Software-Defined Arithmetic.
 */
class Math {
public:
	// ------------------------------------------------------------------------
	// Deterministic Constants (FixedMathCore Q32.32)
	// ------------------------------------------------------------------------
	static inline FixedMathCore pi() { return FixedMathCore::pi(); }
	static inline FixedMathCore tau() { return FixedMathCore(FixedMathCore::TWO_PI_RAW, true); }
	static inline FixedMathCore e() { return FixedMathCore::e(); }

	// ------------------------------------------------------------------------
	// Deterministic Transcendental Functions (FixedMathCore)
	// ------------------------------------------------------------------------
	static inline FixedMathCore sin(FixedMathCore p_x) { return p_x.sin(); }
	static inline FixedMathCore cos(FixedMathCore p_x) { return p_x.cos(); }
	static inline FixedMathCore tan(FixedMathCore p_x) { return p_x.tan(); }
	static inline FixedMathCore atan2(FixedMathCore p_y, FixedMathCore p_x) { return p_y.atan2(p_x); }
	static inline FixedMathCore sqrt(FixedMathCore p_x) { return p_x.square_root(); }
	static inline FixedMathCore pow(FixedMathCore p_base, int32_t p_exp) { return p_base.power(p_exp); }
	static inline FixedMathCore abs(FixedMathCore p_x) { return p_x.absolute(); }

	// ------------------------------------------------------------------------
	// Arbitrary-Precision Math (BigIntCore)
	// ------------------------------------------------------------------------
	static inline BigIntCore abs(BigIntCore p_x) { return p_x.absolute(); }
	static inline BigIntCore sqrt(BigIntCore p_x) { return p_x.square_root(); }
	static inline BigIntCore pow(BigIntCore p_base, BigIntCore p_exp) { return p_base.power(p_exp); }

	// ------------------------------------------------------------------------
	// Batch-Oriented Interpolation (Warp Zero-Copy)
	// ------------------------------------------------------------------------
	static inline FixedMathCore lerp(FixedMathCore p_from, FixedMathCore p_to, FixedMathCore p_weight) {
		return p_from + (p_to - p_from) * p_weight;
	}

	static inline FixedMathCore cubic_interpolate(FixedMathCore p_from, FixedMathCore p_to, FixedMathCore p_pre, FixedMathCore p_post, FixedMathCore p_weight) {
		FixedMathCore p2 = p_weight * p_weight;
		FixedMathCore p3 = p2 * p_weight;
		FixedMathCore half(2147483648LL, true); // 0.5

		return p_from + half * p_weight * (p_to - p_pre + p_weight * (FixedMathCore(2) * p_pre - FixedMathCore(5) * p_from + FixedMathCore(4) * p_to - p_post + p_weight * (FixedMathCore(3) * (p_from - p_to) + p_post - p_pre)));
	}

	// ------------------------------------------------------------------------
	// Utilities & Snapping
	// ------------------------------------------------------------------------
	static inline bool is_equal_approx(FixedMathCore p_a, FixedMathCore p_b) {
		return p_a == p_b; // Deterministic math uses bit-equality
	}

	static inline FixedMathCore snapped(FixedMathCore p_value, FixedMathCore p_step) {
		if (p_step.get_raw() == 0) return p_value;
		FixedMathCore half(2147483648LL, true);
		FixedMathCore div = p_value / p_step;
		// Manual floor by bit-truncation
		int64_t raw_div = (div + half).get_raw();
		int64_t floored = (raw_div >> 32) << 32;
		return FixedMathCore(floored, true) * p_step;
	}

	// Internal random state for PCG (Permuted Congruential Generator)
	static void seed(uint64_t p_seed);
	static uint32_t rand();
	static FixedMathCore randf(); // Returns FixedMathCore in range [0, 1]
};

#endif // MATH_FUNCS_H

--- END OF FILE core/math/math_funcs.h ---
