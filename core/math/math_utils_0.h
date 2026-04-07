--- START OF FILE core/math/math_utils.h ---

#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * MathUtils
 * 
 * High-level deterministic math utilities for the Universal Solver.
 * Every function is designed to be inline-expanded within Warp kernels
 * to maintain 120 FPS during massive procedural simulation sweeps.
 */
class MathUtils {
public:
	// ------------------------------------------------------------------------
	// Deterministic Interpolation & Smoothing
	// ------------------------------------------------------------------------

	static ET_SIMD_INLINE FixedMathCore smoothstep(FixedMathCore p_from, FixedMathCore p_to, FixedMathCore p_weight) {
		if (p_from == p_to) return p_from;
		FixedMathCore zero = MathConstants<FixedMathCore>::zero();
		FixedMathCore one = MathConstants<FixedMathCore>::one();
		FixedMathCore v = CLAMP((p_weight - p_from) / (p_to - p_from), zero, one);
		// 3v^2 - 2v^3
		return v * v * (FixedMathCore(3LL, false) - FixedMathCore(2LL, false) * v);
	}

	static ET_SIMD_INLINE FixedMathCore wrap(FixedMathCore p_value, FixedMathCore p_min, FixedMathCore p_max) {
		FixedMathCore range = p_max - p_min;
		if (range.get_raw() == 0) return p_min;
		FixedMathCore result = p_value - (range * FixedMathCore(static_cast<int64_t>(Math::floor((p_value - p_min) / range).to_int())));
		if (result == p_max) return p_min;
		return result;
	}

	/**
	 * wrap_bigint()
	 * Essential for wrapping galactic coordinates in discrete infinite grids.
	 */
	static ET_SIMD_INLINE BigIntCore wrap_bigint(BigIntCore p_value, BigIntCore p_min, BigIntCore p_max) {
		BigIntCore range = p_max - p_min;
		if (range.is_zero()) return p_min;
		BigIntCore res = (p_value - p_min) % range;
		if (res.sign() < 0) res += range;
		return res + p_min;
	}

	// ------------------------------------------------------------------------
	// Bit-Perfect Bezier Logic
	// ------------------------------------------------------------------------

	static ET_SIMD_INLINE FixedMathCore bezier_interpolate(FixedMathCore p_start, FixedMathCore p_control_1, FixedMathCore p_control_2, FixedMathCore p_end, FixedMathCore p_t) {
		FixedMathCore _1mt = MathConstants<FixedMathCore>::one() - p_t;
		FixedMathCore _1mt2 = _1mt * _1mt;
		FixedMathCore _1mt3 = _1mt2 * _1mt;
		FixedMathCore t2 = p_t * p_t;
		FixedMathCore t3 = t2 * p_t;

		return p_start * _1mt3 + p_control_1 * FixedMathCore(3LL, false) * p_t * _1mt2 + p_control_2 * FixedMathCore(3LL, false) * t2 * _1mt + p_end * t3;
	}

	// ------------------------------------------------------------------------
	// Scientific Precision Helpers
	// ------------------------------------------------------------------------

	/**
	 * lerp_bigint()
	 * Scales arbitrary precision values using a deterministic fixed-point weight.
	 * Used for economic growth models and distance scaling.
	 */
	static inline BigIntCore lerp_bigint(BigIntCore p_from, BigIntCore p_to, FixedMathCore p_weight) {
		if (p_weight.get_raw() <= 0) return p_from;
		if (p_weight.get_raw() >= FixedMathCore::ONE) return p_to;
		
		BigIntCore diff = p_to - p_from;
		// Multiply BigInt by Fixed fractional part
		int64_t w_int = p_weight.to_int();
		uint64_t w_frac = static_cast<uint64_t>(p_weight.get_raw() & 0xFFFFFFFFULL);
		
		BigIntCore result = p_from + (diff * BigIntCore(w_int));
		if (w_frac > 0) {
			// Precision boost for sub-unit interpolation
			result += (diff * BigIntCore(static_cast<int64_t>(w_frac))) / BigIntCore(FixedMathCore::ONE);
		}
		return result;
	}
};

#endif // MATH_UTILS_H

--- END OF FILE core/math/math_utils.h ---
