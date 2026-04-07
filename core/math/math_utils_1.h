--- START OF FILE core/math/math_utils.h ---

#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * MathUtils
 * 
 * High-level deterministic math utilities for the Scale-Aware engine.
 * Engineered for zero-copy execution within EnTT registries.
 */
class MathUtils {
public:
	// ------------------------------------------------------------------------
	// Deterministic Interpolation & Smoothing (120 FPS Optimized)
	// ------------------------------------------------------------------------

	/**
	 * smoothstep()
	 * Returns 0.0 to 1.0 based on a Hermite polynomial.
	 * Used for organic flesh movement and localized balloon expansion.
	 */
	static _FORCE_INLINE_ FixedMathCore smoothstep(FixedMathCore p_from, FixedMathCore p_to, FixedMathCore p_weight) {
		if (p_from == p_to) return p_from;
		FixedMathCore zero = MathConstants<FixedMathCore>::zero();
		FixedMathCore one = MathConstants<FixedMathCore>::one();
		FixedMathCore t = CLAMP((p_weight - p_from) / (p_to - p_from), zero, one);
		// 3t^2 - 2t^3
		return t * t * (FixedMathCore(3LL) - (t * FixedMathCore(2LL)));
	}

	/**
	 * bezier_interpolate()
	 * Bit-perfect Cubic Bezier for cinematic camera paths and robotic arm trajectories.
	 */
	static _FORCE_INLINE_ FixedMathCore bezier_interpolate(FixedMathCore p_start, FixedMathCore p_control_1, FixedMathCore p_control_2, FixedMathCore p_end, FixedMathCore p_t) {
		FixedMathCore _1mt = MathConstants<FixedMathCore>::one() - p_t;
		FixedMathCore _1mt2 = _1mt * _1mt;
		FixedMathCore _1mt3 = _1mt2 * _1mt;
		FixedMathCore t2 = p_t * p_t;
		FixedMathCore t3 = t2 * p_t;

		FixedMathCore term1 = p_start * _1mt3;
		FixedMathCore term2 = p_control_1 * FixedMathCore(3LL) * p_t * _1mt2;
		FixedMathCore term3 = p_control_2 * FixedMathCore(3LL) * t2 * _1mt;
		FixedMathCore term4 = p_end * t3;

		return term1 + term2 + term3 + term4;
	}

	// ------------------------------------------------------------------------
	// Galactic Coordinate Wrapping (Scale-Aware)
	// ------------------------------------------------------------------------

	/**
	 * wrap_fixed()
	 * Wraps a continuous value within a deterministic range [min, max].
	 */
	static _FORCE_INLINE_ FixedMathCore wrap_fixed(FixedMathCore p_value, FixedMathCore p_min, FixedMathCore p_max) {
		FixedMathCore range = p_max - p_min;
		if (unlikely(range.get_raw() == 0)) return p_min;
		FixedMathCore res = p_value - (range * Math::floor((p_value - p_min) / range));
		if (res == p_max) return p_min;
		return res;
	}

	/**
	 * wrap_bigint()
	 * Essential for toroidal galactic coordinate systems and sector paging.
	 */
	static _FORCE_INLINE_ BigIntCore wrap_bigint(const BigIntCore &p_value, const BigIntCore &p_min, const BigIntCore &p_max) {
		BigIntCore range = p_max - p_min;
		if (unlikely(range.is_zero())) return p_min;
		BigIntCore res = (p_value - p_min) % range;
		if (res.sign() < 0) res += range;
		return res + p_min;
	}

	// ------------------------------------------------------------------------
	// Sophisticated Scaling Kernels (Cross-Tier)
	// ------------------------------------------------------------------------

	/**
	 * lerp_bigint()
	 * Scales an arbitrary-precision integer by a deterministic fixed-point weight.
	 * used for economic growth models and planet-scale distance interpolation.
	 */
	static BigIntCore lerp_bigint(const BigIntCore &p_from, const BigIntCore &p_to, const FixedMathCore &p_weight) {
		if (p_weight.get_raw() <= 0) return p_from;
		if (p_weight.get_raw() >= FixedMathCore::ONE_RAW) return p_to;
		
		BigIntCore diff = p_to - p_from;
		// Multiply by raw Q32.32 bits and then right-shift by 32
		BigIntCore scaled_diff = (diff * BigIntCore(p_weight.get_raw())) / BigIntCore(FixedMathCore::ONE_RAW);
		return p_from + scaled_diff;
	}

	/**
	 * remap()
	 * Translates a value from one range to another with bit-perfect accuracy.
	 */
	static _FORCE_INLINE_ FixedMathCore remap(FixedMathCore p_val, FixedMathCore p_istart, FixedMathCore p_istop, FixedMathCore p_ostart, FixedMathCore p_ostop) {
		return p_ostart + (p_ostop - p_ostart) * ((p_val - p_istart) / (p_istop - p_istart));
	}

	// ------------------------------------------------------------------------
	// Celestial Physics Helpers
	// ------------------------------------------------------------------------

	/**
	 * solve_kepler_eccentric_anomaly()
	 * Newton-Raphson solver for M = E - e * sin(E).
	 * Strictly deterministic FixedMath implementation.
	 */
	static FixedMathCore solve_kepler(FixedMathCore p_mean_anomaly, FixedMathCore p_eccentricity);
};

#endif // MATH_UTILS_H

--- END OF FILE core/math/math_utils.h ---
