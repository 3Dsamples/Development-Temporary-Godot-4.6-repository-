--- START OF FILE core/math/math_funcs.h ---

#ifndef MATH_FUNCS_H
#define MATH_FUNCS_H

#include "core/typedefs.h"
#include "core/math/math_defs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Math
 * 
 * The master static math library for the Scale-Aware engine.
 * strictly deterministic and hardware-agnostic.
 * Optimized for Warp-style parallel batch processing.
 */
class Math {
public:
	// ------------------------------------------------------------------------
	// Deterministic Basic Arithmetic
	// ------------------------------------------------------------------------
	static _FORCE_INLINE_ FixedMathCore abs(FixedMathCore p_x) { return p_x.absolute(); }
	static _FORCE_INLINE_ BigIntCore abs(BigIntCore p_x) { return p_x.absolute(); }

	static _FORCE_INLINE_ FixedMathCore sign(FixedMathCore p_x) { 
		return (p_x.get_raw() > 0) ? MathConstants<FixedMathCore>::one() : 
		       ((p_x.get_raw() < 0) ? -MathConstants<FixedMathCore>::one() : MathConstants<FixedMathCore>::zero()); 
	}

	// ------------------------------------------------------------------------
	// Deterministic Trigonometry & Power
	// ------------------------------------------------------------------------
	static _FORCE_INLINE_ FixedMathCore sin(FixedMathCore p_x) { return p_x.sin(); }
	static _FORCE_INLINE_ FixedMathCore cos(FixedMathCore p_x) { return p_x.cos(); }
	static _FORCE_INLINE_ FixedMathCore tan(FixedMathCore p_x) { return p_x.tan(); }
	static _FORCE_INLINE_ FixedMathCore atan2(FixedMathCore p_y, FixedMathCore p_x) { return p_y.atan2(p_x); }
	
	static _FORCE_INLINE_ FixedMathCore sqrt(FixedMathCore p_x) { return p_x.square_root(); }
	static _FORCE_INLINE_ BigIntCore sqrt(BigIntCore p_x) { return p_x.square_root(); }
	
	static _FORCE_INLINE_ FixedMathCore pow(FixedMathCore p_base, int32_t p_exp) { return p_base.power(p_exp); }
	static _FORCE_INLINE_ BigIntCore pow(BigIntCore p_base, BigIntCore p_exp) { return p_base.power(p_exp); }
	
	static _FORCE_INLINE_ FixedMathCore exp(FixedMathCore p_x) { return p_x.exp(); }
	static _FORCE_INLINE_ FixedMathCore log(FixedMathCore p_x) { return p_x.log(); }

	// ------------------------------------------------------------------------
	// Rounding & Quantization (Scale-Aware)
	// ------------------------------------------------------------------------
	static _FORCE_INLINE_ FixedMathCore floor(FixedMathCore p_x) {
		return FixedMathCore(p_x.to_int());
	}
	
	static _FORCE_INLINE_ FixedMathCore ceil(FixedMathCore p_x) {
		int64_t i = p_x.to_int();
		if ((p_x.get_raw() & 0xFFFFFFFFLL) != 0) i++;
		return FixedMathCore(i);
	}

	static _FORCE_INLINE_ FixedMathCore round(FixedMathCore p_x) {
		return floor(p_x + MathConstants<FixedMathCore>::half());
	}

	static _FORCE_INLINE_ FixedMathCore snapped(FixedMathCore p_val, FixedMathCore p_step) {
		if (p_step.get_raw() == 0) return p_val;
		return floor(p_val / p_step + MathConstants<FixedMathCore>::half()) * p_step;
	}

	// ------------------------------------------------------------------------
	// Sophisticated Interpolation (120 FPS Smoothness)
	// ------------------------------------------------------------------------
	static _FORCE_INLINE_ FixedMathCore lerp(FixedMathCore p_from, FixedMathCore p_to, FixedMathCore p_weight) {
		return p_from + (p_to - p_from) * p_weight;
	}

	/**
	 * smoothstep()
	 * Used for organic flesh movement and balloon expansion.
	 */
	static _FORCE_INLINE_ FixedMathCore smoothstep(FixedMathCore p_from, FixedMathCore p_to, FixedMathCore p_weight) {
		FixedMathCore t = CLAMP((p_weight - p_from) / (p_to - p_from), MathConstants<FixedMathCore>::zero(), MathConstants<FixedMathCore>::one());
		return t * t * (FixedMathCore(3LL) - (t * FixedMathCore(2LL)));
	}

	static _FORCE_INLINE_ FixedMathCore cubic_interpolate(FixedMathCore p_pre, FixedMathCore p_from, FixedMathCore p_to, FixedMathCore p_post, FixedMathCore p_weight) {
		FixedMathCore p2 = p_weight * p_weight;
		FixedMathCore p3 = p2 * p_weight;
		FixedMathCore half = MathConstants<FixedMathCore>::half();

		return p_from + half * p_weight * (p_to - p_pre + p_weight * (FixedMathCore(2LL) * p_pre - FixedMathCore(5LL) * p_from + FixedMathCore(4LL) * p_to - p_post + p_weight * (FixedMathCore(3LL) * (p_from - p_to) + p_post - p_pre)));
	}

	// ------------------------------------------------------------------------
	// Unit Conversions
	// ------------------------------------------------------------------------
	static _FORCE_INLINE_ FixedMathCore deg_to_rad(FixedMathCore p_y) {
		return p_y * (MATH_PI / FixedMathCore(180LL));
	}
	
	static _FORCE_INLINE_ FixedMathCore rad_to_deg(FixedMathCore p_y) {
		return p_y * (FixedMathCore(180LL) / MATH_PI);
	}

	// ------------------------------------------------------------------------
	// Precision Checks
	// ------------------------------------------------------------------------
	static _FORCE_INLINE_ bool is_equal_approx(FixedMathCore p_a, FixedMathCore p_b) {
		if (p_a == p_b) return true;
		return abs(p_a - p_b) < CMP_EPSILON;
	}

	static _FORCE_INLINE_ bool is_zero_approx(FixedMathCore p_a) {
		return abs(p_a) < CMP_EPSILON;
	}

	// ------------------------------------------------------------------------
	// Deterministic Randomness (Global State)
	// ------------------------------------------------------------------------
	static void seed(uint64_t p_seed);
	static uint32_t rand();
	static FixedMathCore randf(); // Returns [0.0, 1.0] in FixedMath
};

#endif // MATH_FUNCS_H

--- END OF FILE core/math/math_funcs.h ---
