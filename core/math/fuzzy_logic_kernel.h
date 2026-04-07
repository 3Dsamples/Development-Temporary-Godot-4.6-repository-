--- START OF FILE core/math/fuzzy_logic_kernel.h ---

#ifndef FUZZY_LOGIC_KERNEL_H
#define FUZZY_LOGIC_KERNEL_H

#include "core/typedefs.h"
#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * FuzzyLogicKernel
 * 
 * Provides deterministic fuzzy membership functions and inference operators.
 * Engineered for hardware-agnostic AI state resolution in the Universal Solver.
 * All grades are represented in FixedMathCore [0.0, 1.0].
 */
class ET_ALIGN_32 FuzzyLogicKernel {
public:
	// ------------------------------------------------------------------------
	// Membership Functions (Fuzzification)
	// ------------------------------------------------------------------------

	/**
	 * triangle()
	 * Returns the membership grade for a triangular function.
	 * p_a: start, p_b: peak, p_c: end.
	 */
	static _FORCE_INLINE_ FixedMathCore triangle(FixedMathCore p_x, FixedMathCore p_a, FixedMathCore p_b, FixedMathCore p_c) {
		FixedMathCore zero = MathConstants<FixedMathCore>::zero();
		if (p_x <= p_a || p_x >= p_c) return zero;
		if (p_x == p_b) return MathConstants<FixedMathCore>::one();
		
		if (p_x < p_b) {
			return (p_x - p_a) / (p_b - p_a);
		} else {
			return (p_c - p_x) / (p_c - p_b);
		}
	}

	/**
	 * trapezoid()
	 * Returns the membership grade for a trapezoidal function.
	 * p_a/p_d: feet, p_b/p_c: plateau.
	 */
	static _FORCE_INLINE_ FixedMathCore trapezoid(FixedMathCore p_x, FixedMathCore p_a, FixedMathCore p_b, FixedMathCore p_c, FixedMathCore p_d) {
		FixedMathCore zero = MathConstants<FixedMathCore>::zero();
		FixedMathCore one = MathConstants<FixedMathCore>::one();
		
		FixedMathCore term1 = (p_x - p_a) / (p_b - p_a);
		FixedMathCore term2 = (p_d - p_x) / (p_d - p_c);
		
		// Return max(0, min(term1, 1, term2))
		FixedMathCore result = term1 < one ? term1 : one;
		if (term2 < result) result = term2;
		return result > zero ? result : zero;
	}

	// ------------------------------------------------------------------------
	// Fuzzy Operators (Inference)
	// ------------------------------------------------------------------------

	static _FORCE_INLINE_ FixedMathCore fuzzy_and(FixedMathCore p_a, FixedMathCore p_b) {
		return p_a < p_b ? p_a : p_b; // Zadeh AND (min)
	}

	static _FORCE_INLINE_ FixedMathCore fuzzy_or(FixedMathCore p_a, FixedMathCore p_b) {
		return p_a > p_b ? p_a : p_b; // Zadeh OR (max)
	}

	static _FORCE_INLINE_ FixedMathCore fuzzy_not(FixedMathCore p_a) {
		return MathConstants<FixedMathCore>::one() - p_a;
	}

	// ------------------------------------------------------------------------
	// Defuzzification Kernels (Action)
	// ------------------------------------------------------------------------

	/**
	 * resolve_weighted_average()
	 * 
	 * Calculates the crisp output from multiple fuzzy rules.
	 * Kernel-ready for zero-copy EnTT SoA data streams.
	 */
	static FixedMathCore resolve_weighted_average(const FixedMathCore *p_memberships, const FixedMathCore *p_centroids, uint32_t p_count) {
		FixedMathCore numerator = MathConstants<FixedMathCore>::zero();
		FixedMathCore denominator = MathConstants<FixedMathCore>::zero();

		for (uint32_t i = 0; i < p_count; i++) {
			numerator += p_memberships[i] * p_centroids[i];
			denominator += p_memberships[i];
		}

		if (unlikely(denominator.get_raw() == 0)) {
			return MathConstants<FixedMathCore>::zero();
		}

		return numerator / denominator;
	}

	/**
	 * resolve_macro_priority()
	 * 
	 * Uses BigIntCore to weight fuzzy decisions based on galactic-scale 
	 * resource counts or entity populations.
	 */
	static BigIntCore resolve_macro_priority(const FixedMathCore &p_fuzzy_grade, const BigIntCore &p_scale_factor) {
		// Convert Fixed Grade [0..1] to a discrete multiplier
		return (p_scale_factor * BigIntCore(p_fuzzy_grade.get_raw())) / BigIntCore(FixedMathCore::ONE);
	}
};

#endif // FUZZY_LOGIC_KERNEL_H

--- END OF FILE core/math/fuzzy_logic_kernel.h ---
