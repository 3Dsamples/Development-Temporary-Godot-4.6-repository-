--- START OF FILE core/math/random_pcg.cpp ---

#include "core/math/random_pcg.h"
#include "core/math/math_funcs.h"
#include <string>

/**
 * rand_big()
 * 
 * Generates an arbitrary-precision random integer between p_from and p_to.
 * It populates BigIntCore chunks using the PCG entropy source, ensuring
 * that galactic-scale procedural generation (like star positions in a sector)
 * remains perfectly deterministic regardless of the numeric magnitude.
 */
BigIntCore RandomPCG::rand_big(const BigIntCore &p_from, const BigIntCore &p_to) {
	if (p_from == p_to) return p_from;

	BigIntCore min_v = p_from;
	BigIntCore max_v = p_to;
	if (p_to < p_from) {
		min_v = p_to;
		max_v = p_from;
	}

	BigIntCore range = max_v - min_v + BigIntCore(1);
	std::string range_str = range.to_string();
	
	// Determine how many digits we need to generate
	size_t digits_needed = range_str.length();
	std::string result_str = "";

	// Generate random digits using the PCG source
	for (size_t i = 0; i < digits_needed; ++i) {
		uint32_t digit = rand() % 10;
		result_str += std::to_string(digit);
	}

	BigIntCore random_offset(result_str);
	return (random_offset % range) + min_v;
}

/**
 * rand_gaussian_fixed()
 * 
 * Implements a deterministic Box-Muller transform using FixedMathCore.
 * Provides normal distribution variance for physical simulations (e.g., 
 * material fatigue or particle scattering) without FPU drift.
 */
FixedMathCore rand_gaussian_fixed(RandomPCG &p_pcg, FixedMathCore p_mean, FixedMathCore p_stddev) {
	FixedMathCore u1 = p_pcg.randf();
	FixedMathCore u2 = p_pcg.randf();

	// Ensure u1 is not zero to prevent log(0)
	if (u1.get_raw() == 0) {
		u1 = FixedMathCore(1LL, true);
	}

	// Box-Muller: z0 = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)
	FixedMathCore minus_two(-2LL, false);
	FixedMathCore ln_u1 = Math::log(u1); // Software-defined ln
	FixedMathCore r = Math::sqrt(minus_two * ln_u1);
	FixedMathCore theta = Math::tau() * u2;
	
	return (r * Math::cos(theta)) * p_stddev + p_mean;
}

/**
 * rand_unit_vector_f()
 * 
 * Generates a random 3D unit vector using deterministic math.
 * Essential for Warp-style kernels performing isotropic force distribution.
 */
Vector3f rand_unit_vector_f(RandomPCG &p_pcg) {
	FixedMathCore z = p_pcg.rand_range_fixed(FixedMathCore(-1LL, false), FixedMathCore(1LL, false));
	FixedMathCore phi = p_pcg.rand_range_fixed(FixedMathCore(0LL, true), Math::tau());
	
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore r = Math::sqrt(one - z * z);
	
	return Vector3f(r * Math::cos(phi), r * Math::sin(phi), z);
}

--- END OF FILE core/math/random_pcg.cpp ---
