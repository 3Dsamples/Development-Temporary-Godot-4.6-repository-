--- START OF FILE core/math/random_dist.h ---

#ifndef RANDOM_DIST_H
#define RANDOM_DIST_H

#include "core/math/math_funcs.h"
#include "core/math/random_pcg.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * RandomDist Template
 * 
 * Provides deterministic probability density functions (PDF) and sampling.
 * Built for zero-copy execution within EnTT registries to handle millions of 
 * stochastic entities (particles, stars, micro-fractures) at 120 FPS.
 */
template <typename T>
class RandomDist {
public:
	// ------------------------------------------------------------------------
	// Deterministic Sampling API (FixedMathCore)
	// ------------------------------------------------------------------------

	/**
	 * gaussian()
	 * Returns a normally distributed value using the Box-Muller transform.
	 * Critical for Structural Fatigue Modeling and physical variations.
	 */
	static ET_SIMD_INLINE T gaussian(RandomPCG &p_pcg, T p_mean = MathConstants<T>::zero(), T p_stddev = MathConstants<T>::one()) {
		T u1 = p_pcg.randf();
		T u2 = p_pcg.randf();

		// Prevent log(0)
		if (u1.get_raw() == 0) u1 = FixedMathCore(1LL, true);

		T r = Math::sqrt(FixedMathCore(-2LL, false) * Math::log(u1));
		T theta = Math::tau() * u2;
		return (r * Math::cos(theta)) * p_stddev + p_mean;
	}

	/**
	 * exponential()
	 * Samples from an exponential distribution. 
	 * Used for timing independent events like radioactive decay or economic ticks.
	 */
	static ET_SIMD_INLINE T exponential(RandomPCG &p_pcg, T p_lambda) {
		T u = p_pcg.randf();
		if (u.get_raw() == 0) u = FixedMathCore(1LL, true);
		return -Math::log(u) / p_lambda;
	}

	/**
	 * cauchy()
	 * Samples from a Cauchy distribution (heavy-tailed).
	 * Used for simulating "Black Swan" events in macro-economies.
	 */
	static ET_SIMD_INLINE T cauchy(RandomPCG &p_pcg, T p_median, T p_scale) {
		T u = p_pcg.randf();
		// tan(pi * (u - 0.5))
		T val = Math::tan(Math::pi() * (u - MathConstants<T>::half()));
		return p_median + p_scale * val;
	}

	// ------------------------------------------------------------------------
	// Atmospheric Scattering Distributions
	// ------------------------------------------------------------------------

	/**
	 * rayleigh_phase()
	 * Returns the Rayleigh phase function for a given cosine of the scattering angle.
	 */
	static ET_SIMD_INLINE T rayleigh_phase(T p_cos_theta) {
		T factor = FixedMathCore(3LL, false) / (FixedMathCore(16LL, false) * Math::pi());
		return factor * (MathConstants<T>::one() + p_cos_theta * p_cos_theta);
	}

	/**
	 * mie_phase_hg()
	 * Henyey-Greenstein approximation for Mie scattering.
	 */
	static ET_SIMD_INLINE T mie_phase_hg(T p_cos_theta, T p_g) {
		T g2 = p_g * p_g;
		T one = MathConstants<T>::one();
		T denom = one + g2 - FixedMathCore(2LL, false) * p_g * p_cos_theta;
		T factor = (one - g2) / (FixedMathCore(4LL, false) * Math::pi());
		return factor / (denom * Math::sqrt(denom));
	}
};

typedef RandomDist<FixedMathCore> RandomDistf;

#endif // RANDOM_DIST_H

--- END OF FILE core/math/random_dist.h ---
