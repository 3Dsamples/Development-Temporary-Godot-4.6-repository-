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
 * Engineered for hardware-agnostic execution in the Scale-Aware pipeline.
 * Aligned to 32 bytes for zero-copy EnTT component batching.
 */
template <typename T>
class ET_ALIGN_32 RandomDist {
public:
	// ------------------------------------------------------------------------
	// Deterministic Sampling API (FixedMathCore)
	// ------------------------------------------------------------------------

	/**
	 * gaussian()
	 * 
	 * Returns a normally distributed value using the bit-perfect Box-Muller transform.
	 * Critical for Structural Fatigue Modeling and localized physical variations.
	 * Formula: Z = sqrt(-2 * ln(U1)) * cos(2 * pi * U2)
	 */
	static ET_SIMD_INLINE T gaussian(RandomPCG &p_pcg, T p_mean = MathConstants<T>::zero(), T p_stddev = MathConstants<T>::one()) {
		T u1 = p_pcg.randf();
		T u2 = p_pcg.randf();

		// Ensure u1 is non-zero to prevent log(0) singularity
		if (unlikely(u1.get_raw() == 0)) {
			u1 = T(1LL, true); // Smallest possible non-zero Q32.32
		}

		T minus_two = T(-2LL);
		T r = Math::sqrt(minus_two * Math::log(u1));
		T theta = MathConstants<T>::two_pi() * u2;
		
		return (r * Math::cos(theta)) * p_stddev + p_mean;
	}

	/**
	 * exponential()
	 * 
	 * Samples from an exponential distribution. 
	 * Used for timing independent events like economic ticks or particle decay.
	 * Formula: X = -ln(U) / lambda
	 */
	static ET_SIMD_INLINE T exponential(RandomPCG &p_pcg, T p_lambda) {
		T u = p_pcg.randf();
		if (unlikely(u.get_raw() == 0)) u = T(1LL, true);
		
		if (unlikely(p_lambda.get_raw() == 0)) return MathConstants<T>::zero();
		
		return -(Math::log(u) / p_lambda);
	}

	/**
	 * cauchy()
	 * 
	 * Samples from a Cauchy distribution (heavy-tailed).
	 * Used for simulating extreme outliers in macro-simulation tiers.
	 * Formula: X = median + scale * tan(pi * (U - 0.5))
	 */
	static ET_SIMD_INLINE T cauchy(RandomPCG &p_pcg, T p_median, T p_scale) {
		T u = p_pcg.randf();
		T half = MathConstants<T>::half();
		T pi = MathConstants<T>::pi();
		
		// Map [0,1] to [-pi/2, pi/2]
		T angle = pi * (u - half);
		return p_median + p_scale * Math::tan(angle);
	}

	// ------------------------------------------------------------------------
	// Atmospheric Scattering Distributions (Phase Functions)
	// ------------------------------------------------------------------------

	/**
	 * rayleigh_phase()
	 * 
	 * Analytical phase function for scattering by small molecules.
	 * P(theta) = 3 / (16 * pi) * (1 + cos^2(theta))
	 */
	static ET_SIMD_INLINE T rayleigh_phase(T p_cos_theta) {
		T one = MathConstants<T>::one();
		// Constant 3 / (16 * pi) in bit-perfect FixedMath
		T factor("0.0596831036");
		return factor * (one + p_cos_theta * p_cos_theta);
	}

	/**
	 * mie_phase_hg()
	 * 
	 * Henyey-Greenstein approximation for Mie scattering (haze/clouds).
	 * P(theta) = (1 - g^2) / (4*pi * (1 + g^2 - 2*g*cos(theta))^1.5)
	 */
	static ET_SIMD_INLINE T mie_phase_hg(T p_cos_theta, T p_g) {
		T one = MathConstants<T>::one();
		T two = T(2LL);
		T four_pi = T(4LL) * MathConstants<T>::pi();
		
		T g2 = p_g * p_g;
		T denom_inner = one + g2 - (two * p_g * p_cos_theta);
		
		// x^1.5 = sqrt(x^3)
		T denom = four_pi * Math::sqrt(denom_inner * denom_inner * denom_inner);
		
		if (unlikely(denom.get_raw() == 0)) return one;
		
		return (one - g2) / denom;
	}

	/**
	 * laplace()
	 * 
	 * Double-exponential distribution used in advanced signal processing.
	 */
	static ET_SIMD_INLINE T laplace(RandomPCG &p_pcg, T p_mu, T p_b) {
		T u = p_pcg.randf() - MathConstants<T>::half();
		T sign_u = (u.get_raw() > 0) ? MathConstants<T>::one() : (u.get_raw() < 0 ? -MathConstants<T>::one() : MathConstants<T>::zero());
		
		T inner = MathConstants<T>::one() - (Math::abs(u) * T(2LL));
		if (inner.get_raw() <= 0) inner = T( unit_epsilon_raw, true);
		
		return p_mu - p_b * sign_u * Math::log(inner);
	}
};

// Global typedef for the deterministic physics tier
typedef RandomDist<FixedMathCore> RandomDistf;

#endif // RANDOM_DIST_H

--- END OF FILE core/math/random_dist.h ---
