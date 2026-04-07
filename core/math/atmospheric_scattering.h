--- START OF FILE core/math/atmospheric_scattering.h ---

#ifndef ATMOSPHERIC_SCATTERING_H
#define ATMOSPHERIC_SCATTERING_H

#include "core/typedefs.h"
#include "core/math/vector3.h"
#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * AtmosphericScattering
 * 
 * High-performance deterministic light scattering engine.
 * Solves the radiative transfer equation using bit-perfect FixedMathCore.
 * Aligned for SIMD/Warp batch processing of atmospheric density volumes.
 */
class ET_ALIGN_32 AtmosphericScattering {
public:
	struct ET_ALIGN_32 AtmosphereParams {
		Vector3f rayleigh_coefficients; // Scattering coeffs for RGB wavelengths
		FixedMathCore mie_coefficient;
		FixedMathCore rayleigh_scale_height;
		FixedMathCore mie_scale_height;
		FixedMathCore planet_radius;
		FixedMathCore atmosphere_radius;
		FixedMathCore mie_g; // Asymmetry factor for Henyey-Greenstein
	};

	// ------------------------------------------------------------------------
	// Deterministic Physical Kernels
	// ------------------------------------------------------------------------

	/**
	 * compute_density()
	 * Calculates atmospheric density at a specific altitude using exponential decay.
	 * Uses FixedMathCore::exp() for bit-perfect atmospheric profiling.
	 */
	static _FORCE_INLINE_ FixedMathCore compute_density(FixedMathCore p_altitude, FixedMathCore p_scale_height) {
		if (p_altitude.get_raw() < 0) return MathConstants<FixedMathCore>::one();
		// density = exp(-h / H)
		return Math::exp(-(p_altitude / p_scale_height));
	}

	/**
	 * phase_rayleigh()
	 * 3/16pi * (1 + cos^2(theta))
	 */
	static _FORCE_INLINE_ FixedMathCore phase_rayleigh(FixedMathCore p_cos_theta) {
		FixedMathCore factor = FixedMathCore(3LL, false) / (FixedMathCore(16LL, false) * Math::pi());
		return factor * (MathConstants<FixedMathCore>::one() + p_cos_theta * p_cos_theta);
	}

	/**
	 * phase_mie()
	 * Henyey-Greenstein phase function approximation.
	 */
	static _FORCE_INLINE_ FixedMathCore phase_mie(FixedMathCore p_cos_theta, FixedMathCore p_g) {
		FixedMathCore g2 = p_g * p_g;
		FixedMathCore one = MathConstants<FixedMathCore>::one();
		FixedMathCore two = FixedMathCore(2LL, false);
		
		FixedMathCore denom = one + g2 - two * p_g * p_cos_theta;
		FixedMathCore factor = (one - g2) / (FixedMathCore(4LL, false) * Math::pi());
		
		// result = factor * (1 + cos^2) / ((1 + g^2 - 2gcos)^(3/2))
		// Note: Simplified for performance while maintaining deterministic curvature
		return factor * (one + p_cos_theta * p_cos_theta) / (denom * Math::sqrt(denom));
	}

	// ------------------------------------------------------------------------
	// Batch Integration API (Warp-Kernel Ready)
	// ------------------------------------------------------------------------

	/**
	 * compute_optical_depth()
	 * Integrates density along a ray from point A to B.
	 * Zero-copy logic for EnTT component streams.
	 */
	static FixedMathCore compute_optical_depth(
			const Vector3f &p_origin, 
			const Vector3f &p_direction, 
			FixedMathCore p_limit, 
			FixedMathCore p_scale_height, 
			int p_samples);

	/**
	 * resolve_sky_color()
	 * The master scattering kernel. Combines Rayleigh and Mie contributions.
	 * p_index used for deterministic dithering/noise injection via BigIntCore handles.
	 */
	static Vector3f resolve_sky_color(
			const BigIntCore &p_entity_id,
			const Vector3f &p_view_dir,
			const Vector3f &p_sun_dir,
			const AtmosphereParams &p_params);
};

#endif // ATMOSPHERIC_SCATTERING_H

--- END OF FILE core/math/atmospheric_scattering.h ---
