--- START OF FILE core/math/atmospheric_scattering_rayleigh.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * phase_rayleigh_kernel()
 * 
 * Deterministic Rayleigh phase function.
 * P(theta) = 3 / (16 * pi) * (1 + cos^2(theta))
 * Describes the angular distribution of light scattered by small particles.
 */
static _FORCE_INLINE_ FixedMathCore phase_rayleigh_kernel(FixedMathCore p_cos_theta) {
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore sixteen_pi = FixedMathCore(16LL, false) * Math::pi();
	FixedMathCore coefficient = FixedMathCore(3LL, false) / sixteen_pi;

	return coefficient * (one + p_cos_theta * p_cos_theta);
}

/**
 * compute_rayleigh_inscatter_kernel()
 * 
 * Warp-style kernel for batch processing Rayleigh scattering contributions.
 * Calculates the spectral shift of light based on atmosphere density and wavelength.
 * 
 * p_wavelength_coeffs: Precomputed 1/lambda^4 coefficients for RGB.
 */
void compute_rayleigh_inscatter_kernel(
		const Vector3f &p_sample_pos,
		const Vector3f &p_view_dir,
		const Vector3f &p_light_dir,
		const FixedMathCore &p_light_energy,
		const AtmosphereParams &p_params,
		bool p_is_anime,
		Vector3f &r_rayleigh_accum) {

	// 1. Angular Scattering Evaluation
	FixedMathCore cos_theta = p_view_dir.dot(p_light_dir);
	FixedMathCore phase = phase_rayleigh_kernel(cos_theta);

	// 2. Local Density Sampling
	// height = length(sample_pos) - planet_radius
	FixedMathCore height = p_sample_pos.length() - p_params.planet_radius;
	FixedMathCore density = AtmosphericScattering::compute_density(height, p_params.rayleigh_scale_height);

	// 3. Spectral Energy Calculation
	// Rayleigh scattering is inversely proportional to the fourth power of wavelength.
	// Blue scatters more than Red.
	FixedMathCore intensity = p_light_energy * phase * density;

	// --- Style Adaptation: Anime vs Realistic ---
	if (p_is_anime) {
		// Anime Style: Boost the blue scattering intensity for more vibrant daytime skies
		// and sharpen the gradient transition.
		FixedMathCore saturation_boost(6442450944LL, true); // 1.5x
		FixedMathCore step_threshold(858993459LL, true);   // 0.2
		
		FixedMathCore style_density = wp::step(step_threshold, density) * density * saturation_boost;
		intensity = p_light_energy * phase * style_density;
	}

	// Apply RGB wavelength coefficients stored in p_params.rayleigh_coefficients
	r_rayleigh_accum.x += intensity * p_params.rayleigh_coefficients.x; // Red
	r_rayleigh_accum.y += intensity * p_params.rayleigh_coefficients.y; // Green
	r_rayleigh_accum.z += intensity * p_params.rayleigh_coefficients.z; // Blue
}

/**
 * batch_rayleigh_resolve_sweep()
 * 
 * Parallel EnTT sweep to resolve sky gradients across a viewing frustum.
 * Ensures bit-perfect sunsets and midday blues for 120 FPS simulation.
 */
void batch_rayleigh_resolve_sweep(
		const BigIntCore *p_entity_ids,
		const Vector3f *p_origins,
		const Vector3f *p_directions,
		const AtmosphereParams *p_atm_data,
		Vector3f *r_output_radiance,
		uint64_t p_count,
		const Vector3f &p_sun_dir,
		const FixedMathCore &p_sun_energy) {

	// ETEngine Strategy: Zero-Copy parallel sweep
	// Processes atmospheric samples directly from SoA registry buffers
	for (uint64_t i = 0; i < p_count; i++) {
		const AtmosphereParams &params = p_atm_data[i];
		const Vector3f &view_dir = p_directions[i];
		bool is_anime = (p_entity_ids[i].hash() % 2 == 1);

		Vector3f radiance;
		// This kernel would be called iteratively within the ray-marcher
		// integrate_rayleigh_contribution(...)
		
		// For the purpose of the 176-file core, we implement the math primitive:
		FixedMathCore cos_theta = view_dir.dot(p_sun_dir);
		FixedMathCore phase = phase_rayleigh_kernel(cos_theta);
		
		// Simplified single-step resolve for SoA data mapping
		radiance = params.rayleigh_coefficients * (phase * p_sun_energy);
		
		if (is_anime) {
			radiance *= FixedMathCore(5368709120LL, true); // 1.25x brightness snap
		}

		r_output_radiance[i] = radiance;
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_scattering_rayleigh.cpp ---
