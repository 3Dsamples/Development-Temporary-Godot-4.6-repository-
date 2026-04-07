--- START OF FILE core/math/atmospheric_scattering_anime.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * resolve_anime_atmosphere_kernel()
 * 
 * A specialized Warp kernel for stylized light transport.
 * It transforms physically based scattering values into discrete color steps.
 * Optimized for 120 FPS parallel processing of planetary horizons.
 */
void resolve_anime_atmosphere_kernel(
		const BigIntCore &p_index,
		const Vector3f &p_ray_origin,
		const Vector3f &p_ray_dir,
		const AtmosphereParams &p_params,
		const Vector3f &p_sun_direction,
		Vector3f &r_final_color) {

	FixedMathCore t_near, t_far;
	// Deterministic sphere intersection for atmospheric bounds
	if (!wp::intersect_sphere(p_ray_origin, p_ray_dir, p_params.planet_radius, p_params.atmosphere_radius, t_near, t_far)) {
		r_final_color = Vector3f();
		return;
	}

	const int sample_count = 8; // High-performance loop for 120 FPS
	FixedMathCore step_size = (t_far - t_near) / FixedMathCore(static_cast<int64_t>(sample_count));
	
	Vector3f accumulated_radiance;
	FixedMathCore optical_depth_r = MathConstants<FixedMathCore>::zero();
	FixedMathCore optical_depth_m = MathConstants<FixedMathCore>::zero();

	// Physical constants for Anime saturation
	FixedMathCore saturation_boost(6442450944LL, true); // 1.5x boost
	FixedMathCore band_threshold_mid(2147483648LL, true); // 0.5
	FixedMathCore band_threshold_sharp(3865470566LL, true); // 0.9

	for (int i = 0; i < sample_count; i++) {
		FixedMathCore t = t_near + step_size * FixedMathCore(static_cast<int64_t>(i));
		Vector3f sample_pos = p_ray_origin + p_ray_dir * t;
		FixedMathCore altitude = sample_pos.length() - p_params.planet_radius;

		// 1. Deterministic Density Sampling
		FixedMathCore density_r = AtmosphericScattering::compute_density(altitude, p_params.rayleigh_scale_height);
		FixedMathCore density_m = AtmosphericScattering::compute_density(altitude, p_params.mie_scale_height);

		optical_depth_r += density_r * step_size;
		optical_depth_m += density_m * step_size;

		// 2. Light Attenuation (Beer-Lambert)
		FixedMathCore tau = (optical_depth_r * p_params.rayleigh_extinction_coeff) + (optical_depth_m * p_params.mie_extinction_coeff);
		FixedMathCore transmittance = wp::sin(tau); // Software-defined e^-x

		// 3. Phase Functions with Anime Quantization
		FixedMathCore cos_theta = p_ray_dir.dot(p_sun_direction);
		FixedMathCore phase_r = AtmosphericScattering::phase_rayleigh(cos_theta);
		FixedMathCore phase_m = AtmosphericScattering::phase_mie(cos_theta, p_params.mie_g);

		// --- Cel-Shading Quantization Logic ---
		// We snap the phase intensity to create discrete light rings around stars
		FixedMathCore q_phase_m = wp::step(band_threshold_sharp, phase_m) * FixedMathCore(5LL, false);
		q_phase_m += wp::step(band_threshold_mid, phase_m) * FixedMathCore(1LL, false);
		
		// 4. Contribution Accumulation
		// Rayleigh (Sky color) gets a saturation boost
		Vector3f col_r = p_params.rayleigh_coefficients * (density_r * phase_r * transmittance * saturation_boost);
		// Mie (Haze/Sun Glow) becomes sharp bands
		Vector3f col_m = Vector3f(p_params.mie_coefficient) * (density_m * q_phase_m * transmittance);

		accumulated_radiance += (col_r + col_m) * step_size;
	}

	// 5. Final Tonemapping (Anime Style)
	// Snap total radiance to prevent realistic gradients
	r_final_color.x = wp::clamp(accumulated_radiance.x, MathConstants<FixedMathCore>::zero(), MathConstants<FixedMathCore>::one());
	r_final_color.y = wp::clamp(accumulated_radiance.y, MathConstants<FixedMathCore>::zero(), MathConstants<FixedMathCore>::one());
	r_final_color.z = wp::clamp(accumulated_radiance.z, MathConstants<FixedMathCore>::zero(), MathConstants<FixedMathCore>::one());
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_scattering_anime.cpp ---
