--- START OF FILE core/math/atmospheric_scattering_anime.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: AnimeAtmosphereResolveKernel
 * 
 * Performs a deterministic ray-march through the atmosphere to resolve 
 * stylized "Anime" visuals.
 * 1. Quantizes light intensity into discrete bands (Cel-shading).
 * 2. Applies Relativistic Doppler shifting to sky hues based on ship velocity.
 * 3. Injects exaggerated Mie halos around light sources.
 */
void resolve_anime_atmosphere_kernel(
		const BigIntCore &p_index,
		Vector3f &r_final_color,
		const Vector3f &p_ray_origin,
		const Vector3f &p_ray_dir,
		const Vector3f &p_ship_vel,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_lights) {

	FixedMathCore t_near, t_far;
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Intersection with Atmospheric Shell
	if (!wp::intersect_sphere(p_ray_origin, p_ray_dir, p_params.planet_radius, p_params.atmosphere_radius, t_near, t_far)) {
		r_final_color = Vector3f();
		return;
	}

	// 2. High-Speed Sophistication: Relativistic Doppler Color-Shift
	// sky_lambda = lambda / (1 + v_radial / c)
	FixedMathCore radial_v = p_ray_dir.dot(p_ship_vel);
	FixedMathCore c_fixed(1287458867200LL, true); // Speed of light approx
	FixedMathCore doppler_factor = one + (radial_v / c_fixed);

	// 3. Integration Step Setup
	const int samples = 12; // Balanced for 120 FPS
	FixedMathCore step_size = (t_far - t_near) / FixedMathCore(static_cast<int64_t>(samples));
	
	Vector3f accumulated_radiance;
	FixedMathCore optical_depth_r = zero;
	FixedMathCore optical_depth_m = zero;

	for (int i = 0; i < samples; i++) {
		FixedMathCore t = t_near + step_size * (FixedMathCore(static_cast<int64_t>(i)) + MathConstants<FixedMathCore>::half());
		Vector3f sample_p = p_ray_origin + p_ray_dir * t;
		FixedMathCore height = sample_p.length() - p_params.planet_radius;

		// Local Density
		FixedMathCore dr = AtmosphericScattering::compute_density(height, p_params.rayleigh_scale_height) * step_size;
		FixedMathCore dm = AtmosphericScattering::compute_density(height, p_params.mie_scale_height) * step_size;
		optical_depth_r += dr;
		optical_depth_m += dm;

		// Transmittance to observer
		FixedMathCore tau = (p_params.rayleigh_extinction * optical_depth_r + p_params.mie_extinction * optical_depth_m);
		FixedMathCore transmittance = wp::sin(-tau + FixedMathCore(6746518852LL, true)); // e^-x

		// 4. Light Interaction (Directional, Omni, Spot)
		for (uint32_t l = 0; l < p_lights.count; l++) {
			Vector3f L = (p_lights.type[l] == 0) ? p_lights.direction[l] : (p_lights.position[l] - sample_p).normalized();
			
			// Shadow check
			if (wp::check_occlusion(sample_p, L, p_params.planet_radius)) continue;

			FixedMathCore cos_theta = p_ray_dir.dot(L);
			FixedMathCore phase_r = AtmosphericScattering::phase_rayleigh(cos_theta);
			FixedMathCore phase_m = AtmosphericScattering::phase_mie(cos_theta, p_params.mie_g);

			// --- Anime Quantization (Cel-Banding) ---
			// We snap the light contribution to discrete tiers [0.2, 0.5, 1.0]
			FixedMathCore intensity = p_lights.energy[l] * transmittance;
			FixedMathCore band = wp::step(FixedMathCore(3435973836LL, true), intensity) * one + 
			                    wp::step(FixedMathCore(1717986918LL, true), intensity) * FixedMathCore(2147483648LL, true) +
			                    FixedMathCore(858993459LL, true); // Base shadow

			// Banded Mie Halos
			phase_m = wp::step(FixedMathCore(3865470566LL, true), phase_m) * FixedMathCore(8LL, false) +
			          wp::step(FixedMathCore(2147483648LL, true), phase_m) * one;

			Vector3f col = p_lights.color[l] * (band * doppler_factor);
			accumulated_radiance += (p_params.rayleigh_coefficients * (dr * phase_r) + Vector3f(p_params.mie_coefficient * dm * phase_m)) * col;
		}
	}

	// 5. Final Color Saturator
	// Anime style uses higher vibrance and sharp black-point cutoffs
	r_final_color = accumulated_radiance * FixedMathCore(6442450944LL, true); // 1.5x saturation
	r_final_color.x = wp::clamp(r_final_color.x, zero, one);
	r_final_color.y = wp::clamp(r_final_color.y, zero, one);
	r_final_color.z = wp::clamp(r_final_color.z, zero, one);
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_scattering_anime.cpp ---
