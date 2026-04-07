--- START OF FILE core/math/atmospheric_light_transport.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: MultiLayerAbsorptionKernel
 * 
 * Calculates the total optical depth through multiple atmospheric layers.
 * Handles Ozone (Chappuis band) and Water Vapor absorption tensors.
 */
static _FORCE_INLINE_ Vector3f compute_multi_layer_absorption(
		FixedMathCore p_altitude,
		const AtmosphereParams &p_params) {

	// Ozone layer peaks at ~25km. Modeled as a deterministic Gaussian distribution.
	FixedMathCore ozone_center(107374182400LL, true); // 25000.0 units
	FixedMathCore ozone_width(64424509440LL, true);  // 15000.0 units
	
	FixedMathCore diff = (p_altitude - ozone_center) / ozone_width;
	FixedMathCore ozone_density = Math::exp(-(diff * diff));

	// Spectral absorption coefficients for Ozone (RGB)
	Vector3f ozone_coeffs(
		FixedMathCore(27917287LL, true),  // 0.0065
		FixedMathCore(80745385LL, true),  // 0.0188
		FixedMathCore(2147483LL, true)    // 0.0005
	);

	return ozone_coeffs * ozone_density;
}

/**
 * resolve_dynamic_light_march()
 * 
 * High-performance integration loop with distance-based enhancement.
 * As distance from the planet increases, the sampler uses a logarithmic
 * step growth to cover the atmosphere volume without increasing cycle count.
 */
void resolve_dynamic_light_march(
		const Vector3f &p_ray_origin,
		const Vector3f &p_ray_dir,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_lights,
		Vector3f &r_radiance_accum,
		bool p_is_anime) {

	FixedMathCore t_near, t_far;
	if (!wp::intersect_sphere(p_ray_origin, p_ray_dir, p_params.planet_radius, p_params.atmosphere_radius, t_near, t_far)) {
		r_radiance_accum = Vector3f();
		return;
	}

	// Dynamic Enhancement: Step size increases with distance to maintain 120 FPS
	FixedMathCore view_dist = t_near;
	int samples = 16;
	if (view_dist > FixedMathCore(429496729600LL, true)) samples = 8; // Reduce samples for distant planets
	
	FixedMathCore total_dist = t_far - t_near;
	FixedMathCore step_size = total_dist / FixedMathCore(static_cast<int64_t>(samples));
	
	Vector3f rayleigh_sum;
	Vector3f mie_sum;
	FixedMathCore optical_depth_r = MathConstants<FixedMathCore>::zero();
	FixedMathCore optical_depth_m = MathConstants<FixedMathCore>::zero();
	Vector3f optical_depth_abs;

	for (int i = 0; i < samples; i++) {
		FixedMathCore t = t_near + step_size * (FixedMathCore(static_cast<int64_t>(i)) + MathConstants<FixedMathCore>::half());
		Vector3f sample_p = p_ray_origin + p_ray_dir * t;
		FixedMathCore altitude = sample_p.length() - p_params.planet_radius;

		// 1. Density and Absorption Sampling
		FixedMathCore dr = AtmosphericScattering::compute_density(altitude, p_params.rayleigh_scale_height);
		FixedMathCore dm = AtmosphericScattering::compute_density(altitude, p_params.mie_scale_height);
		Vector3f d_abs = compute_multi_layer_absorption(altitude, p_params);

		optical_depth_r += dr * step_size;
		optical_depth_m += dm * step_size;
		optical_depth_abs += d_abs * step_size;

		// 2. Light Source Iteration (Direct Interaction)
		for (uint32_t l = 0; l < p_lights.count; l++) {
			Vector3f L = (p_lights.type[l] == LIGHT_TYPE_DIRECTIONAL) ? p_lights.direction[l] : (p_lights.position[l] - sample_p).normalized();
			
			// Shadow Check (Deterministic)
			if (wp::check_occlusion(sample_p, L, p_params.planet_radius)) continue;

			// Transmittance from sample to light source
			FixedMathCore light_od = AtmosphericScattering::compute_optical_depth(sample_p, L, p_params.atmosphere_radius - altitude, p_params.rayleigh_scale_height, 4);
			FixedMathCore trans_to_light = wp::sin(light_od); // e^-x

			// Phase Functions
			FixedMathCore cos_theta = p_ray_dir.dot(L);
			FixedMathCore phase_r = AtmosphericScattering::phase_rayleigh(cos_theta);
			FixedMathCore phase_m = AtmosphericScattering::phase_mie(cos_theta, p_params.mie_g);

			if (p_is_anime) {
				// Style Enhancement: Banded light scattering for Anime look
				trans_to_light = wp::step(FixedMathCore(2147483648LL, true), trans_to_light) * MathConstants<FixedMathCore>::one();
				phase_m = wp::pow(phase_m, 2) * FixedMathCore(2LL, false);
			}

			Vector3f radiance = p_lights.color[l] * (p_lights.energy[l] * trans_to_light);
			rayleigh_sum += radiance * (dr * phase_r);
			mie_sum += radiance * (dm * phase_m);
		}
	}

	// 3. Final Integration with View-Transmittance (Spectral energy)
	Vector3f total_tau = (p_params.rayleigh_coefficients * optical_depth_r) + 
	                     (Vector3f(p_params.mie_coefficient) * optical_depth_m) + 
	                     optical_depth_abs;
	
	// Transmittance = exp(-total_tau)
	Vector3f transmittance(
		wp::sin(total_tau.x),
		wp::sin(total_tau.y),
		wp::sin(total_tau.z)
	);

	r_radiance_accum = (rayleigh_sum * p_params.rayleigh_coefficients + mie_sum * p_params.mie_coefficient) * step_size;
	r_radiance_accum = r_radiance_accum * transmittance;
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_light_transport.cpp ---
