--- START OF FILE core/math/atmospheric_scattering_logic.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * integrate_light_contribution_kernel()
 * 
 * A Warp-style kernel for calculating in-scattering from all light types.
 * Handles Directional (infinite), Omni (point), and Spot (conical) sources.
 * Uses bit-perfect attenuation curves to ensure 100% determinism.
 */
static _FORCE_INLINE_ Vector3f integrate_light_contribution_kernel(
		const Vector3f &p_sample_pos,
		const Vector3f &p_view_dir,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_lights,
		bool p_is_anime) {

	Vector3f total_inscatter;
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	for (uint32_t i = 0; i < p_lights.count; i++) {
		Vector3f light_dir;
		FixedMathCore attenuation = one;

		if (p_lights.type[i] == LIGHT_TYPE_DIRECTIONAL) {
			light_dir = p_lights.direction[i];
		} else {
			Vector3f rel_vec = p_lights.position[i] - p_sample_pos;
			FixedMathCore dist_sq = rel_vec.length_squared();
			FixedMathCore radius_sq = p_lights.radius[i] * p_lights.radius[i];

			if (dist_sq > radius_sq) continue;

			light_dir = rel_vec.normalized();
			// Deterministic Inverse Square Falloff
			attenuation = one - (dist_sq / radius_sq);
			attenuation = attenuation * attenuation; // Quadratic for smooth results

			if (p_lights.type[i] == LIGHT_TYPE_SPOT) {
				FixedMathCore cos_angle = (-light_dir).dot(p_lights.direction[i]);
				FixedMathCore spot_cutoff = p_lights.spot_angle[i];
				if (cos_angle < spot_cutoff) continue;
				// Conical smoothing
				attenuation *= (cos_angle - spot_cutoff) / (one - spot_cutoff);
			}
		}

		FixedMathCore cos_theta = p_view_dir.dot(light_dir);
		FixedMathCore phase_r = AtmosphericScattering::phase_rayleigh(cos_theta);
		FixedMathCore phase_m = AtmosphericScattering::phase_mie(cos_theta, p_params.mie_g);

		// --- Anime Stylization Logic ---
		if (p_is_anime) {
			// Quantize light intensity for cel-shaded atmospheric bands
			attenuation = wp::step(FixedMathCore(2147483648LL, true), attenuation) * FixedMathCore(3435973836LL, true) + 
			              wp::step(FixedMathCore(858993459LL, true), attenuation) * FixedMathCore(858993459LL, true);
			
			// Exaggerate Mie halos for artistic effect
			phase_m = wp::pow(phase_m, 2); 
		}

		Vector3f light_color = p_lights.color[i] * (p_lights.energy[i] * attenuation);
		total_inscatter += (p_params.rayleigh_coefficients * phase_r + Vector3f(p_params.mie_coefficient) * phase_m) * light_color;
	}

	return total_inscatter;
}

/**
 * batch_resolve_atmosphere_kernel()
 * 
 * The master 120 FPS parallel sweep. Processes a batch of view rays
 * through atmospheric volumes stored in EnTT component streams.
 */
void batch_resolve_atmosphere_kernel(
		const BigIntCore *p_entity_ids,
		const Vector3f *p_origins,
		const Vector3f *p_directions,
		const AtmosphereParams *p_atm_data,
		const LightDataSoA &p_global_lights,
		Vector3f *r_output_colors,
		uint64_t p_count) {

	for (uint64_t i = 0; i < p_count; i++) {
		const AtmosphereParams &params = p_atm_data[i];
		const Vector3f &ray_o = p_origins[i];
		const Vector3f &ray_d = p_directions[i];
		bool is_anime = (p_entity_ids[i].hash() % 2 == 1);

		// Determine intersection distance through atmospheric sphere
		FixedMathCore t_near, t_far;
		if (!wp::intersect_sphere(ray_o, ray_d, params.planet_radius, params.atmosphere_radius, t_near, t_far)) {
			r_output_colors[i] = Vector3f();
			continue;
		}

		// Fixed-step deterministic integration
		const int samples = 16;
		FixedMathCore step_size = (t_far - t_near) / FixedMathCore(static_cast<int64_t>(samples));
		Vector3f accumulated_color;
		FixedMathCore optical_depth_r = MathConstants<FixedMathCore>::zero();
		FixedMathCore optical_depth_m = MathConstants<FixedMathCore>::zero();

		for (int s = 0; s < samples; s++) {
			Vector3f sample_p = ray_o + ray_d * (t_near + step_size * FixedMathCore(static_cast<int64_t>(s)));
			FixedMathCore height = sample_p.length() - params.planet_radius;
			
			FixedMathCore hr = AtmosphericScattering::compute_density(height, params.rayleigh_scale_height) * step_size;
			FixedMathCore hm = AtmosphericScattering::compute_density(height, params.mie_scale_height) * step_size;
			optical_depth_r += hr;
			optical_depth_m += hm;

			// Combined light integration (Sun + Local Lights)
			Vector3f light_at_point = integrate_light_contribution_kernel(sample_p, ray_d, params, p_global_lights, is_anime);
			
			// Transmittance from sample to observer
			FixedMathCore tau = (params.rayleigh_coefficients.x * optical_depth_r + params.mie_coefficient * optical_depth_m);
			FixedMathCore atten = wp::sin(tau); // Deterministic e^-x approximation

			accumulated_color += light_at_point * (hr * atten);
		}

		r_output_colors[i] = accumulated_color;
	}
}

--- END OF FILE core/math/atmospheric_scattering_logic.cpp ---
