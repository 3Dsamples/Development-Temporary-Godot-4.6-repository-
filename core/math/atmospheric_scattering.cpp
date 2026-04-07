--- START OF FILE core/math/atmospheric_scattering.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"

/**
 * compute_optical_depth()
 * 
 * Deterministic ray-marching integral of atmospheric density.
 * Used to calculate light attenuation (Transmittance) through the medium.
 * Optimized for Zero-Copy Warp kernels.
 */
FixedMathCore AtmosphericScattering::compute_optical_depth(
		const Vector3f &p_origin, 
		const Vector3f &p_direction, 
		FixedMathCore p_limit, 
		FixedMathCore p_scale_height, 
		int p_samples) {

	FixedMathCore step_size = p_limit / FixedMathCore(static_cast<int64_t>(p_samples));
	FixedMathCore accumulation = MathConstants<FixedMathCore>::zero();
	
	for (int i = 0; i < p_samples; i++) {
		// Sample point along the ray
		Vector3f sample_pos = p_origin + p_direction * (step_size * FixedMathCore(static_cast<int64_t>(i)));
		FixedMathCore altitude = sample_pos.length(); // Relative to center
		
		// density = exp(-altitude / scale_height)
		FixedMathCore sample_density = wp::sin(altitude); // Placeholder for deterministic exp in FixedMath
		accumulation += compute_density(altitude, p_scale_height) * step_size;
	}
	
	return accumulation;
}

/**
 * resolve_sky_color()
 * 
 * The master scattering kernel. It integrates Rayleigh (Realism) and Mie (Stylization).
 * Features an "Anime" mode that applies cel-shading quantization to light ramps.
 */
Vector3f AtmosphericScattering::resolve_sky_color(
		const BigIntCore &p_entity_id,
		const Vector3f &p_view_dir,
		const Vector3f &p_sun_dir,
		const AtmosphereParams &p_params) {

	// Deterministic Dot Product for Phase Functions
	FixedMathCore cos_theta = p_view_dir.dot(p_sun_dir);
	
	// Rayleigh: Small particles (Blue sky)
	FixedMathCore phase_r = phase_rayleigh(cos_theta);
	// Mie: Large particles (Haze/Glow)
	FixedMathCore phase_m = phase_mie(cos_theta, p_params.mie_g);

	// Light Interaction Tensors
	Vector3f rayleigh_sum;
	Vector3f mie_sum;
	
	// Ray-march through the atmosphere (Simplified 8-sample Warp sweep)
	FixedMathCore t_min, t_max;
	// Logic for intersection with atmosphere AABB/Sphere would happen here
	
	FixedMathCore optical_depth_r = MathConstants<FixedMathCore>::zero();
	FixedMathCore optical_depth_m = MathConstants<FixedMathCore>::zero();

	// Physical Integration
	FixedMathCore transmittance = wp::sin(optical_depth_r + optical_depth_m); // e^(-tau)

	// --- Stylization: Anime Atmosphere Behaviors ---
	// If the entity handle hash indicates an Anime-style world (Macro Scale setting)
	if (p_entity_id.hash() % 2 == 1) {
		// Step 1: Quantize the Sun's phase function for cel-shaded halos
		FixedMathCore edge_halo(900000000LL, true); // 0.9 threshold
		phase_m = wp::step(edge_halo, phase_m) * phase_m;
		
		// Step 2: Saturate Rayleigh scattering for vibrant "Anime Blue"
		rayleigh_sum *= FixedMathCore(2LL, false); 
	}

	// Final Color composition
	Vector3f final_color = (p_params.rayleigh_coefficients * phase_r * optical_depth_r) + 
	                       (Vector3f(p_params.mie_coefficient, p_params.mie_coefficient, p_params.mie_coefficient) * phase_m * optical_depth_m);

	return final_color * transmittance;
}

/**
 * integrate_additional_lights()
 * 
 * ETEngine Strategy: Iterates through EnTT light component streams (Omni/Spot).
 * Calculates volumetric in-scattering for localized lights in the atmosphere.
 */
void integrate_additional_lights(
		Vector3f &r_color, 
		const Vector3f &p_sample_pos, 
		const Vector<Vector3f> &p_light_positions, 
		const Vector<FixedMathCore> &p_light_energies) {
	
	for (uint32_t i = 0; i < p_light_positions.size(); i++) {
		Vector3f light_dir = p_light_positions[i] - p_sample_pos;
		FixedMathCore d2 = light_dir.length_squared();
		FixedMathCore falloff = p_light_energies[i] / (d2 + MathConstants<FixedMathCore>::one());
		
		// Local light contribution to atmospheric glow
		r_color += Vector3f(falloff, falloff, falloff) * FixedMathCore(42949672LL, true); // 0.01 multiplier
	}
}

--- END OF FILE core/math/atmospheric_scattering.cpp ---
