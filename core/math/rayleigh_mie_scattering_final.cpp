--- START OF FILE core/math/rayleigh_mie_scattering_final.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: FinalRadianceCompositionKernel
 * 
 * Combines all scattering components into the final spectral energy buffer.
 * 1. Merges Rayleigh (sky) and Mie (glow) contributions.
 * 2. Applies accumulated Transmittance (extinction).
 * 3. Injects Indirect Irradiance from the Spherical Harmonics (SH) cache.
 * 4. Normalizes energy levels for bit-perfect HDR integration.
 */
void final_radiance_composition_kernel(
		const BigIntCore &p_index,
		Vector3f &r_final_radiance,
		const Vector3f &p_rayleigh_accum,
		const Vector3f &p_mie_accum,
		const Vector3f &p_transmittance,
		const Vector3f *p_sh_irradiance_coeffs,
		const Vector3f &p_surface_normal,
		const FixedMathCore &p_exposure,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Combine Direct Scattering Tiers
	// Total = (Rayleigh + Mie) * Transmittance
	Vector3f total_scattering = (p_rayleigh_accum + p_mie_accum);

	// 2. Resolve Indirect Ambient (Irradiance)
	// Uses the precomputed SH coefficients to calculate the light received from the whole sky.
	Vector3f ambient_light = AtmosphericScattering::resolve_irradiance_from_sh(p_surface_normal, p_sh_irradiance_coeffs);
	
	// 3. Advanced Behavior: Volumetric Energy Conservation
	// Ensures that the total energy does not exceed the incoming star-light intensity.
	FixedMathCore energy_sum = total_scattering.length() + ambient_light.length();
	if (energy_sum > p_exposure) {
		FixedMathCore dampening = p_exposure / energy_sum;
		total_scattering *= dampening;
		ambient_light *= dampening;
	}

	// 4. Final Spectral Composition
	r_final_radiance = (total_scattering * p_transmittance) + ambient_light;

	// --- Sophisticated Anime Stylization: Spectral Banding ---
	if (p_is_anime) {
		// Anime Technique: "Chromatic Snap". 
		// Instead of smooth gradients, colors are forced into vibrant, highly-saturated bins.
		FixedMathCore luminance = r_final_radiance.get_luminance();
		
		// Band thresholds for cel-shaded atmosphere
		FixedMathCore shadow_band(858993459LL, true); // 0.2
		FixedMathCore highlight_band(3435973836LL, true); // 0.8

		if (luminance < shadow_band) {
			r_final_radiance *= FixedMathCore(214748364LL, true); // Deep shadow tint (0.05)
		} else if (luminance > highlight_band) {
			r_final_radiance *= FixedMathCore(5LL, false); // Saturated highlight boost
		} else {
			// Normal band: slightly increase saturation
			FixedMathCore sat_boost(5153960755LL, true); // 1.2x
			r_final_radiance *= sat_boost;
		}
	}
}

/**
 * execute_final_scattering_resolve()
 * 
 * Orchestrates the parallel composition of the sky and light-matter interactions.
 * Processes EnTT SoA buffers for Rayleigh, Mie, and SH data at 120 FPS.
 */
void execute_final_scattering_resolve(
		const BigIntCore &p_total_entities,
		Vector3f *r_radiance_buffer,
		const Vector3f *p_rayleigh_stream,
		const Vector3f *p_mie_stream,
		const Vector3f *p_transmittance_stream,
		const Vector3f *p_sh_coeffs,
		const Vector3f *p_normals,
		const FixedMathCore &p_exposure_const) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_total_entities.to_string()));
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? total : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				// Deterministic style selection based on entity handle hash
				bool anime_mode = (i % 6 == 0); 

				final_radiance_composition_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_radiance_buffer[i],
					p_rayleigh_stream[i],
					p_mie_stream[i],
					p_transmittance_stream[i],
					&p_sh_coeffs[i * 9], // 9 SH coefficients per entity
					p_normals[i],
					p_exposure_const,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	// Barrier: Ensure visual composition is complete for the 120 FPS frame
	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * apply_galactic_distance_attenuation()
 * 
 * High-Speed Behavioral Technique: 
 * As spaceships move at relativistic speeds between stars, the global 
 * light-field is attenuated to prevent "Flash-Bang" artifacts when 
 * crossing sector boundaries.
 */
void apply_galactic_distance_attenuation(
		Vector3f &r_radiance,
		const BigIntCore &p_dist_to_star_sq,
		const FixedMathCore &p_star_power) {
	
	// Inverse Square Law in BigInt/FixedMath
	BigIntCore four_pi("13"); // Approx
	BigIntCore divisor = p_dist_to_star_sq * four_pi;
	
	FixedMathCore div_f(static_cast<int64_t>(std::stoll(divisor.to_string())));
	FixedMathCore intensity = p_star_power / (div_f + MathConstants<FixedMathCore>::one());
	
	r_radiance *= intensity;
}

} // namespace UniversalSolver

--- END OF FILE core/math/rayleigh_mie_scattering_final.cpp ---
