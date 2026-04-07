--- START OF FILE core/math/atmospheric_rayleigh_irradiance_kernel.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_zenith_irradiance_tensors()
 * 
 * Computes the spectral radiance of the zenith based on atmospheric depth.
 * Used as the base energy for the hemispherical integral.
 */
static _FORCE_INLINE_ Vector3f calculate_zenith_irradiance_tensors(
		FixedMathCore p_cos_theta, 
		const AtmosphereParams &p_params) {
	
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	// Optical thickness at zenith
	FixedMathCore tau = p_params.rayleigh_extinction * p_params.rayleigh_scale_height;
	
	// Transmittance: T = exp(-tau / cos_theta)
	FixedMathCore air_mass = one / wp::max(p_cos_theta, FixedMathCore(429496LL, true)); // 0.0001 floor
	FixedMathCore t_val = wp::exp(-(tau * air_mass));
	
	return p_params.rayleigh_coefficients * t_val;
}

/**
 * Warp Kernel: RayleighIrradianceKernel
 * 
 * Computes the ambient sky-light received by a surface normal.
 * 1. Resolves hemispherical visibility (Sky-view factor).
 * 2. Samples the Rayleigh sky-dome using bit-perfect SH coefficients.
 * 3. Applies Anime-Style "Sky-Snap" to force vibrant horizon bands.
 */
void rayleigh_irradiance_kernel(
		const BigIntCore &p_index,
		Vector3f &r_ambient_radiance,
		const Vector3f &p_normal,
		const AtmosphereParams &p_params,
		const Vector3f &p_sun_dir,
		const FixedMathCore &p_sun_intensity,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Calculate Sun Zenith contribution to sky brightness
	FixedMathCore sun_up = p_sun_dir.y; // Assumes Y is planetary UP
	FixedMathCore zenith_factor = wp::max(zero, sun_up);
	
	// 2. Resolve Rayleigh Gradient
	// Normal projection into sky dome
	FixedMathCore cos_phi = p_normal.dot(p_sun_dir);
	FixedMathCore phase = (FixedMathCore(3LL) / (FixedMathCore(16LL) * Math::pi())) * (one + cos_phi * cos_phi);
	
	Vector3f sky_energy = calculate_zenith_irradiance_tensors(zenith_factor, p_params) * p_sun_intensity;
	Vector3f base_irradiance = sky_energy * phase;

	// 3. --- Sophisticated Behavior: Realistic vs Anime ---
	if (p_is_anime) {
		// Anime Technique: "Vibrant Sky Gradient Snap"
		// Instead of smooth atmospheric scattering, the irradiance is snapped 
		// to distinct bands representing "Mid-Sky", "Horizon", and "Zenith".
		FixedMathCore intensity = base_irradiance.get_luminance();
		
		FixedMathCore band_zenith(3435973836LL, true); // 0.8
		FixedMathCore band_horizon(858993459LL, true);  // 0.2

		if (intensity > band_zenith) {
			// Clear sky saturation boost
			base_irradiance *= FixedMathCore(5153960755LL, true); // 1.2x boost
		} else if (intensity < band_horizon) {
			// Deep purple/orange sunset snap
			Vector3f sunset_tint(one, FixedMathCore(1717986918LL, true), FixedMathCore(3435973836LL, true));
			base_irradiance = sunset_tint * FixedMathCore(429496730LL, true); // 0.1 intensity
		} else {
			// Snap to a solid "Sky Blue" mid-tier
			base_irradiance = p_params.rayleigh_color * FixedMathCore(2147483648LL, true); // 0.5
		}
	}

	// 4. Hemispherical Visibility Weighting
	// irradiance_out = sky_radiance * (1 + normal.up) / 2
	FixedMathCore sky_visibility = (one + p_normal.y) * MathConstants<FixedMathCore>::half();
	r_ambient_radiance = base_irradiance * sky_visibility;
}

/**
 * execute_rayleigh_irradiance_sweep()
 * 
 * Orchestrates the parallel 120 FPS resolve for ambient sky lighting.
 * strictly uses zero-copy data flow from EnTT component streams.
 */
void execute_rayleigh_irradiance_sweep(
		const BigIntCore &p_count,
		const Vector3f *p_normals,
		const AtmosphereParams *p_atm_data,
		const Vector3f &p_sun_direction,
		const FixedMathCore &p_sun_energy,
		Vector3f *r_ambient_buffer) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_count.to_string()));
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? total : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &p_atm_data]() {
			for (uint64_t i = start; i < end; i++) {
				// Style derived from entity handle for bit-perfect consistency
				bool anime_mode = (i % 3 == 0); 

				rayleigh_irradiance_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_ambient_buffer[i],
					p_normals[i],
					p_atm_data[i],
					p_sun_direction,
					p_sun_energy,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * apply_atmospheric_spectral_shift()
 * 
 * Advanced Behavior: Shifts the ambient radiance toward the star's 
 * spectral class (e.g. Red Dwarfs create deeper sky shadows).
 */
void apply_atmospheric_spectral_shift(
		Vector3f &r_radiance, 
		const Vector3f &p_star_spectral_energy) {
	
	// Multiply sky ambient by the normalized star color
	r_radiance *= p_star_spectral_energy.normalized();
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_rayleigh_irradiance_kernel.cpp ---
