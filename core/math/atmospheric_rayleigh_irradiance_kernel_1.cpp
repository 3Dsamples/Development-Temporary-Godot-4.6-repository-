--- START OF FILE core/math/atmospheric_rayleigh_irradiance_kernel.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_zenith_irradiance()
 * 
 * Computes the spectral radiance at the zenith for a planetary atmosphere.
 * T = exp(-tau / cos_theta)
 * Strictly uses FixedMathCore for bit-perfect atmospheric extinction.
 */
static _FORCE_INLINE_ Vector3f calculate_zenith_irradiance(
		FixedMathCore p_sun_cos_theta, 
		const AtmosphereParams &p_params) {
	
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();

	// Calculate optical thickness at zenith
	FixedMathCore tau = p_params.rayleigh_extinction * p_params.rayleigh_scale_height;
	
	// Air Mass approx: 1.0 / cos(theta)
	FixedMathCore mu = wp::max(p_sun_cos_theta, FixedMathCore(429496LL, true)); // 0.0001 floor
	FixedMathCore air_mass = one / mu;
	
	// Deterministic Transmittance: exp(-tau * air_mass)
	FixedMathCore t_val = wp::exp(-(tau * air_mass));
	
	return p_params.rayleigh_coefficients * t_val;
}

/**
 * Warp Kernel: RayleighIrradianceKernel
 * 
 * Resolves the ambient sky light received by a surface normal.
 * 1. Computes the sky visibility factor (Normal-Up dot).
 * 2. Samples the Rayleigh gradient based on sun zenith.
 * 3. Applies Anime-Style "Chromatic Snap" to force vibrant sky bands.
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

	// 1. Sun Position contribution
	FixedMathCore sun_up = p_sun_dir.y; // Assumes Y is planetary UP
	FixedMathCore zenith_factor = wp::max(zero, sun_up);
	
	// 2. Resolve Rayleigh Irradiance Tensors
	// Phase integration for irradiance approximates the hemispherical integral.
	FixedMathCore cos_phi = p_normal.dot(p_sun_dir);
	FixedMathCore phase = (FixedMathCore(3LL) / (FixedMathCore(16LL) * Math::pi())) * (one + cos_phi * cos_phi);
	
	Vector3f zenith_energy = calculate_zenith_irradiance(zenith_factor, p_params) * p_sun_intensity;
	Vector3f base_irradiance = zenith_energy * phase;

	// 3. --- Sophisticated Behavior: Realistic vs Anime ---
	if (p_is_anime) {
		// Anime Technique: "Vibrant Sky Banding". 
		// Instead of smooth atmospheric scattering, the ambient light is forced 
		// into distinct color bins based on the sun's altitude.
		FixedMathCore lum = base_irradiance.get_luminance();
		
		FixedMathCore threshold_day(3006477107LL, true); // 0.7
		FixedMathCore threshold_sunset(858993459LL, true); // 0.2

		if (lum > threshold_day) {
			// Saturated High-noon Blue
			base_irradiance = p_params.rayleigh_color * FixedMathCore(5153960755LL, true); // 1.2x
		} else if (lum < threshold_sunset) {
			// Stylized "Magic Hour" Purple/Orange snap
			Vector3f sunset_tint(one, FixedMathCore(1717986918LL, true), FixedMathCore(3435973836LL, true));
			base_irradiance = sunset_tint * FixedMathCore(429496730LL, true); // 0.1
		} else {
			// Snap to a solid mid-tier band
			base_irradiance *= FixedMathCore(2147483648LL, true); // 0.5
		}
	}

	// 4. Hemispherical Sky-View Factor
	// irradiance_out = base * (1 + cos(alpha)) / 2, where alpha is angle from normal to zenith
	FixedMathCore sky_view = (one + p_normal.y) * MathConstants<FixedMathCore>::half();
	r_ambient_radiance = base_irradiance * sky_view;
}

/**
 * execute_rayleigh_irradiance_sweep()
 * 
 * Orchestrates the parallel 120 FPS resolve for ambient sky lighting.
 * strictly uses zero-copy data flow from EnTT component streams.
 */
void execute_rayleigh_irradiance_sweep(
		KernelRegistry &p_registry,
		const AtmosphereParams &p_params,
		const Vector3f &p_sun_direction,
		const FixedMathCore &p_sun_energy) {

	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &norm_stream = p_registry.get_stream<Vector3f>(COMPONENT_NORMAL);
	auto &ambient_stream = p_registry.get_stream<Vector3f>(COMPONENT_RAYLEIGH_AMBIENT);

	uint64_t count = pos_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &pos_stream, &norm_stream, &ambient_stream, &p_params]() {
			for (uint64_t i = start; i < end; i++) {
				// Deterministic style selection based on entity handle hash (every 5th is anime)
				BigIntCore entity_idx(static_cast<int64_t>(i));
				bool anime_mode = (entity_idx.hash() % 5 == 0); 

				rayleigh_irradiance_kernel(
					entity_idx,
					ambient_stream[i],
					norm_stream[i],
					p_params,
					p_sun_direction,
					p_sun_energy,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_rayleigh_irradiance_kernel.cpp ---
