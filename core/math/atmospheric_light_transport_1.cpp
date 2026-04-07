--- START OF FILE core/math/atmospheric_light_transport.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_transmittance_approx()
 * 
 * Deterministic Taylor-series approximation for exp(-x).
 * res = 1 - x + x^2/2 - x^3/6 + x^4/24
 */
static _FORCE_INLINE_ FixedMathCore calculate_transmittance_approx(FixedMathCore p_tau) {
	FixedMathCore x = p_tau;
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore half(2147483648LL, true);
	FixedMathCore sixth(715827882LL, true);
	FixedMathCore twenty_fourth(178956970LL, true);

	FixedMathCore x2 = x * x;
	FixedMathCore x3 = x2 * x;
	FixedMathCore x4 = x3 * x;

	FixedMathCore res = one - x + (x2 * half) - (x3 * sixth) + (x4 * twenty_fourth);
	return wp::clamp(res, MathConstants<FixedMathCore>::zero(), one);
}

/**
 * Warp Kernel: AtmosphericLightMarchKernel
 * 
 * Master kernel for planetary volume integration.
 * - Distance Adaptive: Reduces sample count for distant celestial bodies to preserve 120 FPS.
 * - Multi-Light Interaction: Aggregates spectral energy from all star-entities.
 * - Style Tensors: Injects cel-shading bands for Anime behavior.
 */
void resolve_dynamic_light_march(
		const BigIntCore &p_entity_id,
		Vector3f &r_radiance,
		const Vector3f &p_ray_origin,
		const Vector3f &p_ray_dir,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_lights,
		const FixedMathCore &p_observer_dist,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Find atmospheric boundaries
	FixedMathCore t_near, t_far;
	if (!wp::intersect_sphere(p_ray_origin, p_ray_dir, p_params.planet_center, p_params.atmosphere_radius, t_near, t_far)) {
		r_radiance = Vector3f();
		return;
	}

	// 2. Dynamic Sampling Refinement
	// Samples are reduced as distance from planet increases (Galactic optimization)
	int sample_count = 16;
	if (p_observer_dist > FixedMathCore(50000LL, false)) sample_count = 8;
	if (p_observer_dist > FixedMathCore(1000000LL, false)) sample_count = 4;

	FixedMathCore total_dist = t_far - t_near;
	FixedMathCore step_size = total_dist / FixedMathCore(static_cast<int64_t>(sample_count));
	
	Vector3f accumulated_inscatter;
	FixedMathCore optical_depth_r = zero;
	FixedMathCore optical_depth_m = zero;

	for (int i = 0; i < sample_count; i++) {
		FixedMathCore t = t_near + step_size * (FixedMathCore(static_cast<int64_t>(i)) + MathConstants<FixedMathCore>::half());
		Vector3f sample_p = p_ray_origin + p_ray_dir * t;
		FixedMathCore altitude = (sample_p - p_params.planet_center).length() - p_params.planet_radius;

		// 3. Physical Extinction (Beer-Lambert)
		FixedMathCore density_r = Math::exp(-(altitude / p_params.rayleigh_scale_height));
		FixedMathCore density_m = Math::exp(-(altitude / p_params.mie_scale_height));
		
		optical_depth_r += density_r * step_size;
		optical_depth_m += density_m * step_size;

		FixedMathCore view_tau = (density_r * p_params.rayleigh_extinction) + (density_m * p_params.mie_extinction);
		FixedMathCore trans_to_obs = calculate_transmittance_approx(view_tau);

		// 4. Multi-Source Irradiance
		for (uint32_t l = 0; l < p_lights.count; l++) {
			Vector3f L = (p_lights.type[l] == 0) ? p_lights.direction[l] : (p_lights.position[l] - sample_p).normalized();
			
			// Shadow check for planetary bulk
			if (wp::check_occlusion_sphere(sample_p, L, p_params.planet_center, p_params.planet_radius)) continue;

			// Phase Functions
			FixedMathCore cos_theta = p_ray_dir.dot(L);
			FixedMathCore phase_r = (FixedMathCore(3LL) / (FixedMathCore(16LL) * Math::pi())) * (one + cos_theta * cos_theta);
			FixedMathCore phase_m = wp::henyey_greenstein_phase(cos_theta, p_params.mie_g);

			// Sophisticated Style logic: Anime Banding
			if (p_is_anime) {
				// Quantize the Sun's phase for cel-shaded halos
				phase_m = wp::step(FixedMathCore(3435973836LL, true), phase_m) * FixedMathCore(5LL, false) + 
				          wp::step(FixedMathCore(858993459LL, true), phase_m) * one;
			}

			// Integrate light color and energy
			Vector3f light_energy = p_lights.color[l] * p_lights.energy[l] * trans_to_obs;
			accumulated_inscatter += (p_params.rayleigh_coefficients * (density_r * phase_r) + 
			                         Vector3f(p_params.mie_coefficient * density_m * phase_m)) * light_energy;
		}
	}

	r_radiance = accumulated_inscatter * step_size;

	// 5. Final Color Normalization
	if (p_is_anime) {
		// Anime Technique: "Vibrance Snap"
		// Clamps colors to highly saturated levels based on luminance
		FixedMathCore lum = r_radiance.get_luminance();
		FixedMathCore snap_weight = wp::step(FixedMathCore(429496730LL, true), lum); // 0.1 threshold
		r_radiance = r_radiance.normalized() * snap_weight;
	}
}

/**
 * execute_light_transport_sweep()
 * 
 * Performs the master parallel atmospheric resolve.
 * Zero-copy: Operates directly on EnTT radiance streams.
 */
void execute_light_transport_sweep(
		KernelRegistry &p_registry,
		const Vector3f &p_obs_pos,
		const BigIntCore &p_obs_sx, const BigIntCore &p_obs_sy, const BigIntCore &p_obs_sz,
		const LightDataSoA &p_lights,
		const FixedMathCore &p_delta) {

	auto &radiance_stream = p_registry.get_stream<Vector3f>(COMPONENT_RADIANCE);
	auto &ray_dir_stream = p_registry.get_stream<Vector3f>(COMPONENT_RAY_DIRECTION);
	auto &atm_params_stream = p_registry.get_stream<AtmosphereParams>(COMPONENT_ATMOSPHERE);

	uint64_t count = radiance_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &radiance_stream, &ray_dir_stream, &atm_params_stream, &p_lights]() {
			for (uint64_t i = start; i < end; i++) {
				const AtmosphereParams &params = atm_params_stream[i];
				
				// Calculate bit-perfect distance to planetary center for LOD/Step adjustment
				Vector3f rel_pos = wp::calculate_galactic_relative_pos(
					p_obs_pos, p_obs_sx, p_obs_sy, p_obs_sz,
					params.planet_center, params.sector_x, params.sector_y, params.sector_z,
					FixedMathCore(10000LL, false)
				);
				FixedMathCore dist = rel_pos.length();

				resolve_dynamic_light_march(
					BigIntCore(static_cast<int64_t>(i)),
					radiance_stream[i],
					p_obs_pos,
					ray_dir_stream[i],
					params,
					p_lights,
					dist,
					(i % 16 == 0) // Deterministic Anime Style trigger
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_light_transport.cpp ---
