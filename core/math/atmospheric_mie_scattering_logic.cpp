--- START OF FILE core/math/atmospheric_mie_scattering_logic.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_mie_phase_hg()
 * 
 * Deterministic Henyey-Greenstein Phase Function.
 * P(g, cos_theta) = (1 - g^2) / (4 * PI * (1 + g^2 - 2 * g * cos_theta)^(1.5))
 * strictly avoids floating-point to ensure bit-perfection across simulation nodes.
 */
static _FORCE_INLINE_ FixedMathCore calculate_mie_phase_hg(FixedMathCore p_cos_theta, FixedMathCore p_g) {
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore g2 = p_g * p_g;
	FixedMathCore four_pi(13493037704LL, true); // 4 * PI in Q32.32

	FixedMathCore numerator = one - g2;
	
	// Denominator inner: (1 + g^2 - 2 * g * cos_theta)
	FixedMathCore denom_inner = one + g2 - (FixedMathCore(2LL) * p_g * p_cos_theta);
	
	// Power 1.5 resolve: x^1.5 = sqrt(x^3) using bit-perfect FixedMath square root
	FixedMathCore inner_cubed = denom_inner * denom_inner * denom_inner;
	FixedMathCore denom_p1_5 = inner_cubed.square_root();
	FixedMathCore denominator = four_pi * denom_p1_5;

	if (unlikely(denominator.get_raw() == 0)) {
		return one; // Fallback for singularity at exact light alignment
	}

	return numerator / denominator;
}

/**
 * Warp Kernel: MieVolumeIntegratorKernel
 * 
 * Performs a deterministic volumetric march to resolve light-matter interaction for haze and clouds.
 * 1. Ray-Marching: Samples atmospheric density along the viewing vector.
 * 2. Multi-Light Interaction: Aggregates spectral energy from stars and local light sources.
 * 3. Shadowing: Bit-perfect occlusion checks for each sample point against planetary bulk.
 * 4. Anime Stylization: Snaps intensities into discrete bands for cel-shaded horizons and halos.
 */
void mie_volume_integrator_kernel(
		const BigIntCore &p_index,
		Vector3f &r_radiance,
		const Vector3f &p_ray_origin,
		const Vector3f &p_ray_dir,
		const FixedMathCore &p_ray_length,
		const Vector3f &p_planet_center,
		const FixedMathCore &p_planet_radius,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_lights,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	
	// Sample count optimized for 120 FPS performance budget
	const int samples = 12;
	FixedMathCore step_size = p_ray_length / FixedMathCore(static_cast<int64_t>(samples));
	Vector3f accumulated_radiance;
	FixedMathCore total_tau = zero;

	for (int i = 0; i < samples; i++) {
		// Calculate sample point in the middle of the current step
		FixedMathCore t = step_size * (FixedMathCore(static_cast<int64_t>(i)) + MathConstants<FixedMathCore>::half());
		Vector3f sample_p = p_ray_origin + p_ray_dir * t;
		
		// Altitude relative to planetary surface
		FixedMathCore altitude = (sample_p - p_planet_center).length() - p_planet_radius;
		if (altitude.get_raw() < 0) continue; // Skip underground samples

		// 1. Resolve Local Density (Exponential Decay)
		FixedMathCore density = (-(altitude / p_params.mie_scale_height)).exp();

		// 2. Multi-Source Light Integration
		Vector3f step_energy;
		for (uint32_t l = 0; l < p_lights.count; l++) {
			Vector3f L;
			FixedMathCore attenuation = one;

			if (p_lights.type[l] == 0) { // Directional (e.g., Star)
				L = p_lights.direction[l];
			} else { // Point / Spot
				Vector3f rel_l = p_lights.position[l] - sample_p;
				FixedMathCore d2 = rel_l.length_squared();
				L = rel_l.normalized();
				attenuation = one / (d2 + one);
			}

			// Deterministic Shadow Check against planetary bulk
			if (wp::check_occlusion_sphere(sample_p, L, p_planet_center, p_planet_radius)) {
				continue;
			}

			// 3. Phase and Transmittance
			FixedMathCore cos_theta = p_ray_dir.dot(L);
			FixedMathCore phase = calculate_mie_phase_hg(cos_theta, p_params.mie_g);
			
			// Transmittance to observer: exp(-tau_accumulated)
			FixedMathCore trans_to_obs = (-total_tau).exp();

			// 4. --- Sophisticated Real-Time Behavior: Anime vs Realistic ---
			if (p_is_anime) {
				// Technique: "Mie Glow Banding". 
				// Instead of a smooth halo, create sharp light rings using step functions.
				FixedMathCore threshold_inner(3435973836LL, true); // 0.8
				FixedMathCore threshold_outer(1288490188LL, true); // 0.3
				
				if (phase > threshold_inner) {
					phase = FixedMathCore(8LL); // Intensified inner halo
				} else if (phase > threshold_outer) {
					phase = FixedMathCore(2LL); // Secondary band
				} else {
					phase = zero; // Sharp cel-shaded cutoff
				}
				
				// Saturate attenuation for high-contrast anime shadows
				attenuation = wp::step(FixedMathCore(2147483648LL, true), attenuation) * one;
			}

			// 5. In-Scattering Accumulation
			FixedMathCore energy_mag = p_lights.energy[l] * phase * density * trans_to_obs * attenuation;
			step_energy += p_lights.color[l] * energy_mag;
		}

		accumulated_radiance += step_energy * step_size;

		// Accumulate optical depth for the next step
		total_tau += density * step_size * p_params.mie_coefficient;
	}

	r_final_radiance = accumulated_radiance * p_params.mie_coefficient;
}

/**
 * execute_mie_scattering_sweep()
 * 
 * Orchestrates the parallel 120 FPS Mie resolve across the EnTT registry.
 * Zero-copy: Operates directly on SoA streams to maintain simulation heartbeat.
 */
void execute_mie_scattering_sweep(
		KernelRegistry &p_registry,
		const Vector3f &p_obs_pos,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_lights) {

	auto &rad_stream = p_registry.get_stream<Vector3f>(COMPONENT_RADIANCE);
	auto &ray_stream = p_registry.get_stream<Vector3f>(COMPONENT_RAY_DIR);
	auto &len_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_RAY_LENGTH);
	
	uint64_t count = rad_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &rad_stream, &ray_stream, &len_stream, &p_params, &p_lights]() {
			for (uint64_t i = start; i < end; i++) {
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 12 == 0);

				mie_volume_integrator_kernel(
					handle,
					rad_stream[i],
					p_obs_pos,
					ray_stream[i],
					len_stream[i],
					p_params.planet_center,
					p_params.planet_radius,
					p_params,
					p_lights,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_mie_scattering_logic.cpp ---
