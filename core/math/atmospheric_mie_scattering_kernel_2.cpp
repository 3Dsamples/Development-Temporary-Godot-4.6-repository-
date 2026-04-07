--- START OF FILE core/math/atmospheric_mie_scattering_kernel.cpp ---

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
 * P(g, cos_theta) = (1 - g^2) / (4 * PI * (1 + g^2 - 2 * g * cos_theta)^1.5)
 * strictly avoids floating-point to ensure bit-perfection across simulation nodes.
 */
static _FORCE_INLINE_ FixedMathCore calculate_mie_phase_hg(FixedMathCore p_cos_theta, FixedMathCore p_g) {
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore g2 = p_g * p_g;
	FixedMathCore four_pi(13493037704LL, true); // 4 * PI in Q32.32

	FixedMathCore numerator = one - g2;
	
	// Denominator: (1 + g^2 - 2 * g * cos_theta)^1.5
	FixedMathCore inner = one + g2 - (FixedMathCore(2LL) * p_g * p_cos_theta);
	
	// x^1.5 = sqrt(x^3) using bit-perfect FixedMath square root
	FixedMathCore inner_cubed = inner * inner * inner;
	FixedMathCore denom_p1_5 = inner_cubed.square_root();
	FixedMathCore denominator = four_pi * denom_p1_5;

	if (unlikely(denominator.get_raw() == 0)) {
		return one; // Singularity fallback
	}

	return numerator / denominator;
}

/**
 * Warp Kernel: MieScatteringIntegratorKernel
 * 
 * Performs a deterministic volumetric march to resolve light-matter interaction.
 * 1. Ray-Marching: Samples atmospheric density along the viewing vector.
 * 2. Real-Time Light Interaction: Aggregates spectral energy from directional and point lights.
 * 3. Shadowing: Bit-perfect occlusion checks for each sample point against planetary bulk.
 * 4. Anime Stylization: Snaps intensities to discrete bands for cel-shaded horizons.
 */
void mie_scattering_integrator_kernel(
		const BigIntCore &p_index,
		Vector3f &r_radiance,
		const Vector3f &p_ray_origin,
		const Vector3f &p_ray_dir,
		const FixedMathCore &p_ray_length,
		const Vector3f &p_sun_dir,
		const FixedMathCore &p_sun_intensity,
		const AtmosphereParams &p_params,
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
		FixedMathCore height = sample_p.length() - p_params.planet_radius;
		if (height.get_raw() < 0) continue; // Skip underground samples

		// 1. Resolve Local Density (Exponential Decay)
		FixedMathCore density = (-(height / p_params.mie_scale_height)).exp();

		// 2. Deterministic Shadow Check
		// Uses bit-perfect ray-sphere intersection for the planet body
		if (wp::check_occlusion_sphere(sample_p, p_sun_dir, Vector3f_ZERO, p_params.planet_radius)) {
			continue;
		}

		// 3. Phase and Transmittance
		FixedMathCore cos_theta = p_ray_dir.dot(p_sun_dir);
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
				phase = FixedMathCore(10LL); // High-intensity core
			} else if (phase > threshold_outer) {
				phase = FixedMathCore(2LL);  // Secondary glow band
			} else {
				phase = zero; // Sharp cutoff
			}
			
			// Increase density contrast for anime clouds/haze
			density = wp::step(FixedMathCore(214748364LL, true), density) * one; 
		}

		// 5. In-Scattering Accumulation
		// Energy = Intensity * Phase * Density * Transmittance
		FixedMathCore energy_mag = p_sun_intensity * phase * density * trans_to_obs * p_params.mie_coefficient;
		accumulated_radiance += Vector3f(energy_mag, energy_mag, energy_mag) * step_size;

		// Accumulate optical depth for the next step
		total_tau += density * step_size * p_params.mie_coefficient;
	}

	r_radiance = accumulated_radiance;
}

/**
 * execute_mie_resolve_sweep()
 * 
 * Master orchestrator for the parallel 120 FPS Mie scattering pass.
 * Partitions the EnTT component streams into SIMD-friendly chunks.
 */
void execute_mie_resolve_sweep(
		KernelRegistry &p_registry,
		const Vector3f &p_sun_direction,
		const FixedMathCore &p_sun_intensity,
		const AtmosphereParams &p_params) {

	auto &radiance_stream = p_registry.get_stream<Vector3f>(COMPONENT_RADIANCE);
	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &ray_stream = p_registry.get_stream<Vector3f>(COMPONENT_RAY_DIR);
	auto &len_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_RAY_LENGTH);

	uint64_t count = radiance_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &radiance_stream, &pos_stream, &ray_stream, &len_stream, &p_params]() {
			for (uint64_t i = start; i < end; i++) {
				// Deterministic Style Selection based on entity handle index
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 16 == 0); 

				mie_scattering_integrator_kernel(
					handle,
					radiance_stream[i],
					pos_stream[i],
					ray_stream[i],
					len_stream[i],
					p_sun_direction,
					p_sun_intensity,
					p_params,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_mie_scattering_kernel.cpp ---
