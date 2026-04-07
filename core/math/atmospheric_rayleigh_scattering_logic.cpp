--- START OF FILE core/math/atmospheric_rayleigh_scattering_logic.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_rayleigh_phase_logic()
 * 
 * Deterministic Rayleigh phase function: P(theta) = 3 / (16 * pi) * (1 + cos^2 theta)
 * Strictly uses Software-Defined Arithmetic to ensure zero-drift results.
 * Constant 3 / (16 * pi) approx: 0.05968310365 in Q32.32
 */
static _FORCE_INLINE_ FixedMathCore calculate_rayleigh_phase_logic(FixedMathCore p_cos_theta) {
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore coeff(256345098LL, true); 
	
	return coeff * (one + (p_cos_theta * p_cos_theta));
}

/**
 * calculate_spectral_beta_rayleigh()
 * 
 * Computes scattering coefficients based on the inverse 4th power of wavelength.
 * Base wavelengths: Red (680nm), Green (550nm), Blue (440nm).
 * Corrected for relativistic Doppler factor.
 */
static _FORCE_INLINE_ Vector3f calculate_spectral_beta_rayleigh(
		const Vector3f &p_base_wavelengths, 
		FixedMathCore p_doppler_factor,
		const FixedMathCore &p_rayleigh_coeff) {
	
	// lambda_shifted = lambda_base / doppler_factor
	Vector3f obs_lambda = p_base_wavelengths / p_doppler_factor;
	
	// beta = 1 / lambda^4.
	FixedMathCore r4 = obs_lambda.x * obs_lambda.x * obs_lambda.x * obs_lambda.x;
	FixedMathCore g4 = obs_lambda.y * obs_lambda.y * obs_lambda.y * obs_lambda.y;
	FixedMathCore b4 = obs_lambda.z * obs_lambda.z * obs_lambda.z * obs_lambda.z;

	// Bit-perfect normalization scale for physical units
	FixedMathCore norm_scale(429496729600000000LL, true); 
	
	Vector3f beta(norm_scale / r4, norm_scale / g4, norm_scale / b4);
	return beta * p_rayleigh_coeff;
}

/**
 * Warp Kernel: RayleighIntegratorKernel
 * 
 * Performs a parallel spectral integration of the sky color.
 * 1. Doppler Shift: Wavelengths are shifted by relativistic spaceship velocity.
 * 2. Ray-Marching: Integrates density profile exp(-h/H) in bit-perfect steps.
 * 3. Multi-Light interaction: Aggregates energy with deterministic shadows.
 * 4. Anime Style: Saturated gradients and hue-snapping for stylized horizons.
 */
void rayleigh_integrator_kernel(
		const BigIntCore &p_index,
		Vector3f &r_radiance,
		const Vector3f &p_ray_origin,
		const Vector3f &p_ray_dir,
		const Vector3f &p_ship_velocity,
		const FixedMathCore &p_ray_length,
		const Vector3f &p_planet_center,
		const FixedMathCore &p_planet_radius,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_lights,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Relativistic Doppler Resolve
	// Factor D = 1 + (v_radial / c)
	FixedMathCore radial_v = p_ray_dir.dot(p_ship_velocity);
	FixedMathCore doppler_factor = one + (radial_v / PHYSICS_C);

	// Precompute Spectral Tensors
	Vector3f base_lambda(FixedMathCore(680LL), FixedMathCore(550LL), FixedMathCore(440LL));
	Vector3f beta_rayleigh = calculate_spectral_beta_rayleigh(base_lambda, doppler_factor, p_params.rayleigh_coefficient);

	// 2. Integration March (16 steps for 120 FPS high-fidelity)
	const int samples = 16;
	FixedMathCore step_size = p_ray_length / FixedMathCore(static_cast<int64_t>(samples));
	Vector3f accumulated_radiance;
	FixedMathCore total_optical_depth = zero;

	for (int i = 0; i < samples; i++) {
		FixedMathCore t = step_size * (FixedMathCore(static_cast<int64_t>(i)) + MathConstants<FixedMathCore>::half());
		Vector3f sample_p = p_ray_origin + p_ray_dir * t;
		FixedMathCore height = (sample_p - p_planet_center).length() - p_planet_radius;

		if (height.get_raw() < 0) continue;

		// Local Density at altitude: exp(-h/H)
		FixedMathCore density = (-(height / p_params.rayleigh_scale_height)).exp();
		total_optical_depth += density * step_size;

		// Transmittance back to observer
		FixedMathCore trans_to_obs = (-(total_optical_depth * p_params.rayleigh_extinction)).exp();

		// 3. Multi-Light Spectral Aggregation
		Vector3f step_energy;
		for (uint32_t l = 0; l < p_lights.count; l++) {
			Vector3f L = (p_lights.type[l] == 0) ? p_lights.direction[l] : (p_lights.position[l] - sample_p).normalized();
			
			// Deterministic Shadowing
			if (wp::check_occlusion_sphere(sample_p, L, p_planet_center, p_planet_radius)) {
				continue;
			}

			// Transmittance from light source to sample point
			FixedMathCore light_od = wp::calculate_path_density(sample_p, L, p_params);
			FixedMathCore trans_to_light = (-(light_od * p_params.rayleigh_extinction)).exp();

			FixedMathCore cos_theta = p_ray_dir.dot(L);
			FixedMathCore phase = calculate_rayleigh_phase_logic(cos_theta);

			// In-Scattering: Beta * Phase * Sun * Density * Transmittance
			Vector3f scatt = beta_rayleigh * (phase * p_lights.energy[l] * density * trans_to_light * trans_to_obs);
			step_energy += scatt;
		}

		// 4. --- Sophisticated Real-Time Behavior: Anime Visuals ---
		if (p_is_anime) {
			// Anime Technique: "Vibrance Snap".
			// Forces colors to extreme saturation bands based on local energy intensity.
			FixedMathCore lum = step_energy.get_luminance();
			FixedMathCore snap_hi(3006477107LL, true); // 0.7
			FixedMathCore snap_lo(858993459LL, true);  // 0.2

			if (lum > snap_hi) {
				step_energy *= FixedMathCore(3LL, false); // Saturated daytime sky
			} else if (lum < snap_lo) {
				step_energy *= FixedMathCore(214748364LL, true); // Deep Indigo snap
			}
		}

		accumulated_radiance += step_energy * step_size;
	}

	r_radiance = accumulated_radiance;
}

/**
 * execute_rayleigh_resolve_wave()
 * 
 * Orchestrates the parallel 120 FPS Rayleigh resolve across the EnTT registry.
 * Zero-copy: Operates directly on SoA radiance and position streams.
 */
void execute_rayleigh_resolve_wave(
		KernelRegistry &p_registry,
		const Vector3f &p_ship_velocity,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_lights) {

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

		SimulationThreadPool::get_singleton()->enqueue_task([=, &radiance_stream, &pos_stream, &ray_stream, &len_stream, &p_params, &p_lights]() {
			for (uint64_t i = start; i < end; i++) {
				// Deterministic Style derivation from Entity handle hash
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 6 == 0); 

				rayleigh_integrator_kernel(
					handle,
					radiance_stream[i],
					pos_stream[i],
					ray_stream[i],
					p_ship_velocity,
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

	// Wait for the synchronization barrier
	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_rayleigh_scattering_logic.cpp ---
