--- START OF FILE core/math/atmospheric_rayleigh_scattering_kernel.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_rayleigh_phase()
 * 
 * Deterministic Rayleigh phase function: P(theta) = 3 / (16 * pi) * (1 + cos^2 theta)
 * Strictly uses Software-Defined Arithmetic to ensure zero-drift results.
 * Constant 3 / (16 * pi) approx: 0.0596831036
 */
static _FORCE_INLINE_ FixedMathCore calculate_rayleigh_phase(FixedMathCore p_cos_theta) {
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore coeff(256345098LL, true); // 0.0596831036 in bit-perfect Q32.32
	
	return coeff * (one + p_cos_theta * p_cos_theta);
}

/**
 * calculate_spectral_coefficients()
 * 
 * Computes scattering coefficients based on the inverse 4th power of wavelength.
 * lambda_r: 680nm, lambda_g: 550nm, lambda_b: 440nm.
 * Shifted by Doppler factor for high-speed ship kinematics.
 */
static _FORCE_INLINE_ Vector3f calculate_spectral_coefficients(
		const Vector3f &p_base_wavelengths, 
		FixedMathCore p_doppler_factor) {
	
	// lambda_obs = lambda_src / doppler_factor
	Vector3f obs_lambda = p_base_wavelengths / p_doppler_factor;
	
	// beta = 1 / lambda^4. Normalized for simulation energy scales.
	FixedMathCore r4 = obs_lambda.x * obs_lambda.x * obs_lambda.x * obs_lambda.x;
	FixedMathCore g4 = obs_lambda.y * obs_lambda.y * obs_lambda.y * obs_lambda.y;
	FixedMathCore b4 = obs_lambda.z * obs_lambda.z * obs_lambda.z * obs_lambda.z;

	FixedMathCore normalizer(429496729600000000LL, true); // bit-perfect unit scale
	
	return Vector3f(normalizer / r4, normalizer / g4, normalizer / b4);
}

/**
 * Warp Kernel: RayleighVolumeIntegratorKernel
 * 
 * Computes the spectral radiance contributed by Rayleigh scattering for a batch of view rays.
 * 1. Doppler Shift: Wavelengths are shifted based on spaceship relative velocity.
 * 2. Ray-Marching: Integrates density profile exp(-h/H) in bit-perfect steps.
 * 3. Light Interaction: Multi-source energy aggregation with deterministic shadows.
 * 4. Anime Style: Injects stylized saturation and hue-snapping for cel-shaded skies.
 */
void rayleigh_volume_integrator_kernel(
		const BigIntCore &p_index,
		Vector3f &r_radiance,
		const Vector3f &p_ray_origin,
		const Vector3f &p_ray_dir,
		const Vector3f &p_ship_velocity,
		const FixedMathCore &p_ray_length,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_lights,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Relativistic Doppler Factor Resolve
	// D = 1 + (v_radial / c)
	FixedMathCore radial_v = p_ray_dir.dot(p_ship_velocity);
	FixedMathCore c_fixed(299792458LL); // Speed of light in units/sec
	FixedMathCore doppler_factor = one + (radial_v / c_fixed);

	// Spectral Coefficients (RGB)
	Vector3f base_lambda(FixedMathCore(680LL), FixedMathCore(550LL), FixedMathCore(440LL));
	Vector3f beta_rayleigh = calculate_spectral_coefficients(base_lambda, doppler_factor);

	// 2. Integration March
	const int samples = 16;
	FixedMathCore step_size = p_ray_length / FixedMathCore(static_cast<int64_t>(samples));
	Vector3f accumulated_radiance;
	FixedMathCore total_optical_depth = zero;

	for (int i = 0; i < samples; i++) {
		FixedMathCore t = step_size * (FixedMathCore(static_cast<int64_t>(i)) + MathConstants<FixedMathCore>::half());
		Vector3f sample_pos = p_ray_origin + p_ray_dir * t;
		FixedMathCore altitude = (sample_pos - p_params.planet_center).length() - p_params.planet_radius;

		if (altitude.get_raw() < 0) continue;

		// Local Density (exp(-h/H))
		FixedMathCore density = wp::exp(-(altitude / p_params.rayleigh_scale_height));
		total_optical_depth += density * step_size;

		// 3. Multi-Light Interaction
		Vector3f step_energy;
		for (uint32_t l = 0; l < p_lights.count; l++) {
			Vector3f L = (p_lights.type[l] == 0) ? p_lights.direction[l] : (p_lights.position[l] - sample_pos).normalized();
			
			// Shadow check against planetary bulk
			if (wp::check_occlusion_sphere(sample_pos, L, p_params.planet_center, p_params.planet_radius)) continue;

			// Transmittance from light to sample point
			FixedMathCore light_od = wp::calculate_path_density(sample_pos, L, p_params);
			FixedMathCore trans_to_light = wp::exp(-(light_od * p_params.rayleigh_extinction));

			FixedMathCore cos_theta = p_ray_dir.dot(L);
			FixedMathCore phase = calculate_rayleigh_phase(cos_theta);

			step_energy += p_lights.color[l] * (p_lights.energy[l] * phase * trans_to_light);
		}

		// 4. View-Attenuation (Absorption between sample and observer)
		FixedMathCore trans_to_view = wp::exp(-(total_optical_depth * p_params.rayleigh_extinction));

		// 5. Sophisticated Anime Behavior
		if (p_is_anime) {
			// Anime Technique: "Atmospheric Saturation".
			// Forces colors to extreme bins if they cross energy thresholds.
			FixedMathCore lum = step_energy.get_luminance();
			FixedMathCore threshold(2147483648LL, true); // 0.5
			if (lum > threshold) {
				step_energy *= FixedMathCore(2LL); // Saturated glow
			} else {
				step_energy *= FixedMathCore(429496730LL, true); // 0.1 deep shadow snap
			}
		}

		accumulated_radiance += (beta_rayleigh * step_energy) * (trans_to_view * step_size);
	}

	r_radiance = accumulated_radiance;
}

/**
 * execute_rayleigh_scattering_wave()
 * 
 * Orchestrates the parallel 120 FPS sweep for atmospheric Rayleigh resolve.
 * Partitions EnTT component streams into SIMD-friendly worker batches.
 */
void execute_rayleigh_scattering_wave(
		KernelRegistry &p_registry,
		const Vector3f &p_ship_velocity,
		const Vector3f &p_sun_direction,
		const FixedMathCore &p_sun_energy,
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
				// Style derived from Entity handle hash for bit-perfect consistency
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 6 == 0); 

				rayleigh_volume_integrator_kernel(
					handle,
					radiance_stream[i],
					pos_stream[i],
					ray_stream[i],
					p_ship_velocity,
					len_stream[i],
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

--- END OF FILE core/math/atmospheric_rayleigh_scattering_kernel.cpp ---
