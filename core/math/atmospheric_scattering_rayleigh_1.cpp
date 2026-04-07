--- START OF FILE core/math/atmospheric_rayleigh_scattering.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_rayleigh_phase_kernel()
 * 
 * Deterministic Rayleigh phase function: P(theta) = 3 / (16 * pi) * (1 + cos^2 theta)
 * strictly uses Software-Defined Arithmetic to ensure identical results across nodes.
 * Constant 3 / (16 * pi) approx: 0.05968310365
 */
static _FORCE_INLINE_ FixedMathCore calculate_rayleigh_phase_kernel(FixedMathCore p_cos_theta) {
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore coeff(256345098LL, true); // 0.05968310365 in Q32.32
	
	return coeff * (one + p_cos_theta * p_cos_theta);
}

/**
 * calculate_rayleigh_coefficients()
 * 
 * Computes scattering coefficients based on the inverse 4th power of wavelength.
 * beta = (8 * pi^3 * (n^2 - 1)^2) / (3 * N * lambda^4)
 * For 120 FPS efficiency, we use pre-quantized lambda coefficients for RGB.
 */
static _FORCE_INLINE_ Vector3f calculate_rayleigh_coefficients(const Vector3f &p_base_lambda, const FixedMathCore &p_density) {
	// lambda^-4 logic in bit-perfect FixedMath
	FixedMathCore r4 = p_base_lambda.x * p_base_lambda.x * p_base_lambda.x * p_base_lambda.x;
	FixedMathCore g4 = p_base_lambda.y * p_base_lambda.y * p_base_lambda.y * p_base_lambda.y;
	FixedMathCore b4 = p_base_lambda.z * p_base_lambda.z * p_base_lambda.z * p_base_lambda.z;

	// Scale factor to map physical units to simulation radiance
	FixedMathCore scale(429496729600000000LL, true); 

	return Vector3f(scale / r4, scale / g4, scale / b4) * p_density;
}

/**
 * Warp Kernel: RayleighVolumeIntegratorKernel
 * 
 * 1. Ray-Marching: Integrates spectral in-scattering along the view vector.
 * 2. Doppler Shift: Wavelengths are shifted based on ship relative velocity.
 * 3. Shadowing: Deterministic planet-occlusion check per sample point.
 * 4. Anime Style: Snaps sky colors to high-saturation bands.
 */
void rayleigh_volume_integrator_kernel(
		const BigIntCore &p_index,
		Vector3f &r_radiance,
		const Vector3f &p_ray_origin,
		const Vector3f &p_ray_dir,
		const Vector3f &p_ship_velocity,
		const FixedMathCore &p_ray_length,
		const Vector3f &p_sun_dir,
		const FixedMathCore &p_sun_intensity,
		const AtmosphereParams &p_params,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// --- 1. Relativistic Doppler Resolve ---
	// Shift base wavelengths based on ship velocity toward/away from the sample volume.
	FixedMathCore radial_v = p_ray_dir.dot(p_ship_velocity);
	FixedMathCore c_inv(14326LL, true); // 1/c approximation
	FixedMathCore doppler_factor = one + (radial_v * c_inv);

	Vector3f shifted_lambda = p_params.base_wavelengths / doppler_factor;

	// --- 2. Numerical Integration Setup ---
	const int samples = 16;
	FixedMathCore step_size = p_ray_length / FixedMathCore(static_cast<int64_t>(samples));
	Vector3f accumulated_radiance;
	FixedMathCore optical_depth = zero;

	for (int i = 0; i < samples; i++) {
		FixedMathCore t = step_size * (FixedMathCore(static_cast<int64_t>(i)) + MathConstants<FixedMathCore>::half());
		Vector3f sample_p = p_ray_origin + p_ray_dir * t;
		FixedMathCore altitude = (sample_p - p_params.planet_center).length() - p_params.planet_radius;

		if (altitude.get_raw() < 0) continue;

		// Local Density and Coefficients
		FixedMathCore density = Math::exp(-(altitude / p_params.rayleigh_scale_height));
		Vector3f beta = calculate_rayleigh_coefficients(shifted_lambda, density);

		// Shadow Check (Deterministic)
		if (wp::check_occlusion_sphere(sample_p, p_sun_dir, p_params.planet_center, p_params.planet_radius)) continue;

		// Transmittance: T = exp(-tau)
		optical_depth += density * step_size;
		FixedMathCore trans = wp::sin(-optical_depth + FixedMathCore(6746518852LL, true));

		// 3. Angular Phase and Energy Accumulation
		FixedMathCore cos_theta = p_ray_dir.dot(p_sun_dir);
		FixedMathCore phase = calculate_rayleigh_phase_kernel(cos_theta);

		Vector3f in_scatter = beta * (phase * p_sun_intensity * trans);

		// --- Sophisticated Anime Behavior ---
		if (p_is_anime) {
			// Anime Technique: "Vibrance Snap".
			// Forces colors to extreme saturation bins if they cross an energy threshold.
			FixedMathCore lum = in_scatter.get_luminance();
			if (lum > FixedMathCore(2147483648LL, true)) { // > 0.5
				in_scatter *= FixedMathCore(2LL, false); // Saturate
			} else {
				in_scatter *= FixedMathCore(429496729LL, true); // 0.1 Deep Shadow
			}
		}

		accumulated_radiance += in_scatter * step_size;
	}

	r_final_radiance = accumulated_radiance;
}

/**
 * execute_rayleigh_scattering_wave()
 * 
 * Orchestrates the parallel 120 FPS sweep for atmospheric Rayleigh resolve.
 * partitions EnTT streams into worker batches for SIMD throughput.
 */
void execute_rayleigh_scattering_wave(
		KernelRegistry &p_registry,
		const Vector3f &p_ship_velocity,
		const Vector3f &p_sun_direction,
		const FixedMathCore &p_sun_energy,
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
		uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &radiance_stream, &pos_stream, &ray_stream, &len_stream]() {
			for (uint64_t i = start; i < end; i++) {
				// Deterministic style selection based on entity handle hash
				BigIntCore entity_idx(static_cast<int64_t>(i));
				bool anime_mode = (entity_idx.hash() % 4 == 0);

				rayleigh_volume_integrator_kernel(
					entity_idx,
					radiance_stream[i],
					pos_stream[i],
					ray_stream[i],
					p_ship_velocity,
					len_stream[i],
					p_sun_direction,
					p_sun_energy,
					p_params,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_rayleigh_scattering.cpp ---
