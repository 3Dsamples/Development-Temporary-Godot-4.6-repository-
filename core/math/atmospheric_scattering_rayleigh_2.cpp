--- START OF FILE core/math/atmospheric_scattering_rayleigh.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_rayleigh_coefficients()
 * 
 * Computes the spectral scattering intensity based on the inverse 4th power law.
 * Red (~680nm), Green (~550nm), Blue (~440nm).
 * Ported to bit-perfect Software-Defined Arithmetic.
 */
static _FORCE_INLINE_ Vector3f calculate_rayleigh_coefficients(const Vector3f &p_wavelengths_nm) {
    // scale factor to normalize units in Q32.32
    FixedMathCore scale(1000000000LL, false); 
    
    FixedMathCore r4 = p_wavelengths_nm.x * p_wavelengths_nm.x * p_wavelengths_nm.x * p_wavelengths_nm.x;
    FixedMathCore g4 = p_wavelengths_nm.y * p_wavelengths_nm.y * p_wavelengths_nm.y * p_wavelengths_nm.y;
    FixedMathCore b4 = p_wavelengths_nm.z * p_wavelengths_nm.z * p_wavelengths_nm.z * p_wavelengths_nm.z;

    // Scattering ~ 1 / lambda^4
    return Vector3f(scale / r4, scale / g4, scale / b4);
}

/**
 * Warp Kernel: RayleighVolumeIntegratorKernel
 * 
 * Performs parallel spectral integration of the sky color.
 * 1. Doppler Shift: Wavelengths are shifted by relativistic spaceship velocity.
 * 2. Density Sample: exp(-h/H) profile.
 * 3. In-Scattering: Phase function P(theta).
 * 4. Anime Style: Saturated gradients and hue snapping.
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

	// 1. Relativistic Doppler Resolve
	// Shift base wavelengths based on ship velocity relative to the sample vector.
	// f_shifted = f * (1 + v/c) -> lambda_shifted = lambda / (1 + v/c)
	FixedMathCore radial_v = p_ray_dir.dot(p_ship_velocity);
	FixedMathCore c_fixed(299792458LL); // speed of light
	FixedMathCore doppler_factor = one + (radial_v / c_fixed);

	// Initial Wavelength Tensors (nm)
	Vector3f base_lambda(FixedMathCore(680LL), FixedMathCore(550LL), FixedMathCore(440LL));
	Vector3f shifted_lambda = base_lambda / doppler_factor;
	Vector3f beta_rayleigh = calculate_rayleigh_coefficients(shifted_lambda);

	// 2. Integration March
	const int samples = 16;
	FixedMathCore step_size = p_ray_length / FixedMathCore(static_cast<int64_t>(samples));
	Vector3f accumulated_radiance;
	FixedMathCore optical_depth = zero;

	for (int i = 0; i < samples; i++) {
		FixedMathCore t = step_size * (FixedMathCore(static_cast<int64_t>(i)) + MathConstants<FixedMathCore>::half());
		Vector3f sample_p = p_ray_origin + p_ray_dir * t;
		FixedMathCore altitude = (sample_p - p_params.planet_center).length() - p_params.planet_radius;

		if (altitude.get_raw() < 0) continue;

		// Local Density at height
		FixedMathCore density = Math::exp(-(altitude / p_params.rayleigh_scale_height));
		optical_depth += density * step_size;

		// Deterministic Shadowing
		if (wp::check_occlusion_sphere(sample_p, p_sun_dir, p_params.planet_center, p_params.planet_radius)) {
			continue;
		}

		// Phase Function: P(theta) = 3/(16pi) * (1 + cos^2 theta)
		FixedMathCore cos_theta = p_ray_dir.dot(p_sun_dir);
		FixedMathCore rayleigh_phase = (FixedMathCore(3LL) / (FixedMathCore(16LL) * MathConstants<FixedMathCore>::pi())) * (one + cos_theta * cos_theta);

		// Transmittance back to ship
		FixedMathCore trans = wp::exp(-(optical_depth * p_params.rayleigh_extinction));

		// In-Scattering calculation
		Vector3f step_radiance = beta_rayleigh * (density * rayleigh_phase * p_sun_intensity * trans);

		// --- Sophisticated Real-Time Behavior: Anime style ---
		if (p_is_anime) {
			// Saturated sky snap: accentuate the blues or oranges
			FixedMathCore lum = step_radiance.get_luminance();
			if (lum > FixedMathCore(2147483648LL, true)) { // > 0.5
				step_radiance *= FixedMathCore(2LL); // Over-saturate daytime sky
			} else {
				step_radiance *= FixedMathCore(214748364LL, true); // 0.05 deep shadow
			}
		}

		accumulated_radiance += step_radiance * step_size;
	}

	r_radiance = accumulated_radiance;
}

/**
 * execute_rayleigh_wave()
 * 
 * Master orchestrator for parallel 120 FPS Rayleigh resolve.
 * Maps EnTT components to Warp Kernels.
 */
void execute_rayleigh_wave(
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
				BigIntCore entity_handle = p_registry.get_entity_at_index(COMPONENT_RADIANCE, i);
				bool anime_mode = (entity_handle.hash() % 6 == 0);

				rayleigh_volume_integrator_kernel(
					entity_handle,
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

--- END OF FILE core/math/atmospheric_scattering_rayleigh.cpp ---
