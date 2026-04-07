--- START OF FILE core/math/atmospheric_rayleigh_phase_kernel.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_rayleigh_phase()
 * 
 * Deterministic Rayleigh phase function: P(theta) = 3 / (16 * pi) * (1 + cos^2 theta)
 * This handles the angular distribution of light scattered by molecules.
 */
static _FORCE_INLINE_ FixedMathCore calculate_rayleigh_phase(FixedMathCore p_cos_theta) {
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	// 3 / (16 * pi) approximated in Q32.32 bit-perfection
	FixedMathCore coefficient(256543160LL, true); // ~0.059683

	return coefficient * (one + p_cos_theta * p_cos_theta);
}

/**
 * Warp Kernel: RayleighScatteringSpectralKernel
 * 
 * Computes the spectral radiance for Rayleigh scattering.
 * 1. Resolves the 1/lambda^4 dependency for RGB wavelengths.
 * 2. Applies the phase function based on the sun's relative position.
 * 3. Injects a Red-Shift tensor for low-altitude (sunset) light paths.
 */
void rayleigh_scattering_spectral_kernel(
		const BigIntCore &p_index,
		Vector3f &r_radiance,
		const Vector3f &p_sample_pos,
		const Vector3f &p_view_dir,
		const Vector3f &p_sun_dir,
		const FixedMathCore &p_sun_intensity,
		const AtmosphereParams &p_params,
		bool p_is_anime) {

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();

	// 1. Angular Phase Calculation
	FixedMathCore cos_theta = p_view_dir.dot(p_sun_dir);
	FixedMathCore phase = calculate_rayleigh_phase(cos_theta);

	// 2. Altitude and Density Resolve
	FixedMathCore altitude = p_sample_pos.length() - p_params.planet_radius;
	FixedMathCore density = AtmosphericScattering::compute_density(altitude, p_params.rayleigh_scale_height);

	// 3. Sunset Red-Shift / Path-Length Correction
	// We calculate how much the light is "filtered" based on the zenith angle.
	FixedMathCore sun_zenith = p_sun_dir.y; // Simplified for planetary up-axis
	FixedMathCore red_shift = one - wp::clamp(sun_zenith + FixedMathCore(429496730LL, true), zero, one); // 0.1 horizon bias
	
	// 4. Spectral Scattering Tensors
	// Basic Rayleigh coefficients (1/lambda^4)
	Vector3f scattering_coeffs = p_params.rayleigh_coefficients;

	if (p_is_anime) {
		// --- Sophisticated Anime Behavior ---
		// Saturate the blue scattering during the day and force sharp 
		// orange/purple transitions during sunsets using cel-banding.
		FixedMathCore day_threshold(2147483648LL, true); // 0.5
		if (sun_zenith > day_threshold) {
			// Boost blue wavelength for "Vibrant Anime Sky"
			scattering_coeffs.z *= FixedMathCore(2LL, false); 
		} else {
			// Banded Sunset: Snap red-shift to discrete values
			red_shift = wp::step(FixedMathCore(3006477107LL, true), red_shift) * one + 
			            wp::step(FixedMathCore(1288490188LL, true), red_shift) * FixedMathCore(2147483648LL, true);
		}
	}

	// 5. Final Spectral Integration
	// I = Sun * Phase * Density * Coeffs * (RedShift Adjustment)
	FixedMathCore base_intensity = p_sun_intensity * phase * density;
	
	r_radiance.x += base_intensity * scattering_coeffs.x * (one + red_shift * FixedMathCore(5LL, false));
	r_radiance.y += base_intensity * scattering_coeffs.y * (one + red_shift);
	r_radiance.z += base_intensity * scattering_coeffs.z * (one - red_shift);
}

/**
 * execute_rayleigh_volume_sweep()
 * 
 * Parallel EnTT sweep to resolve blue-sky gradients.
 * Processes atmospheric density components in contiguous SoA memory.
 */
void execute_rayleigh_volume_sweep(
		const BigIntCore &p_count,
		const Vector3f *p_positions,
		const Vector3f *p_view_dirs,
		const Vector3f &p_sun_dir,
		const FixedMathCore &p_sun_intensity,
		const AtmosphereParams &p_params,
		Vector3f *r_radiance_buffer) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_count.to_string()));
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? total : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &p_params]() {
			for (uint64_t i = start; i < end; i++) {
				bool anime_mode = (i % 4 == 0); // Deterministic style assignment
				
				rayleigh_scattering_spectral_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_radiance_buffer[i],
					p_positions[i],
					p_view_dirs[i],
					p_sun_dir,
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

--- END OF FILE core/math/atmospheric_rayleigh_phase_kernel.cpp ---
