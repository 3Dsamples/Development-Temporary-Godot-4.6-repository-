--- START OF FILE core/math/atmospheric_scattering_mie.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_mie_phase_hg()
 * 
 * Deterministic implementation of the Henyey-Greenstein phase function.
 * Formula: P(g, cos_theta) = (1 - g^2) / (4 * PI * (1 + g^2 - 2 * g * cos_theta)^(1.5))
 * strictly uses Software-Defined Arithmetic to ensure bit-perfection across nodes.
 */
static _FORCE_INLINE_ FixedMathCore calculate_mie_phase_hg(FixedMathCore p_cos_theta, FixedMathCore p_g) {
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore g2 = p_g * p_g;
	FixedMathCore four_pi(13493037704LL, true); // 4 * PI in Q32.32

	FixedMathCore numerator = one - g2;
	
	// Denominator inner: (1 + g^2 - 2 * g * cos_theta)
	FixedMathCore denom_inner = one + g2 - (FixedMathCore(2LL) * p_g * p_cos_theta);
	
	// Power 1.5 resolve: x^1.5 = sqrt(x^3)
	// Guaranteed bit-identical result via FixedMathCore::square_root()
	FixedMathCore denom_p1_5 = Math::sqrt(denom_inner * denom_inner * denom_inner);
	FixedMathCore denominator = four_pi * denom_p1_5;

	if (unlikely(denominator.get_raw() == 0)) {
		return one; // Fallback for singularity
	}

	return numerator / denominator;
}

/**
 * Warp Kernel: MieVolumeIntegratorKernel
 * 
 * Computes the spectral radiance contributed by Mie scattering for a single ray.
 * 1. Ray-Marching: Samples density along the view vector.
 * 2. Sun-Visibility: Performs a bit-perfect shadow check for each sample.
 * 3. Haze Composition: Aggregates phase-shifted radiance.
 * 4. Anime Stylization: Snaps the Mie-glow into discrete bands for cel-shading.
 */
void mie_volume_integrator_kernel(
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
	
	// Optimized for 120 FPS: 8-sample march for narrow-angle Mie glare
	const int sample_count = 8;
	FixedMathCore step_size = p_ray_length / FixedMathCore(static_cast<int64_t>(sample_count));
	Vector3f accumulated_mie;

	for (int i = 0; i < sample_count; i++) {
		FixedMathCore t = step_size * (FixedMathCore(static_cast<int64_t>(i)) + MathConstants<FixedMathCore>::half());
		Vector3f sample_pos = p_ray_origin + p_ray_dir * t;
		FixedMathCore altitude = sample_pos.length() - p_params.planet_radius;

		if (altitude.get_raw() < 0) continue;

		// 1. Resolve local haze density (Exponential decay)
		FixedMathCore density = Math::exp(-(altitude / p_params.mie_scale_height));

		// 2. Deterministic Shadowing
		// Check if the sun is occluded by the planetary bulk at this atmospheric point
		if (wp::check_occlusion_sphere(sample_pos, p_sun_dir, p_params.planet_center, p_params.planet_radius)) {
			continue;
		}

		// 3. Compute Phase Function
		FixedMathCore cos_theta = p_ray_dir.dot(p_sun_dir);
		FixedMathCore phase = calculate_mie_phase_hg(cos_theta, p_params.mie_g);

		// --- Sophisticated Behavior: Realistic vs Anime ---
		if (p_is_anime) {
			// Anime Technique: "Sun Rings". 
			// Instead of a smooth gradient, we quantize the phase into sharp glare bands.
			FixedMathCore band_0_threshold(3865470566LL, true); // 0.9
			FixedMathCore band_1_threshold(2147483648LL, true); // 0.5
			
			if (phase > band_0_threshold) {
				phase = FixedMathCore(10LL); // Intense core
			} else if (phase > band_1_threshold) {
				phase = FixedMathCore(2LL);  // Secondary halo
			} else {
				phase = zero; // Sharp cutoff for anime cel-look
			}
		}

		// 4. Energy Integration
		// S_mie = Sun * Phase * Density * Mie_Coefficient
		FixedMathCore scattering_mag = p_sun_intensity * phase * density * p_params.mie_coefficient;
		accumulated_mie += Vector3f(scattering_mag, scattering_mag, scattering_mag) * step_size;
	}

	r_radiance = accumulated_mie;
}

/**
 * execute_mie_scattering_wave()
 * 
 * Orchestrates the parallel 120 FPS sweep for atmospheric haze.
 * Partitions the EnTT radiance registry into worker chunks.
 * Zero-copy: Operates directly on the aligned memory addresses of the registry.
 */
void execute_mie_scattering_wave(
		KernelRegistry &p_registry,
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
				// Deterministic Style Selection based on entity handle hash
				BigIntCore entity_idx(static_cast<int64_t>(i));
				bool anime_mode = (entity_idx.hash() % 16 == 0);

				mie_volume_integrator_kernel(
					entity_idx,
					radiance_stream[i],
					pos_stream[i],
					ray_stream[i],
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

--- END OF FILE core/math/atmospheric_scattering_mie.cpp ---
