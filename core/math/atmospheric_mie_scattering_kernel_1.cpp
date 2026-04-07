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
 * calculate_mie_phase_kernel()
 * 
 * Bit-perfect Henyey-Greenstein Phase Function.
 * P(theta) = (1 - g^2) / (4*pi * (1 + g^2 - 2*g*cos(theta))^1.5)
 */
static _FORCE_INLINE_ FixedMathCore calculate_mie_phase_kernel(FixedMathCore p_cos_theta, FixedMathCore p_g) {
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore g2 = p_g * p_g;
	FixedMathCore two_g = FixedMathCore(2LL, false) * p_g;
	
	FixedMathCore numerator = one - g2;
	FixedMathCore denom_inner = one + g2 - (two_g * p_cos_theta);
	
	// x^1.5 = sqrt(x^3) via deterministic software-arithmetic
	FixedMathCore denom_pow = (denom_inner * denom_inner * denom_inner).square_root();
	FixedMathCore denominator = FixedMathCore(13493037704LL, true) * denom_pow; // 4 * pi * pow

	if (unlikely(denominator.get_raw() == 0)) return one;
	return numerator / denominator;
}

/**
 * Warp Kernel: MieVolumeIntegrationKernel
 * 
 * Performs a deterministic ray-march through the Mie-density volume.
 * 1. Resolves local density at altitude (exp falloff).
 * 2. Aggregates spectral energy from multiple lights (Stars + Local).
 * 3. Applies Anime-Style quantization to light intensity and halos.
 */
void mie_volume_integration_kernel(
		const BigIntCore &p_index,
		Vector3f &r_radiance,
		const Vector3f &p_ray_origin,
		const Vector3f &p_ray_dir,
		const FixedMathCore &p_ray_length,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_lights,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// Optimized 8-sample march to maintain 120 FPS performance budget
	const int samples = 8;
	FixedMathCore step_size = p_ray_length / FixedMathCore(static_cast<int64_t>(samples));
	Vector3f accumulated_radiance;
	FixedMathCore total_optical_depth = zero;

	for (int i = 0; i < samples; i++) {
		FixedMathCore t = step_size * (FixedMathCore(static_cast<int64_t>(i)) + MathConstants<FixedMathCore>::half());
		Vector3f sample_pos = p_ray_origin + p_ray_dir * t;
		
		// Altitude calculation relative to the planetary anchor
		FixedMathCore altitude = (sample_pos - p_params.planet_center).length() - p_params.planet_radius;
		if (altitude.get_raw() < 0) continue;

		// 1. Resolve local haze density
		FixedMathCore density = wp::exp(-(altitude / p_params.mie_scale_height));
		total_optical_depth += density * step_size;

		// 2. Light Source Interaction Loop
		Vector3f inscatter_energy;
		for (uint32_t l = 0; l < p_lights.count; l++) {
			Vector3f L = (p_lights.type[l] == 0) ? p_lights.direction[l] : (p_lights.position[l] - sample_pos).normalized();
			
			// Shadow check against planet bulk
			if (wp::check_occlusion_sphere(sample_pos, L, p_params.planet_center, p_params.planet_radius)) continue;

			// Transmittance from light to sample
			FixedMathCore light_od = wp::calculate_path_density(sample_pos, L, p_params);
			FixedMathCore light_trans = wp::exp(-light_od);

			// Phase Function Resolve
			FixedMathCore cos_theta = p_ray_dir.dot(L);
			FixedMathCore phase = calculate_mie_phase_kernel(cos_theta, p_params.mie_g);

			// --- Sophisticated Anime Behavior: Quantized Haze ---
			if (p_is_anime) {
				// Snap light transmittance to discrete cel-bands
				light_trans = wp::step(FixedMathCore(2147483648LL, true), light_trans) * one + 
				              wp::step(FixedMathCore(858993459LL, true), light_trans) * FixedMathCore(2147483648LL, true);
				
				// Exaggerated sharp Mie-Halo
				phase = wp::step(FixedMathCore(3865470566LL, true), phase) * FixedMathCore(10LL) + one;
			}

			FixedMathCore scattering = p_lights.energy[l] * phase * density * light_trans;
			inscatter_energy += p_lights.color[l] * scattering;
		}

		// 3. Extinction to Observer
		FixedMathCore view_trans = wp::exp(-(total_optical_depth * p_params.mie_coefficient));
		accumulated_radiance += inscatter_energy * (view_trans * step_size);
	}

	r_radiance = accumulated_radiance * p_params.mie_coefficient;
}

/**
 * execute_mie_volume_sweep()
 * 
 * Orchestrates the parallel 120 FPS resolve for all viewing rays in the EnTT registry.
 * Zero-copy: operates directly on radiance and ray-direction SoA buffers.
 */
void execute_mie_volume_sweep(
		KernelRegistry &p_registry,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_lights) {

	auto &radiance_stream = p_registry.get_stream<Vector3f>(COMPONENT_RADIANCE);
	auto &ray_dir_stream = p_registry.get_stream<Vector3f>(COMPONENT_RAY_DIR);
	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &len_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_RAY_LENGTH);

	uint64_t count = radiance_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &radiance_stream, &ray_dir_stream, &pos_stream, &len_stream, &p_params, &p_lights]() {
			for (uint64_t i = start; i < end; i++) {
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 12 == 0);

				mie_volume_integration_kernel(
					handle,
					radiance_stream[i],
					pos_stream[i],
					ray_dir_stream[i],
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

--- END OF FILE core/math/atmospheric_mie_scattering_kernel.cpp ---
