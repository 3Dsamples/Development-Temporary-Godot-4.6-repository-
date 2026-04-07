--- START OF FILE core/math/spectral_radiance_scattering_kernel.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/cloud_voxel_kernel.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_anisotropic_phase_kernel()
 * 
 * Deterministic Henyey-Greenstein phase function for volume scattering.
 * p_g: Asymmetry factor [-1..1]. Positive values simulate forward scattering.
 */
static _FORCE_INLINE_ FixedMathCore calculate_anisotropic_phase_kernel(FixedMathCore p_cos_theta, FixedMathCore p_g) {
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore g2 = p_g * p_g;
	FixedMathCore denom_base = one + g2 - (FixedMathCore(2LL, false) * p_g * p_cos_theta);
	
	// denom = (1 + g^2 - 2g cos(theta))^1.5
	FixedMathCore denom = denom_base * Math::sqrt(denom_base);
	if (denom.get_raw() == 0) return one;

	FixedMathCore factor = (one - g2) / (FixedMathCore(13493037704LL, true) * denom); // 4*PI * denom
	return factor;
}

/**
 * Warp Kernel: VolumetricScatteringIntegrator
 * 
 * Marches through a dense EnTT voxel registry to compute light in-scattering and absorption.
 * 1. Resolves shadowing from sun-light to voxel (Light-Transmittance).
 * 2. Resolves in-scattering toward the observer (View-Transmittance).
 * 3. Applies localized spectral albedo for nebular coloring.
 */
void volumetric_scattering_integrator_kernel(
		const BigIntCore &p_index,
		Vector3f &r_accumulated_radiance,
		const Vector3f &p_ray_origin,
		const Vector3f &p_ray_dir,
		const FixedMathCore &p_max_dist,
		const CloudVoxel *p_voxel_data,
		const Vector3f &p_sun_dir,
		const FixedMathCore &p_sun_intensity,
		const FixedMathCore &p_step_size,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	
	Vector3f in_scatter;
	FixedMathCore optical_depth = zero;

	// Deterministic 16-step march for 120 FPS budget
	const int step_count = 16;
	
	for (int i = 0; i < step_count; i++) {
		FixedMathCore t = p_step_size * (FixedMathCore(static_cast<int64_t>(i), false) + MathConstants<FixedMathCore>::half());
		if (t > p_max_dist) break;

		Vector3f sample_p = p_ray_origin + p_ray_dir * t;
		
		// Map 3D position to BigInt grid index
		// (Simplified coordinate mapping for core logic)
		uint64_t voxel_idx = static_cast<uint64_t>(std::stoll(p_index.to_string())) + i; 
		const CloudVoxel &v = p_voxel_data[voxel_idx];

		if (v.density <= zero) continue;

		// 1. Calculate Sun-to-Voxel Shadowing (Beer's Law)
		// We use a bit-perfect shadow factor pre-calculated in the shadow sweep
		FixedMathCore sun_transmittance = Math::exp(-v.light_absorption);

		// 2. Compute Scattering Contribution
		FixedMathCore cos_theta = p_ray_dir.dot(p_sun_dir);
		FixedMathCore phase = calculate_anisotropic_phase_kernel(cos_theta, FixedMathCore(3435973836LL, true)); // g=0.8

		// 3. --- Sophisticated Real-Time Behavior: Spectral Albedo Shifting ---
		// Nebulae change color based on internal temperature (FixedMathCore)
		Vector3f albedo_color = Vector3f(one, FixedMathCore(2147483648LL, true), one); // Base pink/blue
		FixedMathCore heat_factor = wp::clamp(v.temperature / FixedMathCore(5000LL, false), zero, one);
		albedo_color = wp::lerp(albedo_color, Vector3f(one, zero, zero), heat_factor); // Shift to red

		if (p_is_anime) {
			// Anime Style: Force vibrant albedo snaps and hard-edged shadows
			sun_transmittance = wp::step(FixedMathCore(2147483648LL, true), sun_transmittance);
			albedo_color *= FixedMathCore(2LL, false); // Saturated colors
		}

		// 4. Accumulate In-scattering
		// S_local = Sun * Phase * Density * Albedo * SunTransmittance
		Vector3f local_scatter = albedo_color * (p_sun_intensity * phase * v.density * sun_transmittance);
		
		// 5. Apply View-Attenuation (Absorption between voxel and observer)
		FixedMathCore view_transmittance = Math::exp(-optical_depth);
		in_scatter += local_scatter * (view_transmittance * p_step_size);

		// Accumulate total depth for next step
		optical_depth += v.density * p_step_size;
	}

	r_accumulated_radiance = in_scatter;
}

/**
 * execute_volumetric_scattering_sweep()
 * 
 * Orchestrates the parallel light transport simulation for all EnTT voxels.
 * Zero-copy logic: Directly reads/writes to registry radiance buffers.
 */
void execute_volumetric_scattering_sweep(
		const BigIntCore &p_total_rays,
		const Vector3f *p_ray_origins,
		const Vector3f *p_ray_dirs,
		const CloudVoxel *p_voxels,
		Vector3f *r_radiance_output,
		const Vector3f &p_sun_dir,
		const FixedMathCore &p_sun_energy) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_total_rays.to_string()));
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? total : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				// Deterministic style selection based on entity handle
				bool anime_mode = (i % 3 == 0); 

				volumetric_scattering_integrator_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_radiance_output[i],
					p_ray_origins[i],
					p_ray_dirs[i],
					FixedMathCore(10000LL, false), // 10k range
					p_voxels,
					p_sun_dir,
					p_sun_energy,
					FixedMathCore(100LL, false), // 100 unit steps
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/spectral_radiance_scattering_kernel.cpp ---
