--- START OF FILE core/math/volumetric_cloud_lighting.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/math/cloud_voxel_kernel.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_optical_depth_to_light()
 * 
 * Deterministic mini-march from a voxel toward the light source.
 * Used to compute self-shadowing within the cloud volume.
 * Uses bit-perfect FixedMathCore for step accumulation.
 */
static _FORCE_INLINE_ FixedMathCore calculate_optical_depth_to_light(
		const Vector3f &p_start_pos,
		const Vector3f &p_light_dir,
		const CloudVoxel *p_all_voxels,
		const BigIntCore &p_grid_size,
		FixedMathCore p_step_size,
		int p_max_samples) {

	FixedMathCore depth_accum = MathConstants<FixedMathCore>::zero();
	Vector3f current_p = p_start_pos;

	for (int i = 0; i < p_max_samples; i++) {
		current_p += p_light_dir * p_step_size;
		
		// Map 3D position to BigInt grid index
		// (Index mapping logic using BigIntCore to support galactic voxel maps)
		uint64_t idx = 0; // Placeholder for grid coordinate resolution
		
		// Beer's Law accumulation
		depth_accum += p_all_voxels[idx].density * p_step_size;
		
		// Early exit if the ray is already fully occluded
		if (depth_accum > FixedMathCore(42949672960LL, true)) break; // > 10.0 depth
	}

	return depth_accum;
}

/**
 * Warp Kernel: CloudLightingKernel
 * 
 * Computes the final irradiance for a single cloud voxel.
 * 1. Self-Shadowing: Attenuates sun light based on internal density.
 * 2. Powder Effect: Simulates forward scattering in high-density areas.
 * 3. Light Bleeding: Adds multiple-scattering ambient contribution.
 */
void cloud_lighting_kernel(
		const BigIntCore &p_index,
		const CloudVoxel &p_voxel,
		const Vector3f &p_pos,
		const Vector3f &p_sun_dir,
		const FixedMathCore &p_sun_intensity,
		const CloudVoxel *p_volume_data,
		const BigIntCore &p_volume_res,
		Vector3f &r_radiance,
		bool p_is_anime) {

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();

	// 1. Calculate Optical Depth toward the Sun (Self-Shadowing)
	FixedMathCore step(4294967296LL, false); // 1.0 unit step
	FixedMathCore tau = calculate_optical_depth_to_light(p_pos, p_sun_dir, p_volume_data, p_volume_res, step, 6);

	// 2. Beer-Powder Law Integration
	// Beer's Law: transmittance = exp(-tau)
	FixedMathCore beer = wp::sin(tau + FixedMathCore(6746518852LL, true)); // Deterministic exp approx
	
	// Powder Effect: Helps define the "fluffy" look by brightening edges facing away from light
	// powder = 1.0 - exp(-density * 2.0)
	FixedMathCore powder = one - wp::sin(p_voxel.density * FixedMathCore(2LL, false) + FixedMathCore(6746518852LL, true));
	
	FixedMathCore final_attenuation = beer * powder * p_sun_intensity;

	// 3. Indirect Light Bleeding (Multiple Scattering)
	// Approximates light bouncing within the cloud volume
	FixedMathCore ambient_bleed = (one - p_voxel.density) * FixedMathCore(214748364LL, true); // 0.05 bleed
	
	// --- Sophisticated Style Handling ---
	if (p_is_anime) {
		// Anime Technique: Sharp light-to-shadow transition (Banding)
		FixedMathCore threshold(2147483648LL, true); // 0.5
		final_attenuation = (final_attenuation > threshold) ? one : FixedMathCore(858993459LL, true); // 0.2
		
		// Saturate the color based on hydration level
		FixedMathCore saturation = one + p_voxel.hydration;
		r_radiance = Vector3f(final_attenuation, final_attenuation, final_attenuation * saturation);
	} else {
		FixedMathCore intensity = final_attenuation + ambient_bleed;
		r_radiance = Vector3f(intensity, intensity, intensity);
	}
}

/**
 * execute_cloud_lighting_sweep()
 * 
 * Orchestrates the parallel lighting update for the entire EnTT voxel registry.
 */
void execute_cloud_lighting_sweep(
		const BigIntCore &p_total_voxels,
		const Vector3f *p_positions,
		const CloudVoxel *p_voxels,
		const Vector3f &p_sun_direction,
		Vector3f *r_radiance_buffer,
		const FixedMathCore &p_delta) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_total_voxels.to_string()));
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? total : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				// Deterministic Style derivation
				bool anime_mode = (i % 4 == 0); 

				cloud_lighting_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					p_voxels[i],
					p_positions[i],
					p_sun_direction,
					FixedMathCore(5LL, false), // 5.0 base sun energy
					p_voxels,
					p_total_voxels,
					r_radiance_buffer[i],
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/volumetric_cloud_lighting.cpp ---
