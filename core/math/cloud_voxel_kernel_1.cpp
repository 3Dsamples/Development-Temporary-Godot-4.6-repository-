--- START OF FILE core/math/cloud_voxel_kernel.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_shadow_depth_kernel()
 * 
 * Performs a deterministic micro-march from a voxel toward a light source.
 * Computes internal occlusion within the EnTT component stream.
 */
static _FORCE_INLINE_ FixedMathCore calculate_shadow_depth_kernel(
		const Vector3f &p_start_pos,
		const Vector3f &p_light_dir,
		const FixedMathCore *p_density_stream,
		uint64_t p_voxel_count,
		const FixedMathCore &p_step_size) {

	FixedMathCore depth_acc = MathConstants<FixedMathCore>::zero();
	Vector3f current_p = p_start_pos;

	// ETEngine Strategy: 6-sample shadow march to maintain 120 FPS budget
	for (int i = 0; i < 6; i++) {
		current_p += p_light_dir * p_step_size;
		
		// Map 3D pos to linear stream index (Assuming grid-aligned SoA)
		uint64_t idx = static_cast<uint64_t>(current_p.length().to_int()) % p_voxel_count;
		
		depth_acc += p_density_stream[idx] * p_step_size;
		
		// Optimization: Early exit for fully opaque regions
		if (depth_acc > FixedMathCore(42949672960LL, true)) break; // > 10.0 depth
	}

	return depth_acc;
}

/**
 * Warp Kernel: VolumetricCloudIrradianceKernel
 * 
 * Computes the internal lighting of a cloud voxel.
 * 1. Self-Shadowing: Beer's Law (e^-tau).
 * 2. Forward Scattering: Powder Effect (1.0 - e^-(2*density)).
 * 3. Anime Quantization: Snaps radiance into cel-shaded bands.
 */
void cloud_irradiance_kernel(
		const BigIntCore &p_index,
		Vector3f &r_radiance,
		const FixedMathCore &p_density,
		const Vector3f &p_pos,
		const Vector3f &p_sun_dir,
		const FixedMathCore &p_sun_intensity,
		const FixedMathCore *p_all_densities,
		uint64_t p_count,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Calculate Optical Depth to Sun
	FixedMathCore step(4294967296LL, false); // 1.0 unit
	FixedMathCore tau = calculate_shadow_depth_kernel(p_pos, p_sun_dir, p_all_densities, p_count, step);

	// 2. Resolve Beer-Powder Irradiance
	// Beer's Law: Absorption
	FixedMathCore beer = wp::exp(-tau);
	
	// Powder Law: Simulates multiple scattering in dark regions
	FixedMathCore powder = one - wp::exp(-(p_density * FixedMathCore(2LL, false)));
	
	FixedMathCore irradiance_mag = beer * powder * p_sun_intensity;

	// 3. --- Sophisticated Behavior: Realistic vs Anime ---
	if (p_is_anime) {
		// Anime Technique: Sharp Light-to-Shadow transition (Step Bands)
		FixedMathCore threshold(2147483648LL, true); // 0.5
		irradiance_mag = wp::step(threshold, irradiance_mag) * one + 
		                 (one - wp::step(threshold, irradiance_mag)) * FixedMathCore(858993459LL, true); // 0.2
		
		// Color shift based on density (Deeper blues in shadows)
		r_radiance = Vector3f(irradiance_mag, irradiance_mag, irradiance_mag * (one + p_density));
	} else {
		// Realistic: Smooth energy conservation
		r_radiance = Vector3f(irradiance_mag, irradiance_mag, irradiance_mag);
	}
}

/**
 * execute_cloud_volume_sweep()
 * 
 * Master orchestrator for parallel cloud lighting.
 * Partitions the EnTT voxel registry into SIMD-friendly worker batches.
 */
void execute_cloud_volume_sweep(
		KernelRegistry &p_registry,
		const Vector3f &p_sun_dir,
		const FixedMathCore &p_sun_energy) {

	auto &density_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_DENSITY);
	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &radiance_stream = p_registry.get_stream<Vector3f>(COMPONENT_RADIANCE);
	
	uint64_t total_voxels = density_stream.size();
	if (total_voxels == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total_voxels / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? total_voxels : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &density_stream, &pos_stream, &radiance_stream]() {
			for (uint64_t i = start; i < end; i++) {
				// Deterministic style selection based on entity handle hash
				bool anime_mode = (i % 4 == 0); 

				cloud_irradiance_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					radiance_stream[i],
					density_stream[i],
					pos_stream[i],
					p_sun_dir,
					p_sun_energy,
					density_stream.get_base_ptr(),
					total_voxels,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * apply_cloud_buoyancy_physics()
 * 
 * Updates cloud voxel velocities based on thermal interaction with light.
 * Hotter (illuminated) voxels rise; cooler (shadowed) voxels sink.
 */
void apply_cloud_buoyancy_physics(
		Vector3f *r_velocities,
		const FixedMathCore *p_temperatures,
		uint64_t p_count,
		const FixedMathCore &p_delta) {
	
	FixedMathCore ambient_temp(12376175411LL, true); // 288.15K
	FixedMathCore buoyancy_k(429496730LL, true);    // 0.1 coeff

	for (uint64_t i = 0; i < p_count; i++) {
		FixedMathCore lift = (p_temperatures[i] - ambient_temp) * buoyancy_k;
		r_velocities[i].y += lift * p_delta;
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/cloud_voxel_kernel.cpp ---
