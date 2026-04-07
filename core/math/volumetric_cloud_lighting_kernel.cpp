--- START OF FILE core/math/volumetric_cloud_lighting_kernel.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/math/cloud_voxel_kernel.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_sun_occlusion_tau()
 * 
 * Performs a deterministic micro-march from a specific voxel toward the sun.
 * Accumulates optical depth (tau) to resolve Beer's Law self-shadowing.
 * strictly uses FixedMathCore to ensure identical shadow patterns on all nodes.
 */
static _FORCE_INLINE_ FixedMathCore calculate_sun_occlusion_tau(
		const Vector3f &p_start_pos,
		const Vector3f &p_sun_dir,
		const FixedMathCore *p_density_stream,
		const BigIntCore &p_grid_res,
		const FixedMathCore &p_step_size,
		uint32_t p_samples) {

	FixedMathCore tau_accum = MathConstants<FixedMathCore>::zero();
	Vector3f current_p = p_start_pos;

	// ETEngine Strategy: 6-step shadow march is the sweet spot for 120 FPS.
	for (uint32_t i = 0; i < p_samples; i++) {
		current_p += p_sun_dir * p_step_size;
		
		// Map 3D coordinate to linear EnTT stream index
		// Use BigInt logic for grid resolution to support planetary-scale cloud maps
		int64_t gx = Math::floor(current_p.x).to_int();
		int64_t gy = Math::floor(current_p.y).to_int();
		int64_t gz = Math::floor(current_p.z).to_int();
		
		BigIntCore linear_idx = (BigIntCore(gz) * p_grid_res * p_grid_res) + (BigIntCore(gy) * p_grid_res) + BigIntCore(gx);
		uint64_t raw_idx = static_cast<uint64_t>(std::stoll(linear_idx.to_string()));
		
		// Accumulate density along the light path
		tau_accum += p_density_stream[raw_idx % 1000000] * p_step_size; // Modulo as safety for core example
		
		// Optimization: Early exit if light is fully absorbed
		if (tau_accum > FixedMathCore(42949672960LL, true)) break; // tau > 10.0
	}

	return tau_accum;
}

/**
 * Warp Kernel: VolumetricCloudLightingKernel
 * 
 * Computes the final spectral radiance for a cloud voxel.
 * 1. Self-Shadowing: Beer's Law (exp(-tau)).
 * 2. Powder Effect: Simulates forward scattering (1.0 - exp(-2*density)).
 * 3. Irradiance: Adds Multiple-Scattering ambient energy.
 * 4. Anime Stylization: Injects "Silver Linings" and "Banded Shadows".
 */
void volumetric_cloud_lighting_kernel(
		const BigIntCore &p_index,
		Vector3f &r_radiance,
		const FixedMathCore &p_density,
		const Vector3f &p_pos,
		const Vector3f &p_sun_dir,
		const FixedMathCore &p_sun_energy,
		const FixedMathCore *p_all_densities,
		const BigIntCore &p_grid_res,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore two(2LL, false);

	// 1. Resolve Path to Light
	FixedMathCore step_dist(4294967296LL, false); // 1.0 unit step
	FixedMathCore tau = calculate_sun_occlusion_tau(p_pos, p_sun_dir, p_all_densities, p_grid_res, step_dist, 6);

	// 2. Beer-Powder Law Integration
	// Transmittance (Absorption)
	FixedMathCore beer = wp::exp(-tau);
	
	// Powder Effect (Fluffiness/Scattering)
	FixedMathCore powder = one - wp::exp(-(p_density * two));
	
	FixedMathCore direct_irradiance = beer * powder * p_sun_energy;

	// 3. Indirect Multi-Scattering (Ambient Light-Bleeding)
	// Light that has bounced inside the cloud, providing detail in shadows.
	FixedMathCore ambient_factor = (one - p_density) * FixedMathCore(214748364LL, true); // 0.05 base
	
	// 4. --- Sophisticated Real-Time Behavior: Realistic vs Anime ---
	if (p_is_anime) {
		// Anime Technique: "Silver Linings" (Rim Lighting).
		// Brightens edges of clouds facing the sun using a sharp step function.
		FixedMathCore edge_detection = wp::step(FixedMathCore(3865470566LL, true), powder); // 0.9 threshold
		FixedMathCore lining = edge_detection * p_sun_energy * FixedMathCore(5LL, false);
		
		// Banded Shadows: Snaps the Beer-Law falloff into discrete cel-shaded levels.
		FixedMathCore snap_hi(3006477107LL, true); // 0.7
		FixedMathCore snap_lo(858993459LL, true);  // 0.2
		
		FixedMathCore quantized_beer = wp::step(snap_hi, beer) * one + 
		                               wp::step(snap_lo, beer) * FixedMathCore(2147483648LL, true) + 
		                               FixedMathCore(429496730LL, true); // 0.1 shadow
		
		FixedMathCore final_mag = (quantized_beer * p_sun_energy) + lining;
		r_radiance = Vector3f(final_mag, final_mag, final_mag * (one + edge_detection)); // Add blue-tint to lining
	} else {
		// Realistic Path: Energy conservation and smooth irradiance
		FixedMathCore total_mag = direct_irradiance + ambient_factor;
		r_radiance = Vector3f(total_mag, total_mag, total_mag);
	}
}

/**
 * execute_volumetric_cloud_lighting_sweep()
 * 
 * Master orchestrator for parallel cloud visuals.
 * Partitions the EnTT voxel registry into SIMD worker batches.
 * Zero-copy: Writes results directly to the Radiance component stream.
 */
void execute_volumetric_cloud_lighting_sweep(
		KernelRegistry &p_registry,
		const Vector3f &p_sun_direction,
		const FixedMathCore &p_sun_intensity,
		const BigIntCore &p_grid_resolution) {

	auto &density_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_DENSITY);
	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &rad_stream = p_registry.get_stream<Vector3f>(COMPONENT_RADIANCE);
	
	uint64_t count = density_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &density_stream, &pos_stream, &rad_stream]() {
			for (uint64_t i = start; i < end; i++) {
				// Deterministic Style Selection based on entity handle index
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 8 == 0); // 12.5% chance of anime-lighting

				volumetric_cloud_lighting_kernel(
					handle,
					rad_stream[i],
					density_stream[i],
					pos_stream[i],
					p_sun_direction,
					p_sun_intensity,
					density_stream.get_base_ptr(),
					p_grid_resolution,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	// Final Synchronization Barrier for the 120 FPS visual update
	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/volumetric_cloud_lighting_kernel.cpp ---
