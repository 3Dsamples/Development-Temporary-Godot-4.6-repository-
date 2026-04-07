--- START OF FILE core/math/atmospheric_mie_scattering_advanced.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * check_planetary_ring_shadow_kernel()
 * 
 * Deterministic intersection logic for planetary rings (e.g. Saturn-scale).
 * Rings are modeled as an infinite plane with an inner and outer circular bounds.
 * Strictly uses FixedMathCore to ensure identical shadowing on all clients.
 * 
 * Returns 0 if shadowed, 1 if illuminated.
 */
static _FORCE_INLINE_ FixedMathCore check_planetary_ring_shadow_kernel(
		const Vector3f &p_sample_pos,
		const Vector3f &p_light_dir,
		const FixedMathCore &p_ring_inner_r2,
		const FixedMathCore &p_ring_outer_r2,
		const Vector3f &p_ring_normal) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Ray-Plane Intersection: t = (dot(N, (PointOnPlane - RayOrigin))) / dot(N, RayDir)
	// PointOnPlane is the planet center (Vector3f_ZERO)
	FixedMathCore denom = p_ring_normal.dot(p_light_dir);
	
	// If light is parallel to the ring plane, it is not occluded by the rings.
	if (wp::abs(denom) < FixedMathCore(42949LL, true)) return one; 

	FixedMathCore t = -p_ring_normal.dot(p_sample_pos) / denom;
	
	// If t is negative, the ring is behind the light ray relative to the sample.
	if (t < zero) return one; 

	// 2. Proximity check in the plane of the ring
	Vector3f intersection_point = p_sample_pos + (p_light_dir * t);
	FixedMathCore dist_sq = intersection_point.length_squared();

	// If the intersection is within the ring gap, return shadowed (zero)
	if (dist_sq >= p_ring_inner_r2 && dist_sq <= p_ring_outer_r2) {
		return zero;
	}

	return one;
}

/**
 * Warp Kernel: AdvancedMieMultiLightKernel
 * 
 * Aggregates Mie scattering from all active light sources in the EnTT registry.
 * 1. Resolves shadowing from Planet bulk, Rings, and local Occluders.
 * 2. Applies Henyey-Greenstein phase for each light independently.
 * 3. Injects sophisticated "Anime Bloom" quantization for stylized light interaction.
 * 4. Handles high-speed spaceship "Glow Compression" based on observer velocity.
 */
void advanced_mie_multi_light_kernel(
		const BigIntCore &p_index,
		Vector3f &r_mie_accum,
		const Vector3f &p_sample_pos,
		const Vector3f &p_view_dir,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_lights,
		bool p_is_anime) {

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	Vector3f total_energy;

	// Loop through all light sources in the current galactic sector
	for (uint32_t l = 0; l < p_lights.count; l++) {
		Vector3f light_dir = (p_lights.type[l] == 0) ? 
		                     p_lights.direction[l] : 
		                     (p_lights.position[l] - p_sample_pos).normalized();

		// 1. Shadowing: Check Planet bulk (Primary Occluder)
		if (wp::check_occlusion_sphere(p_sample_pos, light_dir, Vector3f_ZERO, p_params.planet_radius)) continue;
		
		// 2. Shadowing: Check Ring systems (Secondary Occluder)
		FixedMathCore ring_shadow = check_planetary_ring_shadow_kernel(
			p_sample_pos, 
			light_dir, 
			p_params.ring_inner_radius_sq, 
			p_params.ring_outer_radius_sq, 
			p_params.ring_normal
		);

		if (ring_shadow.get_raw() == 0) continue;

		// 3. Phase Function Resolve (Henyey-Greenstein)
		FixedMathCore cos_theta = p_view_dir.dot(light_dir);
		FixedMathCore phase = wp::henyey_greenstein_phase(cos_theta, p_params.mie_g);

		// 4. Sophisticated Behavior: Dynamic Velocity-Based Flare
		// For high-speed ships, the forward glare is compressed and intensified.
		FixedMathCore ship_speed = p_params.ship_velocity.length();
		FixedMathCore relativistic_boost = one + (ship_speed / PHYSICS_C) * phase;

		// --- Anime Stylization: Banding and Color-Snap ---
		if (p_is_anime) {
			// Technique: "Mie Glow Tiers".
			// Quantize haze intensity for cel-shaded horizons.
			FixedMathCore threshold_hi(3435973836LL, true); // 0.8
			FixedMathCore threshold_lo(858993459LL, true);  // 0.2
			
			if (phase > threshold_hi) phase = FixedMathCore(5LL, false); 
			else if (phase > threshold_lo) phase = one;
			else phase = zero;
			
			relativistic_boost *= FixedMathCore(2LL, false); // Exaggerated speed flares
		}

		// 5. Accumulate Spectral Energy
		FixedMathCore altitude = wp::max(zero, p_sample_pos.length() - p_params.planet_radius);
		FixedMathCore density = AtmosphericScattering::compute_density(altitude, p_params.mie_scale_height);

		FixedMathCore mag = p_lights.energy[l] * phase * density * relativistic_boost * p_params.mie_coefficient;
		total_energy += p_lights.color[l] * (mag * ring_shadow);
	}

	r_mie_accum = total_energy;
}

/**
 * execute_advanced_mie_sweep()
 * 
 * Orchestrates the parallel 120 FPS multi-light atmospheric resolve.
 * Optimized for high-fidelity planetary rendering and robotic sensor simulation.
 * Zero-copy: Operates on EnTT component streams for light energy and positions.
 */
void execute_advanced_mie_sweep(
		KernelRegistry &p_registry,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_global_lights) {

	auto &radiance_stream = p_registry.get_stream<Vector3f>(COMPONENT_RADIANCE);
	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &ray_stream = p_registry.get_stream<Vector3f>(COMPONENT_RAY_DIR);
	
	uint64_t count = radiance_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &radiance_stream, &pos_stream, &ray_stream, &p_params, &p_global_lights]() {
			for (uint64_t i = start; i < end; i++) {
				// Style derived from entity handle hash (BigIntCore)
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 14 == 0);

				advanced_mie_multi_light_kernel(
					handle,
					radiance_stream[i],
					pos_stream[i],
					ray_stream[i],
					p_params,
					p_global_lights,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_mie_scattering_advanced.cpp ---
