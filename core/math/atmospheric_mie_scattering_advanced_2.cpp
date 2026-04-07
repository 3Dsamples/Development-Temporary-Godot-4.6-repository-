--- START OF FILE core/math/atmospheric_mie_scattering_advanced.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * check_planetary_ring_shadow_kernel()
 * 
 * Deterministic intersection logic for planetary rings (e.g., Saturn-scale).
 * Rings are modeled as an infinite plane with inner and outer circular bounds.
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
	// Plane passes through the planet center (0,0,0) in local space.
	FixedMathCore denom = p_ring_normal.dot(p_light_dir);
	
	// If light is parallel to the ring plane, it is not occluded by the rings.
	if (wp::abs(denom) < FixedMathCore(42949LL, true)) {
		return one; 
	}

	// Calculate distance t to the plane. Plane equation: dot(N, P) = 0.
	// t = -dot(N, sample_pos) / dot(N, light_dir)
	FixedMathCore t = (-p_ring_normal.dot(p_sample_pos)) / denom;
	
	// If t is negative, the ring plane is behind the light ray relative to the sample.
	if (t < zero) {
		return one; 
	}

	// 2. Proximity check in the plane of the ring
	Vector3f intersection_point = p_sample_pos + (p_light_dir * t);
	FixedMathCore dist_sq = intersection_point.length_squared();

	// If the intersection is within the ring bounds, return shadowed (zero)
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

	// Loop through all light sources in the current simulation wave
	for (uint32_t l = 0; l < p_lights.count; l++) {
		Vector3f light_dir;
		FixedMathCore distance_attenuation = one;

		if (p_lights.type[l] == 0) { // Directional (e.g., Star)
			light_dir = p_lights.direction[l];
		} else { // Point / Spot
			Vector3f rel_l = p_lights.position[l] - p_sample_pos;
			FixedMathCore d2 = rel_l.length_squared();
			light_dir = rel_l.normalized();
			// Deterministic falloff: 1 / (d^2 + 1)
			distance_attenuation = one / (d2 + one);
		}

		// 1. Shadowing Resolve: Check Planet bulk (Primary Occluder)
		if (wp::check_occlusion_sphere(p_sample_pos, light_dir, Vector3f_ZERO, p_params.planet_radius)) {
			continue;
		}
		
		// 2. Shadowing Resolve: Check Ring systems (Secondary Occluder)
		FixedMathCore ring_shadow = check_planetary_ring_shadow_kernel(
			p_sample_pos, 
			light_dir, 
			p_params.ring_inner_radius_sq, 
			p_params.ring_outer_radius_sq, 
			p_params.ring_normal
		);

		if (ring_shadow.get_raw() == 0) {
			continue;
		}

		// 3. Phase Function Resolve (Henyey-Greenstein)
		FixedMathCore cos_theta = p_view_dir.dot(light_dir);
		FixedMathCore phase = wp::henyey_greenstein_phase(cos_theta, p_params.mie_g);

		// 4. --- Sophisticated Behavior: Realistic vs Anime ---
		FixedMathCore light_intensity = p_lights.energy[l] * distance_attenuation;
		
		if (p_is_anime) {
			// Anime Technique: "Banded Light Tiers".
			// Quantize haze intensity for cel-shaded horizons.
			FixedMathCore tier_hi(3435973836LL, true); // 0.8
			FixedMathCore tier_lo(858993459LL, true);  // 0.2
			
			if (phase > tier_hi) {
				phase = FixedMathCore(5LL, false); // Strong inner glow
			} else if (phase > tier_lo) {
				phase = one; // Medium band
			} else {
				phase = zero; // Sharp cutoff
			}
			
			light_intensity = wp::step(FixedMathCore(2147483648LL, true), light_intensity) * p_lights.energy[l];
		}

		// 5. Density Integration
		FixedMathCore altitude = wp::max(zero, p_sample_pos.length() - p_params.planet_radius);
		FixedMathCore density = AtmosphericScattering::compute_density(altitude, p_params.mie_scale_height);

		FixedMathCore mag = light_intensity * phase * density * p_params.mie_coefficient;
		total_energy += p_lights.color[l] * mag;
	}

	r_mie_accum = total_energy;
}

/**
 * execute_advanced_mie_sweep()
 * 
 * Orchestrates the parallel 120 FPS multi-light atmospheric resolve.
 * Partitions the voxel stream across worker threads.
 */
void execute_advanced_mie_sweep(
		KernelRegistry &p_registry,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_global_lights) {

	auto &radiance_stream = p_registry.get_stream<Vector3f>(COMPONENT_MIE_RADIANCE);
	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_SAMPLE_POS);
	auto &view_stream = p_registry.get_stream<Vector3f>(COMPONENT_VIEW_DIR);
	
	uint64_t count = radiance_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &radiance_stream, &pos_stream, &view_stream, &p_params, &p_global_lights]() {
			for (uint64_t i = start; i < end; i++) {
				// Style flag: use entity hash for deterministic anime look
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 12 == 0);

				advanced_mie_multi_light_kernel(
					handle,
					radiance_stream[i],
					pos_stream[i],
					view_stream[i],
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
