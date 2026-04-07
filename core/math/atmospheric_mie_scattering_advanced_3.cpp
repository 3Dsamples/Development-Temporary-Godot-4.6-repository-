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
 * Deterministic intersection logic for planar ring systems (e.g., planetary disks).
 * Rings are modeled as a 2D disc with inner and outer radii.
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

	// 1. Ray-Plane Intersection: t = -(dot(N, sample_pos)) / dot(N, light_dir)
	// The ring plane is centered at (0,0,0) in local planet space.
	FixedMathCore denom = p_ring_normal.dot(p_light_dir);
	
	// If light is parallel to the ring plane, it cannot be occluded.
	if (wp::abs(denom) < FixedMathCore(42949LL, true)) {
		return one; 
	}

	FixedMathCore t = (-p_ring_normal.dot(p_sample_pos)) / denom;
	
	// If t is negative, the ring is behind the light ray relative to the sample point.
	if (t < zero) {
		return one; 
	}

	// 2. Resolve intersection point and distance from center
	Vector3f intersection_point = p_sample_pos + (p_light_dir * t);
	FixedMathCore dist_sq = intersection_point.length_squared();

	// If the intersection occurs between the inner and outer boundaries, it is shadowed.
	if (dist_sq >= p_ring_inner_r2 && dist_sq <= p_ring_outer_r2) {
		return zero;
	}

	return one;
}

/**
 * Warp Kernel: AdvancedMieMultiLightKernel
 * 
 * Aggregates Mie scattering from all active light sources in the EnTT registry.
 * 1. Resolves shadowing from Planet bulk, Rings, and local occluders.
 * 2. Applies Henyey-Greenstein phase for each light independently.
 * 3. Injects sophisticated "Anime Bloom" quantization for stylized light interaction.
 * 4. strictly deterministic for 120 FPS cross-platform synchronization.
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
	Vector3f total_spectral_energy;

	// Loop through all light sources (Stars, Pulsars, Local Lights)
	for (uint32_t l = 0; l < p_lights.count; l++) {
		Vector3f light_dir;
		FixedMathCore distance_attenuation = one;

		if (p_lights.type[l] == 0) { // Directional (Star)
			light_dir = p_lights.direction[l];
		} else { // Point / Spot
			Vector3f rel_light_vec = p_lights.position[l] - p_sample_pos;
			FixedMathCore d2 = rel_light_vec.length_squared();
			light_dir = rel_light_vec.normalized();
			// Deterministic 1/r^2 falloff
			distance_attenuation = one / (d2 + one);
		}

		// 1. Shadowing: Check Planet Body (Primary Occluder)
		if (wp::check_occlusion_sphere(p_sample_pos, light_dir, Vector3f_ZERO, p_params.planet_radius)) {
			continue;
		}
		
		// 2. Shadowing: Check Planetary Rings (Secondary Occluder)
		FixedMathCore ring_shadow_factor = check_planetary_ring_shadow_kernel(
			p_sample_pos, 
			light_dir, 
			p_params.ring_inner_radius_sq, 
			p_params.ring_outer_radius_sq, 
			p_params.ring_normal
		);

		if (ring_shadow_factor.get_raw() == 0) {
			continue;
		}

		// 3. Resolve Local Particle Density
		FixedMathCore altitude = wp::max(zero, p_sample_pos.length() - p_params.planet_radius);
		FixedMathCore density = AtmosphericScattering::compute_density(altitude, p_params.mie_scale_height);

		// 4. Compute Phase Function (Henyey-Greenstein)
		FixedMathCore cos_theta = p_view_dir.dot(light_dir);
		FixedMathCore phase = wp::henyey_greenstein_phase(cos_theta, p_params.mie_g);

		// 5. --- Sophisticated Real-Time Behavior: Anime vs Realistic ---
		FixedMathCore light_intensity = p_lights.energy[l] * distance_attenuation;
		
		if (p_is_anime) {
			// Anime Technique: "Mie Glow Tiers". 
			// Snaps the phase intensity to create sharp, dramatic halos.
			FixedMathCore tier_hi(3435973836LL, true); // 0.8
			FixedMathCore tier_lo(858993459LL, true);  // 0.2
			
			if (phase > tier_hi) {
				phase = FixedMathCore(10LL, false); // Saturated sun core
			} else if (phase > tier_lo) {
				phase = one; // Standard secondary ring
			} else {
				phase = zero; // Sharp cel-shaded cutoff
			}
			
			// Force light intensity into binary steps
			light_intensity = wp::step(FixedMathCore(2147483648LL, true), light_intensity) * p_lights.energy[l];
		}

		// 6. Accumulate Spectral Radiance
		// I = Light_E * Phase * Density * Mie_Coeff
		FixedMathCore mag = light_intensity * phase * density * p_params.mie_coefficient;
		total_spectral_energy += p_lights.color[l] * (mag * ring_shadow_factor);
	}

	r_mie_accum = total_spectral_energy;
}

/**
 * execute_advanced_mie_sweep()
 * 
 * Master orchestrator for the parallel 120 FPS multi-light atmospheric resolve.
 * Partitions the EnTT SoA component streams across worker threads.
 * Zero-copy: Operates directly on the aligned memory addresses of the registry.
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
				// Style flag: derived from deterministic entity handle
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 16 == 0);

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

	// Wait for the 120 FPS synchronization barrier
	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_mie_scattering_advanced.cpp ---
