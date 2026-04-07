--- START OF FILE core/math/atmospheric_mie_scattering_advanced.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * check_planetary_ring_shadow()
 * 
 * Deterministic intersection logic for planetary rings (e.g. Saturn).
 * Rings are modeled as a plane with an inner and outer radius.
 * Strictly uses FixedMathCore for bit-perfect shadowing in atmospheric volumes.
 */
static _FORCE_INLINE_ FixedMathCore check_planetary_ring_shadow(
		const Vector3f &p_sample_pos,
		const Vector3f &p_light_dir,
		const FixedMathCore &p_ring_inner_r2,
		const FixedMathCore &p_ring_outer_r2,
		const Vector3f &p_ring_normal) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Ray-Plane Intersection
	// t = (d - n.o) / (n.d)
	FixedMathCore denom = p_ring_normal.dot(p_light_dir);
	if (wp::abs(denom) < FixedMathCore(42949LL, true)) return one; // Parallel

	FixedMathCore t = -p_ring_normal.dot(p_sample_pos) / denom;
	if (t < zero) return one; // Intersection is behind sample

	// 2. Proximity Check
	Vector3f intersection_point = p_sample_pos + (p_light_dir * t);
	FixedMathCore dist_sq = intersection_point.length_squared();

	if (dist_sq >= p_ring_inner_r2 && dist_sq <= p_ring_outer_r2) {
		// Sample is in the ring's shadow
		return zero;
	}

	return one;
}

/**
 * Warp Kernel: AdvancedMieMultiLightKernel
 * 
 * Aggregates Mie scattering from all active light sources in the EnTT registry.
 * 1. Resolves shadowing from Planet, Rings, and local Occluders.
 * 2. Applies Henyey-Greenstein phase for each light.
 * 3. Injects "Anime Bloom" quantization for stylized light interaction.
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

	for (uint32_t l = 0; l < p_lights.count; l++) {
		Vector3f light_dir = (p_lights.type[l] == LIGHT_TYPE_DIRECTIONAL) ? 
		                     p_lights.direction[l] : 
		                     (p_lights.position[l] - p_sample_pos).normalized();

		// 1. Shadowing: Check Planet bulk and Ring systems
		if (wp::check_occlusion_sphere(p_sample_pos, light_dir, p_params.planet_radius)) continue;
		
		FixedMathCore ring_shadow = check_planetary_ring_shadow(
			p_sample_pos, 
			light_dir, 
			p_params.ring_inner_radius_sq, 
			p_params.ring_outer_radius_sq, 
			p_params.ring_normal
		);

		if (ring_shadow.get_raw() == 0) continue;

		// 2. Light Interaction
		FixedMathCore cos_theta = p_view_dir.dot(light_dir);
		FixedMathCore phase = AtmosphericScattering::phase_mie(cos_theta, p_params.mie_g);

		// 3. Sophisticated Behavior: Dynamic High-Speed Flare
		// Adjust Mie intensity based on observer velocity (Glow Compression)
		FixedMathCore ship_speed_factor = p_params.observer_velocity.length() / FixedMathCore(50000LL, false);
		FixedMathCore intensity_boost = one + (ship_speed_factor * phase);

		// --- Anime Stylization: Step Ramps ---
		if (p_is_anime) {
			// Quantize haze intensity for cel-shaded horizons
			intensity_boost = wp::step(FixedMathCore(2147483648LL, true), intensity_boost) * one + 
			                  wp::step(FixedMathCore(858993459LL, true), intensity_boost) * FixedMathCore(2147483648LL, true);
		}

		FixedMathCore altitude = wp::max(zero, p_sample_pos.length() - p_params.planet_radius);
		FixedMathCore density = AtmosphericScattering::compute_density(altitude, p_params.mie_scale_height);

		FixedMathCore mag = p_lights.energy[l] * phase * density * intensity_boost * p_params.mie_coefficient;
		r_mie_accum += p_lights.color[l] * mag;
	}
}

/**
 * execute_advanced_mie_sweep()
 * 
 * Performs parallel multi-light atmospheric resolve.
 * Optimized for 120 FPS by partitioning rays across worker threads.
 */
void execute_advanced_mie_sweep(
		const BigIntCore &p_ray_count,
		const Vector3f *p_positions,
		const Vector3f *p_ray_dirs,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_global_lights,
		Vector3f *r_radiance_buffer) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_ray_count.to_string()));
	uint32_t worker_threads = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total / worker_threads;

	for (uint32_t w = 0; w < worker_threads; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == worker_threads - 1) ? total : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &p_params, &p_global_lights]() {
			for (uint64_t i = start; i < end; i++) {
				// Style derived from entity handle hash
				bool is_anime = (i % 2 == 0);

				advanced_mie_multi_light_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_radiance_buffer[i],
					p_positions[i],
					p_ray_dirs[i],
					p_params,
					p_global_lights,
					is_anime
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_mie_scattering_advanced.cpp ---
