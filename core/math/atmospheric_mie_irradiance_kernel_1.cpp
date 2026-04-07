--- START OF FILE core/math/atmospheric_mie_irradiance_kernel.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_mie_ms_factor()
 * 
 * Computes the Multiple-Scattering (MS) energy compensation factor.
 * Approximates the infinite geometric series of light bounces in haze.
 * MS = 1.0 / (1.0 - Albedo * (1.0 - exp(-optical_depth)))
 */
static _FORCE_INLINE_ FixedMathCore calculate_mie_ms_factor(
		const FixedMathCore &p_albedo, 
		const FixedMathCore &p_tau) {
	
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	// Deterministic exponential approximation for extinction
	FixedMathCore transmittance = wp::exp(-p_tau);
	FixedMathCore bounce_prob = p_albedo * (one - transmittance);
	
	// Clamp to prevent singularity in extremely dense nebulae (Max 0.98 albedo)
	FixedMathCore max_albedo(4209067950LL, true); 
	FixedMathCore denom = one - wp::min(bounce_prob, max_albedo);
	
	return one / denom;
}

/**
 * Warp Kernel: MieIrradianceKernel
 * 
 * Calculates the indirect planetary glow (Irradiance) received at a point.
 * 1. Integrates in-scattered Mie light from the entire sky dome.
 * 2. Applies the Multiple-Scattering boost for "Thick" atmospheres.
 * 3. Quantizes radiance into stylized "Anime" bands if the Style-Tensor is active.
 */
void mie_irradiance_kernel(
		const BigIntCore &p_index,
		Vector3f &r_indirect_glow,
		const Vector3f &p_sample_pos,
		const Vector3f &p_normal,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_lights,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	Vector3f total_irradiance;

	// 1. Resolve Local Altitude Density
	FixedMathCore height = wp::max(zero, p_sample_pos.length() - p_params.planet_radius);
	FixedMathCore density = AtmosphericScattering::compute_density(height, p_params.mie_scale_height);

	// 2. Iterate through all star-sources in the EnTT registry
	for (uint32_t l = 0; l < p_lights.count; l++) {
		Vector3f L = (p_lights.type[l] == 0) ? p_lights.direction[l] : (p_lights.position[l] - p_sample_pos).normalized();
		
		// Deterministic Shadowing: Check if sample is in planetary shadow
		if (wp::check_occlusion_sphere(p_sample_pos, L, p_params.planet_center, p_params.planet_radius)) continue;

		// Compute Direct-to-Indirect Energy Transfer
		// Phase function for irradiance uses the normal as the "view" direction
		FixedMathCore cos_theta = p_normal.dot(L);
		FixedMathCore phase = wp::henyey_greenstein_phase(cos_theta, p_params.mie_g);
		
		// 3. Multi-Scattering Scaling
		// optical_depth = density * scale_height
		FixedMathCore tau = density * p_params.mie_scale_height;
		FixedMathCore ms_factor = calculate_mie_ms_factor(p_params.mie_albedo, tau);

		FixedMathCore radiance_mag = p_lights.energy[l] * phase * density * ms_factor * p_params.mie_coefficient;
		total_irradiance += p_lights.color[l] * radiance_mag;
	}

	// 4. --- Sophisticated Behavior: Realistic vs Anime ---
	if (p_is_anime) {
		// Anime Technique: "Glow Bleeding". 
		// Radiance is snapped to discrete tiers to create the look of hand-drawn cell layers.
		FixedMathCore intensity = total_irradiance.get_luminance();
		
		FixedMathCore band_hi(3435973836LL, true); // 0.8
		FixedMathCore band_lo(858993459LL, true);  // 0.2

		if (intensity > band_hi) {
			r_indirect_glow = total_irradiance.normalized() * p_params.sun_intensity_ref;
		} else if (intensity > band_lo) {
			r_indirect_glow = total_irradiance * FixedMathCore(2147483648LL, true); // 0.5 mid-band
		} else {
			// Anime Deep Shadow Glow
			r_indirect_glow = p_params.mie_color * FixedMathCore(429496730LL, true); // 0.1 shadow
		}
		
		// Saturation boost
		r_indirect_glow *= FixedMathCore(5153960755LL, true); // 1.2x
	} else {
		// Physically Correct: Direct assignment from the integrated sum
		r_indirect_glow = total_irradiance;
	}
}

/**
 * execute_mie_irradiance_sweep()
 * 
 * Orchestrates the parallel 120 FPS resolve for atmospheric glow.
 * zero-copy: Operates directly on SoA buffers for normals, positions, and results.
 */
void execute_mie_irradiance_sweep(
		KernelRegistry &p_registry,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_lights) {

	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &norm_stream = p_registry.get_stream<Vector3f>(COMPONENT_NORMAL);
	auto &glow_stream = p_registry.get_stream<Vector3f>(COMPONENT_MIE_GLOW);

	uint64_t count = pos_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &pos_stream, &norm_stream, &glow_stream, &p_params, &p_lights]() {
			for (uint64_t i = start; i < end; i++) {
				// Deterministic style assignment based on entity handle hash
				BigIntCore entity_idx(static_cast<int64_t>(i));
				bool anime_mode = (entity_idx.hash() % 4 == 0); 

				mie_irradiance_kernel(
					entity_idx,
					glow_stream[i],
					pos_stream[i],
					norm_stream[i],
					p_params,
					p_lights,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * apply_mie_bloom_feedback()
 * 
 * sophisticated light-interaction: Hazy planetary horizons produce bloom halos 
 * in the visual spectral energy buffer.
 */
void apply_mie_bloom_feedback(
		Vector3f &r_radiance, 
		const Vector3f &p_mie_glow, 
		const FixedMathCore &p_bloom_k) {
	
	FixedMathCore lum = p_mie_glow.get_luminance();
	FixedMathCore threshold(3006477107LL, true); // 0.7
	
	if (lum > threshold) {
		r_radiance += p_mie_glow * p_bloom_k;
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_mie_irradiance_kernel.cpp ---
