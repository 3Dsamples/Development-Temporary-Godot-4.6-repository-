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
	// exp(-tau) approximation for Q32.32
	FixedMathCore transmittance = wp::exp(-p_tau);
	FixedMathCore bounce_prob = p_albedo * (one - transmittance);
	
	// Clamp to prevent singularity in extremely dense nebulae
	FixedMathCore denom = one - wp::min(bounce_prob, FixedMathCore(4080218931LL, true)); // 0.95 cap
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
		const Vector3f &p_sun_dir,
		const FixedMathCore &p_sun_energy,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Resolve Local Altitude Density
	FixedMathCore height = wp::max(zero, p_sample_pos.length() - p_params.planet_radius);
	FixedMathCore density = AtmosphericScattering::compute_density(height, p_params.mie_scale_height);

	// 2. Compute Direct-to-Indirect Energy Transfer
	// In-scattering is proportional to Sun Energy * Phase * Density
	FixedMathCore cos_theta = p_normal.dot(p_sun_dir);
	FixedMathCore phase = wp::henyey_greenstein_phase(cos_theta, p_params.mie_g);
	
	FixedMathCore optical_depth = density * p_params.atmosphere_radius;
	FixedMathCore ms_factor = calculate_mie_ms_factor(p_params.mie_albedo, optical_depth);

	// 3. --- Sophisticated Behavior: Realistic vs Anime ---
	FixedMathCore final_radiance;
	if (p_is_anime) {
		// Anime Technique: "Glow Bleeding". 
		// Instead of realistic falloff, haze "leaks" across shadows in sharp steps.
		FixedMathCore intensity = (p_sun_energy * phase * density * ms_factor);
		
		// Band thresholds for stylized cel-shading
		FixedMathCore band_threshold(1288490188LL, true); // 0.3
		if (intensity > band_threshold) {
			final_radiance = p_sun_energy * FixedMathCore(2147483648LL, true); // 0.5 snap
		} else {
			final_radiance = intensity * FixedMathCore(429496730LL, true); // 0.1 shadow glow
		}
		
		// Saturation shift: Anime haze is often tinted by planetary albedo
		r_indirect_glow = p_params.mie_color * final_radiance * FixedMathCore(6442450944LL, true); // 1.5x saturation
	} else {
		// Physically Correct: Energy Conservation
		final_radiance = p_sun_energy * phase * density * ms_factor * p_params.mie_coefficient;
		r_indirect_glow = p_params.mie_color * final_radiance;
	}
}

/**
 * execute_mie_irradiance_sweep()
 * 
 * Orchestrates the parallel 120 FPS resolve for atmospheric glow.
 * Zero-copy: Operates on EnTT component streams for normals and positions.
 */
void execute_mie_irradiance_sweep(
		const BigIntCore &p_count,
		const Vector3f *p_positions,
		const Vector3f *p_normals,
		const AtmosphereParams *p_atm_data,
		const Vector3f &p_sun_direction,
		const FixedMathCore &p_sun_energy,
		Vector3f *r_glow_buffer) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_count.to_string()));
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? total : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &p_atm_data]() {
			for (uint64_t i = start; i < end; i++) {
				// Deterministic style selection: even indices are anime-styled
				bool anime_mode = (i % 2 == 0); 

				mie_irradiance_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_glow_buffer[i],
					p_positions[i],
					p_normals[i],
					p_atm_data[i],
					p_sun_direction,
					p_sun_energy,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	// Final Synchronization Barrier
	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * apply_atmospheric_bloom_feedback()
 * 
 * Injects irradiance results into the visual spectral bloom pipeline.
 * Ensures that hazy planetary horizons produce "Halo" artifacts in bit-perfect HDR.
 */
void apply_atmospheric_bloom_feedback(
		Vector3f &r_radiance, 
		const Vector3f &p_indirect_glow, 
		const FixedMathCore &p_bloom_intensity) {
	
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore threshold(3435973836LL, true); // 0.8
	
	FixedMathCore lum = p_indirect_glow.get_luminance();
	if (lum > threshold) {
		r_radiance += p_indirect_glow * p_bloom_intensity;
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_mie_irradiance_kernel.cpp ---
