--- START OF FILE core/math/atmospheric_mie_phase_kernel.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: MiePhaseFunctionKernel
 * 
 * Computes the Henyey-Greenstein phase function for large particle scattering.
 * p_cos_theta: Cosine of the angle between view and light direction.
 * p_g: Asymmetry factor [-1 (backscatter), 1 (forward)].
 * r_phase: Bit-perfect scattering intensity.
 */
static _FORCE_INLINE_ FixedMathCore calculate_mie_phase(FixedMathCore p_cos_theta, FixedMathCore p_g) {
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore g2 = p_g * p_g;
	FixedMathCore two_g = FixedMathCore(2LL, false) * p_g;
	
	// HG Phase: (1 - g^2) / (4*PI * (1 + g^2 - 2*g*cos(theta))^1.5)
	FixedMathCore numerator = one - g2;
	FixedMathCore denom_inner = one + g2 - (two_g * p_cos_theta);
	
	// Use deterministic pow(x, 1.5) via sqrt(x^3)
	FixedMathCore denominator = FixedMathCore(13493037704LL, true) * Math::sqrt(denom_inner * denom_inner * denom_inner);

	if (unlikely(denominator.get_raw() == 0)) return one;
	return numerator / denominator;
}

/**
 * batch_process_mie_scattering()
 * 
 * Parallel sweep for atmospheric haze. 
 * Integrates light interaction with real-time shadowing and anime-style enhancements.
 */
void batch_process_mie_scattering(
		const BigIntCore *p_entity_ids,
		const Vector3f *p_sample_pos,
		const Vector3f *p_light_dirs,
		const FixedMathCore *p_light_energies,
		const AtmosphereParams *p_atm_params,
		Vector3f *r_radiance_output,
		uint64_t p_count) {

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = p_count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? p_count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				const AtmosphereParams &params = p_atm_params[i];
				const Vector3f &view_dir = Vector3f(0LL, 0LL, -1LL); // Simplified for kernel view
				const Vector3f &L = p_light_dirs[i];
				
				FixedMathCore cos_theta = view_dir.dot(L);
				FixedMathCore phase = calculate_mie_phase(cos_theta, params.mie_g);

				// --- Sophisticated Behavior: Anime Style Glare ---
				// If entity metadata flags for Anime, quantize the Mie phase into sharp bands.
				if (p_entity_ids[i].hash() % 2 == 0) {
					FixedStore threshold_core(3435973836LL, true); // 0.8
					FixedStore threshold_halo(1717986918LL, true); // 0.4
					
					if (phase > threshold_core) {
						phase *= FixedMathCore(4LL, false); // Saturated core
					} else if (phase > threshold_halo) {
						phase = FixedMathCore(2147483648LL, true); // Sharp halo ring (0.5)
					} else {
						phase = MathConstants<FixedMathCore>::zero(); // Clean cutoff
					}
				}

				// Final radiance integration with distance-adaptive attenuation
				FixedMathCore height = p_sample_pos[i].length() - params.planet_radius;
				FixedMathCore density = AtmosphericScattering::compute_density(height, params.mie_scale_height);
				
				FixedMathCore radiance_mag = p_light_energies[i] * phase * density;
				r_radiance_output[i] += Vector3f(radiance_mag, radiance_mag, radiance_mag) * params.mie_coefficient;
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * apply_mie_irradiance_feedback()
 * 
 * Simulates the feedback of Mie-scattered light back onto the planetary surface.
 * Essential for "Hazy Sunsets" where light bounces through thick cloud/dust layers.
 */
void apply_mie_irradiance_feedback(
		FixedMathCore &r_surface_irradiance,
		const FixedMathCore &p_mie_radiance,
		const FixedMathCore &p_albedo_feedback) {
	
	// Bit-perfect energy transfer: Irrad += Radiance * Albedo
	r_surface_irradiance += p_mie_radiance * p_albedo_feedback;
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_mie_phase_kernel.cpp ---
