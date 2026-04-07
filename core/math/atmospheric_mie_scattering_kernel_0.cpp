--- START OF FILE core/math/atmospheric_mie_scattering_kernel.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: MieScatteringIntegralKernel
 * 
 * Computes the Mie scattering contribution for a batch of atmospheric samples.
 * Optimized for high-albedo particles (Haze, Fog, Clouds).
 * 
 * p_samples: SoA stream of sample positions in local space.
 * p_light_dirs: Directions to the light source (e.g. Sun).
 * p_energy: Light intensity tensors.
 * r_out_mie: Output spectral radiance.
 */
void mie_scattering_integral_kernel(
		const BigIntCore &p_index,
		const Vector3f &p_sample_pos,
		const Vector3f &p_view_dir,
		const Vector3f &p_light_dir,
		const FixedMathCore &p_light_energy,
		const AtmosphericScattering::AtmosphereParams &p_params,
		const bool p_is_anime,
		Vector3f &r_out_mie) {

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();

	// 1. Calculate Phase Function (Henyey-Greenstein)
	// P(theta) = (1 - g^2) / (4pi * (1 + g^2 - 2g cos(theta))^1.5)
	FixedMathCore cos_theta = p_view_dir.dot(p_light_dir);
	FixedMathCore g = p_params.mie_g;
	FixedMathCore g2 = g * g;

	FixedMathCore numerator = one - g2;
	FixedMathCore denominator_base = one + g2 - (FixedMathCore(2LL, false) * g * cos_theta);
	
	// x^1.5 = sqrt(x^3). Calculated via deterministic bit-perfect FixedMath.
	FixedMathCore denominator = FixedMathCore(13493037704LL, true) * Math::sqrt(denominator_base * denominator_base * denominator_base); // 4*pi * base^1.5

	FixedMathCore phase = (denominator.get_raw() == 0) ? one : (numerator / denominator);

	// 2. Resolve Density at Altitude
	FixedMathCore altitude = p_sample_pos.length() - p_params.planet_radius;
	FixedMathCore density = AtmosphericScattering::compute_density(altitude, p_params.mie_scale_height);

	// 3. --- Stylization Behavioral Logic ---
	if (p_is_anime) {
		// Anime Technique: Light Halos. 
		// Quantize the Mie phase into sharp bands to create stylized sun circles.
		FixedMathCore outer_step = wp::step(FixedMathCore(858993459LL, true), phase); // 0.2
		FixedMathCore inner_step = wp::step(FixedMathCore(3435973836LL, true), phase); // 0.8
		
		phase = (outer_step * FixedMathCore(2LL, false)) + (inner_step * FixedMathCore(8LL, false));
		
		// Force color saturation for anime haze
		density = wp::clamp(density * FixedMathCore(2LL, false), zero, one);
	}

	// 4. Final Radiance Accumulation
	// S = L_light * Phase * Density * Mie_Coeff
	FixedMathCore magnitude = p_light_energy * phase * density * p_params.mie_coefficient;
	r_out_mie = Vector3f(magnitude, magnitude, magnitude);
}

/**
 * execute_mie_volume_sweep()
 * 
 * Orchestrates a parallel sweep over EnTT atmospheric component buffers.
 * Maintains 120 FPS by partitioning the atmospheric shell into SIMD-friendly chunks.
 */
void execute_mie_volume_sweep(
		const BigIntCore &p_total_samples,
		const Vector3f *p_positions,
		const Vector3f *p_view_dirs,
		const Vector3f *p_light_dirs,
		const FixedMathCore *p_energies,
		const AtmosphericScattering::AtmosphereParams &p_params,
		Vector3f *r_radiance_buffer) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_total_samples.to_string()));
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? total : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &p_params]() {
			for (uint64_t i = start; i < end; i++) {
				// Style flag derived from entity index for deterministic look-consistency
				bool anime_mode = (i % 2 == 0); 

				mie_scattering_integral_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					p_positions[i],
					p_view_dirs[i],
					p_light_dirs[i],
					p_energies[i],
					p_params,
					anime_mode,
					r_radiance_buffer[i]
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	// Synchronization barrier for zero-copy result availability
	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_mie_scattering_kernel.cpp ---
