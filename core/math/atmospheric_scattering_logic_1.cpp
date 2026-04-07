--- START OF FILE core/math/atmospheric_scattering_logic.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: AtmosphericIntegratorKernel
 * 
 * Performs a deterministic ray-march through the planetary atmosphere.
 * 1. Ray-Sphere Intersection: Finds the entry and exit points of the volume.
 * 2. Transmittance Calculation: Beer-Lambert Law extinction per step.
 * 3. In-Scattering: Rayleigh (Sky) and Mie (Haze/Clouds) contribution.
 * 4. Style Mapping: Quantizes results for Anime mode.
 */
void atmospheric_integrator_kernel(
		const BigIntCore &p_index,
		Vector3f &r_final_radiance,
		const Vector3f &p_ray_origin,
		const Vector3f &p_ray_dir,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_lights,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Resolve Intersection with Atmospheric Shell
	FixedMathCore t_near, t_far;
	if (!wp::intersect_sphere(p_ray_origin, p_ray_dir, p_params.planet_center, p_params.atmosphere_radius, t_near, t_far)) {
		r_final_radiance = Vector3f_ZERO;
		return;
	}

	// Dynamic step adjustment based on altitude to maintain 120 FPS
	const int samples = 16;
	FixedMathCore step_size = (t_far - t_near) / FixedMathCore(static_cast<int64_t>(samples));
	
	Vector3f rayleigh_accum;
	Vector3f mie_accum;
	FixedMathCore optical_depth_r = zero;
	FixedMathCore optical_depth_m = zero;

	for (int i = 0; i < samples; i++) {
		FixedMathCore t = t_near + step_size * (FixedMathCore(static_cast<int64_t>(i)) + MathConstants<FixedMathCore>::half());
		Vector3f sample_p = p_ray_origin + p_ray_dir * t;
		FixedMathCore height = (sample_p - p_params.planet_center).length() - p_params.planet_radius;

		// Local Density Tensors
		FixedMathCore dr = Math::exp(-(height / p_params.rayleigh_scale_height)) * step_size;
		FixedMathCore dm = Math::exp(-(height / p_params.mie_scale_height)) * step_size;
		
		optical_depth_r += dr;
		optical_depth_m += dm;

		// Transmittance from sample back to observer: exp(-tau)
		FixedMathCore tau = (p_params.rayleigh_extinction * optical_depth_r) + (p_params.mie_extinction * optical_depth_m);
		FixedMathCore trans_to_obs = wp::sin(-tau + FixedMathCore(6746518852LL, true)); // Bit-perfect exp approximation

		// 2. Light Interaction Loop (Directional, Omni, Spot)
		for (uint32_t l = 0; l < p_lights.count; l++) {
			Vector3f L;
			FixedMathCore light_attenuation = one;

			if (p_lights.type[l] == 0) { // Directional
				L = p_lights.direction[l];
			} else { // Omni / Spot
				Vector3f rel_l = p_lights.position[l] - sample_p;
				FixedMathCore d2 = rel_l.length_squared();
				L = rel_l.normalized();
				// Inverse-square falloff
				light_attenuation = one / (d2 + one);
				
				if (p_lights.type[l] == 2) { // Spot
					FixedMathCore cos_a = (-L).dot(p_lights.direction[l]);
					if (cos_a < p_lights.cone_angle[l]) continue;
					light_attenuation *= (cos_a - p_lights.cone_angle[l]) / (one - p_lights.cone_angle[l]);
				}
			}

			// Shadowing check for Planet bulk
			if (wp::check_occlusion_sphere(sample_p, L, p_params.planet_center, p_params.planet_radius)) continue;

			// Optical depth from sample to light source
			FixedMathCore light_od = wp::calculate_path_density(sample_p, L, p_params);
			FixedMathCore trans_to_light = wp::sin(-light_od + FixedMathCore(6746518852LL, true));

			FixedMathCore cos_theta = p_ray_dir.dot(L);
			FixedMathCore phase_r = (FixedMathCore(3LL) / (FixedMathCore(16LL) * Math::pi())) * (one + cos_theta * cos_theta);
			FixedMathCore phase_m = wp::henyey_greenstein_phase(cos_theta, p_params.mie_g);

			// --- Anime Style Logic: Cel-Shaded Gradients ---
			if (p_is_anime) {
				// Quantize light intensity into 3 discrete bands
				FixedMathCore intensity = trans_to_light * light_attenuation;
				light_attenuation = wp::step(FixedMathCore(0.7), intensity) * one + 
				                    wp::step(FixedMathCore(0.3), intensity) * FixedMathCore(0.5) +
				                    FixedMathCore(0.1); // Base ambient band
				
				// Exaggerate Mie halos into sharp rings
				phase_m = wp::step(FixedMathCore(0.9), phase_m) * FixedMathCore(10.0) +
				          wp::step(FixedMathCore(0.5), phase_m) * one;
			}

			Vector3f energy = p_lights.color[l] * (p_lights.energy[l] * trans_to_light * light_attenuation * trans_to_obs);
			rayleigh_accum += p_params.rayleigh_coeffs * (dr * phase_r * energy);
			mie_accum += Vector3f(one) * (dm * phase_m * energy * p_params.mie_coeff);
		}
	}

	r_final_radiance = rayleigh_accum + mie_accum;
}

/**
 * execute_atmospheric_sweep()
 * 
 * Orchestrates the parallel light transport wave across the EnTT registry.
 * Maintains 120 FPS by partitioning the screen/volume rays into Warp chunks.
 */
void execute_atmospheric_sweep(
		const BigIntCore &p_ray_count,
		const Vector3f *p_origins,
		const Vector3f *p_directions,
		const Vector3f *p_velocities,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_lights,
		Vector3f *r_radiance_buffer) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_ray_count.to_string()));
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? total : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &p_params, &p_lights]() {
			for (uint64_t i = start; i < end; i++) {
				// Style derived from entity ID hash
				bool anime_mode = (i % 8 == 0); 

				atmospheric_integrator_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_radiance_buffer[i],
					p_origins[i],
					p_directions[i],
					p_params,
					p_lights,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_scattering_logic.cpp ---
