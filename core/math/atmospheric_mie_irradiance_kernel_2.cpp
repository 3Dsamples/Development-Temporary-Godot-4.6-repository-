--- START OF FILE core/math/atmospheric_mie_irradiance_kernel.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_mie_multiple_scattering_tensor()
 * 
 * Computes the energy compensation factor for light that has bounced multiple times.
 * In a real atmosphere, Mie scattering is highly forward-peaked.
 * MS = 1.0 / (1.0 - MieAlbedo * (1.0 - exp(-optical_depth)))
 */
static _FORCE_INLINE_ FixedMathCore calculate_mie_multiple_scattering_tensor(
		const FixedMathCore &p_albedo, 
		const FixedMathCore &p_tau) {
	
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	// exp(-tau) approximation for Q32.32 bit-perfection
	FixedMathCore transmittance = wp::exp(-p_tau);
	
	// Energy preserved in the system after multiple bounces
	FixedMathCore bounce_retention = p_albedo * (one - transmittance);
	
	// Safety clamp at 0.99 to avoid division by zero in hyper-dense nebulae
	FixedMathCore safety_cap(4252017623LL, true); 
	FixedMathCore denom = one - wp::min(bounce_retention, safety_cap);
	
	return one / denom;
}

/**
 * Warp Kernel: MieIrradianceKernel
 * 
 * Computes the indirect light contribution (Irradiance) for a surface normal.
 * 1. Resolves local atmospheric density based on altitude.
 * 2. Aggregates multi-scattered light from all star-entities in the EnTT registry.
 * 3. Applies stylized "Anime" quantization for cel-shaded planetary glows.
 */
void mie_irradiance_kernel(
		const BigIntCore &p_index,
		Vector3f &r_indirect_radiance,
		const Vector3f &p_sample_pos,
		const Vector3f &p_normal,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_lights,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	Vector3f accumulated_energy;

	// 1. Resolve Local Voxel Density
	// height = |pos - planet_center| - planet_radius
	FixedMathCore height = wp::max(zero, (p_sample_pos - p_params.planet_center).length() - p_params.planet_radius);
	FixedMathCore density = wp::exp(-(height / p_params.mie_scale_height));

	// 2. Loop through all light sources (EnTT LightDataSoA)
	for (uint32_t l = 0; l < p_lights.count; l++) {
		Vector3f L;
		if (p_lights.type[l] == 0) { // Directional (Star)
			L = p_lights.direction[l];
		} else { // Omni / Spot
			L = (p_lights.position[l] - p_sample_pos).normalized();
		}

		// Geometric Shadowing (Planet Bulk)
		if (wp::check_occlusion_sphere(p_sample_pos, L, p_params.planet_center, p_params.planet_radius)) {
			continue;
		}

		// Hemispherical Cosine Weight (Diffuse term)
		FixedMathCore cos_theta = p_normal.dot(L);
		if (cos_theta <= zero) continue;

		// 3. Compute Scattering Strength
		// phase = Henyey-Greenstein(normal dot light)
		FixedMathCore phase = wp::henyey_greenstein_phase(cos_theta, p_params.mie_g);
		
		// Optical depth approximation for the MS factor
		FixedMathCore tau = density * p_params.mie_scale_height;
		FixedMathCore ms_factor = calculate_mie_multiple_scattering_tensor(p_params.mie_albedo, tau);

		// Energy Integration: I = Light * Phase * Density * MultipleScattering
		FixedMathCore intensity = p_lights.energy[l] * phase * density * ms_factor * p_params.mie_coefficient;
		
		accumulated_energy += p_lights.color[l] * intensity;
	}

	// 4. --- Sophisticated Style Resolution ---
	if (p_is_anime) {
		// Anime Technique: "Glow Quantization".
		// Instead of smooth atmospheric gradients, snap the irradiance to 
		// 3 discrete levels: Saturated, Mid, and Shadow.
		FixedMathCore total_lum = accumulated_energy.get_luminance();
		
		FixedMathCore band_0(3435973836LL, true); // 0.8
		FixedMathCore band_1(1288490188LL, true); // 0.3

		FixedMathCore multiplier;
		if (total_lum > band_0) {
			multiplier = FixedMathCore(12874588672LL, true); // 3.0 boost
		} else if (total_lum > band_1) {
			multiplier = one;
		} else {
			multiplier = FixedMathCore(429496730LL, true); // 0.1 base ambient
		}
		
		r_indirect_radiance = accumulated_energy.normalized() * multiplier;
	} else {
		// Realistic Path
		r_indirect_radiance = accumulated_energy;
	}
}

/**
 * execute_mie_irradiance_sweep()
 * 
 * Orchestrates the 120 FPS parallel resolve for indirect atmospheric light.
 * Processes the entire EnTT component pool for surface-aware entities.
 */
void execute_mie_irradiance_sweep(
		KernelRegistry &p_registry,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_lights) {

	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &norm_stream = p_registry.get_stream<Vector3f>(COMPONENT_NORMAL);
	auto &radiance_stream = p_registry.get_stream<Vector3f>(COMPONENT_MIE_IRRADIANCE);

	uint64_t entity_count = pos_stream.size();
	if (entity_count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = entity_count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? entity_count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &pos_stream, &norm_stream, &radiance_stream, &p_params, &p_lights]() {
			for (uint64_t i = start; i < end; i++) {
				// Deterministic Style derivation from Entity ID hash
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 16 == 0); 

				mie_irradiance_kernel(
					handle,
					radiance_stream[i],
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

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_mie_irradiance_kernel.cpp ---
