--- START OF FILE core/math/atmospheric_scattering_mie_final.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_energy_conservation_tensor()
 * 
 * Ensures that the sum of single and multiple scattering does not exceed 
 * the incoming light energy. strictly uses FixedMathCore for bit-perfection.
 */
static _FORCE_INLINE_ FixedMathCore calculate_energy_conservation_tensor(
		const FixedMathCore &p_single,
		const FixedMathCore &p_multiple,
		const FixedMathCore &p_source_energy) {
	
	FixedMathCore total = p_single + p_multiple;
	if (total > p_source_energy && total.get_raw() > 0) {
		return p_source_energy / total;
	}
	return MathConstants<FixedMathCore>::one();
}

/**
 * Warp Kernel: MieRadianceCompositionKernel
 * 
 * Finalizes the Mie spectral energy for an atmospheric volume.
 * 1. Merges Single-Scattering and Multi-Scattering (Irradiance) tiers.
 * 2. Applies Thermodynamic Correction: haze density shifts with temperature.
 * 3. Energy Conservation: Normalizes output against the source star intensity.
 * 4. Anime Style: Quantizes the final spectral radiance into high-saturation bands.
 */
void mie_radiance_composition_kernel(
		const BigIntCore &p_index,
		Vector3f &r_final_mie_radiance,
		const Vector3f &p_single_scatter,
		const Vector3f &p_multi_scatter,
		const FixedMathCore &p_shadow_factor,
		const FixedMathCore &p_temperature,
		const FixedMathCore &p_source_energy,
		const AtmosphereParams &p_params,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Resolve Thermodynamic Density Shift
	// Sophisticated Behavior: Haze becomes more opaque as temperature rises (Evaporation/Steam).
	// T_ref = 373.15K (100C). Factor scales density by exp(T/T_ref).
	FixedMathCore t_ref(16026725580LL, true); 
	FixedMathCore thermo_multiplier = wp::exp(p_temperature / t_ref);
	
	Vector3f total_mie = (p_single_scatter + p_multi_scatter) * thermo_multiplier;

	// 2. Apply Deterministic Shadowing
	total_mie *= p_shadow_factor;

	// 3. Resolve Energy Conservation
	FixedMathCore lum = total_mie.get_luminance();
	FixedMathCore conservation_k = calculate_energy_conservation_tensor(lum, zero, p_source_energy);
	total_mie *= conservation_k;

	// 4. --- Sophisticated Real-Time Behavior: Anime vs Realistic ---
	if (p_is_anime) {
		// Anime Technique: "Spectral Banding".
		// Instead of a smooth haze gradient, colors are snapped into 3 sharp tiers.
		FixedMathCore intensity = total_mie.get_luminance();
		FixedMathCore band_hi(3435973836LL, true); // 0.8
		FixedMathCore band_lo(1073741824LL, true); // 0.25

		FixedMathCore multiplier;
		if (intensity > band_hi) {
			multiplier = FixedMathCore(5LL, false); // Saturated Highlight
		} else if (intensity > band_lo) {
			multiplier = one; // Normal Haze
		} else {
			multiplier = FixedMathCore(214748364LL, true); // 0.05 Deep Shadow
		}
		
		// Force saturation toward the Mie color constant
		r_final_mie_radiance = p_params.mie_color * (intensity * multiplier);
	} else {
		// Realistic Path: Direct assignment of conserved energy
		r_final_mie_radiance = total_mie;
	}
}

/**
 * execute_mie_final_composition_wave()
 * 
 * Orchestrates the absolute final parallel resolve for atmospheric haze at 120 FPS.
 * Partitions EnTT component streams for single/multi scattering and thermal state.
 */
void execute_mie_final_composition_wave(
		KernelRegistry &p_registry,
		const AtmosphereParams &p_params,
		const FixedMathCore &p_sun_energy) {

	auto &single_stream = p_registry.get_stream<Vector3f>(COMPONENT_MIE_SINGLE);
	auto &multi_stream = p_registry.get_stream<Vector3f>(COMPONENT_MIE_MULTI);
	auto &shadow_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_SHADOW_FACTOR);
	auto &temp_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_TEMPERATURE);
	auto &final_stream = p_registry.get_stream<Vector3f>(COMPONENT_MIE_FINAL_RADIANCE);

	uint64_t count = final_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &single_stream, &multi_stream, &shadow_stream, &temp_stream, &final_stream, &p_params]() {
			for (uint64_t i = start; i < end; i++) {
				// Style derived from Entity ID handle hash
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 12 == 0);

				mie_radiance_composition_kernel(
					handle,
					final_stream[i],
					single_stream[i],
					multi_stream[i],
					shadow_stream[i],
					temp_stream[i],
					p_sun_energy,
					p_params,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	// Final Synchronization Barrier for the visual frame conclude
	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_scattering_mie_final.cpp ---
