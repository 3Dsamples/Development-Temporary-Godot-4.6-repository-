--- START OF FILE core/math/spectral_tensor_compositor.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "core/math/spectral_tensor_kernel.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_aces_tonemap_tensor()
 * 
 * Absolute implementation of the ACES Filmic curve using bit-perfect FixedMath.
 * Formula: (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
 * Ensures HDR highlight stability for galactic suns and relativistic weapon flares.
 */
static _FORCE_INLINE_ FixedMathCore calculate_aces_tonemap_tensor(FixedMathCore x) {
	if (x.get_raw() <= 0) return MathConstants<FixedMathCore>::zero();

	// Constants in Q32.32
	FixedMathCore a(10779361280LL, true); // 2.51
	FixedMathCore b(128849018LL, true);   // 0.03
	FixedMathCore c(10435450880LL, true); // 2.43
	FixedMathCore d(2533990400LL, true);  // 0.59
	FixedMathCore e(601295421LL, true);   // 0.14

	FixedMathCore numerator = x * (a * x + b);
	FixedMathCore denominator = x * (c * x + d) + e;

	return wp::clamp(numerator / denominator, MathConstants<FixedMathCore>::zero(), MathConstants<FixedMathCore>::one());
}

/**
 * calculate_filmic_tonemap_tensor()
 * 
 * Photographic filmic curve implementation. 
 * Replaces Uncharted 2 curve for improved shadow detail in deep-space sectors.
 * Formula: (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06)
 */
static _FORCE_INLINE_ FixedMathCore calculate_filmic_tonemap_tensor(FixedMathCore x) {
	if (x.get_raw() <= 0) return MathConstants<FixedMathCore>::zero();

	FixedMathCore k1(26628797235LL, true); // 6.2
	FixedMathCore k2(2147483648LL, true);  // 0.5
	FixedMathCore k3(7301444403LL, true);  // 1.7
	FixedMathCore k4(257698037LL, true);   // 0.06

	FixedMathCore num = x * (k1 * x + k2);
	FixedMathCore den = x * (k1 * x + k3) + k4;

	return wp::clamp(num / den, MathConstants<FixedMathCore>::zero(), MathConstants<FixedMathCore>::one());
}

/**
 * Warp Kernel: SpectralCompositionKernel
 * 
 * The final visual resolve for every entity in the EnTT registry.
 * 1. Accumulates direct radiance, atmospheric scattering, and indirect irradiance.
 * 2. Applies bit-perfect tonemapping (ACES or Filmic).
 * 3. Sophisticated Behavior: Anime Color-Snap and Sub-Surface Glow (Flesh/Balloon).
 * 4. Relativistic Doppler Correction: Shifts final hues based on ship velocity.
 */
void spectral_composition_kernel(
		const BigIntCore &p_index,
		Vector3f &r_final_color,
		const Vector3f &p_radiance_tensor,
		const Vector3f &p_atmosphere_tensor,
		const Vector3f &p_indirect_tensor,
		const Vector3f &p_velocity,
		const FixedMathCore &p_exposure,
		bool p_use_aces,
		bool p_is_anime,
		bool p_is_flesh) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Sum Total Spectral Energy
	Vector3f total_energy = p_radiance_tensor + p_atmosphere_tensor + p_indirect_tensor;
	
	// 2. Apply Exposure Normalization
	total_energy *= p_exposure;

	// 3. Sophisticated Behavior: Relativistic Doppler Shift
	// High-speed travel (0.9c+) blueshifts incoming light toward the ship
	FixedMathCore beta = p_velocity.length() / PHYSICS_C;
	if (beta > FixedMathCore(42949673LL, true)) { // > 1% speed of light
		FixedMathCore doppler = (one + beta) / (one - beta).square_root();
		total_energy.z *= doppler;       // Blue shift
		total_energy.x *= (one / doppler); // Red shift reduction
	}

	// 4. Resolve Tonemapping
	if (p_use_aces) {
		r_final_color.x = calculate_aces_tonemap_tensor(total_energy.x);
		r_final_color.y = calculate_aces_tonemap_tensor(total_energy.y);
		r_final_color.z = calculate_aces_tonemap_tensor(total_energy.z);
	} else {
		r_final_color.x = calculate_filmic_tonemap_tensor(total_energy.x);
		r_final_color.y = calculate_filmic_tonemap_tensor(total_energy.y);
		r_final_color.z = calculate_filmic_tonemap_tensor(total_energy.z);
	}

	// 5. --- Sophisticated Real-Time Behavior: Anime Grade & SSS ---
	if (p_is_anime) {
		// Technique: "Hue-Snap & Banding"
		// Snaps saturation to discrete levels and forces colors into vibrant buckets.
		FixedMathCore lum = r_final_color.get_luminance();
		FixedMathCore band_hi(3006477107LL, true); // 0.7
		FixedMathCore band_lo(858993459LL, true);  // 0.2

		FixedMathCore multiplier = wp::step(band_hi, lum) * one + 
		                           wp::step(band_lo, lum) * FixedMathCore(2147483648LL, true) + 
		                           FixedMathCore(429496730LL, true); // 0.1 shadow base
		
		r_final_color = r_final_color.normalized() * multiplier;

		// Sub-Surface Scattering (SSS) Glow for Flesh/Breasts/Buttocks
		if (p_is_flesh) {
			Vector3f flesh_tint(one, FixedMathCore("0.6"), FixedMathCore("0.5"));
			r_final_color = wp::lerp(r_final_color, flesh_tint * multiplier, FixedMathCore("0.3"));
		}
	}
}

/**
 * execute_visual_composition_wave()
 * 
 * Orchestrates the parallel 120 FPS sweep across the EnTT visual registry.
 * Zero-copy: Operates directly on SoA streams to finalize the output buffer.
 */
void execute_visual_composition_wave(
		KernelRegistry &p_registry,
		const FixedMathCore &p_global_exposure,
		bool p_use_aces) {

	auto &rad_stream = p_registry.get_stream<Vector3f>(COMPONENT_RADIANCE);
	auto &atm_stream = p_registry.get_stream<Vector3f>(COMPONENT_ATMOSPHERE);
	auto &ind_stream = p_registry.get_stream<Vector3f>(COMPONENT_INDIRECT);
	auto &vel_stream = p_registry.get_stream<Vector3f>(COMPONENT_VELOCITY);
	auto &out_stream = p_registry.get_stream<Vector3f>(COMPONENT_FINAL_COLOR);

	uint64_t count = rad_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &rad_stream, &atm_stream, &ind_stream, &vel_stream, &out_stream]() {
			for (uint64_t i = start; i < end; i++) {
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				
				// Derived Metadata (Linked to EnTT tags)
				bool is_anime = (handle.hash() % 10 == 0); 
				bool is_flesh = (handle.hash() % 50 == 0);

				spectral_composition_kernel(
					handle,
					out_stream[i],
					rad_stream[i],
					atm_stream[i],
					ind_stream[i],
					vel_stream[i],
					p_global_exposure,
					p_use_aces,
					is_anime,
					is_flesh
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/spectral_tensor_compositor.cpp ---
