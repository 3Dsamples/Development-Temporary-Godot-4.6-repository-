--- START OF FILE core/math/spectral_radiance_bloom_kernel.cpp ---

#include "core/math/spectral_tensor_kernel.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: LuminanceThresholdKernel
 * 
 * Extracts pixels exceeding the HDR threshold to be processed by the bloom pipeline.
 * Strictly bit-perfect to ensure glare patterns are identical across simulation nodes.
 */
static _FORCE_INLINE_ FixedMathCore extract_luminance_threshold(
		const Vector3f &p_energy,
		const FixedMathCore &p_threshold,
		const FixedMathCore &p_soft_knee) {

	// standard Rec. 709 luminance coefficients in FixedMath
	FixedMathCore lum = p_energy.x * FixedMathCore(913142732LL, true) + 
	                    p_energy.y * FixedMathCore(3071746048LL, true) + 
	                    p_energy.z * FixedMathCore(310118991LL, true);

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// Quadratic knee for smooth HDR transitions
	FixedMathCore rq = wp::clamp(lum - p_threshold + p_soft_knee, zero, p_soft_knee * FixedMathCore(2LL, false));
	rq = (rq * rq) / (p_soft_knee * FixedMathCore(4LL, false) + FixedMathCore(4294LL, true)); // small epsilon
	
	FixedMathCore factor = wp::max(rq, lum - p_threshold) / wp::max(lum, FixedMathCore(4294LL, true));
	return factor;
}

/**
 * Warp Kernel: AnisotropicGlareKernel
 * 
 * Simulates light streaks (starbursts) radiating from high-energy points.
 * Features specialized Anime-Style sharp streaks and realistic diffraction spikes.
 */
void anisotropic_glare_kernel(
		const BigIntCore &p_index,
		Vector3f &r_bloom_accum,
		const Vector3f &p_source_energy,
		const Vector3f &p_pixel_coord,
		const Vector3f &p_light_pos,
		const FixedMathCore &p_streak_count,
		const FixedMathCore &p_streak_width,
		bool p_is_anime) {

	Vector3f dir_to_light = p_pixel_coord - p_light_pos;
	FixedMathCore dist = dir_to_light.length();
	if (dist.get_raw() == 0) return;

	Vector3f n_dir = dir_to_light / dist;
	
	// Deterministic Angular Check for streaks
	FixedMathCore angle = Math::atan2(n_dir.y, n_dir.x);
	
	// Calculate star-flare intensity: cos(angle * count / 2) ^ width
	FixedMathCore flare_val = Math::cos(angle * p_streak_count * MathConstants<FixedMathCore>::half());
	flare_val = wp::max(MathConstants<FixedMathCore>::zero(), flare_val);
	
	// Sophisticated Anime Feature: Sharp banding of light streaks
	if (p_is_anime) {
		// Quantize the flare into sharp "beams" using step functions
		flare_val = wp::step(FixedMathCore(3865470566LL, true), flare_val) * MathConstants<FixedMathCore>::one(); // 0.9 threshold
	} else {
		// Realistic smooth exponential decay for diffraction
		flare_val = wp::pow(flare_val, 8); 
	}

	FixedMathCore attenuation = MathConstants<FixedMathCore>::one() / (dist + MathConstants<FixedMathCore>::one());
	r_bloom_accum += p_source_energy * (flare_val * attenuation);
}

/**
 * execute_spectral_bloom_sweep()
 * 
 * Master parallel sweep for visual post-processing.
 * 1. Parallel Luminance Extraction.
 * 2. Multi-threaded Glare/Star-flare generation.
 * 3. EnTT stream accumulation for 120 FPS performance.
 */
void execute_spectral_bloom_sweep(
		const BigIntCore &p_total_pixels,
		Vector3f *r_radiance_stream,
		const Vector3f *p_light_positions,
		const Vector3f *p_light_energies,
		uint32_t p_light_count,
		const FixedMathCore &p_threshold,
		const FixedMathCore &p_delta) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_total_pixels.to_string()));
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? total : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				Vector3f &pixel = r_radiance_stream[i];
				
				// 1. Extract bloom weight via FixedMath threshold
				FixedMathCore weight = extract_luminance_threshold(pixel, p_threshold, FixedMathCore(2147483648LL, true));
				
				// 2. Process localized light interactions (Glare/Flares)
				Vector3f bloom_acc;
				for (uint32_t l = 0; l < p_light_count; l++) {
					// Use pixel index to derive deterministic screen coordinates
					Vector3f coord(FixedMathCore(static_cast<int64_t>(i % 1920), false), 
					               FixedMathCore(static_cast<int64_t>(i / 1920), false), 
					               MathConstants<FixedMathCore>::zero());

					anisotropic_glare_kernel(
						BigIntCore(static_cast<int64_t>(i)),
						bloom_acc,
						p_light_energies[l],
						coord,
						p_light_positions[l],
						FixedMathCore(6LL, false), // 6-point star
						FixedMathCore(8LL, false),  // Streak width power
						(i % 12 == 0) // Deterministic Anime Style
					);
				}

				// Final Zero-Copy accumulation into the EnTT radiance stream
				pixel += bloom_acc * weight;
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/spectral_radiance_bloom_kernel.cpp ---
