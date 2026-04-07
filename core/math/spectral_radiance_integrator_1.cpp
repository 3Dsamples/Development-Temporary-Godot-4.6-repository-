--- START OF FILE core/math/spectral_radiance_integrator.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: ACES_Filmic_Tonemap
 * 
 * Implements the standard ACES filmic response curve in bit-perfect Q32.32.
 * Formula: f(x) = (x * (a * x + b)) / (x * (c * x + d) + e)
 */
static _FORCE_INLINE_ FixedMathCore apply_aces_tonemap(FixedMathCore x) {
	if (x.get_raw() <= 0) return MathConstants<FixedMathCore>::zero();

	// ACES Curve Coefficients in FixedMath (Q32.32)
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
 * Warp Kernel: Filmic_Generic_Tonemap
 * 
 * A highly sophisticated photographic filmic response curve.
 * Engineered for deep shadow retention and highlight roll-off in space environments.
 */
static _FORCE_INLINE_ FixedMathCore apply_filmic_tonemap(FixedMathCore x) {
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	if (x.get_raw() <= 0) return zero;

	// Filmic curve: (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06)
	FixedMathCore k_6_2(26628797235LL, true); // 6.2
	FixedMathCore k_0_5(2147483648LL, true);  // 0.5
	FixedMathCore k_1_7(7301444403LL, true);  // 1.7
	FixedMathCore k_0_06(257698037LL, true);  // 0.06

	FixedMathCore num = x * (k_6_2 * x + k_0_5);
	FixedMathCore den = x * (k_6_2 * x + k_1_7) + k_0_06;

	return wp::clamp(num / den, zero, one);
}

/**
 * execute_visual_composition_sweep()
 * 
 * The master 120 FPS visual finalizer.
 * 1. Parallel Exposure Normalization.
 * 2. Deterministic Tonemapping (ACES or Filmic).
 * 3. Sophisticated Anime grading (Hue snapping and saturation quantization).
 */
void execute_visual_composition_sweep(
		KernelRegistry &p_registry,
		const FixedMathCore &p_exposure_val,
		bool p_use_aces,
		bool p_is_anime) {

	auto &radiance_stream = p_registry.get_stream<Vector3f>(COMPONENT_RADIANCE);
	uint64_t entity_count = radiance_stream.size();
	if (entity_count == 0) return;

	uint32_t worker_threads = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk_size = entity_count / worker_threads;

	for (uint32_t w = 0; w < worker_threads; w++) {
		uint64_t start = w * chunk_size;
		uint64_t end = (w == worker_threads - 1) ? entity_count : (w + 1) * chunk_size;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &radiance_stream]() {
			for (uint64_t i = start; i < end; i++) {
				Vector3f &color = radiance_stream[i];

				// 1. Apply Exposure Tensor
				color *= p_exposure_val;

				// 2. Resolve Tonemapping Wave
				if (p_use_aces) {
					color.x = apply_aces_tonemap(color.x);
					color.y = apply_aces_tonemap(color.y);
					color.z = apply_aces_tonemap(color.z);
				} else {
					color.x = apply_filmic_tonemap(color.x);
					color.y = apply_filmic_tonemap(color.y);
					color.z = apply_filmic_tonemap(color.z);
				}

				// 3. --- Sophisticated Behavior: Anime Stylization ---
				if (p_is_anime) {
					// Snap luminance to discrete cel-shaded levels (Banding)
					FixedMathCore lum = color.get_luminance();
					FixedMathCore band_hi(3006477107LL, true); // 0.7
					FixedMathCore band_lo(1288490188LL, true); // 0.3
					
					FixedMathCore snap = wp::step(band_hi, lum) * MathConstants<FixedMathCore>::one() + 
					                    wp::step(band_lo, lum) * FixedMathCore(2147483648LL, true) + 
					                    FixedMathCore(429496730LL, true); // 0.1 shadow base
					
					color = color.normalized() * snap;
					
					// Saturation Boost for vibrancy
					color *= FixedMathCore(5153960755LL, true); // 1.2x boost
				}
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * compute_auto_exposure_reduction()
 * 
 * Global parallel reduction to find the average luminance across all galactic entities.
 * Returns a bit-perfect FixedMathCore value used to adjust p_exposure_val.
 */
FixedMathCore compute_auto_exposure_reduction(const Vector3f *p_radiance_stream, uint64_t p_count) {
	if (p_count == 0) return MathConstants<FixedMathCore>::one();

	BigIntCore total_lum_raw(0LL);
	for (uint64_t i = 0; i < p_count; i++) {
		total_lum_raw += BigIntCore(p_radiance_stream[i].get_luminance().get_raw());
	}

	BigIntCore count_bi(static_cast<int64_t>(p_count));
	FixedMathCore avg_lum(static_cast<int64_t>((total_lum_raw / count_bi).operator int64_t()), true);
	
	// Target luminance normalization (Middle Gray 0.18)
	FixedMathCore target_gray(773094113LL, true); 
	return target_gray / (avg_lum + FixedMathCore(4294LL, true)); // Epsilon to prevent division by zero
}

} // namespace UniversalSolver

--- END OF FILE core/math/spectral_radiance_integrator.cpp ---
