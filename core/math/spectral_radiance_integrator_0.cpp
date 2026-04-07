--- START OF FILE core/math/spectral_radiance_integrator.cpp ---

#include "core/math/spectral_tensor_kernel.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: ACES_ToneMapping_Kernel
 * 
 * Ported ACES Filmic Tone Mapping to bit-perfect FixedMathCore.
 * Formula: (x(ax+b)) / (x(cx+d)+e)
 * Ensures high-dynamic-range (HDR) stability for galactic suns and 
 * stylized anime highlights without floating-point artifacts.
 */
static _FORCE_INLINE_ FixedMathCore apply_aces_tonemap(FixedMathCore x) {
	if (x.get_raw() <= 0) return MathConstants<FixedMathCore>::zero();

	// ACES Constants in Q32.32
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
 * integrate_spectral_radiance_batch()
 * 
 * The final 120 FPS sweep for the visual pipeline.
 * 1. Applies Exposure compensation.
 * 2. Processes ACES Tone Mapping per color channel.
 * 3. Injects distance-based spectral shifting (Atmospheric perspective).
 */
void integrate_spectral_radiance_batch(
		const BigIntCore &p_count,
		Vector3f *r_radiance_stream,
		const FixedMathCore &p_exposure,
		const FixedMathCore &p_observer_dist,
		bool p_is_anime) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_count.to_string()));
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? total : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				Vector3f &energy = r_radiance_stream[i];

				// 1. Exposure Adjustment
				energy *= p_exposure;

				// 2. Deterministic Tone Mapping
				energy.x = apply_aces_tonemap(energy.x);
				energy.y = apply_aces_tonemap(energy.y);
				energy.z = apply_aces_tonemap(energy.z);

				// 3. Distance-Based Enhancement
				// For galactic scales, we slightly blue-shift distant low-intensity pixels
				if (p_observer_dist > FixedMathCore(50000LL, false)) {
					FixedMathCore blue_shift = wp::min(p_observer_dist / FixedMathCore(1000000LL, false), FixedMathCore(429496729LL, true)); // Max 0.1
					energy.z = wp::max(energy.z, blue_shift * FixedMathCore(2147483648LL, true));
				}

				// --- Anime Style Final Pass ---
				if (p_is_anime) {
					// Snap luminance to discrete bands for final output
					FixedMathCore luminance = energy.dot(Vector3f(FixedMathCore(913142732LL, true), FixedMathCore(3071746048LL, true), FixedMathCore(310118991LL, true)));
					FixedMathCore factor = wp::step(FixedMathCore(2147483648LL, true), luminance) ? MathConstants<FixedMathCore>::one() : FixedMathCore(2147483648LL, true);
					energy *= factor;
				}
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * compute_global_luminance_reduction()
 * 
 * Parallel reduction to find average scene brightness for Auto-Exposure.
 * Strictly bit-perfect to prevent exposure flickering.
 */
FixedMathCore compute_global_luminance_reduction(const Vector3f *p_radiance, uint64_t p_count) {
	FixedMathCore sum = MathConstants<FixedMathCore>::zero();
	for (uint64_t i = 0; i < p_count; i++) {
		sum += p_radiance[i].length_squared();
	}
	return (p_count > 0) ? Math::sqrt(sum / FixedMathCore(static_cast<int64_t>(p_count))) : MathConstants<FixedMathCore>::one();
}

} // namespace UniversalSolver

--- END OF FILE core/math/spectral_radiance_integrator.cpp ---
