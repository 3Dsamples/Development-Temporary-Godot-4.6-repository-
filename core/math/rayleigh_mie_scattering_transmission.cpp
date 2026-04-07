--- START OF FILE core/math/rayleigh_mie_scattering_transmission.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * exp_fixed_kernel()
 * 
 * High-performance deterministic exponential function (e^x) for Q32.32.
 * Uses a 5th-order Taylor series approximation for the range [-8, 0].
 * Essential for bit-perfect light extinction and atmospheric density profiles.
 */
static _FORCE_INLINE_ FixedMathCore exp_fixed_kernel(FixedMathCore p_x) {
	if (p_x.get_raw() >= 0) return MathConstants<FixedMathCore>::one();
	if (p_x < FixedMathCore(-34359738368LL, true)) return MathConstants<FixedMathCore>::zero(); // e^-8 approx cutoff

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore x = p_x;
	FixedMathCore x2 = x * x;
	FixedMathCore x3 = x2 * x;
	FixedMathCore x4 = x3 * x;
	FixedMathCore x5 = x4 * x;

	// 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5!
	FixedMathCore res = one + x 
		+ (x2 * FixedMathCore(2147483648LL, true)) 
		+ (x3 * FixedMathCore(715827882LL, true)) 
		+ (x4 * FixedMathCore(178956970LL, true)) 
		+ (x5 * FixedMathCore(35791394LL, true));

	return wp::max(MathConstants<FixedMathCore>::zero(), res);
}

/**
 * Warp Kernel: AtmosphericTransmissionKernel
 * 
 * Resolves the light extinction between two points in space.
 * 1. Integrates the extinction coefficient (Scattering + Absorption).
 * 2. Applies Style-Tensors for Anime-specific color bands.
 * 3. Injects turbulence-based density fluctuations for real-time sky "shimmer".
 */
void atmospheric_transmission_kernel(
		const BigIntCore &p_index,
		Vector3f &r_transmittance,
		const Vector3f &p_origin,
		const Vector3f &p_target,
		const AtmosphereParams &p_params,
		const FixedMathCore &p_turbulence_seed,
		bool p_is_anime) {

	Vector3f ray_vec = p_target - p_origin;
	FixedMathCore dist = ray_vec.length();
	Vector3f dir = ray_vec / dist;

	const int samples = 8;
	FixedMathCore step_size = dist / FixedMathCore(static_cast<int64_t>(samples));
	Vector3f optical_depth;

	for (int i = 0; i < samples; i++) {
		FixedMathCore t = step_size * (FixedMathCore(static_cast<int64_t>(i)) + MathConstants<FixedMathCore>::half());
		Vector3f sample_p = p_origin + dir * t;
		FixedMathCore height = wp::max(MathConstants<FixedMathCore>::zero(), sample_p.length() - p_params.planet_radius);

		// 1. Resolve Local Density with Shimmer/Turbulence
		// Simulation behavior: High-speed ships experience "Aero-Jitter" density shifts.
		FixedMathCore density_mod = MathConstants<FixedMathCore>::one() + wp::sin(height * p_turbulence_seed) * FixedMathCore(42949673LL, true);
		
		FixedMathCore dr = AtmosphericScattering::compute_density(height, p_params.rayleigh_scale_height) * density_mod;
		FixedMathCore dm = AtmosphericScattering::compute_density(height, p_params.mie_scale_height) * density_mod;

		// 2. Accumulate Extinction Tensors (Rayleigh + Mie + Absorption)
		optical_depth += (p_params.rayleigh_extinction_coeffs * dr + Vector3f(p_params.mie_extinction_coeff * dm)) * step_size;
	}

	// 3. Final Transmittance Resolve: T = exp(-tau)
	r_transmittance.x = exp_fixed_kernel(-optical_depth.x);
	r_transmittance.y = exp_fixed_kernel(-optical_depth.y);
	r_transmittance.z = exp_fixed_kernel(-optical_depth.z);

	// --- Sophisticated Anime Behavior ---
	if (p_is_anime) {
		// Anime Technique: "Atmospheric Depth Slicing".
		// Forces transmission to snap to specific intensity bands to create 
		// the look of layered background cels.
		FixedMathCore avg_t = (r_transmittance.x + r_transmittance.y + r_transmittance.z) / FixedMathCore(3LL, false);
		FixedMathCore snap = wp::step(FixedMathCore(3006477107LL, true), avg_t) * MathConstants<FixedMathCore>::one() +
		                    wp::step(FixedMathCore(1288490188LL, true), avg_t) * FixedMathCore(2147483648LL, true);
		
		r_transmittance = Vector3f(snap, snap, snap);
	}
}

/**
 * execute_transmission_sweep()
 * 
 * Parallel orchestrator for resolving global light extinction.
 * Used for shadows, volumetric god-rays, and distant visibility.
 */
void execute_transmission_sweep(
		const BigIntCore &p_count,
		const Vector3f *p_origins,
		const Vector3f *p_targets,
		const AtmosphereParams &p_params,
		Vector3f *r_results_buffer) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_count.to_string()));
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? total : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &p_params]() {
			for (uint64_t i = start; i < end; i++) {
				// Style derived from entity handle for bit-perfect consistency
				bool anime_mode = (i % 4 == 0);
				FixedMathCore seed_val(static_cast<int64_t>(i % 100), false);

				atmospheric_transmission_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_results_buffer[i],
					p_origins[i],
					p_targets[i],
					p_params,
					seed_val,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/rayleigh_mie_scattering_transmission.cpp ---
