--- START OF FILE core/math/atmospheric_scattering_transmission.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * deterministic_exp_kernel()
 * 
 * Computes e^x using a 7th-order Taylor series for the range [-8, 0].
 * Essential for bit-perfect light absorption and density profiles.
 * Strictly integer-based FixedMath to prevent FPU clock drift.
 */
static _FORCE_INLINE_ FixedMathCore deterministic_exp_kernel(FixedMathCore p_x) {
	if (p_x.get_raw() >= 0) return MathConstants<FixedMathCore>::one();
	if (p_x < FixedMathCore(-34359738368LL, true)) return MathConstants<FixedMathCore>::zero(); // Cutoff at e^-8

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore x = p_x;
	FixedMathCore x2 = x * x;
	FixedMathCore x3 = x2 * x;
	FixedMathCore x4 = x3 * x;
	FixedMathCore x5 = x4 * x;
	FixedMathCore x6 = x5 * x;

	// coefficients: 1/2!, 1/3!, 1/4!, 1/5!, 1/6!
	FixedMathCore c2(2147483648LL, true); 
	FixedMathCore c3(715827882LL, true);
	FixedMathCore c4(178956970LL, true);
	FixedMathCore c5(35791394LL, true);
	FixedMathCore c6(5965232LL, true);

	FixedMathCore res = one + x + (x2 * c2) + (x3 * c3) + (x4 * c4) + (x5 * c5) + (x6 * c6);
	return wp::max(MathConstants<FixedMathCore>::zero(), res);
}

/**
 * Warp Kernel: AtmosphericTransmissionKernel
 * 
 * Resolves the light transmission between two coordinates.
 * 1. Integrates the extinction tensor (Rayleigh + Mie + Ozone).
 * 2. Applies Style-Tensors for Anime-specific banding.
 * 3. Injects "Turbulence Noise" into the density field for real-time shimmering.
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
	FixedMathCore total_dist = ray_vec.length();
	if (total_dist.get_raw() == 0) {
		r_transmittance = Vector3f(MathConstants<FixedMathCore>::one());
		return;
	}
	Vector3f dir = ray_vec / total_dist;

	// Distance-Adaptive Sampling: maintain 120 FPS performance
	int samples = 8;
	if (total_dist > FixedMathCore(100000LL, false)) samples = 16; // Higher detail for long paths

	FixedMathCore step_size = total_dist / FixedMathCore(static_cast<int64_t>(samples));
	Vector3f optical_depth_accum;

	for (int i = 0; i < samples; i++) {
		FixedMathCore t = step_size * (FixedMathCore(static_cast<int64_t>(i)) + MathConstants<FixedMathCore>::half());
		Vector3f sample_p = p_origin + dir * t;
		
		FixedMathCore altitude = wp::max(MathConstants<FixedMathCore>::zero(), (sample_p - p_params.planet_center).length() - p_params.planet_radius);

		// Sophisticated Behavior: Dynamic Turbulence (Shimmer)
		// Simulates heat-waves and atmospheric distortion at 120 FPS.
		FixedMathCore jitter = wp::sin(altitude * p_turbulence_seed + FixedMathCore(static_cast<int64_t>(i))) * FixedMathCore(42949673LL, true); // 0.01 factor
		FixedMathCore density_mod = MathConstants<FixedMathCore>::one() + jitter;

		FixedMathCore dr = AtmosphericScattering::compute_density(altitude, p_params.rayleigh_scale_height) * density_mod;
		FixedMathCore dm = AtmosphericScattering::compute_density(altitude, p_params.mie_scale_height) * density_mod;

		// Accumulate Spectral Extinction (Wavelength dependent)
		optical_depth_accum += (p_params.rayleigh_extinction_coeffs * dr + Vector3f(p_params.mie_extinction_coeff * dm)) * step_size;
	}

	// 3. Final Resolve: T = exp(-tau)
	r_transmittance.x = deterministic_exp_kernel(-optical_depth_accum.x);
	r_transmittance.y = deterministic_exp_kernel(-optical_depth_accum.y);
	r_transmittance.z = deterministic_exp_kernel(-optical_depth_accum.z);

	// --- Sophisticated Anime Behavior: Transmission Banding ---
	if (p_is_anime) {
		// Anime Technique: "Optical Slicing". 
		// Instead of smooth gradients, light extinction occurs in discrete color cels.
		FixedMathCore avg_t = (r_transmittance.x + r_transmittance.y + r_transmittance.z) / FixedMathCore(3LL);
		
		FixedMathCore band_hi(3435973836LL, true); // 0.8
		FixedMathCore band_lo(858993459LL, true);  // 0.2

		FixedMathCore snap = wp::step(band_hi, avg_t) * MathConstants<FixedMathCore>::one() + 
		                    wp::step(band_lo, avg_t) * FixedMathCore(2147483648LL, true) + 
		                    FixedMathCore(214748364LL, true); // base 0.05 shadow
		
		r_transmittance = Vector3f(snap, snap, snap);
	}
}

/**
 * execute_transmission_sweep_parallel()
 * 
 * Orchestrates the master 120 FPS parallel resolve for light transport.
 * Processes EnTT SoA component streams for origins, targets, and extinction data.
 */
void execute_transmission_sweep_parallel(
		KernelRegistry &p_registry,
		const AtmosphereParams &p_params,
		const BigIntCore &p_world_seed) {

	auto &origin_stream = p_registry.get_stream<Vector3f>(COMPONENT_RAY_ORIGIN);
	auto &target_stream = p_registry.get_stream<Vector3f>(COMPONENT_RAY_TARGET);
	auto &trans_stream = p_registry.get_stream<Vector3f>(COMPONENT_TRANSMITTANCE);
	
	uint64_t count = origin_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &origin_stream, &target_stream, &trans_stream, &p_params]() {
			for (uint64_t i = start; i < end; i++) {
				// Style derived from Entity ID for bit-perfect look-consistency
				bool anime_mode = (i % 6 == 0); 
				FixedMathCore turb_seed = FixedMathCore(static_cast<int64_t>(p_world_seed.hash() ^ i));

				atmospheric_transmission_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					trans_stream[i],
					origin_stream[i],
					target_stream[i],
					p_params,
					turb_seed,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_scattering_transmission.cpp ---
