--- START OF FILE core/math/atmospheric_scattering_transmission_kernel.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_spectral_extinction_tensors()
 * 
 * Computes the extinction coefficients for RGB wavelengths.
 * Rayleigh: beta ~ 1/lambda^4
 * Mie: beta ~ constant (scaled by particle size)
 * Doppler Correction: lambda_obs = lambda_src / (1 + v/c)
 */
static _FORCE_INLINE_ Vector3f calculate_spectral_extinction_tensors(
		const Vector3f &p_base_lambda,
		const FixedMathCore &p_doppler_factor,
		const FixedMathCore &p_rayleigh_density,
		const FixedMathCore &p_mie_density,
		const AtmosphereParams &p_params) {

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	
	// Shift wavelengths for high-speed observers
	Vector3f obs_lambda = p_base_lambda / p_doppler_factor;
	
	// Rayleigh Extinction: Power law integration
	FixedMathCore r4 = obs_lambda.x * obs_lambda.x * obs_lambda.x * obs_lambda.x;
	FixedMathCore g4 = obs_lambda.y * obs_lambda.y * obs_lambda.y * obs_lambda.y;
	FixedMathCore b4 = obs_lambda.z * obs_lambda.z * obs_lambda.z * obs_lambda.z;

	FixedMathCore norm_scale(429496729600000000LL, true);
	Vector3f r_ext = Vector3f(norm_scale / r4, norm_scale / g4, norm_scale / b4) * p_rayleigh_density;
	
	// Mie Extinction: Isotropic particle absorption
	Vector3f m_ext = Vector3f(p_params.mie_extinction_coeff) * p_mie_density;

	return r_ext + m_ext;
}

/**
 * Warp Kernel: AtmosphericTransmissionKernel
 * 
 * Computes the transmittance (0.0 to 1.0) along a path through the atmosphere.
 * 1. Ray-Marching: Samples the multi-layer density field.
 * 2. Shadow Resolve: Checks for occlusions by stars, planets, and EnTT entities.
 * 3. Beer's Law: Resolves T = exp(-Sum(extinction * step)).
 * 4. Anime Style: Snaps transmittance into discrete photographic bands.
 */
void atmospheric_transmission_kernel(
		const BigIntCore &p_index,
		Vector3f &r_transmittance,
		const Vector3f &p_origin,
		const Vector3f &p_target,
		const Vector3f &p_ship_velocity,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_lights,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	Vector3f path_vec = p_target - p_origin;
	FixedMathCore dist = path_vec.length();
	if (unlikely(dist.get_raw() == 0)) {
		r_transmittance = Vector3f(one, one, one);
		return;
	}
	Vector3f path_dir = path_vec / dist;

	// 1. Relativistic Doppler Factor
	FixedMathCore radial_v = path_dir.dot(p_ship_velocity);
	FixedMathCore c_fixed(299792458LL);
	FixedMathCore doppler = one + (radial_v / c_fixed);
	Vector3f base_lambda(FixedMathCore(680LL), FixedMathCore(550LL), FixedMathCore(440LL));

	// 2. Deterministic Integration (16-sample march for 120 FPS stability)
	const int samples = 16;
	FixedMathCore step_size = dist / FixedMathCore(static_cast<int64_t>(samples));
	Vector3f total_optical_depth;

	for (int i = 0; i < samples; i++) {
		FixedMathCore t = step_size * (FixedMathCore(static_cast<int64_t>(i)) + MathConstants<FixedMathCore>::half());
		Vector3f sample_p = p_origin + path_dir * t;
		
		FixedMathCore altitude = (sample_p - p_params.planet_center).length() - p_params.planet_radius;
		if (altitude.get_raw() < 0) {
			// Path goes through planet bulk: Total Extinction
			r_transmittance = Vector3f(zero, zero, zero);
			return;
		}

		// Resolve Local Densities
		FixedMathCore dr = wp::exp(-(altitude / p_params.rayleigh_scale_height));
		FixedMathCore dm = wp::exp(-(altitude / p_params.mie_scale_height));

		// Accumulate Extinction Tensor
		Vector3f step_ext = calculate_spectral_extinction_tensors(base_lambda, doppler, dr, dm, p_params);
		total_optical_depth += step_ext * step_size;
	}

	// 3. Final Transmittance Resolve
	// res = exp(-tau)
	r_transmittance.x = wp::exp(-total_optical_depth.x);
	r_transmittance.y = wp::exp(-total_optical_depth.y);
	r_transmittance.z = wp::exp(-total_optical_depth.z);

	// 4. --- Sophisticated Behavior: Shadow-Integral ---
	// Check for shadows cast by other lights in the sector (e.g. eclipses)
	for (uint32_t l = 0; l < p_lights.count; l++) {
		Vector3f L = (p_lights.type[l] == 0) ? p_lights.direction[l] : (p_lights.position[l] - p_target).normalized();
		if (wp::check_occlusion_sphere(p_target, L, p_params.planet_center, p_params.planet_radius)) {
			// Apply ambient shadow tint
			FixedMathCore shadow_factor(214748364LL, true); // 0.05
			r_transmittance *= shadow_factor;
		}
	}

	// 5. --- Anime Style Quantization ---
	if (p_is_anime) {
		// Anime Technique: "Depth Slicing". 
		// Transmittance is snapped to 4 discrete tiers to match hand-painted cel layers.
		auto snap_band = [&](FixedMathCore val) -> FixedMathCore {
			if (val > FixedMathCore(3435973836LL, true)) return one; // Clear
			if (val > FixedMathCore(2147483648LL, true)) return FixedMathCore(3006477107LL, true); // 0.7
			if (val > FixedMathCore(858993459LL, true)) return FixedMathCore(1288490188LL, true);  // 0.3
			return FixedMathCore(214748364LL, true); // 0.05 Shadow
		};

		r_transmittance.x = snap_band(r_transmittance.x);
		r_transmittance.y = snap_band(r_transmittance.y);
		r_transmittance.z = snap_band(r_transmittance.z);
	}
}

/**
 * execute_transmission_resolve_sweep()
 * 
 * Master parallel sweep for the 120 FPS optical finalizer.
 * Partitions the EnTT component registry for global light-extinction resolve.
 */
void execute_transmission_resolve_sweep(
		KernelRegistry &p_registry,
		const Vector3f &p_ship_velocity,
		const AtmosphereParams &p_params,
		const LightDataSoA &p_lights) {

	auto &origin_stream = p_registry.get_stream<Vector3f>(COMPONENT_RAY_ORIGIN);
	auto &target_stream = p_registry.get_stream<Vector3f>(COMPONENT_RAY_TARGET);
	auto &trans_stream = p_registry.get_stream<Vector3f>(COMPONENT_TRANSMITTANCE);

	uint64_t count = origin_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &origin_stream, &target_stream, &trans_stream, &p_params, &p_lights]() {
			for (uint64_t i = start; i < end; i++) {
				// Style flag linked to Entity handle
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 6 == 0);

				atmospheric_transmission_kernel(
					handle,
					trans_stream[i],
					origin_stream[i],
					target_stream[i],
					p_ship_velocity,
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

--- END OF FILE core/math/atmospheric_scattering_transmission_kernel.cpp ---
