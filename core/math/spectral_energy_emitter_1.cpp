--- START OF FILE core/math/spectral_energy_emitter.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_spectral_color_kernel()
 * 
 * Maps a black-body temperature to a bit-perfect RGB spectral tensor.
 * Strictly uses FixedMathCore polynomial approximations of the CIE color space.
 * T is in Kelvin.
 */
static _FORCE_INLINE_ Vector3f calculate_spectral_color_kernel(const FixedMathCore &p_temperature) {
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// Normalized temp scale for color lookup [1000K to 40000K]
	FixedMathCore t = p_temperature / FixedMathCore(40000LL, false);
	t = wp::clamp(t, zero, one);

	Vector3f color;
	// Red component: high at low temps, decreases at high temps
	color.x = one - wp::smoothstep(FixedMathCore("0.1"), FixedMathCore("0.4"), t);
	
	// Green component: peaks in mid-range (Solar G-type stars)
	color.y = wp::smoothstep(zero, FixedMathCore("0.2"), t) * (one - wp::smoothstep(FixedMathCore("0.5"), one, t));
	
	// Blue component: high at massive O-type stars
	color.z = wp::smoothstep(FixedMathCore("0.3"), one, t);

	return color.normalized();
}

/**
 * Warp Kernel: StarLightEmissionKernel
 * 
 * Computes the total radiant flux and spectral energy for a high-mass entity.
 * 1. Stefan-Boltzmann: P = sigma * A * T^4
 * 2. Wien's Law: lambda_max = b / T (Wavelength selection)
 * 3. Sophisticated Behavior: Anime-Flare Quantization.
 */
void starlight_emission_kernel(
		const BigIntCore &p_index,
		Vector3f &r_spectral_radiance,
		FixedMathCore &r_total_flux,
		const FixedMathCore &p_temperature,
		const BigIntCore &p_surface_area_sq_km,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Calculate Total Radiant Power (Stefan-Boltzmann)
	// sigma approx 5.67e-8. Scaled for bit-perfection in Q32.32
	FixedMathCore sigma("0.000000056703");
	FixedMathCore t2 = p_temperature * p_temperature;
	FixedMathCore t4 = t2 * t2;
	
	// Convert BigInt area to FixedMath for flux calculation
	FixedMathCore area_f(static_cast<int64_t>(std::stoll(p_surface_area_sq_km.to_string())));
	r_total_flux = sigma * area_f * t4;

	// 2. Resolve Spectral Distribution
	r_spectral_radiance = calculate_spectral_color_kernel(p_temperature);

	// 3. --- Sophisticated Real-Time Behavior: Realistic vs Anime ---
	if (p_is_anime) {
		// Anime Technique: "Energy Banding". 
		// Quantizes the flux into discrete "Power Tiers" to prevent smooth HDR gradients
		// and create the look of traditional hand-drawn star flares.
		FixedMathCore tier_step(1000000LL, false); 
		r_total_flux = Math::snapped(r_total_flux, tier_step);

		// Saturation boost for anime star hues (Vibrant purples and blues)
		r_spectral_radiance *= FixedMathCore(5153960755LL, true); // 1.2x boost
	}

	// 4. Relativistic Luminosity Correction
	// (Placeholder for ship-velocity relative boost resolved in finalizer pass)
}

/**
 * execute_spectral_emission_sweep()
 * 
 * Orchestrates the parallel 120 FPS sweep for all light-emitting celestial bodies.
 * processes star-tensors in EnTT SoA streams.
 */
void execute_spectral_emission_sweep(
		KernelRegistry &p_registry,
		const FixedMathCore &p_delta) {

	auto &temp_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_TEMPERATURE);
	auto &area_stream = p_registry.get_stream<BigIntCore>(COMPONENT_SURFACE_AREA);
	auto &radiance_stream = p_registry.get_stream<Vector3f>(COMPONENT_SPECTRAL_RADIANCE);
	auto &flux_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_TOTAL_FLUX);

	uint64_t count = temp_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &temp_stream, &area_stream, &radiance_stream, &flux_stream]() {
			for (uint64_t i = start; i < end; i++) {
				// Deterministic Style Selection: 1 in 10 emitters use anime flare logic
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 10 == 0);

				starlight_emission_kernel(
					handle,
					radiance_stream[i],
					flux_stream[i],
					temp_stream[i],
					area_stream[i],
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * calculate_distance_irradiance()
 * 
 * Bit-perfect Inverse Square Law implementation.
 * E = P / (4 * pi * r^2)
 * strictly uses BigIntCore for distance squared to handle galactic ranges.
 */
FixedMathCore calculate_distance_irradiance(
		const FixedMathCore &p_source_flux,
		const BigIntCore &p_dist_km_sq) {
	
	if (p_dist_km_sq.is_zero()) return p_source_flux;

	// 4 * pi in Q32.32
	FixedMathCore four_pi(53972150816LL, true); 
	
	BigIntCore divisor_bi = p_dist_km_sq * BigIntCore(four_pi.get_raw());
	// Precision-safe shift to FixedMathCore
	FixedMathCore divisor_f(static_cast<int64_t>((divisor_bi / BigIntCore(FixedMathCore::ONE_RAW)).operator int64_t()), true);

	return p_source_flux / divisor_f;
}

} // namespace UniversalSolver

--- END OF FILE core/math/spectral_energy_emitter.cpp ---
