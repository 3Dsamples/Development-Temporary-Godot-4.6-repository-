--- START OF FILE core/math/spectral_energy_conduction.cpp ---

#include "core/math/spectral_tensor_kernel.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: MaterialThermodynamicsKernel
 * 
 * Simulates the physical evolution of a material based on its thermal state.
 * 1. Radiative Cooling: Objects lose heat to the vacuum of space.
 * 2. Thermal Softening: Yield strength decreases as temperature rises.
 * 3. Incandescence: Albedo shifts toward "Glow" colors at high heat.
 */
void material_thermodynamics_kernel(
		const BigIntCore &p_index,
		FixedMathCore &r_temperature,
		FixedMathCore &r_yield_strength,
		Vector3f &r_albedo,
		Vector3f &r_emission,
		const FixedMathCore &p_thermal_conductivity,
		const FixedMathCore &p_delta,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore ambient_temp(12591030272LL, true); // 293.15 K

	// 1. Deterministic Radiative Cooling (Stefan-Boltzmann approx)
	// Heat loss is proportional to T^4. In FixedMath, we use a simplified power-curve.
	FixedMathCore t_norm = r_temperature / FixedMathCore(2000LL, false); // Normalized to 2000K
	FixedMathCore cooling_power = wp::pow(t_norm, 4) * FixedMathCore(42949672LL, true); // 0.01 sigma
	r_temperature -= cooling_power * p_delta;

	// 2. Thermal Softening Behavior
	// material weakens as it approaches melting point (approx 1800K for steel-base)
	FixedMathCore melting_point(7730941132800LL, true); // 1800 K
	FixedMathCore integrity_factor = wp::clamp((melting_point - r_temperature) / melting_point, zero, one);
	r_yield_strength *= integrity_factor;

	// 3. Spectral Incandescence (Black-body Glow)
	if (r_temperature > FixedMathCore(800LL, false)) {
		FixedMathCore glow_intensity = (r_temperature - FixedMathCore(800LL, false)) / FixedMathCore(1000LL, false);
		Vector3f heat_color(one, FixedMathCore(2147483648LL, true), FixedMathCore(858993459LL, true)); // Orange-red glow
		
		if (p_is_anime) {
			// Anime Style: Quantize the glow into sharp spectral bands
			FixedMathCore step = wp::step(FixedMathCore(2147483648LL, true), glow_intensity);
			r_emission = heat_color * (step * FixedMathCore(5LL, false));
		} else {
			r_emission += heat_color * glow_intensity;
		}
	}
}

/**
 * resolve_spectral_conduction_sweep()
 * 
 * Parallel sweep to diffuse heat between adjacent material tensors.
 * Uses EnTT adjacency streams to map zero-copy heat transfer.
 */
void resolve_spectral_conduction_sweep(
		const BigIntCore &p_total_entities,
		FixedMathCore *r_temperatures,
		const uint32_t *p_adjacency_map, // Indices of neighboring entities
		const uint32_t p_neighbors_per_entity,
		const FixedMathCore &p_conductivity,
		const FixedMathCore &p_delta) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_total_entities.to_string()));
	uint32_t worker_threads = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk_size = total / worker_threads;

	for (uint32_t w = 0; w < worker_threads; w++) {
		uint64_t start = w * chunk_size;
		uint64_t end = (w == worker_threads - 1) ? total : (w + 1) * chunk_size;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				FixedMathCore local_temp = r_temperatures[i];
				FixedMathCore conduction_accum = MathConstants<FixedMathCore>::zero();

				for (uint32_t n = 0; n < p_neighbors_per_entity; n++) {
					uint32_t neighbor_idx = p_adjacency_map[i * p_neighbors_per_entity + n];
					if (neighbor_idx == 0xFFFFFFFF) continue;

					FixedMathCore neighbor_temp = r_temperatures[neighbor_idx];
					// Fourier's Law: dQ = k * dT
					conduction_accum += (neighbor_temp - local_temp) * p_conductivity;
				}
				
				r_temperatures[i] += conduction_accum * p_delta;
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * apply_albedo_wear_kernel()
 * 
 * Simulates real-time material "Scuffing" or "Oxidation".
 * Changes spectral reflectance based on structural fatigue components.
 */
void apply_albedo_wear_kernel(
		Vector3f &r_albedo,
		const FixedMathCore &p_fatigue,
		const FixedMathCore &p_oxidation_rate) {
	
	FixedMathCore wear_factor = p_fatigue * p_oxidation_rate;
	Vector3f wear_color(FixedMathCore(429496729LL, true), FixedMathCore(429496729LL, true), FixedMathCore(429496729LL, true)); // Dusty grey
	
	r_albedo = r_albedo.lerp(wear_color, wp::min(wear_factor, MathConstants<FixedMathCore>::one()));
}

} // namespace UniversalSolver

--- END OF FILE core/math/spectral_energy_conduction.cpp ---
