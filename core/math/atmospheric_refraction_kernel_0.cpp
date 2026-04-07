--- START OF FILE core/math/atmospheric_refraction_kernel.cpp ---

#include "core/math/atmospheric_scattering.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_refractive_index()
 * 
 * Computes the index of refraction (n) based on Gladstone-Dale relation.
 * n = 1 + k * rho, where rho is the local FixedMath density.
 */
static _FORCE_INLINE_ FixedMathCore calculate_refractive_index(
		FixedMathCore p_density, 
		const FixedMathCore &p_refractivity_const) {
	
	// refractivity_const for air is approx 0.000293
	return MathConstants<FixedMathCore>::one() + (p_density * p_refractivity_const);
}

/**
 * Warp Kernel: AtmosphericRefractionKernel
 * 
 * Simulates the bending of light rays as they pass through varying density layers.
 * 1. Computes the density gradient (surface normal).
 * 2. Applies the vector form of Snell's Law.
 * 3. Injects "Mirage Tensors" for temperature-inversion layers (Anime/Realistic).
 */
void atmospheric_refraction_kernel(
		const BigIntCore &p_index,
		Vector3f &r_ray_direction,
		const Vector3f &p_sample_pos,
		const FixedMathCore &p_density,
		const FixedMathCore &p_next_density,
		const AtmosphereParams &p_params,
		bool p_is_anime) {

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	
	// 1. Determine local Refractive Indices
	FixedMathCore n1 = calculate_refractive_index(p_density, p_params.refractivity_k);
	FixedMathCore n2 = calculate_refractive_index(p_next_density, p_params.refractivity_k);
	
	FixedMathCore eta = n1 / n2;

	// 2. Resolve Gradient Normal (points toward planet center)
	Vector3f normal = (-p_sample_pos).normalized();

	// 3. Vector Snell's Law: r = eta*v + (eta*cos1 - sqrt(1 - eta^2*(1 - cos1^2))) * n
	FixedMathCore cos1 = -normal.dot(r_ray_direction);
	FixedMathCore sin2_theta2 = eta * eta * (one - cos1 * cos1);

	// Total internal reflection check (unlikely in atmosphere but kept for bit-perfection)
	if (sin2_theta2 > one) return;

	FixedMathCore cos2 = Math::sqrt(one - sin2_theta2);
	r_ray_direction = (r_ray_direction * eta) + (normal * (eta * cos1 - cos2));
	r_ray_direction = r_ray_direction.normalized();

	// --- Sophisticated Behavior: Mirage / Fata Morgana ---
	if (p_is_anime) {
		// Anime Style: Quantize refraction into discrete "Heat Ripple" bands
		// This creates the iconic "staircase" distortion seen in stylized horizons
		FixedMathCore ripple = wp::sin(p_sample_pos.y * FixedMathCore(10LL, false));
		if (ripple > FixedMathCore(2147483648LL, true)) { // 0.5 threshold
			r_ray_direction.y += FixedMathCore(42949673LL, true); // 0.01 nudge
		}
	}
}

/**
 * execute_refractive_ray_march()
 * 
 * Parallel sweep to bend view rays through the atmosphere.
 * Necessary for realistic "Horizon Looming" where the sun stays visible 
 * after it has technically set.
 */
void execute_refractive_ray_march(
		const BigIntCore &p_count,
		Vector3f *r_ray_directions,
		const Vector3f *p_sample_positions,
		const FixedMathCore *p_densities,
		const AtmosphereParams &p_params) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_count.to_string()));
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? total : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &p_params]() {
			for (uint64_t i = start; i < end; i++) {
				// Use p_densities[i] and a look-ahead for p_next_density
				FixedMathCore d_next = p_densities[i] * FixedMathCore(4252017623LL, true); // 0.99 approx
				
				atmospheric_refraction_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_ray_directions[i],
					p_sample_positions[i],
					p_densities[i],
					d_next,
					p_params,
					(i % 16 == 0) // Deterministic Anime Style
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/atmospheric_refraction_kernel.cpp ---
