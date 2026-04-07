--- START OF FILE core/math/relativistic_time_sync.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: RelativisticClockSyncKernel
 * 
 * Synchronizes the local clock of an entity with the Universal Galactic Time.
 * 1. Calculates the Lorentz Factor based on bit-perfect velocity.
 * 2. Integrates Local Proper Time (LPT) using dilated deltas.
 * 3. Handles "Time-Jumping" for entities exceeding warp thresholds.
 */
void relativistic_clock_sync_kernel(
		const BigIntCore &p_index,
		BigIntCore &r_local_ticks,
		FixedMathCore &r_proper_time_fraction,
		const Vector3f &p_velocity,
		const FixedMathCore &p_c_sq,
		const FixedMathCore &p_universal_delta,
		bool p_is_anime_dilation) {

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	
	// 1. Calculate Lorentz Factor (Gamma)
	// beta^2 = v^2 / c^2
	FixedMathCore v2 = p_velocity.length_squared();
	FixedMathCore beta2 = wp::min(v2 / p_c_sq, FixedMathCore(4294967290LL, true)); // Clamp below c
	
	// gamma = 1 / sqrt(1 - beta^2)
	FixedMathCore gamma = one / (one - beta2).square_root();

	// 2. Sophisticated Behavior: Anime Time Dilation
	// In stylized mode, high acceleration or "special moves" trigger an extra
	// temporal multiplier to simulate "bullet time" or dramatic slowdowns.
	if (p_is_anime_dilation) {
		FixedMathCore style_multiplier = FixedMathCore(5LL, false); // 5x dilation boost
		gamma *= style_multiplier;
	}

	// 3. Proper Time Integration (dt_proper = dt_world / gamma)
	FixedMathCore proper_delta = p_universal_delta / gamma;
	r_proper_time_accum += proper_delta;

	// 4. Proper Time to BigInt Tick Transition
	// When the fractional part exceeds 1.0, we increment the discrete BigInt tick count.
	// This prevents floating-point clock drift over years of real-time simulation.
	if (r_proper_time_accum >= one) {
		int64_t full_ticks = r_proper_time_accum.to_int();
		r_local_ticks += BigIntCore(full_ticks);
		r_proper_time_accum -= FixedMathCore(full_ticks);
	}
}

/**
 * execute_galactic_time_sweep()
 * 
 * Parallel sweep over all EnTT entities to synchronize temporal frames.
 * Ensures that every robot, spaceship, and planet is in the correct 
 * relativistic state before the 120 FPS physics sweep.
 */
void execute_galactic_time_sweep(
		BigIntCore *r_entity_ticks,
		FixedMathCore *r_entity_fractions,
		const Vector3f *p_velocities,
		uint64_t p_count,
		const FixedMathCore &p_c,
		const FixedMathCore &p_delta) {

	FixedMathCore c2 = p_c * p_c;
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = p_count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? p_count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				// Deterministic style selection based on entity handle
				bool anime_dilation = (i % 32 == 0); 

				relativistic_clock_sync_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_entity_ticks[i],
					r_entity_fractions[i],
					p_velocities[i],
					c2,
					p_delta,
					anime_dilation
				);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * calculate_time_divergence()
 * 
 * Returns the BigIntCore difference between the Universal Clock and an Entity Clock.
 * Used for AI prediction and sensor lag simulation in robotic entities.
 */
BigIntCore calculate_time_divergence(
		const BigIntCore &p_universal_ticks,
		const BigIntCore &p_entity_ticks) {
	
	return p_universal_ticks - p_entity_ticks;
}

} // namespace UniversalSolver

--- END OF FILE core/math/relativistic_time_sync.cpp ---
