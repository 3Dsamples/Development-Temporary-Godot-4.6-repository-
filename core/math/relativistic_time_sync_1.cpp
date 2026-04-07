--- START OF FILE core/math/relativistic_time_sync.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/simulation/simulation_manager.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: TimeDilationResolveKernel
 * 
 * Performs a bit-perfect update of the entity's local clock.
 * 1. Computes Lorentz Factor: gamma = 1 / sqrt(1 - v^2/c^2).
 * 2. Proper Time Delta: dt_proper = dt_universal / gamma.
 * 3. Fractional Accumulation: Manages sub-tick precision in FixedMath.
 * 4. Tick Promotion: Increments BigIntCore clock when fraction >= 1.0.
 */
void time_dilation_resolve_kernel(
		const BigIntCore &p_index,
		BigIntCore &r_local_ticks,
		FixedMathCore &r_fractional_accumulator,
		const Vector3f &p_velocity,
		const FixedMathCore &p_c_sq,
		const FixedMathCore &p_universal_delta,
		bool p_is_anime_dilation) {

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();

	// 1. Calculate Velocity Magnitude and Beta
	FixedMathCore v2 = p_velocity.length_squared();
	// Clamp v^2 to 0.999999c^2 to prevent imaginary time or division by zero
	FixedMathCore beta2 = wp::min(v2 / p_c_sq, FixedMathCore(4294967290LL, true));

	// 2. Resolve Lorentz Factor (Gamma)
	// gamma = 1 / sqrt(1 - beta^2)
	FixedMathCore inv_gamma = (one - beta2).square_root();
	
	// Sophisticated Behavior: Anime Bullet-Time
	// In stylized combat, we multiply the dilation effect to simulate extreme speed.
	if (p_is_anime_dilation) {
		inv_gamma *= FixedMathCore(214748364LL, true); // 0.05x multiplier (20x dilation)
	}

	// 3. Proper Time Step Calculation
	// Proper time is the time experienced by the entity
	FixedMathCore proper_delta = p_universal_delta * inv_gamma;

	// 4. Update Fractional Accumulator (Q32.32)
	r_fractional_accumulator += proper_delta;

	// 5. Promote full seconds/ticks to BigIntCore
	// Ensures we never lose time precision over millions of years of simulation
	if (r_fractional_accumulator >= one) {
		int64_t full_units = r_fractional_accumulator.to_int();
		r_local_ticks += BigIntCore(full_units);
		r_fractional_accumulator -= FixedMathCore(full_units);
	}
}

/**
 * execute_galactic_time_sync_sweep()
 * 
 * The master 120 FPS parallel orchestrator for relativistic clocks.
 * Partitions the EnTT registry and updates every entity's temporal frame.
 */
void execute_galactic_time_sync_sweep(
		KernelRegistry &p_registry,
		const FixedMathCore &p_universal_delta) {

	auto &vel_stream = p_registry.get_stream<Vector3f>(COMPONENT_VELOCITY);
	auto &tick_stream = p_registry.get_stream<BigIntCore>(COMPONENT_LOCAL_TICKS);
	auto &frac_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_TIME_FRACTION);

	uint64_t entity_count = vel_stream.size();
	if (entity_count == 0) return;

	// Physical Constant: Speed of Light squared in Q32.32
	FixedMathCore c_sq = PHYSICS_C * PHYSICS_C;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = entity_count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? entity_count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &vel_stream, &tick_stream, &frac_stream]() {
			for (uint64_t i = start; i < end; i++) {
				// Deterministic Style derivation: 1 in 32 entities use Anime dilation
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 32 == 0);

				time_dilation_resolve_kernel(
					handle,
					tick_stream[i],
					frac_stream[i],
					vel_stream[i],
					c_sq,
					p_universal_delta,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * calculate_causal_latency()
 * 
 * Computes the time difference (Latency) between the observer and target 
 * caused by relativistic dilation. Used for robotic perception offset.
 */
BigIntCore calculate_causal_latency(
		const BigIntCore &p_universal_now,
		const BigIntCore &p_entity_local_ticks) {
	
	// Returns the number of ticks the entity has "traveled into the future"
	// relative to the universal clock.
	return p_universal_now - p_entity_local_ticks;
}

/**
 * synchronize_galactic_clocks()
 * 
 * Post-integration cleanup. Re-aligns sub-tick accumulators to 
 * prevent bit-rounding overflow in long-running simulations.
 */
void synchronize_galactic_clocks(FixedMathCore *r_fractions, uint64_t p_count) {
	FixedMathCore max_frac(429496729600LL, false); // 100.0 threshold
	for (uint64_t i = 0; i < p_count; i++) {
		if (r_fractions[i] > max_frac) {
			// This represents a simulation error; force reset to maintenance mode
			r_fractions[i] = MathConstants<FixedMathCore>::zero();
		}
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/relativistic_time_sync.cpp ---
