--- START OF FILE core/math/structural_fatigue_kernel.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: StructuralFatigueKernel
 * 
 * Simulates the accumulation of damage due to cyclic loading.
 * 1. Endurance Limit: Stresses below this threshold cause zero damage.
 * 2. Damage Increment: Non-linear accumulation based on stress amplitude.
 * 3. Snap-Point Logic: Triggers an immediate state transition to 'Fractured' 
 *    when integrity reaches zero.
 */
void structural_fatigue_kernel(
		const BigIntCore &p_index,
		FixedMathCore &r_fatigue_accum,
		FixedMathCore &r_integrity,
		FixedMathCore &r_yield_strength,
		const FixedMathCore &p_current_stress,
		const FixedMathCore &p_endurance_limit,
		const FixedMathCore &p_toughness,
		const FixedMathCore &p_delta,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Check for valid stress cycle
	if (p_current_stress <= p_endurance_limit) {
		// Relaxation: Materials slightly recover if below endurance limit
		FixedMathCore recovery_rate = FixedMathCore(4294967LL, true); // 0.001
		r_fatigue_accum = wp::max(zero, r_fatigue_accum - (recovery_rate * p_delta));
		return;
	}

	// 2. Calculate Damage Increment (Non-Linear)
	// stress_ratio = current_stress / yield_strength
	FixedMathCore stress_ratio = p_current_stress / (r_yield_strength + FixedMathCore(1LL, true));
	
	// damage = (stress_ratio)^4 * delta (Aggressive exponent for structural snapping)
	FixedMathCore ratio2 = stress_ratio * stress_ratio;
	FixedMathCore damage = ratio2 * ratio2 * p_delta;

	// 3. Accumulate Fatigue
	r_fatigue_accum += damage;

	// 4. Update Integrity
	// Integrity = 1.0 - (Fatigue / Toughness)
	FixedMathCore integrity_loss = r_fatigue_accum / p_toughness;
	r_integrity = (integrity_loss < one) ? (one - integrity_loss) : zero;

	// 5. Sophisticated Behavior: Snap-Point / Softening
	// As integrity drops, the material yield strength decreases (Softening)
	r_yield_strength = r_yield_strength * (one - (damage * FixedMathCore(10LL, false)));

	// --- Anime Style Snap ---
	if (p_is_anime) {
		// Quantize integrity for dramatic "Shield Crack" or "Armor Snap" effects
		FixedMathCore threshold(1073741824LL, true); // 0.25
		if (r_integrity < threshold && r_integrity > zero) {
			// Instant snap to near-failure for dramatic impact
			r_integrity = FixedMathCore(42949673LL, true); // 0.01
			r_fatigue_accum = p_toughness;
		}
	}
}

/**
 * execute_fatigue_simulation_sweep()
 * 
 * Master parallel sweep for material health.
 * Processes EnTT SoA streams for fatigue, integrity, and local stress.
 */
void execute_fatigue_simulation_sweep(
		const BigIntCore &p_entity_count,
		FixedMathCore *r_fatigue_stream,
		FixedMathCore *r_integrity_stream,
		FixedMathCore *r_yield_stream,
		const FixedMathCore *p_stress_stream,
		const FixedMathCore *p_endurance_stream,
		const FixedMathCore &p_toughness_const,
		const FixedMathCore &p_delta) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_entity_count.to_string()));
	uint32_t worker_threads = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk_size = total / worker_threads;

	for (uint32_t w = 0; w < worker_threads; w++) {
		uint64_t start = w * chunk_size;
		uint64_t end = (w == worker_threads - 1) ? total : (w + 1) * chunk_size;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				// Deterministic style assignment
				bool is_anime = (i % 10 == 0);

				structural_fatigue_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_fatigue_stream[i],
					r_integrity_stream[i],
					r_yield_stream[i],
					p_stress_stream[i],
					p_endurance_stream[i],
					p_toughness_const,
					p_delta,
					is_anime
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/structural_fatigue_kernel.cpp ---
