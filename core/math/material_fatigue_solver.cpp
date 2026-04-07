--- START OF FILE core/math/material_fatigue_solver.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: MaterialFailureKernel
 * 
 * Evaluates the structural health of a geometric component.
 * 1. Calculates von Mises Equivalent Stress from the internal stress tensor.
 * 2. Accumulates fatigue damage based on cyclic loading.
 * 3. Triggers permanent "Set" or "Fracture" state based on yield strength.
 */
void material_failure_kernel(
		const BigIntCore &p_index,
		FixedMathCore &r_integrity,
		FixedMathCore &r_fatigue,
		FixedMathCore &r_yield_strength,
		const Vector3f &p_stress_tensor_diag, // Principal stresses
		const FixedMathCore &p_toughness,
		const FixedMathCore &p_delta) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore two = FixedMathCore(2LL, false);

	// 1. Calculate von Mises Equivalent Stress (Deterministic)
	// sigma_e = sqrt(0.5 * ((s1-s2)^2 + (s2-s3)^2 + (s3-s1)^2))
	FixedMathCore s1 = p_stress_tensor_diag.x;
	FixedMathCore s2 = p_stress_tensor_diag.y;
	FixedMathCore s3 = p_stress_tensor_diag.z;

	FixedMathCore diff1 = s1 - s2;
	FixedMathCore diff2 = s2 - s3;
	FixedMathCore diff3 = s3 - s1;

	FixedMathCore sum_sq = (diff1 * diff1) + (diff2 * diff2) + (diff3 * diff3);
	FixedMathCore equivalent_stress = (MathConstants<FixedMathCore>::half() * sum_sq).square_root();

	// 2. Fatigue Accumulation (Miner's Rule Approximation)
	// Damage += (Stress / Yield)^FatigueExponent * dt
	// We use a fixed exponent of 3 for common ductile metals in FixedMath
	FixedMathCore stress_ratio = equivalent_stress / (r_yield_strength + FixedMathCore(1LL, true));
	FixedMathCore damage_cycle = stress_ratio * stress_ratio * stress_ratio;
	
	r_fatigue += damage_cycle * p_delta;

	// 3. Structural Integrity Decay
	// Integrity is the inverse of fatigue scaled by material toughness
	FixedMathCore integrity_loss = r_fatigue / p_toughness;
	r_integrity = (integrity_loss < one) ? (one - integrity_loss) : zero;

	// 4. Brittle Transition (High-Speed Behavior)
	// If yield strength is exceeded, the material "softens" (plasticity)
	if (equivalent_stress > r_yield_strength) {
		FixedMathCore excess = equivalent_stress - r_yield_strength;
		r_yield_strength -= excess * FixedMathCore(42949673LL, true); // 0.01 hardening/softening factor
	}
}

/**
 * solve_material_fatigue_batch()
 * 
 * Parallel sweep over EnTT material components.
 * Processes millions of micro-stresses per frame for 120 FPS high-fidelity destruction.
 */
void solve_material_fatigue_batch(
		FixedMathCore *r_integrity_stream,
		FixedMathCore *r_fatigue_stream,
		FixedMathCore *r_yield_stream,
		const Vector3f *p_stress_tensors,
		uint64_t p_count,
		const FixedMathCore &p_toughness_constant,
		const FixedMathCore &p_delta) {

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = p_count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? p_count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				material_failure_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_integrity_stream[i],
					r_fatigue_stream[i],
					r_yield_stream[i],
					p_stress_tensors[i],
					p_toughness_constant,
					p_delta
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * check_fracture_condition()
 * 
 * Returns a list of entity indices that have failed structurally.
 * Used to trigger the Voronoi Shatter Kernel for real-time destruction.
 */
void check_fracture_condition(
		const FixedMathCore *p_integrity_stream,
		uint64_t p_count,
		Vector<uint64_t> &r_failed_indices) {
	
	FixedMathCore failure_threshold(429496730LL, true); // 0.1 integrity remains

	for (uint64_t i = 0; i < p_count; i++) {
		if (p_integrity_stream[i] < failure_threshold) {
			r_failed_indices.push_back(i);
		}
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/material_fatigue_solver.cpp ---
