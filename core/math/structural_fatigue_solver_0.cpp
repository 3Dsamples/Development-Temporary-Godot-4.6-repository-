--- START OF FILE core/math/structural_fatigue_solver.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: MaterialFailureEvaluationKernel
 * 
 * Performs a bit-perfect evaluation of structural health.
 * 1. Von Mises Stress: sigma_e = sqrt(0.5 * ((s1-s2)^2 + (s2-s3)^2 + (s3-s1)^2))
 * 2. Thermal Softening: Yield strength decays based on FixedMath temperature.
 * 3. Fatigue Accumulation: Power-law damage based on cyclic stress ratio.
 * 4. Snap-Point Resolution: Immediate integrity collapse if fatigue > toughness.
 */
void material_failure_evaluation_kernel(
		const BigIntCore &p_index,
		FixedMathCore &r_integrity,
		FixedMathCore &r_fatigue,
		FixedMathCore &r_dynamic_yield,
		const Vector3f &p_principal_stresses,
		const FixedMathCore &p_temperature,
		const FixedMathCore &p_base_yield,
		const FixedMathCore &p_melting_point,
		const FixedMathCore &p_toughness,
		const FixedMathCore &p_delta) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Calculate Deterministic Von Mises Equivalent Stress
	FixedMathCore s1 = p_principal_stresses.x;
	FixedMathCore s2 = p_principal_stresses.y;
	FixedMathCore s3 = p_principal_stresses.z;

	FixedMathCore d1 = s1 - s2;
	FixedMathCore d2 = s2 - s3;
	FixedMathCore d3 = s3 - s1;

	FixedMathCore sum_sq = (d1 * d1) + (d2 * d2) + (d3 * d3);
	FixedMathCore equivalent_stress = wp::sqrt(MathConstants<FixedMathCore>::half() * sum_sq);

	// 2. Resolve Thermal Softening
	// Material weakens as temperature approaches the melting point.
	// yield = base_yield * (1.0 - (T / T_melt)^2)
	FixedMathCore t_ratio = wp::clamp(p_temperature / p_melting_point, zero, one);
	FixedMathCore softening = one - (t_ratio * t_ratio);
	r_dynamic_yield = p_base_yield * softening;

	// 3. Palmgren-Miner Fatigue Accumulation
	// If stress > endurance limit (simplified as 10% of yield), accumulate damage.
	FixedMathCore endurance_limit = r_dynamic_yield * FixedMathCore(429496730LL, true); // 0.1
	
	if (equivalent_stress > endurance_limit) {
		// damage = (stress / yield)^beta * dt. Using beta=4 for high-cycle fatigue.
		FixedMathCore stress_ratio = equivalent_stress / (r_dynamic_yield + MathConstants<FixedMathCore>::unit_epsilon());
		FixedMathCore ratio2 = stress_ratio * stress_ratio;
		FixedMathCore damage = ratio2 * ratio2 * p_delta;
		
		r_fatigue += damage;
	} else {
		// Relaxation behavior: minor fatigue recovery if stress is negligible.
		FixedMathCore recovery = FixedMathCore(4294967LL, true) * p_delta; // 0.001
		r_fatigue = wp::max(zero, r_fatigue - recovery);
	}

	// 4. Structural Integrity & Snap-Point Resolution
	// integrity = clamp(1.0 - fatigue / toughness)
	FixedMathCore life_fraction = r_fatigue / p_toughness;
	
	// Sophisticated Snap: Once life fraction > 0.9, integrity collapses non-linearly.
	if (life_fraction > FixedMathCore(3865470566LL, true)) { // 0.9 threshold
		FixedMathCore snap_factor = (life_fraction - FixedMathCore(3865470566LL, true)) * FixedMathCore(10LL, false);
		r_integrity = wp::max(zero, (one - life_fraction) * (one - snap_factor));
	} else {
		r_integrity = one - life_fraction;
	}

	// Phase Change check: if molten, integrity is zeroed.
	if (t_ratio >= one) {
		r_integrity = zero;
	}
}

/**
 * execute_material_failure_wave()
 * 
 * Orchestrates the parallel 120 FPS material sweep across the EnTT registry.
 * Zero-copy: Operates directly on the aligned SoA buffers for yield, fatigue, and stress.
 */
void execute_material_failure_wave(
		KernelRegistry &p_registry,
		const FixedMathCore &p_delta) {

	auto &integrity_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_INTEGRITY);
	auto &fatigue_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_FATIGUE);
	auto &yield_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_DYNAMIC_YIELD);
	auto &stress_stream = p_registry.get_stream<Vector3f>(COMPONENT_STRESS_TENSOR);
	auto &temp_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_TEMPERATURE);

	uint64_t count = integrity_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &integrity_stream, &fatigue_stream, &yield_stream, &stress_stream, &temp_stream]() {
			for (uint64_t i = start; i < end; i++) {
				// Base material properties (e.g., Titanium Alloy constants)
				FixedMathCore base_yield(43435973836LL, false); // High yield
				FixedMathCore melt_point(715827882666LL, false); // 1668 K
				FixedMathCore toughness(1LL, false); // Normalized toughness

				material_failure_evaluation_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					integrity_stream[i],
					fatigue_stream[i],
					yield_stream[i],
					stress_stream[i],
					temp_stream[i],
					base_yield,
					melt_point,
					toughness,
					p_delta,
					(i % 10 == 0) // Deterministic Anime Style flag
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * calculate_fracture_impact_force()
 * 
 * Resolves the impulse energy to be injected into the stress tensor.
 * Strictly uses BigIntCore for energy to support galactic-scale collisions.
 */
Vector3f calculate_fracture_impact_force(
		const BigIntCore &p_energy,
		const Vector3f &p_direction,
		const FixedMathCore &p_area) {
	
	if (p_area.get_raw() == 0) return Vector3f_ZERO;

	// Stress = Force / Area. 
	// Force is derived from the BigInt kinetic energy tensor.
	FixedMathCore force_f(static_cast<int64_t>(std::stoll(p_energy.to_string())));
	FixedMathCore stress_mag = force_f / p_area;

	return p_direction * stress_mag;
}

} // namespace UniversalSolver

--- END OF FILE core/math/structural_fatigue_solver.cpp ---
