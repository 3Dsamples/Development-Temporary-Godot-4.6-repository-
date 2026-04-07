--- START OF FILE core/math/material_softening_kernel.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: MaterialThermalStateKernel
 * 
 * Step-wise update for material property tensors based on local temperature.
 * 1. Thermal Yield Softening: Dynamic reduction of yield strength.
 * 2. Elasticity Modulation: Stiffness (Young's Modulus) decay.
 * 3. Thermal Buckling: Induction of internal stress from expansion.
 * 4. Phase Transition: Integrity collapse at melting point.
 */
void material_thermal_state_kernel(
		const BigIntCore &p_index,
		FixedMathCore &r_dynamic_yield,
		FixedMathCore &r_elasticity,
		FixedMathCore &r_integrity,
		Vector3f &r_internal_stress,
		const FixedMathCore &p_temperature,
		const FixedMathCore &p_melting_point,
		const FixedMathCore &p_base_yield,
		const FixedMathCore &p_base_elasticity,
		const FixedMathCore &p_expansion_coeff,
		const FixedMathCore &p_delta,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	
	// Ambient reference temperature (293.15 K) in Q32.32
	FixedMathCore t_ref(12591030272LL, true); 

	// 1. Calculate Thermal Saturation Ratio [0.0, 1.0]
	// t_ratio = (T - T_ref) / (T_melt - T_ref)
	FixedMathCore t_range = p_melting_point - t_ref;
	if (unlikely(t_range.get_raw() <= 0)) t_range = FixedMathCore(1LL, false); // Safety floor
	
	FixedMathCore t_ratio = wp::clamp((p_temperature - t_ref) / t_range, zero, one);

	// 2. Deterministic Softening Curve
	// yield_mult = (1.0 - t_ratio^2)
	// This ensures rapid structural failure as the material approaches molten state.
	FixedMathCore softening_mult = one - (t_ratio * t_ratio);
	
	// --- Sophisticated Behavior: Realistic vs Anime ---
	if (p_is_anime) {
		// Anime Technique: "Thermal Snapping". 
		// Instead of a smooth curve, yield strength is maintained until 
		// specific "Damage Tiers" (0.5, 0.8, 0.95), then drops instantly.
		FixedMathCore tier1(2147483648LL, true); // 0.5
		FixedMathCore tier2(3435973836LL, true); // 0.8
		
		if (t_ratio > tier2) softening_mult = FixedMathCore(42949673LL, true); // 0.01 (Near liquid)
		else if (t_ratio > tier1) softening_mult = FixedMathCore(1073741824LL, true); // 0.25 (Pliable)
		else softening_mult = one; // Rigid
	}

	r_dynamic_yield = p_base_yield * softening_mult;
	r_elasticity = p_base_elasticity * softening_mult;

	// 3. Thermal Buckling Resolve
	// stress_expansion = Elasticity * Alpha * delta_T
	FixedMathCore dt = p_temperature - t_ref;
	FixedMathCore expansion_stress_mag = r_elasticity * p_expansion_coeff * dt;
	
	// Induce internal stress tensors in local Z (assumed buckling axis for thin shells)
	r_internal_stress.z += expansion_stress_mag * p_delta;

	// 4. Integrity Decay / Phase Transition
	// If thermal stress exceeds dynamic yield, the material fractures/buckles.
	if (expansion_stress_mag > r_dynamic_yield && r_dynamic_yield > zero) {
		FixedMathCore damage = (expansion_stress_mag / r_dynamic_yield) * p_delta;
		r_integrity = (r_integrity > damage) ? r_integrity - damage : zero;
	}

	// Melt Trigger: Instant integrity collapse at melting point.
	if (t_ratio >= one) {
		r_integrity = zero;
		r_dynamic_yield = zero;
	}
}

/**
 * execute_thermal_softening_sweep()
 * 
 * Orchestrates the parallel 120 FPS material wave.
 * Zero-copy: Operates directly on the EnTT registry streams for all physical bodies.
 */
void execute_thermal_softening_sweep(
		KernelRegistry &p_registry,
		const FixedMathCore &p_delta) {

	auto &temp_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_TEMPERATURE);
	auto &yield_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_DYNAMIC_YIELD);
	auto &elastic_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_ELASTICITY);
	auto &integrity_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_INTEGRITY);
	auto &stress_stream = p_registry.get_stream<Vector3f>(COMPONENT_STRESS_TENSOR);

	uint64_t count = temp_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &temp_stream, &yield_stream, &elastic_stream, &integrity_stream, &stress_stream]() {
			// Material Constant Tensors (Steel-Alloy base)
			const FixedMathCore base_yield(350000000LL, false); // 350 MPa
			const FixedMathCore base_elastic(200000000000LL, false); // 200 GPa
			const FixedMathCore melt_point(1800LL << 32, true); // 1800 K
			const FixedMathCore alpha("0.000012"); // Thermal expansion coeff

			for (uint64_t i = start; i < end; i++) {
				// Style derived from entity handle for bit-perfect look-consistency
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 14 == 0);

				material_thermal_state_kernel(
					handle,
					yield_stream[i],
					elastic_stream[i],
					integrity_stream[i],
					stress_stream[i],
					temp_stream[i],
					melt_point,
					base_yield,
					base_elastic,
					alpha,
					p_delta,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/material_softening_kernel.cpp ---
