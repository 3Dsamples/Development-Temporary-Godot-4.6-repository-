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
 * 1. Von Mises Stress: Calculates effective stress from principal components.
 * 2. Thermal Softening: Reduces yield strength based on temperature.
 * 3. Fatigue Accumulation: Power-law damage based on cyclic stress ratio.
 * 4. Snap-Point Resolution: Immediate integrity collapse if damage > toughness.
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
	// sigma_e = sqrt(0.5 * ((s1-s2)^2 + (s2-s3)^2 + (s3-s1)^2))
	FixedMathCore s1 = p_principal_stresses.x;
	FixedMathCore s2 = p_principal_stresses.y;
	FixedMathCore s3 = p_principal_stresses.z;

	FixedMathCore d1 = s1 - s2;
	FixedMathCore d2 = s2 - s3;
	FixedMathCore d3 = s3 - s1;

	FixedMathCore sum_sq = (d1 * d1) + (d2 * d2) + (d3 * d3);
	FixedMathCore equivalent_stress = wp::sqrt(MathConstants<FixedMathCore>::half() * sum_sq);

	// 2. Resolve Thermal Softening
	// yield = base_yield * (1.0 - (T / T_melt)^2)
	FixedMathCore t_ratio = wp::clamp(p_temperature / (p_melting_point + FixedMathCore(1LL, true)), zero, one);
	FixedMathCore softening = one - (t_ratio * t_ratio);
	r_dynamic_yield = p_base_yield * softening;

	// 3. Palmgren-Miner Fatigue Accumulation
	// Only accumulate damage if stress exceeds the endurance limit (10% of yield)
	FixedMathCore endurance_limit = r_dynamic_yield * FixedMathCore(429496730LL, true); 
	
	if (equivalent_stress > endurance_limit) {
		// damage = (stress / yield)^beta * dt. Using beta=4 for high-cycle fatigue.
		FixedMathCore stress_ratio = equivalent_stress / (r_dynamic_yield + FixedMathCore(1LL, true));
		FixedMathCore ratio2 = stress_ratio * stress_ratio;
		FixedMathCore damage = ratio2 * ratio2 * p_delta;
		
		r_fatigue += damage;
	} else {
		// Relaxation behavior: minor structural recovery for ductile materials
		FixedMathCore recovery = FixedMathCore(4294967LL, true) * p_delta; // 0.001
		r_fatigue = wp::max(zero, r_fatigue - recovery);
	}

	// 4. Structural Integrity & Snap-Point Resolution
	// integrity = clamp(1.0 - fatigue / toughness)
	FixedMathCore life_fraction = r_fatigue / (p_toughness + FixedMathCore(1LL, true));
	
	// Deterministic Snap: Non-linear collapse when reaching critical fatigue
	if (life_fraction > FixedMathCore(3865470566LL, true)) { // 0.9 threshold
		FixedMathCore snap_multiplier = (life_fraction - FixedMathCore(3865470566LL, true)) * FixedMathCore(10LL, false);
		r_integrity = wp::max(zero, (one - life_fraction) * (one - snap_multiplier));
	} else {
		r_integrity = one - life_fraction;
	}

	// If melting point is reached, integrity is instantly zeroed
	if (t_ratio >= one) {
		r_integrity = zero;
	}
}

/**
 * execute_material_failure_wave()
 * 
 * Orchestrates the parallel 120 FPS material sweep across the EnTT registry.
 * Zero-copy: Operates directly on the aligned SoA buffers.
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
			// Material Constant Tensors (e.g. Carbon-Steel base)
			const FixedMathCore base_yield(250000000LL, false); // 250 MPa
			const FixedMathCore melt_point(1811LL << 32, true);  // 1811 K
			const FixedMathCore toughness(100LL, false);        // Joules/m^2 scale

			for (uint64_t i = start; i < end; i++) {
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
					(i % 12 == 0) // Stylized Anime snap flag
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * calculate_fracture_impulse_energy()
 * 
 * Maps BigIntCore kinetic energy into FixedMathCore stress magnitude.
 * Used during collision epicenters to calculate the initial stress spike.
 */
FixedMathCore calculate_fracture_impulse_energy(
		const BigIntCore &p_kinetic_energy,
		const FixedMathCore &p_surface_area) {
	
	if (unlikely(p_surface_area.get_raw() == 0)) return MathConstants<FixedMathCore>::zero();

	// Force = sqrt(2 * Energy * mass) / dt (Simplified for stress mapping)
	// We convert the BigInt energy to FixedMath to resolve the local stress intensity.
	FixedMathCore energy_f(static_cast<int64_t>(std::stoll(p_kinetic_energy.to_string())));
	FixedMathCore stress_magnitude = energy_f / p_surface_area;

	return stress_magnitude;
}

} // namespace UniversalSolver

--- END OF FILE core/math/structural_fatigue_solver.cpp ---
