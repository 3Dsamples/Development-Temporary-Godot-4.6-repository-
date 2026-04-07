--- START OF FILE core/math/material_softening_kernel.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: ThermalYieldSofteningKernel
 * 
 * Simulates the physical weakening of a material due to heat.
 * 1. Normalized Temperature: Calculate how close the material is to melting.
 * 2. Softening Curve: Yield strength decays via a bit-perfect quadratic ramp.
 * 3. Thermal Buckling: Induces internal stress from uneven thermal expansion.
 * 4. Phase Transition: Flags the entity as 'Molten' if T > T_melt.
 */
void material_thermal_yield_kernel(
		const BigIntCore &p_index,
		FixedMathCore &r_yield_strength,
		FixedMathCore &r_elasticity,
		FixedMathCore &r_integrity,
		const FixedMathCore &p_temperature,
		const FixedMathCore &p_melting_point,
		const FixedMathCore &p_base_yield,
		const FixedMathCore &p_thermal_expansion_coeff,
		const FixedMathCore &p_delta,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Calculate Thermal Saturation [0..1]
	FixedMathCore t_norm = wp::clamp(p_temperature / p_melting_point, zero, one);

	// 2. Deterministic Softening: Yield = BaseYield * (1 - t_norm^2)
	// This models the rapid loss of structural integrity near the melting point.
	FixedMathCore softening_factor = one - (t_norm * t_norm);
	
	if (p_is_anime) {
		// Anime Style: Banded softening. Strength stays high, then snaps at specific heat thresholds.
		FixedMathCore threshold(2147483648LL, true); // 0.5
		softening_factor = wp::step(threshold, softening_factor) * one + (one - wp::step(threshold, softening_factor)) * FixedMathCore(429496730LL, true);
	}

	r_yield_strength = p_base_yield * softening_factor;

	// 3. Dynamic Elasticity (The "Gooey" Behavior)
	// As material softens, elasticity (stiffness) decreases, creating a "Molten" or "Balloon" effect.
	r_elasticity = MathConstants<FixedMathCore>::half() * softening_factor;

	// 4. Thermal Stress Accumulation
	// sigma_thermal = E * alpha * delta_T
	// High delta_T in rigid materials triggers structural failure (buckling).
	FixedMathCore thermal_stress = r_elasticity * p_thermal_expansion_coeff * p_temperature;
	
	if (thermal_stress > r_yield_strength && r_yield_strength > zero) {
		FixedMathCore damage = (thermal_stress / r_yield_strength) * p_delta;
		r_integrity = (r_integrity > damage) ? r_integrity - damage : zero;
	}

	// 5. Phase Transition logic
	// If T >= T_melt, structural integrity is bypassed (object is now a fluid/gas).
	if (t_norm >= one) {
		r_integrity = zero;
		r_yield_strength = zero;
	}
}

/**
 * execute_thermal_softening_sweep()
 * 
 * Master parallel sweep for material state updates.
 * Processes EnTT SoA streams for temperature and structural tensors.
 */
void execute_thermal_softening_sweep(
		const BigIntCore &p_entity_count,
		FixedMathCore *r_yield_strengths,
		FixedMathCore *r_elasticities,
		FixedMathCore *r_integrities,
		const FixedMathCore *p_temperatures,
		const FixedMathCore *p_melting_points,
		const FixedMathCore *p_base_yields,
		const FixedMathCore &p_expansion_coeff,
		const FixedMathCore &p_delta) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_entity_count.to_string()));
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? total : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				// Style flag: use entity hash for deterministic anime look
				bool anime_mode = (i % 8 == 0);

				material_thermal_yield_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_yield_strengths[i],
					r_elasticities[i],
					r_integrities[i],
					p_temperatures[i],
					p_melting_points[i],
					p_base_yields[i],
					p_expansion_coeff,
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
