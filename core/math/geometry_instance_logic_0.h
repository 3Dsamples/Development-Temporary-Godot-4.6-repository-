--- START OF FILE core/math/geometry_instance_logic.h ---

#ifndef GEOMETRY_INSTANCE_LOGIC_H
#define GEOMETRY_INSTANCE_LOGIC_H

#include "core/math/geometry_instance.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * GeometryInstanceLogic
 * 
 * Static logic kernels for processing batches of GeometryInstances.
 * Designed to be called by Warp-style execution sweeps over EnTT registries.
 * Eliminates per-object logic overhead to maintain 120 FPS simulation throughput.
 */
class GeometryInstanceLogic {
public:
	// ------------------------------------------------------------------------
	// Material Simulation Kernels (Deterministic)
	// ------------------------------------------------------------------------

	/**
	 * process_fatigue_relaxation()
	 * Simulates the elastic recovery of material stress over time.
	 * Kernel-ready for parallel execution in the SimulationThreadPool.
	 */
	static _FORCE_INLINE_ void process_fatigue_relaxation(FixedMathCore &r_integrity, FixedMathCore &r_fatigue, const FixedMathCore &p_recovery_rate, const FixedMathCore &p_delta) {
		if (r_fatigue > MathConstants<FixedMathCore>::zero()) {
			FixedMathCore relaxation = p_recovery_rate * p_delta;
			r_fatigue = (r_fatigue > relaxation) ? r_fatigue - relaxation : MathConstants<FixedMathCore>::zero();
			
			// Integrity recovers inversely to fatigue
			r_integrity = MathConstants<FixedMathCore>::one() - r_fatigue;
		}
	}

	/**
	 * process_thermal_conduction()
	 * Simulates heat dissipation into the environment (convection/radiation).
	 */
	static _FORCE_INLINE_ void process_thermal_conduction(FixedMathCore &r_temperature, const FixedMathCore &p_ambient, const FixedMathCore &p_conductivity, const FixedMathCore &p_delta) {
		FixedMathCore temp_diff = r_temperature - p_ambient;
		r_temperature -= temp_diff * p_conductivity * p_delta;
	}

	// ------------------------------------------------------------------------
	// Galactic Scale Management Kernels
	// ------------------------------------------------------------------------

	/**
	 * apply_galactic_correction()
	 * The "Drift Correction" kernel. Checks if local coordinates exceed precision limits.
	 * If so, shifts the BigIntCore sector coordinates and recenters the local origin.
	 */
	static void apply_galactic_correction(Vector3f &r_local_pos, BigIntCore &r_sx, BigIntCore &r_sy, BigIntCore &r_sz, const FixedMathCore &p_threshold);

	// ------------------------------------------------------------------------
	// State Machine Logic
	// ------------------------------------------------------------------------

	/**
	 * evaluate_simulation_state()
	 * Determines the current SimulationState based on integrity and stress levels.
	 */
	static _FORCE_INLINE_ GeometryInstance::SimulationState evaluate_simulation_state(const FixedMathCore &p_integrity, const FixedMathCore &p_yield_strength) {
		if (p_integrity <= MathConstants<FixedMathCore>::zero()) {
			return GeometryInstance::STATE_DESTROYED;
		}
		if (p_integrity < FixedMathCore(1288490188LL, true)) { // 0.3 Threshold
			return GeometryInstance::STATE_FRACTURING;
		}
		if (p_integrity < FixedMathCore(3435973836LL, true)) { // 0.8 Threshold
			return GeometryInstance::STATE_DEFORMING;
		}
		return GeometryInstance::STATE_STABLE;
	}
};

#endif // GEOMETRY_INSTANCE_LOGIC_H

--- END OF FILE core/math/geometry_instance_logic.h ---
