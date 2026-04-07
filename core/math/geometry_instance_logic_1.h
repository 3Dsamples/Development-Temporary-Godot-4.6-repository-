--- START OF FILE core/math/geometry_instance_logic.h ---

#ifndef GEOMETRY_INSTANCE_LOGIC_H
#define GEOMETRY_INSTANCE_LOGIC_H

#include "core/math/math_defs.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * GeometryInstanceLogic
 * 
 * Provides sophisticated logical kernels for GeometryInstance physical state.
 * These functions are designed to be invoked by Warp kernels during EnTT sweeps.
 */
class GeometryInstanceLogic {
public:
	// ------------------------------------------------------------------------
	// Simulation Tier Management
	// ------------------------------------------------------------------------

	/**
	 * determine_simulation_tier()
	 * 
	 * High-performance heuristic to decide if an entity should be in
	 * TIER_DETERMINISTIC (Active Physics) or TIER_MACRO (Background Logic).
	 * Based on distance to observer and internal energy tensors.
	 */
	static _FORCE_INLINE_ SimulationTier determine_simulation_tier(
			const FixedMathCore &p_dist_sq,
			const FixedMathCore &p_velocity_sq,
			const FixedMathCore &p_integrity) {

		FixedMathCore macro_threshold(1000000LL, false); // 1km^2
		FixedMathCore active_energy_threshold(100LL, false);

		if (p_dist_sq > macro_threshold && p_velocity_sq < active_energy_threshold && p_integrity == MathConstants<FixedMathCore>::one()) {
			return TIER_MACRO_ECONOMY;
		}
		return TIER_DETERMINISTIC;
	}

	// ------------------------------------------------------------------------
	// Material Behavior Logic
	// ------------------------------------------------------------------------

	/**
	 * apply_deformation_clamping()
	 * 
	 * Ensures that the Balloon and Flesh effects do not exceed the structural
	 * snap-point of the material. strictly deterministic.
	 */
	static _FORCE_INLINE_ void apply_deformation_clamping(
			Vector3f &r_pos,
			const Vector3f &p_rest_pos,
			const FixedMathCore &p_max_stretch) {

		Vector3f diff = r_pos - p_rest_pos;
		FixedMathCore dist = diff.length();
		if (dist > p_max_stretch) {
			r_pos = p_rest_pos + diff.normalized() * p_max_stretch;
		}
	}

	/**
	 * evaluate_fracture_trigger()
	 * 
	 * Determines if the current stress and fatigue levels require the 
	 * body to be converted into shards.
	 */
	static _FORCE_INLINE_ bool evaluate_fracture_trigger(
			const FixedMathCore &p_fatigue,
			const FixedMathCore &p_integrity,
			const FixedMathCore &p_temperature,
			const FixedMathCore &p_melting_point) {

		// Condition 1: Temperature exceeds melting point
		if (p_temperature >= p_melting_point) return true;
		
		// Condition 2: Structural integrity collapsed below brittle threshold (0.05)
		if (p_integrity < FixedMathCore(214748364LL, true)) return true;

		return false;
	}

	// ------------------------------------------------------------------------
	// Galactic Coordinate Sync Logic
	// ------------------------------------------------------------------------

	/**
	 * resolve_galactic_sector_sync()
	 * 
	 * Computes the correct BigIntCore sector coordinates for a local position.
	 * strictly uses bit-perfect coordinate normalization.
	 */
	static _FORCE_INLINE_ void resolve_galactic_sector_sync(
			Vector3f &r_local_pos,
			BigIntCore &r_sx,
			BigIntCore &r_sy,
			BigIntCore &r_sz,
			const FixedMathCore &p_sector_size) {

		int64_t move_x = Math::floor(r_local_pos.x / p_sector_size).to_int();
		int64_t move_y = Math::floor(r_local_pos.y / p_sector_size).to_int();
		int64_t move_z = Math::floor(r_local_pos.z / p_sector_size).to_int();

		if (move_x != 0 || move_y != 0 || move_z != 0) {
			r_sx += BigIntCore(move_x);
			r_sy += BigIntCore(move_y);
			r_sz += BigIntCore(move_z);

			r_local_pos.x -= p_sector_size * FixedMathCore(move_x);
			r_local_pos.y -= p_sector_size * FixedMathCore(move_y);
			r_local_pos.z -= p_sector_size * FixedMathCore(move_z);
		}
	}

	// ------------------------------------------------------------------------
	// Physics Interaction Behavior
	// ------------------------------------------------------------------------

	/**
	 * compute_buoyancy_tensor()
	 * 
	 * Calculates the upward force for bodies interacting with fluids.
	 * Used for atmospheric and aquatic simulation.
	 */
	static _FORCE_INLINE_ Vector3f compute_buoyancy_tensor(
			const FixedMathCore &p_submerged_vol,
			const FixedMathCore &p_fluid_density,
			const Vector3f &p_gravity_vec) {

		// F_buoyancy = -rho * V_submerged * g
		return p_gravity_vec * (p_fluid_density * p_submerged_vol * (-MathConstants<FixedMathCore>::one()));
	}
};

#endif // GEOMETRY_INSTANCE_LOGIC_H

--- END OF FILE core/math/geometry_instance_logic.h ---
