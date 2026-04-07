--- START OF FILE core/simulation/galactic_coordinator_logic.h ---

#ifndef GALACTIC_COORDINATOR_LOGIC_H
#define GALACTIC_COORDINATOR_LOGIC_H

#include "core/math/vector3.h"
#include "core/math/math_defs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * GalacticCoordinatorLogic
 * 
 * Centralized static logic for high-precision celestial mechanics.
 * Orchestrates the transition between discrete sector logic and continuous physics.
 * 
 * Features:
 * - 🪐 N-Body Gravity kernels using BigIntCore mass.
 * - 🛰️ Keplerian Orbit prediction via universal variables.
 * - 🌌 Sector-aware visibility and culling filters.
 */
class GalacticCoordinatorLogic {
public:
	// ------------------------------------------------------------------------
	// Celestial Physics Kernels (Deterministic)
	// ------------------------------------------------------------------------

	/**
	 * calculate_gravity_force()
	 * Computes the force vector between two celestial bodies.
	 * f = G * (M1 * M2) / r^2
	 * Uses BigIntCore to prevent overflow with stellar masses.
	 */
	static _FORCE_INLINE_ Vector3f calculate_gravity_force(
			const Vector3f &p_pos_a, const BigIntCore &p_mass_a,
			const Vector3f &p_pos_b, const BigIntCore &p_mass_b,
			const FixedMathCore &p_g_constant) {
		
		Vector3f diff = p_pos_b - p_pos_a;
		FixedMathCore dist_sq = diff.length_squared();
		
		if (dist_sq.get_raw() == 0) return Vector3f(0LL, 0LL, 0LL);

		// Force = G * (M1 * M2) / r^2
		// We calculate f = (G / r^2) * (M1 * M2) to manage scale
		BigIntCore combined_mass = p_mass_a * p_mass_b;
		FixedMathCore factor = p_g_constant / dist_sq;
		
		// Scale-Aware conversion of BigInt product to FixedMath vector magnitude
		FixedMathCore magnitude_f(static_cast<int64_t>((combined_mass / BigIntCore(FixedMathCore::ONE_RAW)).operator int64_t()), true);
		
		return diff.normalized() * (factor * magnitude_f);
	}

	/**
	 * solve_eccentric_anomaly()
	 * Iteratively solves M = E - e * sin(E) for Keplerian propagation.
	 * strictly deterministic to prevent orbital decay over millennia.
	 */
	static FixedMathCore solve_eccentric_anomaly(
			const FixedMathCore &p_mean_anomaly,
			const FixedMathCore &p_eccentricity);

	// ------------------------------------------------------------------------
	// Sector Visibility & Paging
	// ------------------------------------------------------------------------

	/**
	 * is_sector_in_range()
	 * Bit-perfect visibility check between two BigIntCore sectors.
	 */
	static _FORCE_INLINE_ bool is_sector_in_range(
			const BigIntCore &p_sx1, const BigIntCore &p_sy1, const BigIntCore &p_sz1,
			const BigIntCore &p_sx2, const BigIntCore &p_sy2, const BigIntCore &p_sz2,
			const BigIntCore &p_render_distance_sectors) {
		
		BigIntCore dx = (p_sx1 - p_sx2).absolute();
		BigIntCore dy = (p_sy1 - p_sy2).absolute();
		BigIntCore dz = (p_sz1 - p_sz2).absolute();

		return (dx <= p_render_distance_sectors && 
				dy <= p_render_distance_sectors && 
				dz <= p_render_distance_sectors);
	}

	/**
	 * generate_sector_seed()
	 * Creates a deterministic BigIntCore seed for a sector based on 
	 * global universe seed and 3D sector coordinates.
	 */
	static BigIntCore generate_sector_seed(
			const BigIntCore &p_universe_seed,
			const BigIntCore &p_sx,
			const BigIntCore &p_sy,
			const BigIntCore &p_sz);

	// ------------------------------------------------------------------------
	// Orbit Prediction (Non-Iterative)
	// ------------------------------------------------------------------------

	/**
	 * get_position_at_time()
	 * Reconstructs Cartesian position from Keplerian elements at a specific BigInt time.
	 */
	static Vector3f get_position_at_time(
			const Vector3f &p_periapsis_dir,
			const Vector3f &p_orbit_normal,
			const FixedMathCore &p_semi_major_axis,
			const FixedMathCore &p_eccentricity,
			const FixedMathCore &p_eccentric_anomaly);
};

#endif // GALACTIC_COORDINATOR_LOGIC_H

--- END OF FILE core/simulation/galactic_coordinator_logic.h ---
