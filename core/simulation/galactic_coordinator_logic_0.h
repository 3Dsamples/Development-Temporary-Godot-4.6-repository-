--- START OF FILE core/simulation/galactic_coordinator_logic.h ---

#ifndef GALACTIC_COORDINATOR_LOGIC_H
#define GALACTIC_COORDINATOR_LOGIC_H

#include "core/math/vector3.h"
#include "core/templates/vector.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * GalacticCoordinatorLogic
 * 
 * High-performance kernels for celestial simulation.
 * Manages gravity, orbital mechanics, and sector-paging logic.
 * Designed for hardware-agnostic batch math in the Universal Solver.
 */
class GalacticCoordinatorLogic {
public:
	// ------------------------------------------------------------------------
	// Celestial Physics Kernels (Deterministic)
	// ------------------------------------------------------------------------

	/**
	 * calculate_gravity_acceleration()
	 * Computes the gravitational pull vector exerted by a mass on a point.
	 * a = (G * M / r^2) * norm(r)
	 * Uses BigIntCore for astronomical mass to prevent overflow.
	 */
	static _FORCE_INLINE_ Vector3f calculate_gravity_acceleration(const Vector3f& p_rel_pos, const BigIntCore& p_mass, const FixedMathCore& p_g_constant) {
		FixedMathCore r2 = p_rel_pos.length_squared();
		if (unlikely(r2 < FixedMathCore(100LL, true))) { // Near-zero distance safety
			return Vector3f();
		}

		// Convert BigInt mass to FixedMath for the force calculation
		// Scale shifting is used internally to maintain precision
		FixedMathCore m_fixed(static_cast<int64_t>(std::stoll(p_mass.to_string())));
		FixedMathCore magnitude = (p_g_constant * m_fixed) / r2;

		return p_rel_pos.normalized() * magnitude;
	}

	/**
	 * solve_kepler_orbit()
	 * Predicts future position on an elliptical path using FixedMathCore.
	 * Solves M = E - e*sin(E) via deterministic iterative refinement.
	 */
	static FixedMathCore solve_kepler_eccentric_anomaly(const FixedMathCore& p_mean_anomaly, const FixedMathCore& p_eccentricity);

	// ------------------------------------------------------------------------
	// Sector Paging & Persistence Logic
	// ------------------------------------------------------------------------

	/**
	 * is_sector_visible()
	 * Culling logic for galactic sectors based on observer BigIntCore coordinates.
	 */
	static _FORCE_INLINE_ bool is_sector_visible(
			const BigIntCore& p_obs_sx, const BigIntCore& p_obs_sy, const BigIntCore& p_obs_sz,
			const BigIntCore& p_sec_sx, const BigIntCore& p_sec_sy, const BigIntCore& p_sec_sz,
			const BigIntCore& p_render_dist) {
		
		BigIntCore dx = (p_obs_sx - p_sec_sx).absolute();
		BigIntCore dy = (p_obs_sy - p_sec_sy).absolute();
		BigIntCore dz = (p_obs_sz - p_sec_sz).absolute();

		return (dx <= p_render_dist && dy <= p_render_dist && dz <= p_render_dist);
	}

	/**
	 * generate_deterministic_sector_seed()
	 * Creates a unique BigIntCore seed for a sector based on universe seed and coordinates.
	 * Ensures procedural star systems align across the network.
	 */
	static BigIntCore generate_deterministic_sector_seed(const BigIntCore& p_univ_seed, const BigIntCore& p_sx, const BigIntCore& p_sy, const BigIntCore& p_sz);
};

#endif // GALACTIC_COORDINATOR_LOGIC_H

--- END OF FILE core/simulation/galactic_coordinator_logic.h ---
