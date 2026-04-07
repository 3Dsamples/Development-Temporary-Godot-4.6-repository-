--- START OF FILE core/math/kepler_orbit_solver.h ---

#ifndef KEPLER_ORBIT_SOLVER_H
#define KEPLER_ORBIT_SOLVER_H

#include "core/typedefs.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * KeplerOrbitSolver
 * 
 * Deterministic propagator for celestial mechanics.
 * Solves the 2-body problem using Universal Variables (Stumpff functions).
 * Aligned for 120 FPS Warp-style parallel sweeps across EnTT celestial components.
 */
class ET_ALIGN_32 KeplerOrbitSolver {
public:
	struct ET_ALIGN_32 OrbitState {
		Vector3f position; // Relative to parent
		Vector3f velocity;
		BigIntCore parent_mass;
		FixedMathCore G;   // Gravitational constant scaled for the system
	};

	struct ET_ALIGN_32 OrbitElements {
		FixedMathCore semi_major_axis;
		FixedMathCore eccentricity;
		FixedMathCore inclination;
		FixedMathCore longitude_ascending_node;
		FixedMathCore argument_periapsis;
		FixedMathCore true_anomaly;
		BigIntCore epoch_time;
	};

private:
	// Numerical kernels for Universal Variable iteration
	static ET_SIMD_INLINE FixedMathCore _stumpff_c2(const FixedMathCore &p_psi);
	static ET_SIMD_INLINE FixedMathCore _stumpff_c3(const FixedMathCore &p_psi);

public:
	/**
	 * propagate()
	 * Advances an orbital state by a deterministic time step.
	 * Uses Newtonian laws of motion implemented in bit-perfect FixedMath.
	 */
	static void propagate(OrbitState &r_state, const FixedMathCore &p_delta_time);

	/**
	 * state_to_elements()
	 * Converts Cartesian state (pos/vel) to Keplerian elements.
	 */
	static OrbitElements state_to_elements(const OrbitState &p_state);

	/**
	 * elements_to_state()
	 * Reconstructs Cartesian state from orbital elements at a specific time.
	 */
	static OrbitState elements_to_state(const OrbitElements &p_elems, const BigIntCore &p_current_time);

	/**
	 * calculate_gravitational_parameter()
	 * mu = G * (m1 + m2). Returns BigIntCore to prevent overflow.
	 */
	static BigIntCore calculate_mu(const BigIntCore &p_m1, const BigIntCore &p_m2, const FixedMathCore &p_g);
};

#endif // KEPLER_ORBIT_SOLVER_H

--- END OF FILE core/math/kepler_orbit_solver.h ---
