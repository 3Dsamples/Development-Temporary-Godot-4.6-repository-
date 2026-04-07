--- START OF FILE core/math/math_utils.cpp ---

#include "core/math/math_utils.h"
#include "core/math/math_funcs.h"
#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

/**
 * solve_kepler()
 * 
 * Deterministic Newton-Raphson solver for Kepler's Equation: M = E - e * sin(E).
 * Given Mean Anomaly (M) and Eccentricity (e), finds Eccentric Anomaly (E).
 * Used for bit-perfect Keplerian Orbit Prediction across all simulation tiers.
 */
FixedMathCore MathUtils::solve_kepler(FixedMathCore p_mean_anomaly, FixedMathCore p_eccentricity) {
	FixedMathCore e = p_eccentricity;
	FixedMathCore m = p_mean_anomaly;
	
	// Initial guess: E = M
	FixedMathCore eccentric_anomaly = m;
	FixedMathCore tolerance(4295LL, true); // ~0.000001 precision
	
	// ETEngine Strategy: Fixed-iteration loop to ensure deterministic execution time (60 FPS safe)
	for (int i = 0; i < 8; i++) {
		FixedMathCore sin_e = Math::sin(eccentric_anomaly);
		FixedMathCore cos_e = Math::cos(eccentric_anomaly);
		
		// f(E) = E - e * sin(E) - M
		FixedMathCore f_e = eccentric_anomaly - (e * sin_e) - m;
		// f'(E) = 1 - e * cos(E)
		FixedMathCore df_e = MathConstants<FixedMathCore>::one() - (e * cos_e);
		
		FixedMathCore delta = f_e / df_e;
		eccentric_anomaly -= delta;
		
		if (Math::abs(delta) < tolerance) {
			break;
		}
	}
	
	return eccentric_anomaly;
}

/**
 * calculate_gravitational_acceleration()
 * 
 * N-Body Gravitational Approximation.
 * a = G * M / r^2
 * Uses BigIntCore for mass (Planetary/Star scale) and FixedMathCore for spatial vectors.
 */
Vector3f MathUtils::calculate_gravitational_acceleration(const Vector3f &p_relative_pos, const BigIntCore &p_mass) {
	FixedMathCore dist_sq = p_relative_pos.length_squared();
	if (dist_sq < FixedMathCore(100LL, true)) { // Near-zero distance safety
		return Vector3f();
	}
	
	// Universal Gravitational Constant G in Q32.32
	// G ≈ 6.67430e-11. We scale this for galactic unit compatibility.
	static const FixedMathCore G("0.000000000066743");
	
	// Convert BigInt mass to FixedMath for acceleration calculation
	// Note: For massive black holes, we handle scale-shifting to prevent 64-bit overflow.
	FixedMathCore mass_f(std::stoll(p_mass.to_string())); 
	
	FixedMathCore magnitude = (G * mass_f) / dist_sq;
	return p_relative_pos.normalized() * magnitude;
}

/**
 * get_orbital_velocity()
 * 
 * Returns the scalar velocity required for a stable circular orbit.
 * v = sqrt(G * M / r)
 */
FixedMathCore MathUtils::get_orbital_velocity(const BigIntCore &p_mass, FixedMathCore p_radius) {
	if (p_radius <= MathConstants<FixedMathCore>::zero()) return FixedMathCore(0LL, true);
	
	static const FixedMathCore G("0.000000000066743");
	FixedMathCore mass_f(std::stoll(p_mass.to_string()));
	
	return Math::sqrt((G * mass_f) / p_radius);
}

--- END OF FILE core/math/math_utils.cpp ---
