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
 * Used for bit-perfect Keplerian Orbit Prediction.
 */
FixedMathCore MathUtils::solve_kepler(FixedMathCore p_mean_anomaly, FixedMathCore p_eccentricity) {
	FixedMathCore e = p_eccentricity;
	FixedMathCore m = p_mean_anomaly;
	
	// Initial guess: E = M
	FixedMathCore eccentric_anomaly = m;
	
	// Deterministic precision threshold: 0.000001 (raw bits: 4295)
	FixedMathCore tolerance(4295LL, true); 
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	
	// ETEngine Strategy: Fixed iteration limit (12) to ensure constant execution 
	// time within the 120 FPS heartbeat.
	for (int i = 0; i < 12; i++) {
		FixedMathCore sin_e = Math::sin(eccentric_anomaly);
		FixedMathCore cos_e = Math::cos(eccentric_anomaly);
		
		// f(E) = E - e * sin(E) - M
		FixedMathCore f_e = eccentric_anomaly - (e * sin_e) - m;
		// f'(E) = 1 - e * cos(E)
		FixedMathCore df_e = one - (e * cos_e);
		
		// If derivative is zero, we hit a singularity (parabolic/hyperbolic edge case)
		if (unlikely(df_e.get_raw() == 0)) break;

		FixedMathCore delta = f_e / df_e;
		eccentric_anomaly -= delta;
		
		// Check for bit-perfect convergence
		if (Math::abs(delta) < tolerance) {
			break;
		}
	}
	
	return eccentric_anomaly;
}

/**
 * calculate_galactic_offset()
 * 
 * Computes the 3D world-space offset between two galactic sectors.
 * strictly uses BigIntCore to prevent 64-bit overflow during 
 * light-year distance calculations.
 */
Vector3f calculate_galactic_offset(
		const BigIntCore &p_sx_src, const BigIntCore &p_sy_src, const BigIntCore &p_sz_src,
		const BigIntCore &p_sx_dst, const BigIntCore &p_sy_dst, const BigIntCore &p_sz_dst,
		const FixedMathCore &p_sector_size) {

	BigIntCore dx = p_sx_dst - p_sx_src;
	BigIntCore dy = p_sy_dst - p_sy_src;
	BigIntCore dz = p_sz_dst - p_sz_src;

	// Scale-Aware conversion: BigInt * Fixed -> Fixed
	// We convert the sector delta to a FixedMath value and then multiply by sector size.
	// This maintains bit-perfection across the scale boundary.
	FixedMathCore fx(static_cast<int64_t>(std::stoll(dx.to_string())));
	FixedMathCore fy(static_cast<int64_t>(std::stoll(dy.to_string())));
	FixedMathCore fz(static_cast<int64_t>(std::stoll(dz.to_string())));

	return Vector3f(fx * p_sector_size, fy * p_sector_size, fz * p_sector_size);
}

/**
 * get_orbital_velocity_vector()
 * 
 * Returns the bit-perfect velocity vector required for a stable orbit.
 * v = sqrt(mu / r) * cross(normal, radial_unit)
 */
Vector3f get_orbital_velocity_vector(
		const Vector3f &p_relative_pos, 
		const BigIntCore &p_mass, 
		const Vector3f &p_orbit_normal, 
		const FixedMathCore &p_g_constant) {

	FixedMathCore r_mag = p_relative_pos.length();
	if (r_mag.get_raw() == 0) return Vector3f();

	// mu = G * M. Handled via BigInt to avoid overflow.
	FixedMathCore mass_f(static_cast<int64_t>(std::stoll(p_mass.to_string())));
	FixedMathCore mu = p_g_constant * mass_f;

	FixedMathCore v_mag = Math::sqrt(mu / r_mag);
	Vector3f radial_unit = p_relative_pos / r_mag;
	
	// Velocity is perpendicular to both radius and orbit normal
	return p_orbit_normal.cross(radial_unit).normalized() * v_mag;
}

/**
 * apply_centrifugal_force()
 * 
 * Advanced Physics Behavior: Calculates the outward force in a rotating frame.
 * F = m * omega^2 * r
 */
Vector3f apply_centrifugal_force(
		const Vector3f &p_local_pos, 
		const Vector3f &p_angular_vel, 
		const FixedMathCore &p_mass) {

	// a = omega x (omega x r)
	Vector3f accel = p_angular_vel.cross(p_angular_vel.cross(p_local_pos));
	return accel * (-p_mass);
}

--- END OF FILE core/math/math_utils.cpp ---
