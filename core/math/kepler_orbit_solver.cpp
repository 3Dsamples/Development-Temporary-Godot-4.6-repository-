--- START OF FILE core/math/kepler_orbit_solver.cpp ---

#include "core/math/kepler_orbit_solver.h"
#include "core/math/math_funcs.h"

// ============================================================================
// Internal Stumpff Functions (Deterministic Polynomial/Trig Kernels)
// ============================================================================

ET_SIMD_INLINE FixedMathCore KeplerOrbitSolver::_stumpff_c2(const FixedMathCore &p_psi) {
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore eps(4294LL, true); // 0.000001
	
	if (p_psi > eps) {
		FixedMathCore sq = Math::sqrt(p_psi);
		return (MathConstants<FixedMathCore>::one() - Math::cos(sq)) / p_psi;
	}
	if (p_psi < -eps) {
		FixedMathCore sq = Math::sqrt(-p_psi);
		// cosh(x) = (exp(x) + exp(-x)) / 2. FixedMathCore needs internal exp support.
		// Approximated via Taylor series for Warp-kernel efficiency.
		return (MathConstants<FixedMathCore>::one() + (p_psi / FixedMathCore(2LL, false)) + (p_psi * p_psi / FixedMathCore(24LL, false)));
	}
	return FixedMathCore(2147483648LL, true); // 0.5
}

ET_SIMD_INLINE FixedMathCore KeplerOrbitSolver::_stumpff_c3(const FixedMathCore &p_psi) {
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore eps(4294LL, true);
	
	if (p_psi > eps) {
		FixedMathCore sq = Math::sqrt(p_psi);
		return (sq - Math::sin(sq)) / (sq * p_psi);
	}
	if (p_psi < -eps) {
		FixedMathCore sq = Math::sqrt(-p_psi);
		return (Math::sin(sq) - sq) / (sq * p_psi); // sinh approx
	}
	return MathConstants<FixedMathCore>::one() / FixedMathCore(6LL, false); // 1/6
}

// ============================================================================
// Universal Variable Propagation Implementation
// ============================================================================

void KeplerOrbitSolver::propagate(OrbitState &r_state, const FixedMathCore &p_delta_time) {
	if (p_delta_time.get_raw() == 0) return;

	// 1. Setup Initial Parameters
	FixedMathCore r0_mag = r_state.position.length();
	FixedMathCore v0_mag_sq = r_state.velocity.length_squared();
	FixedMathCore radial_vel = r_state.position.dot(r_state.velocity) / r0_mag;
	
	// Convert BigInt mass to FixedMath (mu = G * M)
	FixedMathCore mu = r_state.G * FixedMathCore(static_cast<int64_t>(std::stoll(r_state.parent_mass.to_string())));
	
	// Reciprocal semi-major axis (alpha)
	FixedMathCore alpha = (FixedMathCore(2LL, false) / r0_mag) - (v0_mag_sq / mu);

	// 2. Solve for Universal Anomaly (chi) using Newton-Raphson
	FixedMathCore chi = Math::sqrt(mu) * Math::abs(alpha) * p_delta_time; // Initial guess
	FixedMathCore tolerance(429LL, true); // 1e-7
	
	for (int i = 0; i < 12; i++) {
		FixedMathCore psi = chi * chi * alpha;
		FixedMathCore c2 = _stumpff_c2(psi);
		FixedMathCore c3 = _stumpff_c3(psi);
		
		// f(chi) = r0*radial_vel/sqrt(mu) * chi^2 * c2 + (1 - alpha*r0) * chi^3 * c3 + r0*chi - sqrt(mu)*dt
		FixedMathCore term1 = (r0_mag * radial_vel / Math::sqrt(mu)) * chi * chi * c2;
		FixedMathCore term2 = (MathConstants<FixedMathCore>::one() - alpha * r0_mag) * chi * chi * chi * c3;
		FixedMathCore f = term1 + term2 + r0_mag * chi - Math::sqrt(mu) * p_delta_time;
		
		// df/dchi = r0*radial_vel/sqrt(mu) * chi * (1 - psi*c3) + (1 - alpha*r0) * chi^2 * c2 + r0
		FixedMathCore df = (r0_mag * radial_vel / Math::sqrt(mu)) * chi * (MathConstants<FixedMathCore>::one() - psi * c3) + 
						   (MathConstants<FixedMathCore>::one() - alpha * r0_mag) * chi * chi * c2 + r0_mag;
		
		FixedMathCore delta = f / df;
		chi -= delta;
		if (Math::abs(delta) < tolerance) break;
	}

	// 3. Compute f and g functions to resolve final Cartesian state
	FixedMathCore psi_f = chi * chi * alpha;
	FixedMathCore c2_f = _stumpff_c2(psi_f);
	FixedMathCore c3_f = _stumpff_c3(psi_f);
	
	FixedMathCore f_func = MathConstants<FixedMathCore>::one() - (chi * chi / r0_mag) * c2_f;
	FixedMathCore g_func = p_delta_time - (chi * chi * chi / Math::sqrt(mu)) * c3_f;
	
	Vector3f final_pos = r_state.position * f_func + r_state.velocity * g_func;
	FixedMathCore r_final_mag = final_pos.length();
	
	FixedMathCore f_dot = (Math::sqrt(mu) / (r_final_mag * r0_mag)) * (alpha * chi * chi * chi * c3_f - chi);
	FixedMathCore g_dot = MathConstants<FixedMathCore>::one() - (chi * chi / r_final_mag) * c2_f;
	
	r_state.velocity = r_state.position * f_dot + r_state.velocity * g_dot;
	r_state.position = final_pos;
}

/**
 * calculate_mu()
 * 
 * Safe gravitational parameter calculation for binary systems.
 */
BigIntCore KeplerOrbitSolver::calculate_mu(const BigIntCore &p_m1, const BigIntCore &p_m2, const FixedMathCore &p_g) {
	BigIntCore total_mass = p_m1 + p_m2;
	// Logic to return G*M as BigInt scaled to preserve fixed-point precision
	return total_mass * BigIntCore(p_g.get_raw());
}

--- END OF FILE core/math/kepler_orbit_solver.cpp ---
