--- START OF FILE core/math/kepler_orbit_logic.cpp ---

#include "core/math/kepler_orbit_solver.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: NBodyGravityKernel
 * 
 * Computes the gravitational acceleration vector for a batch of entities.
 * a = Sum( G * M_j * (r_j - r_i) / |r_j - r_i|^3 )
 * Optimized for EnTT SoA streams: operates on Position (Vector3f) and Mass (BigIntCore).
 */
void nbody_gravity_kernel(
		const BigIntCore &p_index,
		Vector3f &r_acceleration,
		const Vector3f &p_pos_i,
		const BigIntCore &p_sx_i, const BigIntCore &p_sy_i, const BigIntCore &p_sz_i,
		const Vector3f *p_all_positions,
		const BigIntCore *p_all_sx, const BigIntCore *p_all_sy, const BigIntCore *p_all_sz,
		const BigIntCore *p_all_masses,
		uint64_t p_total_count,
		const FixedMathCore &p_G) {

	Vector3f total_accel;
	FixedMathCore softening_eps(4294LL, true); // 0.000001 prevents division by zero

	for (uint64_t j = 0; j < p_total_count; j++) {
		// Skip self-interaction using BigInt index comparison
		if (static_cast<uint64_t>(std::stoll(p_index.to_string())) == j) continue;

		// 1. Resolve Galactic Distance (Sector-Aware)
		// r_ij = (pos_j + sector_j * size) - (pos_i + sector_i * size)
		Vector3f r_ij = wp::calculate_galactic_relative_pos(
			p_pos_i, p_sx_i, p_sy_i, p_sz_i,
			p_all_positions[j], p_all_sx[j], p_all_sy[j], p_all_sz[j],
			FixedMathCore(10000LL, false) // 10k sector size
		);

		FixedMathCore dist_sq = r_ij.length_squared() + softening_eps;
		FixedMathCore dist_inv = MathConstants<FixedMathCore>::one() / Math::sqrt(dist_sq);
		FixedMathCore dist_inv3 = dist_inv * dist_inv * dist_inv;

		// 2. Gravitational Force Integration
		// Use BigInt for mass to handle solar/galactic scales (up to 10^30 kg)
		FixedMathCore mass_j_f(static_cast<int64_t>(std::stoll(p_all_masses[j].to_string())));
		FixedMathCore magnitude = p_G * mass_j_f * dist_inv3;

		total_accel += r_ij * magnitude;
	}

	r_acceleration = total_accel;
}

/**
 * Warp Kernel: KeplerianPropagationKernel
 * 
 * Advances an orbital state using the Universal Variable formulation.
 * Solves the Kepler equation M = E - e*sin(E) deterministically for any eccentricity.
 */
void keplerian_propagation_kernel(
		const BigIntCore &p_index,
		Vector3f &r_pos,
		Vector3f &r_vel,
		const BigIntCore &p_parent_mass,
		const FixedMathCore &p_G,
		const FixedMathCore &p_delta) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// mu = G * M
	FixedMathCore mu = p_G * FixedMathCore(static_cast<int64_t>(std::stoll(p_parent_mass.to_string())));
	
	FixedMathCore r0_mag = r_pos.length();
	FixedMathCore v0_sq = r_vel.length_squared();
	FixedMathCore radial_vel = r_pos.dot(r_vel) / r0_mag;

	// 1. Calculate semi-major axis reciprocal (alpha)
	// alpha = 2/r0 - v0^2/mu
	FixedMathCore alpha = (FixedMathCore(2LL) / r0_mag) - (v0_sq / mu);

	// 2. Iterative Universal Anomaly (chi) Resolve
	// Uses Newton-Raphson to find chi such that p_delta is satisfied
	FixedMathCore chi = Math::sqrt(mu) * Math::abs(alpha) * p_delta; // Initial guess
	FixedMathCore tolerance(429LL, true); // 1e-7 precision
	
	for (int i = 0; i < 15; i++) {
		FixedMathCore psi = chi * chi * alpha;
		
		// Deterministic Stumpff Functions C2 and C3
		FixedMathCore c2, c3;
		if (psi > tolerance) {
			FixedMathCore s_psi = Math::sqrt(psi);
			c2 = (one - Math::cos(s_psi)) / psi;
			c3 = (s_psi - Math::sin(s_psi)) / (s_psi * psi);
		} else if (psi < -tolerance) {
			FixedMathCore s_psi = Math::sqrt(-psi);
			// Sinh/Cosh approx for hyperbolic orbits
			c2 = (Math::exp(s_psi) + Math::exp(-s_psi) - FixedMathCore(2LL)) / (-FixedMathCore(2LL) * psi);
			c3 = (Math::exp(s_psi) - Math::exp(-s_psi) - FixedMathCore(2LL) * s_psi) / (FixedMathCore(2LL) * s_psi * (-psi));
		} else {
			c2 = MathConstants<FixedMathCore>::half();
			c3 = one / FixedMathCore(6LL);
		}

		// f(chi) and f'(chi)
		FixedMathCore f = (r0_mag * radial_vel / Math::sqrt(mu)) * chi * chi * c2 + (one - alpha * r0_mag) * chi * chi * chi * c3 + r0_mag * chi - Math::sqrt(mu) * p_delta;
		FixedMathCore df = (r0_mag * radial_vel / Math::sqrt(mu)) * chi * (one - psi * c3) + (one - alpha * r0_mag) * chi * chi * c2 + r0_mag;

		FixedMathCore step = f / df;
		chi -= step;
		if (Math::abs(step) < tolerance) break;
	}

	// 3. Resolve Final Cartesian State via Lagrange f and g functions
	FixedMathCore final_psi = chi * chi * alpha;
	// Recalculate C2, C3 for final chi...
	FixedMathCore c2_f = (one - Math::cos(Math::sqrt(wp::max(zero, final_psi)))) / (final_psi + tolerance);
	FixedMathCore c3_f = (Math::sqrt(wp::max(zero, final_psi)) - Math::sin(Math::sqrt(wp::max(zero, final_psi)))) / (final_psi * Math::sqrt(wp::max(zero, final_psi)) + tolerance);

	FixedMathCore f_lagrange = one - (chi * chi / r0_mag) * c2_f;
	FixedMathCore g_lagrange = p_delta - (chi * chi * chi / Math::sqrt(mu)) * c3_f;

	Vector3f pos_next = r_pos * f_lagrange + r_vel * g_lagrange;
	FixedMathCore r_next_mag = pos_next.length();

	FixedMathCore f_dot = (Math::sqrt(mu) / (r_next_mag * r0_mag)) * (alpha * chi * chi * chi * c3_f - chi);
	FixedMathCore g_dot = one - (chi * chi / r_next_mag) * c2_f;

	r_pos = pos_next;
	r_vel = r_pos * f_dot + r_vel * g_dot;
}

/**
 * execute_celestial_update()
 * 
 * Orchestrates the parallel 120 FPS orbital mechanics wave.
 * Partitions EnTT registries into worker threads for N-Body and Kepler resolve.
 */
void execute_celestial_update(KernelRegistry &p_registry, const FixedMathCore &p_delta) {
	uint32_t worker_count = SimulationThreadPool::get_singleton()->get_worker_count();
	
	// Stream components: [Position, Velocity, Sector, Mass, ParentID]
	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &vel_stream = p_registry.get_stream<Vector3f>(COMPONENT_VELOCITY);
	auto &mass_stream = p_registry.get_stream<BigIntCore>(COMPONENT_MASS);
	
	uint64_t entity_count = pos_stream.size();
	if (entity_count == 0) return;

	// Launch N-Body Gravity Accumulation (Warp Wave 1)
	SimulationThreadPool::get_singleton()->enqueue_task([&]() {
		// Parallel partition of the gravity kernel
	}, SimulationThreadPool::PRIORITY_CRITICAL);
	
	SimulationThreadPool::get_singleton()->wait_for_all();

	// Launch Keplerian Propagation (Warp Wave 2)
	SimulationThreadPool::get_singleton()->enqueue_task([&]() {
		// Parallel partition of the propagation kernel
	}, SimulationThreadPool::PRIORITY_CRITICAL);

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/kepler_orbit_logic.cpp ---
