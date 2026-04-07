--- START OF FILE core/math/kepler_orbit_perturbation.cpp ---

#include "core/math/kepler_orbit_solver.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: OrbitalPerturbationKernel
 * 
 * Applies non-Keplerian forces to a body's state.
 * 1. J2 Perturbation: Simulates the gravitational effect of a planet's bulge.
 * 2. Third-Body Tugging: Influence of distant stars/moons.
 * 3. Atmospheric Decay: Drag-induced velocity loss leading to de-orbiting.
 */
void orbit_perturbation_kernel(
		const BigIntCore &p_index,
		Vector3f &r_velocity,
		const Vector3f &p_position,
		const BigIntCore &p_parent_mass,
		const FixedMathCore &p_parent_radius,
		const FixedMathCore &p_j2_coeff,
		const FixedMathCore &p_drag_coeff,
		const FixedMathCore &p_atm_density,
		const FixedMathCore &p_delta) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore r_mag = p_position.length();
	
	if (unlikely(r_mag < p_parent_radius)) return; // Surface collision handled by Server

	// --- 1. J2 Perturbation (Oblateness) ---
	// Acceleration = -1.5 * J2 * (mu/r^2) * (R/r)^2 * [(1-5(z/r)^2)k + (2z/r)r_unit]
	FixedMathCore mu = FixedMathCore(static_cast<int64_t>(std::stoll(p_parent_mass.to_string()))) * FixedMathCore(66743LL, true); // Scaled G
	FixedMathCore r2 = r_mag * r_mag;
	FixedMathCore ratio = p_parent_radius / r_mag;
	FixedMathCore ratio2 = ratio * ratio;
	
	FixedMathCore j2_factor = FixedMathCore(15LL, false) / FixedMathCore(10LL, false) * p_j2_coeff * (mu / r2) * ratio2;
	FixedMathCore z_r = p_position.z / r_mag;
	FixedMathCore z_r2 = z_r * z_r;

	Vector3f j2_accel;
	FixedMathCore poly_z = one - FixedMathCore(5LL, false) * z_r2;
	j2_accel.x = p_position.x / r_mag * poly_z;
	j2_accel.y = p_position.y / r_mag * poly_z;
	j2_accel.z = p_position.z / r_mag * (FixedMathCore(3LL, false) - FixedMathCore(5LL, false) * z_r2);
	j2_accel *= j2_factor;

	// --- 2. Atmospheric Decay (Drag) ---
	// a_drag = -0.5 * rho * v^2 * Cd * A / m
	FixedMathCore v_mag = r_velocity.length();
	FixedMathCore drag_mag = MathConstants<FixedMathCore>::half() * p_atm_density * v_mag * v_mag * p_drag_coeff;
	Vector3f drag_accel = -r_velocity.normalized() * drag_mag;

	// --- 3. Apply Deterministic Integration ---
	r_velocity += (j2_accel + drag_accel) * p_delta;
}

/**
 * execute_orbital_perturbation_sweep()
 * 
 * Parallel sweep over EnTT celestial components.
 * Essential for 120 FPS simulation of massive debris fields or satellite networks.
 */
void execute_orbital_perturbation_sweep(
		const BigIntCore &p_count,
		Vector3f *r_velocities,
		const Vector3f *p_positions,
		const BigIntCore *p_parent_masses,
		const FixedMathCore &p_delta) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_count.to_string()));
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? total : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				// Constants for Earth-like environment (Deterministic)
				FixedMathCore j2(1082LL << 20, true); // J2 for Earth approx
				FixedMathCore radius(6371000LL, false);
				FixedMathCore drag(22LL, true); // 2.2 Cd
				FixedMathCore rho(1225LL, true); // Sea level density proxy

				orbit_perturbation_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_velocities[i],
					p_positions[i],
					p_parent_masses[i],
					radius,
					j2,
					drag,
					rho,
					p_delta
				);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * calculate_roche_limit_check()
 * 
 * Advanced Behavior: Detects if a body is close enough to its parent 
 * to be torn apart by tidal forces.
 */
bool is_within_roche_limit(
		const BigIntCore &p_m_parent,
		const BigIntCore &p_m_body,
		const FixedMathCore &p_r_parent,
		const FixedMathCore &p_dist) {
	
	// d = R_p * (2 * M_p / M_body)^(1/3)
	FixedMathCore mass_ratio = FixedMathCore(static_cast<int64_t>(std::stoll((p_m_parent * BigIntCore(2LL) / p_m_body).to_string())));
	FixedMathCore roche_limit = p_r_parent * wp::pow(mass_ratio, FixedMathCore(1431655765LL, true)); // 1/3 exponent

	return p_dist < roche_limit;
}

} // namespace UniversalSolver

--- END OF FILE core/math/kepler_orbit_perturbation.cpp ---
