--- START OF FILE core/math/kepler_orbit_perturbation.cpp ---

#include "core/math/kepler_orbit_solver.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_j2_acceleration()
 * 
 * Computes the gravitational perturbation caused by a planet's non-spherical shape.
 * Acceleration = -1.5 * J2 * (mu/r^2) * (R/r)^2 * [(1-5(z/r)^2) * r_hat + 2(z/r) * k_hat]
 */
static _FORCE_INLINE_ Vector3f calculate_j2_acceleration(
		const Vector3f &p_pos,
		const FixedMathCore &p_mu,
		const FixedMathCore &p_planet_radius,
		const FixedMathCore &p_j2_coeff) {

	FixedMathCore r_mag = p_pos.length();
	if (unlikely(r_mag.get_raw() == 0)) return Vector3f_ZERO;

	FixedMathCore r2 = r_mag * r_mag;
	FixedMathCore r5 = r2 * r2 * r_mag;
	FixedMathCore planet_r2 = p_planet_radius * p_planet_radius;
	
	// Pre-factor: (1.5 * J2 * mu * R^2) / r^5
	FixedMathCore factor = (FixedMathCore("1.5") * p_j2_coeff * p_mu * planet_r2) / r5;

	FixedMathCore z = p_pos.z;
	FixedMathCore z_r_sq = (z * z) / r2;
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore five(5LL);
	FixedMathCore two(2LL);

	// Components based on geopotential derivation
	FixedMathCore common_term = one - (five * z_r_sq);
	
	Vector3f accel;
	accel.x = p_pos.x * common_term;
	accel.y = p_pos.y * common_term;
	accel.z = p_pos.z * (one - (five * z_r_sq) + two); // (3 - 5z^2/r^2)

	return accel * (-factor);
}

/**
 * Warp Kernel: OrbitalPerturbationKernel
 * 
 * Step-wise update for non-Keplerian dynamics.
 * 1. J2 Effect: Shifts the nodes and perigee of the orbit.
 * 2. Drag: Simulates energy loss in the upper atmosphere.
 * 3. N-Body: Resolves tugging from distant moons/stars in the same sector.
 */
void orbital_perturbation_kernel(
		const BigIntCore &p_index,
		Vector3f &r_velocity,
		const Vector3f &p_position,
		const BigIntCore &p_parent_mass,
		const FixedMathCore &p_parent_radius,
		const FixedMathCore &p_j2_coeff,
		const FixedMathCore &p_drag_area_mass_ratio,
		const FixedMathCore &p_atm_density,
		const FixedMathCore &p_delta,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	
	// Convert BigInt mass to FixedMath Gravitational Parameter (mu = G * M)
	FixedMathCore mu = PHYSICS_G * FixedMathCore(static_cast<int64_t>(std::stoll(p_parent_mass.to_string())));

	// 1. Calculate J2 Perturbation
	Vector3f a_j2 = calculate_j2_acceleration(p_position, mu, p_parent_radius, p_j2_coeff);

	// 2. Calculate Atmospheric Drag (Decay)
	// a_drag = -0.5 * rho * v^2 * (A/m) * Cd * v_dir
	FixedMathCore v_mag = r_velocity.length();
	FixedMathCore drag_coeff(22LL, true); // 2.2 Cd for satellites
	FixedMathCore drag_mag = MathConstants<FixedMathCore>::half() * p_atm_density * v_mag * v_mag * p_drag_area_mass_ratio * drag_coeff;
	Vector3f a_drag = r_velocity.normalized() * (-drag_mag);

	// 3. --- Sophisticated Behavior: Realistic vs Anime ---
	if (p_is_anime) {
		// Anime Technique: "Kinetic Trails". 
		// Forces orbital perturbations to be more visible by exaggerating the J2 nodal shift.
		a_j2 *= FixedMathCore(5LL); 
		// Clamp drag to prevent complete de-orbiting for "Cool Factor"
		a_drag *= FixedMathCore("0.1"); 
	}

	// 4. Update Velocity Tensor
	r_velocity += (a_j2 + a_drag) * p_delta;
}

/**
 * execute_orbital_perturbation_sweep()
 * 
 * Orchestrates the parallel 120 FPS perturbation wave.
 * Zero-copy: Operates directly on the EnTT registry SoA streams.
 */
void execute_orbital_perturbation_sweep(
		KernelRegistry &p_registry,
		const BigIntCore &p_parent_mass,
		const FixedMathCore &p_parent_radius,
		const FixedMathCore &p_delta) {

	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &vel_stream = p_registry.get_stream<Vector3f>(COMPONENT_VELOCITY);
	auto &drag_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_DRAG_RATIO);
	auto &density_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_ATM_DENSITY);

	uint64_t count = pos_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	// Global J2 constant for Earth-like bodies
	FixedMathCore j2_const("0.00108263");

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &pos_stream, &vel_stream, &drag_stream, &density_stream]() {
			for (uint64_t i = start; i < end; i++) {
				// Deterministic style selection: every 10th satellite uses anime physics
				bool anime_mode = (i % 10 == 0);

				orbital_perturbation_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					vel_stream[i],
					pos_stream[i],
					p_parent_mass,
					p_parent_radius,
					j2_const,
					drag_stream[i],
					density_stream[i],
					p_delta,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * calculate_orbital_lifetime()
 * 
 * Sophisticated Feature: Predicts how many BigInt ticks remain before
 * atmospheric entry based on current decay rate.
 */
BigIntCore calculate_orbital_lifetime(
		const FixedMathCore &p_energy,
		const FixedMathCore &p_decay_rate_per_tick) {
	
	if (p_decay_rate_per_tick.get_raw() <= 0) return BigIntCore("999999999999");
	
	FixedMathCore ticks_f = p_energy / p_decay_rate_per_tick;
	return BigIntCore(static_cast<int64_t>(ticks_f.to_int()));
}

} // namespace UniversalSolver

--- END OF FILE core/math/kepler_orbit_perturbation.cpp ---
