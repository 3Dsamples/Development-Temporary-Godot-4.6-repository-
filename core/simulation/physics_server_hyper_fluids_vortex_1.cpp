--- START OF FILE core/simulation/physics_server_hyper_fluids_vortex.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: AerodynamicWakeKernel
 * 
 * Simulates the interaction between a high-speed body (spaceship) and a fluid.
 * 1. Calculates the local wind produced by ship velocity and rotation.
 * 2. Applies momentum transfer to fluid particles within the ship's wake.
 * 3. Injects vorticity (swirl) based on the ship's trailing edges.
 */
void aerodynamic_wake_kernel(
		Vector3f &r_fluid_vel,
		Vector3f &r_fluid_omega,
		const Vector3f &p_fluid_pos,
		const Vector3f &p_ship_pos,
		const Vector3f &p_ship_vel,
		const Vector3f &p_ship_ang_vel,
		const FixedMathCore &p_wake_radius,
		const FixedMathCore &p_transfer_coeff,
		const FixedMathCore &p_delta) {

	Vector3f rel_pos = p_fluid_pos - p_ship_pos;
	FixedMathCore dist_sq = rel_pos.length_squared();
	FixedMathCore r2 = p_wake_radius * p_wake_radius;

	if (dist_sq >= r2 || dist_sq.get_raw() == 0) return;

	FixedMathCore dist = Math::sqrt(dist_sq);
	FixedMathCore falloff = (p_wake_radius - dist) / p_wake_radius;
	FixedMathCore effect_strength = falloff * falloff * p_transfer_coeff;

	// 1. Linear Momentum Transfer
	// The ship "pushes" air/gas in its direction of travel
	r_fluid_vel = wp::lerp(r_fluid_vel, p_ship_vel, effect_strength * p_delta);

	// 2. Angular Momentum Transfer (Vortex Generation)
	// v_tangent = omega x r
	Vector3f tangent_vel = p_ship_ang_vel.cross(rel_pos);
	r_fluid_vel += tangent_vel * (effect_strength * p_delta);

	// 3. Vorticity Injection
	// Creating turbulent "swirls" in the wake
	Vector3f swirl = p_ship_vel.cross(rel_pos).normalized() * (p_ship_vel.length() * effect_strength);
	r_fluid_omega += swirl * p_delta;
}

/**
 * execute_vortex_dynamics_sweep()
 * 
 * Orchestrates the full 120 FPS turbulence pipeline.
 * Iterates through all active Rigid/Deformable bodies to update the fluid registry.
 */
void PhysicsServerHyper::execute_vortex_dynamics_sweep(const FixedMathCore &p_delta) {
	auto &registry = get_kernel_registry();
	uint64_t particle_count = registry.get_stream<Vector3f>(COMPONENT_VELOCITY).size();
	if (particle_count == 0) return;

	// 1. Interaction Pass: High-Speed Wakes
	// We iterate through every dynamic body (Ships) and influence nearby fluid particles
	for (uint32_t b_idx = 0; b_idx < body_owner.get_rid_count(); b_idx++) {
		Body *ship = body_owner.get_raw_data_ptr()[b_idx];
		if (!ship->active || ship->linear_velocity.length_squared() < FixedMathCore(100LL, false)) continue;

		FixedMathCore wake_radius = ship->mesh_deterministic->get_aabb().size.length() * FixedMathCore(2LL, false);
		FixedMathCore transfer_k(2147483648LL, true); // 0.5 strength

		// Dispatch Warp Sweep for the fluid particles in this ship's vicinity
		SimulationThreadPool::get_singleton()->enqueue_task([=, &registry]() {
			Vector3f *vel_stream = registry.get_stream<Vector3f>(COMPONENT_VELOCITY).get_base_ptr();
			Vector3f *omega_stream = registry.get_stream<Vector3f>(COMPONENT_VORTICITY).get_base_ptr();
			const Vector3f *pos_stream = registry.get_stream<Vector3f>(COMPONENT_POSITION).get_base_ptr();

			for (uint64_t i = 0; i < particle_count; i++) {
				aerodynamic_wake_kernel(
					vel_stream[i],
					omega_stream[i],
					pos_stream[i],
					ship->transform.origin,
					ship->linear_velocity,
					ship->angular_velocity,
					wake_radius,
					transfer_k,
					p_delta
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}
	SimulationThreadPool::get_singleton()->wait_for_all();

	// 2. Confinement Pass: Restore lost energy to vortices
	SimulationThreadPool::get_singleton()->enqueue_task([&]() {
		Vector3f *accel_stream = registry.get_stream<Vector3f>(COMPONENT_ACCELERATION).get_base_ptr();
		const Vector3f *pos_stream = registry.get_stream<Vector3f>(COMPONENT_POSITION).get_base_ptr();
		const Vector3f *omega_stream = registry.get_stream<Vector3f>(COMPONENT_VORTICITY).get_base_ptr();
		const BigIntCore *sx_stream = registry.get_stream<BigIntCore>(COMPONENT_SECTOR_X).get_base_ptr();
		const BigIntCore *sy_stream = registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Y).get_base_ptr();
		const BigIntCore *sz_stream = registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Z).get_base_ptr();

		FixedMathCore epsilon(429496730LL, true); // 0.1 confinement strength
		FixedMathCore h(2LL, false);

		for (uint64_t i = 0; i < particle_count; i++) {
			BigIntCore idx(static_cast<int64_t>(i));
			apply_vorticity_confinement_kernel(
				idx,
				accel_stream[i],
				pos_stream[i],
				omega_stream[i],
				sx_stream[i], sy_stream[i], sz_stream[i],
				pos_stream,
				omega_stream,
				sx_stream, sy_stream, sz_stream,
				particle_count,
				h,
				epsilon,
				(i % 10 == 0) // Anime style swizzle
			);
		}
	}, SimulationThreadPool::PRIORITY_HIGH);

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_fluids_vortex.cpp ---
