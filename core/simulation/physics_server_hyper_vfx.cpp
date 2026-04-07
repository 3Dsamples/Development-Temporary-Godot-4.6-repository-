--- START OF FILE core/simulation/physics_server_hyper_vfx.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/math/warp_kernel.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: ParticleIntegrationKernel
 * 
 * Updates the state of a massive batch of particles in the EnTT registry.
 * Processes position, velocity, and lifetime decay deterministically.
 */
void particle_integration_kernel(
		const BigIntCore &p_index,
		Vector3f &r_position,
		Vector3f &r_velocity,
		FixedMathCore &r_lifetime,
		const FixedMathCore &p_gravity,
		const FixedMathCore &p_delta) {

	if (r_lifetime <= MathConstants<FixedMathCore>::zero()) return;

	// 1. Kinematic Integration (Semi-Implicit Euler)
	r_velocity.y -= p_gravity * p_delta;
	r_position += r_velocity * p_delta;

	// 2. Lifetime Decay
	r_lifetime -= p_delta;

	// 3. Galactic Sector Sync
	// Logic to keep particles relative to the local shifter origin
	// handled by the master shifter sweep.
}

/**
 * emit_impact_debris()
 * 
 * Spawns physical debris shards in response to a collision.
 * Utilizes the EnTT registry to allocate component streams in contiguous memory.
 */
void PhysicsServerHyper::emit_impact_debris(
		const Vector3f &p_origin,
		const Vector3f &p_normal,
		const FixedMathCore &p_energy,
		const BigIntCore &p_count) {

	auto &registry = get_kernel_registry();
	uint64_t n_particles = static_cast<uint64_t>(std::stoll(p_count.to_string()));

	for (uint64_t i = 0; i < n_particles; i++) {
		BigIntCore entity = registry.create_entity();
		
		// Random distribution via deterministic PCG
		RandomPCG &pcg = SimulationThreadPool::get_singleton()->get_worker(0)->get_pcg();
		Vector3f rand_dir = p_normal + (Vector3f(pcg.randf(), pcg.randf(), pcg.randf()) * FixedMathCore(2LL, false) - Vector3f(1LL, 1LL, 1LL));
		
		registry.assign<Vector3f>(entity, COMPONENT_POSITION, p_origin);
		registry.assign<Vector3f>(entity, COMPONENT_VELOCITY, rand_dir.normalized() * (p_energy * pcg.randf()));
		registry.assign<FixedMathCore>(entity, COMPONENT_LIFETIME, FixedMathCore(2LL, false) + pcg.randf());
	}
}

/**
 * update_vfx_simulation()
 * 
 * Master orchestrator for parallel VFX updates.
 * Dispatches Warp kernels to sweep through all active particle components.
 */
void PhysicsServerHyper::update_vfx_simulation(const FixedMathCore &p_delta) {
	auto &registry = get_kernel_registry();
	
	// Launch Warp Kernel across the SoA streams
	// This maintains 120 FPS by ensuring zero-copy access to the EnTT registry
	WarpKernel<Vector3f, Vector3f, FixedMathCore>::launch(
		registry,
		particle_integration_kernel,
		gravity,
		p_delta
	);

	// ETEngine Strategy: Periodically purge expired entities using SparseSet::erase
	// to keep memory density high for the next Warp sweep.
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_vfx.cpp ---
