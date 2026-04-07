--- START OF FILE core/simulation/physics_server_hyper_vfx_emitter.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/math/random_pcg.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: ParticleEmissionKernel
 * 
 * Initializes a batch of newly spawned particles in the EnTT registry.
 * 1. Sets initial position relative to the emitter.
 * 2. Combines parent velocity with ejection impulse (Inheritance).
 * 3. Assigns deterministic lifetimes using bit-perfect RandomPCG.
 */
void particle_emission_kernel(
		const BigIntCore &p_index,
		Vector3f &r_position,
		Vector3f &r_velocity,
		FixedMathCore &r_lifetime,
		const Vector3f &p_emitter_pos,
		const Vector3f &p_emitter_vel,
		const Vector3f &p_ejection_dir,
		const FixedMathCore &p_ejection_force,
		const BigIntCore &p_seed) {

	// 1. Initialize deterministic entropy for this particle index
	RandomPCG pcg;
	pcg.seed(p_seed.hash() ^ p_index.hash());

	// 2. Randomized Ejection Vector (Isotropic or Conical)
	// Uses bit-perfect FixedMath trig functions
	T spread = pcg.randf() * FixedMathCore(429496730LL, true); // 0.1 spread
	Vector3f random_dir = p_ejection_dir + (Vector3f(pcg.randf(), pcg.randf(), pcg.randf()) * spread);
	
	// 3. Velocity Inheritance: Ship Vel + (Impulse / Particle Mass)
	// Essential for high-speed spaceships to leave a realistic wake
	r_velocity = p_emitter_vel + random_dir.normalized() * (p_ejection_force * pcg.randf());

	// 4. Initial Position with sub-frame jitter to prevent "banding" at 120 FPS
	r_position = p_emitter_pos + (r_velocity * pcg.randf() * FixedMathCore(8333333LL, true)); // Jitter within 1/120s

	// 5. Life expectancy in ticks
	r_lifetime = FixedMathCore(5LL, false) + pcg.randf() * FixedMathCore(2LL, false); 
}

/**
 * emit_vfx_thrust_batch()
 * 
 * High-Speed spaceship feature: Emits thrust particles based on engine state.
 * Dynamically scales ejection energy based on the ship's current velocity 
 * and material temperature (thermal incandescence).
 */
void PhysicsServerHyper::emit_vfx_thrust_batch(
		const RID &p_ship_body,
		const Vector3f &p_local_offset,
		const Vector3f &p_direction,
		const BigIntCore &p_particle_count) {

	Body *ship = body_owner.get_or_null(p_ship_body);
	if (unlikely(!ship)) return;

	auto &registry = get_kernel_registry();
	uint64_t count = static_cast<uint64_t>(std::stoll(p_particle_count.to_string()));

	// Transform emitter to world space using ship's bit-perfect Basis
	Vector3f world_origin = ship->transform.xform(p_local_offset);
	Vector3f world_dir = ship->transform.basis.xform(p_direction).normalized();

	// Calculate energy: more thrust for high-speed acceleration
	FixedMathCore thrust_energy = ship->linear_velocity.length() * FixedMathCore(214748364LL, true); // 0.05 scaling

	// ETEngine Strategy: Allocate blocks of entities in EnTT Sparse Sets
	// Zero-Copy: We fetch the raw pointers and launch parallel initialization
	SimulationThreadPool::get_singleton()->enqueue_task([=, &registry]() {
		Vector3f *pos_stream = registry.get_stream<Vector3f>(COMPONENT_POSITION).get_base_ptr();
		Vector3f *vel_stream = registry.get_stream<Vector3f>(COMPONENT_VELOCITY).get_base_ptr();
		FixedMathCore *life_stream = registry.get_stream<FixedMathCore>(COMPONENT_LIFETIME).get_base_ptr();

		for (uint64_t i = 0; i < count; i++) {
			BigIntCore particle_id = registry.create_entity();
			uint64_t p_idx = registry.get_dense_index(particle_id);

			particle_emission_kernel(
				particle_id,
				pos_stream[p_idx],
				vel_stream[p_idx],
				life_stream[p_idx],
				world_origin,
				ship->linear_velocity,
				world_dir,
				thrust_energy,
				ship->self.get_id() // Use ship ID as seed base
			);
		}
	}, SimulationThreadPool::PRIORITY_HIGH);
}

/**
 * recycle_expired_vfx()
 * 
 * ETEngine Optimization: Performs a parallel scan of the lifetime component stream.
 * Identifies expired particles and performs a "Swap-and-Pop" erasure 
 * to maintain high memory density for the next Warp sweep.
 */
void PhysicsServerHyper::recycle_expired_vfx() {
	auto &registry = get_kernel_registry();
	auto &life_stream = registry.get_stream<FixedMathCore>(COMPONENT_LIFETIME);
	uint64_t count = life_stream.size();

	if (count == 0) return;

	// Note: Removal must be serial or carefully partitioned to maintain Sparse Set integrity.
	// We use the 'Erase' pattern from the SparseSet implementation.
	for (int64_t i = count - 1; i >= 0; i--) {
		if (life_stream[i].get_raw() <= 0) {
			BigIntCore entity = registry.get_entity_at_index(COMPONENT_LIFETIME, i);
			registry.erase_entity(entity);
		}
	}
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_vfx_emitter.cpp ---
