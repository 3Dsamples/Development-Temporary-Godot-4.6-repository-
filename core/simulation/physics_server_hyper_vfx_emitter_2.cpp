--- START OF FILE core/simulation/physics_server_hyper_vfx_emitter.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/math/random_pcg.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: ParticleInitializationKernel
 * 
 * Performs parallel initialization of a newly allocated block of particles in the EnTT registry.
 * 1. World-Space Positioning: Corrects for galactic sector offsets.
 * 2. Velocity Inheritance: Combines parent spaceship velocity with ejection impulse.
 * 3. Deterministic Lifetime: Assigns bit-perfect lifespan using RandomPCG.
 * 4. Recoil Force: (Resolved in the emission orchestrator).
 */
void particle_initialization_kernel(
		const BigIntCore &p_index,
		Vector3f &r_position,
		Vector3f &r_velocity,
		BigIntCore &r_sx, BigIntCore &r_sy, BigIntCore &r_sz,
		FixedMathCore &r_lifetime,
		FixedMathCore &r_fatigue,
		const Vector3f &p_emitter_pos,
		const Vector3f &p_emitter_vel,
		const BigIntCore &p_esx, const BigIntCore &p_esy, const BigIntCore &p_esz,
		const Vector3f &p_ejection_dir,
		const FixedMathCore &p_ejection_force,
		const FixedMathCore &p_spread,
		const BigIntCore &p_seed) {

	// 1. Resolve Deterministic Entropy for this specific particle index
	RandomPCG pcg;
	pcg.seed_big(p_seed);
	pcg.seed(pcg.rand64() ^ p_index.hash());

	// 2. Spatial Sector Anchoring
	// Particles start in the same galactic sector as the emitter ship/impact
	r_sx = p_esx;
	r_sy = p_esy;
	r_sz = p_esz;

	// 3. Velocity Resolve (High-Speed Inheritance)
	// v_p = v_ship + (EjectionDir + RandomJitter) * Force
	FixedMathCore half = MathConstants<FixedMathCore>::half();
	Vector3f jitter(
		(pcg.randf() - half) * p_spread,
		(pcg.randf() - half) * p_spread,
		(pcg.randf() - half) * p_spread
	);
	
	Vector3f final_dir = (p_ejection_dir + jitter).normalized();
	r_velocity = p_emitter_vel + (final_dir * p_ejection_force * pcg.randf());

	// 4. Initial Position with Sub-Frame Jitter
	// Prevents "Banding" artifacts at 120 FPS by distributing particles within the time-step.
	FixedMathCore sub_tick = pcg.randf() * FixedMathCore(8333333LL, true); // 1/120s approx
	r_position = p_emitter_pos + (r_velocity * sub_tick);

	// 5. Material Tensors
	// Lifespan: [1.0, 3.0] seconds in deterministic FixedMath
	r_lifetime = MathConstants<FixedMathCore>::one() + (pcg.randf() * FixedMathCore(2LL, false));
	r_fatigue = MathConstants<FixedMathCore>::zero();
}

/**
 * execute_vfx_emission_batch()
 * 
 * Master orchestrator for high-throughput particle spawning.
 * Used for high-speed ship engine flares, weapon impacts, and structural shattering.
 * Strictly uses zero-copy EnTT stream access.
 */
void PhysicsServerHyper::execute_vfx_emission_batch(
		const RID &p_emitter_id,
		const Vector3f &p_local_offset,
		const Vector3f &p_direction,
		const FixedMathCore &p_force,
		const BigIntCore &p_count) {

	Body *emitter = body_owner.get_or_null(p_emitter_id);
	if (unlikely(!emitter)) return;

	KernelRegistry &registry = get_kernel_registry();
	uint64_t n_to_spawn = static_cast<uint64_t>(std::stoll(p_count.to_string()));
	
	// Determine world-space emission point relative to the ship's current sector
	Vector3f world_origin = emitter->transform.xform(p_local_offset);
	Vector3f world_dir = emitter->transform.basis.xform(p_direction).normalized();

	// Sophisticated Behavior: Newton's Third Law (Recoil)
	// Injects a bit-perfect opposing impulse into the parent body.
	FixedMathCore total_impulse = p_force * FixedMathCore(static_cast<int64_t>(n_to_spawn), false);
	emitter->linear_velocity -= (world_dir * total_impulse) / (emitter->mass + MathConstants<FixedMathCore>::unit_epsilon());

	// Parallel Batch Initialization Wave
	uint32_t worker_threads = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk_size = n_to_spawn / worker_threads;

	for (uint32_t w = 0; w < worker_threads; w++) {
		uint64_t start = w * chunk_size;
		uint64_t end = (w == worker_threads - 1) ? n_to_spawn : (start + chunk_size);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &registry, &emitter]() {
			// Zero-Copy access to SoA streams
			auto &pos_stream = registry.get_stream<Vector3f>(COMPONENT_POSITION);
			auto &vel_stream = registry.get_stream<Vector3f>(COMPONENT_VELOCITY);
			auto &sx_stream = registry.get_stream<BigIntCore>(COMPONENT_SECTOR_X);
			auto &sy_stream = registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Y);
			auto &sz_stream = registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Z);
			auto &life_stream = registry.get_stream<FixedMathCore>(COMPONENT_LIFETIME);

			for (uint64_t i = start; i < end; i++) {
				// Allocate Entity ID in O(1)
				BigIntCore particle_id = registry.create_entity();
				uint64_t dense_idx = registry.index_of(particle_id);

				particle_initialization_kernel(
					particle_id,
					pos_stream[dense_idx],
					vel_stream[dense_idx],
					sx_stream[dense_idx], sy_stream[dense_idx], sz_stream[dense_idx],
					life_stream[dense_idx],
					registry.get_stream<FixedMathCore>(COMPONENT_FATIGUE)[dense_idx],
					world_origin,
					emitter->linear_velocity,
					emitter->sector_x, emitter->sector_y, emitter->sector_z,
					world_dir,
					p_force,
					FixedMathCore("0.1"), // 10% Spread constant
					emitter->self.get_id() // Use parent handle as entropy seed
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * recycle_expired_vfx()
 * 
 * Performs a high-speed parallel scan of the lifetime component stream.
 * strictly uses Swap-and-Pop to maintain EnTT SoA stream density for 120 FPS.
 */
void PhysicsServerHyper::recycle_expired_vfx() {
	KernelRegistry &registry = get_kernel_registry();
	auto &life_stream = registry.get_stream<FixedMathCore>(COMPONENT_LIFETIME);
	uint64_t count = life_stream.size();
	if (count == 0) return;

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();

	// Process in reverse to ensure indices remain valid during pop
	for (int64_t i = count - 1; i >= 0; i--) {
		if (life_stream[i] <= zero) {
			BigIntCore handle = registry.get_entity_at_index(COMPONENT_LIFETIME, i);
			registry.destroy_entity(handle);
		}
	}
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_vfx_emitter.cpp ---
