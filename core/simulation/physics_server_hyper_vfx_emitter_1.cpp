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
 * Initializes a batch of newly allocated entities in the EnTT registry.
 * 1. Sets world-space position relative to the emitter and galactic sector.
 * 2. Inherits parent velocity and applies ejection impulse.
 * 3. Assigns deterministic lifetimes and spectral energy tensors.
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

	// 1. Initialize Deterministic Entropy for this Warp lane
	RandomPCG pcg;
	pcg.seed_big(p_seed);
	pcg.seed(pcg.rand64() ^ p_index.hash());

	// 2. Spatial Anchoring
	// Inherit galactic sector from emitter
	r_sx = p_esx;
	r_sy = p_esy;
	r_sz = p_esz;

	// 3. Velocity Resolve (High-Speed Inheritance)
	// v_particle = v_parent + (Dir + Jitter) * Force
	Vector3f jitter(
		(pcg.randf() - MathConstants<FixedMathCore>::half()) * p_spread,
		(pcg.randf() - MathConstants<FixedMathCore>::half()) * p_spread,
		(pcg.randf() - MathConstants<FixedMathCore>::half()) * p_spread
	);
	
	Vector3f final_dir = (p_ejection_dir + jitter).normalized();
	r_velocity = p_emitter_vel + (final_dir * p_ejection_force * pcg.randf());

	// 4. Initial Position with Sub-Frame Jitter
	// Prevents "Banding" artifacts at 120 FPS by interpolating within the emission tick.
	FixedMathCore sub_tick = pcg.randf() * MathDefs::get_fixed_step();
	r_position = p_emitter_pos + (r_velocity * sub_tick);

	// 5. Material Tensors
	// Lifespan: [2.0, 5.0] seconds in bit-perfect FixedMath
	r_lifetime = FixedMathCore(2LL, false) + (pcg.randf() * FixedMathCore(3LL, false));
	r_fatigue = MathConstants<FixedMathCore>::zero();
}

/**
 * execute_vfx_emission_batch()
 * 
 * Orchestrates the parallel allocation and initialization of simulation particles.
 * Triggered by high-speed ship engines, weapon impacts, or structural fractures.
 * strictly uses zero-copy EnTT stream access.
 */
void PhysicsServerHyper::execute_vfx_emission_batch(
		const BigIntCore &p_emitter_id,
		const Vector3f &p_local_offset,
		const Vector3f &p_direction,
		const FixedMathCore &p_force,
		const BigIntCore &p_count) {

	Body *emitter = body_owner.get_or_null(RID(p_emitter_id));
	if (unlikely(!emitter)) return;

	KernelRegistry &registry = get_kernel_registry();
	uint64_t n_to_spawn = static_cast<uint64_t>(std::stoll(p_count.to_string()));
	
	// Determine world-space emission point across sectors
	Vector3f world_origin = emitter->transform.xform(p_local_offset);
	Vector3f world_dir = emitter->transform.basis.xform(p_direction).normalized();

	// Sophisticated Behavior: Recoil Force
	// Injects an equal and opposite impulse into the emitter body.
	emitter->linear_velocity -= (world_dir * p_force * FixedMathCore(static_cast<int64_t>(n_to_spawn), false)) / (emitter->mass + MathConstants<FixedMathCore>::one());

	// Parallel Batch Initialization
	uint32_t worker_threads = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk_size = n_to_spawn / worker_threads;

	for (uint32_t w = 0; w < worker_threads; w++) {
		uint64_t start = w * chunk_size;
		uint64_t end = (w == worker_threads - 1) ? n_to_spawn : (start + chunk_size);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &registry, &emitter]() {
			// Get raw pointers for the newly allocated entities
			auto &pos_stream = registry.get_stream<Vector3f>(COMPONENT_POSITION);
			auto &vel_stream = registry.get_stream<Vector3f>(COMPONENT_VELOCITY);
			auto &sx_stream = registry.get_stream<BigIntCore>(COMPONENT_SECTOR_X);
			auto &sy_stream = registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Y);
			auto &sz_stream = registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Z);
			auto &life_stream = registry.get_stream<FixedMathCore>(COMPONENT_LIFETIME);
			auto &fatigue_stream = registry.get_stream<FixedMathCore>(COMPONENT_FATIGUE);

			for (uint64_t i = start; i < end; i++) {
				// Allocate Entity in O(1) from the free-list or incrementing counter
				BigIntCore new_p_id = registry.create_entity();
				uint64_t dense_idx = registry.index_of(new_p_id);

				particle_initialization_kernel(
					new_p_id,
					pos_stream[dense_idx],
					vel_stream[dense_idx],
					sx_stream[dense_idx], sy_stream[dense_idx], sz_stream[dense_idx],
					life_stream[dense_idx],
					fatigue_stream[dense_idx],
					world_origin,
					emitter->linear_velocity,
					emitter->sector_x, emitter->sector_y, emitter->sector_z,
					world_dir,
					p_force,
					FixedMathCore("0.05"), // Spread
					emitter->self.get_id()  // Seed
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * recycle_expired_entities()
 * 
 * Performs a high-speed parallel scan to identify expired particles.
 * strictly uses Swap-and-Pop to maintain EnTT SoA stream density for the next 120 FPS wave.
 */
void PhysicsServerHyper::recycle_expired_entities() {
	KernelRegistry &registry = get_kernel_registry();
	auto &life_stream = registry.get_stream<FixedMathCore>(COMPONENT_LIFETIME);
	uint64_t count = life_stream.size();
	if (count == 0) return;

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();

	// Process in reverse order to ensure indices remain valid during Swap-and-Pop
	for (int64_t i = count - 1; i >= 0; i--) {
		if (life_stream[i] <= zero) {
			BigIntCore handle = registry.get_entity_at_index(COMPONENT_LIFETIME, i);
			registry.destroy_entity(handle);
		}
	}
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_vfx_emitter.cpp ---
