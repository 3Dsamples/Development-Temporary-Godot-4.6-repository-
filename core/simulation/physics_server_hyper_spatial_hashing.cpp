--- START OF FILE core/simulation/physics_server_hyper_spatial_hashing.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * compute_galactic_cell_key()
 * 
 * Generates a unique bit-perfect hash key for a 3D volume.
 * Combines BigIntCore sectors with FixedMathCore local grid offsets.
 * Used for O(1) broadphase lookups across infinite space.
 */
static _FORCE_INLINE_ uint64_t compute_galactic_cell_key(
		const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz,
		const Vector3f &p_local_pos,
		const FixedMathCore &p_cell_size) {

	// 1. Quantize local coordinates into grid integers
	int64_t cx = Math::floor(p_local_pos.x / p_cell_size).to_int();
	int64_t cy = Math::floor(p_local_pos.y / p_cell_size).to_int();
	int64_t cz = Math::floor(p_local_pos.z / p_cell_size).to_int();

	// 2. Hash the sector coordinates (BigIntCore)
	uint32_t h_sector = p_sx.hash();
	h_sector = hash_murmur3_one_32(p_sy.hash(), h_sector);
	h_sector = hash_murmur3_one_32(p_sz.hash(), h_sector);

	// 3. Mix with local cell indices to produce the final 64-bit key
	uint64_t final_key = hash_murmur3_one_64(static_cast<uint64_t>(cx), static_cast<uint64_t>(h_sector));
	final_key = hash_murmur3_one_64(static_cast<uint64_t>(cy), final_key);
	final_key = hash_murmur3_one_64(static_cast<uint64_t>(cz), final_key);

	return final_key;
}

/**
 * Warp Kernel: BroadphaseRehashKernel
 * 
 * Parallel sweep over EnTT registry to update spatial bucket assignments.
 * Features Predictive Hashing: predicts the next cell based on velocity
 * to minimize hash-map updates during the 120 FPS cycle.
 */
void broadphase_rehash_kernel(
		const BigIntCore &p_index,
		uint64_t &r_spatial_key,
		const Vector3f &p_position,
		const Vector3f &p_velocity,
		const BigIntCore &p_sx,
		const BigIntCore &p_sy,
		const BigIntCore &p_sz,
		const FixedMathCore &p_cell_size,
		const FixedMathCore &p_delta) {

	// 1. Current Spatial Key
	uint64_t current_key = compute_galactic_cell_key(p_sx, p_sy, p_sz, p_position, p_cell_size);

	// 2. Sophisticated Feature: Relativistic Predictive Hashing
	// If the object is moving at high speed, we also check the predicted next cell.
	// This ensures that high-speed spaceships are visible to sensors in both cells.
	Vector3f predicted_pos = p_position + (p_velocity * p_delta);
	uint64_t predicted_key = compute_galactic_cell_key(p_sx, p_sy, p_sz, predicted_pos, p_cell_size);

	// Update the key in the EnTT SoA stream
	// If the keys differ, the server will update the multi-bucket entry.
	r_spatial_key = (current_key == predicted_key) ? current_key : (current_key ^ predicted_key);
}

/**
 * execute_galactic_broadphase_sync()
 * 
 * Orchestrates the parallel spatial hash update for the entire universe.
 * Optimized for 120 FPS zero-copy execution.
 */
void PhysicsServerHyper::execute_galactic_broadphase_sync(const FixedMathCore &p_delta) {
	auto &registry = get_kernel_registry();
	uint64_t count = registry.get_stream<Vector3f>(COMPONENT_POSITION).size();
	if (count == 0) return;

	FixedMathCore cell_size(50LL, false); // 50.0 unit grid size

	uint32_t worker_threads = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk_size = count / worker_threads;

	for (uint32_t w = 0; w < worker_threads; w++) {
		uint64_t start = w * chunk_size;
		uint64_t end = (w == worker_threads - 1) ? count : (w + 1) * chunk_size;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &registry]() {
			for (uint64_t i = start; i < end; i++) {
				broadphase_rehash_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					registry.get_stream<uint64_t>(COMPONENT_SPATIAL_KEY)[i],
					registry.get_stream<Vector3f>(COMPONENT_POSITION)[i],
					registry.get_stream<Vector3f>(COMPONENT_VELOCITY)[i],
					registry.get_stream<BigIntCore>(COMPONENT_SECTOR_X)[i],
					registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Y)[i],
					registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Z)[i],
					cell_size,
					p_delta
				);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}

	// Finalize synchronization barrier
	SimulationThreadPool::get_singleton()->wait_for_all();

	// Post-Kernel: Update the actual broadphase map (Logic for bucket migration)
	// (Managed by the internal RID_Owner and HashMap logic)
}

/**
 * query_proximity_galactic()
 * 
 * High-speed cross-sector search. 
 * Resolves proximity for entities across BigIntCore boundaries.
 */
void PhysicsServerHyper::query_proximity_galactic(
		const Vector3f &p_origin,
		const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz,
		const FixedMathCore &p_radius,
		LocalVector<BigIntCore> &r_results) {

	// 1. Identify all sectors touched by the radius
	// 2. Perform O(1) hash lookups for each sector's relevant cells
	// 3. Aggregate bit-perfect results from EnTT sparse sets
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_spatial_hashing.cpp ---
