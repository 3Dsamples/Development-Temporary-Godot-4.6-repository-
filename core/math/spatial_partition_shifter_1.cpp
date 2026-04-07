--- START OF FILE core/math/spatial_partition_shifter.cpp ---

#include "core/math/spatial_partition.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: SpatialOriginShiftKernel
 * 
 * Performs a zero-copy update of spatial components during a world-recenter.
 * 1. Adjusts local FixedMathCore positions by the shift vector.
 * 2. Increments/Decrements BigIntCore sector coordinates.
 * 3. Pre-calculates the new Morton/Hash key for the next broadphase pass.
 */
void spatial_origin_shift_kernel(
		const BigIntCore &p_index,
		Vector3f &r_local_pos,
		BigIntCore &r_sx, BigIntCore &r_sy, BigIntCore &r_sz,
		uint64_t &r_spatial_key,
		const Vector3f &p_shift_offset,
		const BigIntCore &p_dsx, const BigIntCore &p_dsy, const BigIntCore &p_dsz,
		const FixedMathCore &p_cell_size) {

	// 1. Shift Local Offset (Deterministic FixedMath Subtraction)
	r_local_pos -= p_shift_offset;

	// 2. Adjust Sector Coordinates (Arbitrary Precision BigInt Addition)
	r_sx += p_dsx;
	r_sy += p_dsy;
	r_sz += p_dsz;

	// 3. Re-calculate Spatial Key (Z-Order Morton or Hash)
	// ensures that the broadphase remains coherent after the shift.
	int64_t cx = Math::floor(r_local_pos.x / p_cell_size).to_int();
	int64_t cy = Math::floor(r_local_pos.y / p_cell_size).to_int();
	int64_t cz = Math::floor(r_local_pos.z / p_cell_size).to_int();

	uint32_t h = r_sx.hash();
	h = hash_murmur3_one_32(r_sy.hash(), h);
	h = hash_murmur3_one_32(r_sz.hash(), h);
	
	uint64_t final_key = hash_murmur3_one_64(static_cast<uint64_t>(cx), static_cast<uint64_t>(h));
	final_key = hash_murmur3_one_64(static_cast<uint64_t>(cy), final_key);
	final_key = hash_murmur3_one_64(static_cast<uint64_t>(cz), final_key);

	r_spatial_key = final_key;
}

/**
 * shift_galactic_origin_wave()
 * 
 * Master orchestrator for global origin migration.
 * Triggered when a spaceship or player crosses a sector boundary.
 * partitions the entire EnTT registry and executes the shift kernel in parallel.
 */
void shift_galactic_origin_wave(
		KernelRegistry &p_registry,
		const Vector3f &p_shift_offset,
		const BigIntCore &p_dsx, const BigIntCore &p_dsy, const BigIntCore &p_dsz,
		const FixedMathCore &p_cell_size) {

	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &sx_stream = p_registry.get_stream<BigIntCore>(COMPONENT_SECTOR_X);
	auto &sy_stream = p_registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Y);
	auto &sz_stream = p_registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Z);
	auto &key_stream = p_registry.get_stream<uint64_t>(COMPONENT_SPATIAL_KEY);

	uint64_t entity_count = pos_stream.size();
	if (entity_count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = entity_count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? entity_count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &pos_stream, &sx_stream, &sy_stream, &sz_stream, &key_stream]() {
			for (uint64_t i = start; i < end; i++) {
				spatial_origin_shift_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					pos_stream[i],
					sx_stream[i], sy_stream[i], sz_stream[i],
					key_stream[i],
					p_shift_offset,
					p_dsx, p_dsy, p_dsz,
					p_cell_size
				);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}

	// Final Synchronization Barrier: Ensure every entity in the universe has moved.
	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * migrate_spatial_partition_buckets()
 * 
 * Rebuilds the hash grid buckets from the updated EnTT spatial keys.
 * strictly deterministic O(N) pass.
 */
template <typename UserData>
void migrate_spatial_partition_buckets(
		SpatialPartition<UserData> &r_partition,
		const uint64_t *p_new_keys,
		const UserData *p_data_handles,
		uint64_t p_count) {

	// 1. Clear old bucket pointers (Zero-Copy reset)
	r_partition.clear_grid_only();

	// 2. Re-assign handles to new bit-perfect buckets
	for (uint64_t i = 0; i < p_count; i++) {
		r_partition.direct_insert_to_bucket(p_new_keys[i], p_data_handles[i]);
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/spatial_partition_shifter.cpp ---
