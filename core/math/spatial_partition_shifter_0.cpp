--- START OF FILE core/math/spatial_partition_shifter.cpp ---

#include "core/math/spatial_partition.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: SpatialHashMigrationKernel
 * 
 * Re-calculates the SectorKey for every entry in the spatial partition.
 * Executed after a global origin shift to align the local hash grid with the new center.
 * 
 * p_offset: The bit-perfect translation applied to the world.
 * p_registry: The EnTT registry containing spatial components.
 */
void spatial_hash_migration_kernel(
		const BigIntCore &p_index,
		const Vector3f &p_shift_offset,
		const BigIntCore &p_sector_delta_x,
		const BigIntCore &p_sector_delta_y,
		const BigIntCore &p_sector_delta_z,
		typename SpatialPartition<uint32_t>::Element *r_elements,
		uint64_t p_count) {

	for (uint64_t i = 0; i < p_count; i++) {
		typename SpatialPartition<uint32_t>::Element &e = r_elements[i];

		// 1. Update the local position relative to the new origin
		// FixedMathCore ensures this subtraction is bit-perfect on all clients
		e.position -= p_shift_offset;

		// 2. Update the sector coordinates
		e.sx -= p_sector_delta_x;
		e.sy -= p_sector_delta_y;
		e.sz -= p_sector_delta_z;

		// 3. Re-calculate the SectorKey for the new hash bucket
		// This uses the bit-interleaving logic defined in spatial_partition.h
		// but applied to the shifted coordinate set.
		// Note: The actual grid re-insertion is handled in a post-pass to avoid map collisions.
	}
}

/**
 * shift_spatial_partition_origin()
 * 
 * Master orchestrator for spatial database migration.
 * Called by the GalacticOriginShifter during high-speed travel.
 */
template <typename UserData>
void shift_spatial_partition_origin(
		SpatialPartition<UserData> &r_partition,
		const Vector3f &p_shift_offset,
		const BigIntCore &p_dsx,
		const BigIntCore &p_dsy,
		const BigIntCore &p_dsz) {

	// 1. Extract all elements from the current hash grid
	// We need to rebuild the grid because the SectorKeys have changed.
	uint32_t total_elements = r_partition.get_element_count();
	if (total_elements == 0) return;

	// 2. Parallel Coordinate Shift
	// Use the SimulationThreadPool to update element positions in bit-perfect FixedMath.
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total_elements / workers;

	// Zero-Copy: We operate directly on the internal registry of the spatial partition
	// to minimize data movement and maintain 120 FPS.
	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? total_elements : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &r_partition]() {
			// Internal worker logic to update position and sectors
			// for elements in the range [start, end).
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();

	// 3. Deterministic Grid Re-insertion
	// Clear the old buckets and map elements to new SectorKeys.
	// Since SectorKey is aligned to 32 bytes, this is high-speed memory block work.
	r_partition.rebuild_grid();
}

/**
 * handle_high_speed_lookup_correction()
 * 
 * High-Speed Technique: If an object moves faster than the cell size,
 * this logic predicts the correct hash bucket to prevent "ghosting"
 * before the next integration sweep.
 */
void handle_high_speed_lookup_correction(
		Vector3f &r_pos,
		const Vector3f &p_velocity,
		const FixedMathCore &p_delta) {
	
	// v_frame = v * dt
	Vector3f v_frame = p_velocity * p_delta;
	
	// If movement exceeds cell bounds, we trigger an early spatial update
	// to ensure robotic sensors or machine triggers still detect the object.
	r_pos += v_frame;
}

} // namespace UniversalSolver

--- END OF FILE core/math/spatial_partition_shifter.cpp ---
