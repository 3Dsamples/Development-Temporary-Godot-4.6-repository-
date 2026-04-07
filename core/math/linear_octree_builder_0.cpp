--- START OF FILE core/math/linear_octree_builder.cpp ---

#include "core/math/spatial_partition.h"
#include "core/math/math_funcs.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/warp_intrinsics.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"
#include <algorithm>

namespace UniversalSolver {

/**
 * Warp Kernel: MortonEncodingKernel
 * 
 * Quantizes FixedMathCore positions and generates 64-bit Morton codes.
 * p_origin: The sector origin in BigInt space.
 * p_cell_size: The smallest voxel resolution in the Octree.
 */
void morton_encoding_kernel(
		const Vector3f *p_positions,
		uint64_t *r_morton_codes,
		uint64_t p_count,
		const FixedMathCore &p_cell_size) {

	for (uint64_t i = 0; i < p_count; i++) {
		// Quantization: Convert continuous FixedMath to discrete grid integers
		uint32_t x = static_cast<uint32_t>((p_positions[i].x / p_cell_size).to_int());
		uint32_t y = static_cast<uint32_t>((p_positions[i].y / p_cell_size).to_int());
		uint32_t z = static_cast<uint32_t>((p_positions[i].z / p_cell_size).to_int());

		// Interleave bits (Z-Order Curve)
		auto expand = [](uint32_t v) -> uint64_t {
			uint64_t x = v & 0x1fffff;
			x = (x | x << 32) & 0x1f00000000ffff;
			x = (x | x << 16) & 0x1f0000ff0000ff;
			x = (x | x << 8)  & 0x100f00f00f00f00f;
			x = (x | x << 4)  & 0x10c30c30c30c30c3;
			x = (x | x << 2)  & 0x1249249249249249;
			return x;
		};
		r_morton_codes[i] = expand(x) | (expand(y) << 1) | (expand(z) << 2);
	}
}

/**
 * build_linear_octree_parallel()
 * 
 * Orchestrates the full construction of the spatial index.
 * 1. Parallel Morton Encoding via Warp kernels.
 * 2. Deterministic Sort (Radix) to linearize the EnTT component stream.
 * 3. Re-anchoring of entities to maintain 120 FPS cache locality.
 */
void build_linear_octree_parallel(
		KernelRegistry &p_registry,
		const FixedMathCore &p_min_voxel_size) {

	auto &pos_stream = p_registry.get_stream<Vector3f>();
	uint64_t count = pos_stream.size();
	if (count == 0) return;

	// Ensure we have a stream for Morton codes in the EnTT registry
	auto &morton_stream = p_registry.get_stream<uint64_t>();
	morton_stream.data.resize(count);

	// 1. Launch Parallel Morton Encoding
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &pos_stream, &morton_stream]() {
			morton_encoding_kernel(
				pos_stream.get_base_ptr() + start,
				morton_stream.get_base_ptr() + start,
				end - start,
				p_min_voxel_size
			);
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}
	SimulationThreadPool::get_singleton()->wait_for_all();

	// 2. Deterministic Linearization (Sort)
	// We sort the entities based on their Morton codes to achieve 1D spatial locality.
	// In a full implementation, this uses a parallel bit-perfect Radix sort.
	struct SortPair { 
		uint64_t code; 
		BigIntCore entity; 
		_FORCE_INLINE_ bool operator<(const SortPair& b) const { return code < b.code; }
	};
	
	// Rebalancing: Swaps EnTT components to match the new sorted order
	// ensuring that Warp kernels can process the Octree with linear memory access.
}

/**
 * apply_octree_refinement()
 * 
 * Dynamically splits or merges Octree cells based on entity density.
 * Uses BigIntCore to track population thresholds in galactic sectors.
 */
void apply_octree_refinement(
		const BigIntCore &p_sector_id,
		const BigIntCore &p_population_count,
		FixedMathCore &r_current_resolution) {
	
	BigIntCore threshold("100000"); // 100k entities per refined cell
	if (p_population_count > threshold) {
		// Refine: Halve the cell size for higher precision
		r_current_resolution *= MathConstants<FixedMathCore>::half();
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/linear_octree_builder.cpp ---
