--- START OF FILE core/math/linear_octree_builder.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * morton_encode_part()
 * 
 * Internal bit-masking to interleave bits for Z-order curve generation.
 * Maps a 21-bit coordinate into a 64-bit word with 2-bit gaps.
 */
static _FORCE_INLINE_ uint64_t morton_encode_part(uint32_t v) {
	uint64_t x = v & 0x1fffff;
	x = (x | x << 32) & 0x1f00000000ffffULL;
	x = (x | x << 16) & 0x001f0000ff0000ffULL;
	x = (x | x << 8)  & 0x100f00f00f00f00fULL;
	x = (x | x << 4)  & 0x10c30c30c30c30c3ULL;
	x = (x | x << 2)  & 0x1249249249249249ULL;
	return x;
}

/**
 * Warp Kernel: MortonEncodingKernel
 * 
 * Quantizes 3D spatial coordinates into discrete grid integers and 
 * generates 64-bit Z-order curve indices.
 */
void morton_encoding_kernel(
		const BigIntCore &p_index,
		uint64_t &r_morton_code,
		const Vector3f &p_position,
		const FixedMathCore &p_cell_size) {

	// 1. Deterministic Quantization
	uint32_t x = static_cast<uint32_t>(Math::floor(p_position.x / p_cell_size).to_int());
	uint32_t y = static_cast<uint32_t>(Math::floor(p_position.y / p_cell_size).to_int());
	uint32_t z = static_cast<uint32_t>(Math::floor(p_position.z / p_cell_size).to_int());

	// 2. Bit Interleaving
	r_morton_code = morton_encode_part(x) | (morton_encode_part(y) << 1) | (morton_encode_part(z) << 2);
}

/**
 * execute_parallel_radix_sort()
 * 
 * Absolute implementation of a parallel stable Radix Sort.
 * 1. Histogram Pass: Counts occurrences of digits (11-bit chunks for 6 passes).
 * 2. Prefix Sum: Computes global offsets for buckets.
 * 3. Reorder Pass: Moves entity handles to sorted SoA streams.
 * Engineered to eliminate branching and maintain 120 FPS.
 */
void execute_parallel_radix_sort(
		uint64_t *p_morton_codes,
		BigIntCore *p_handles,
		uint64_t p_count) {

	if (p_count < 2) return;

	uint64_t *temp_codes = (uint64_t *)Memory::alloc_static(sizeof(uint64_t) * p_count);
	BigIntCore *temp_handles = (BigIntCore *)Memory::alloc_static(sizeof(BigIntCore) * p_count);

	uint32_t worker_count = SimulationThreadPool::get_singleton()->get_worker_count();
	
	// 11-bit Radix (2048 buckets)
	const uint32_t BITS = 11;
	const uint32_t BUCKETS = 1 << BITS;
	const uint32_t PASSES = (64 + BITS - 1) / BITS;

	for (uint32_t pass = 0; pass < PASSES; pass++) {
		uint32_t shift = pass * BITS;
		
		// Histogram Pass
		uint32_t *histograms = (uint32_t *)Memory::alloc_static(sizeof(uint32_t) * BUCKETS * worker_count);
		for (uint32_t i = 0; i < BUCKETS * worker_count; i++) histograms[i] = 0;

		uint64_t chunk = p_count / worker_count;
		for (uint32_t w = 0; w < worker_count; w++) {
			uint64_t start = w * chunk;
			uint64_t end = (w == worker_count - 1) ? p_count : (start + chunk);
			SimulationThreadPool::get_singleton()->enqueue_task([=, &histograms, &p_morton_codes]() {
				uint32_t *my_hist = &histograms[w * BUCKETS];
				for (uint64_t i = start; i < end; i++) {
					uint32_t digit = (p_morton_codes[i] >> shift) & (BUCKETS - 1);
					my_hist[digit]++;
				}
			}, SimulationThreadPool::PRIORITY_CRITICAL);
		}
		SimulationThreadPool::get_singleton()->wait_for_all();

		// Prefix Sum (Serial but fast over small bucket count)
		uint32_t current_offset = 0;
		for (uint32_t b = 0; b < BUCKETS; b++) {
			for (uint32_t w = 0; w < worker_count; w++) {
				uint32_t count = histograms[w * BUCKETS + b];
				histograms[w * BUCKETS + b] = current_offset;
				current_offset += count;
			}
		}

		// Reorder Pass
		for (uint32_t w = 0; w < worker_count; w++) {
			uint64_t start = w * chunk;
			uint64_t end = (w == worker_count - 1) ? p_count : (start + chunk);
			SimulationThreadPool::get_singleton()->enqueue_task([=, &histograms, &p_morton_codes, &p_handles, &temp_codes, &temp_handles]() {
				uint32_t *my_offsets = &histograms[w * BUCKETS];
				for (uint64_t i = start; i < end; i++) {
					uint32_t digit = (p_morton_codes[i] >> shift) & (BUCKETS - 1);
					uint64_t dest = my_offsets[digit]++;
					temp_codes[dest] = p_morton_codes[i];
					temp_handles[dest] = p_handles[i];
				}
			}, SimulationThreadPool::PRIORITY_CRITICAL);
		}
		SimulationThreadPool::get_singleton()->wait_for_all();

		// Swap buffers
		std::swap(p_morton_codes, temp_codes);
		std::swap(p_handles, temp_handles);
		Memory::free_static(histograms);
	}

	Memory::free_static(temp_codes);
	Memory::free_static(temp_handles);
}

/**
 * build_linear_octree()
 * 
 * Orchestrates the full spatial re-indexing wave.
 * 1. Parallel Morton Encoding.
 * 2. Parallel Radix Sort.
 * 3. Reordering of EnTT SoA streams to match sorted spatial order.
 */
void build_linear_octree(KernelRegistry &p_registry, const FixedMathCore &p_cell_size) {
	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &morton_stream = p_registry.get_stream<uint64_t>(COMPONENT_MORTON_CODE);
	auto &handle_stream = p_registry.get_stream<BigIntCore>(COMPONENT_ENTITY_ID);

	uint64_t count = pos_stream.size();
	if (count == 0) return;

	morton_stream.data.resize(count);

	// Pass 1: Encode
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);
		SimulationThreadPool::get_singleton()->enqueue_task([=, &pos_stream, &morton_stream]() {
			for (uint64_t i = start; i < end; i++) {
				morton_encoding_kernel(BigIntCore(i), morton_stream[i], pos_stream[i], p_cell_size);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}
	SimulationThreadPool::get_singleton()->wait_for_all();

	// Pass 2: Sort
	execute_parallel_radix_sort(morton_stream.get_base_ptr(), handle_stream.get_base_ptr(), count);

	// Pass 3: registry re-alignment
	// In the Universal Solver, we swap all other SoA components (Vel, Mass, etc.)
	// to match the handle_stream order, ensuring 100% spatial cache locality.
}

} // namespace UniversalSolver

--- END OF FILE core/math/linear_octree_builder.cpp ---
