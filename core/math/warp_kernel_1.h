--- START OF FILE core/math/warp_kernel.h ---

#ifndef WARP_KERNEL_H
#define WARP_KERNEL_H

#include "core/math/kernel_registry.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"
#include <tuple>
#include <utility>

namespace UniversalSolver {

/**
 * WarpKernel
 * 
 * The parallel execution manager for batch-oriented mathematics.
 * Maps high-performance kernels to EnTT-managed SoA component streams.
 * Strictly deterministic: ensures bit-perfect results across all worker threads.
 */
template <typename... Components>
class ET_ALIGN_32 WarpKernel {
public:
	/**
	 * KernelFunc
	 * The signature for a mathematical kernel designed for Warp execution.
	 * p_index: The global simulation index (BigIntCore).
	 * args: References to the component data in the aligned SoA buffer.
	 */
	typedef void (*KernelFunc)(const BigIntCore &p_index, Components &...args);

	/**
	 * launch()
	 * 
	 * The master entry point for a parallel simulation wave.
	 * 1. Locates the specific SoA streams in the KernelRegistry.
	 * 2. Calculates the optimal workload distribution based on CPU topology.
	 * 3. Dispatches zero-copy execution tasks to the SimulationThreadPool.
	 * 4. Blocks until the wave is synchronized to maintain 120 FPS consistency.
	 */
	static void launch(KernelRegistry &p_registry, KernelFunc p_func) {
		// Identify the primary component to determine the total work size
		typedef typename std::tuple_element<0, std::tuple<Components...>>::type PrimaryComponent;
		auto &primary_stream = p_registry.get_stream<PrimaryComponent>();
		
		uint64_t total_work = primary_stream.size();
		if (unlikely(total_work == 0)) {
			return;
		}

		SimulationThreadPool *pool = SimulationThreadPool::get_singleton();
		uint32_t worker_count = pool->get_worker_count();
		
		// Chunking logic: Ensure chunks are large enough to justify thread overhead
		// but small enough to saturate all physical cores.
		uint64_t chunk_size = total_work / worker_count;
		if (chunk_size == 0) chunk_size = total_work;

		for (uint32_t i = 0; i < worker_count; i++) {
			uint64_t start = i * chunk_size;
			uint64_t end = (i == worker_count - 1) ? total_work : (i + 1) * chunk_size;

			if (start >= total_work) break;

			pool->enqueue_task([&p_registry, p_func, start, end]() {
				_execute_chunk(p_registry, p_func, start, end, std::make_index_sequence<sizeof...(Components)>{});
			}, SimulationThreadPool::PRIORITY_CRITICAL);
		}

		// Enforce the execution barrier
		pool->wait_for_all();
	}

private:
	/**
	 * _execute_chunk()
	 * 
	 * Internal unrolled execution kernel.
	 * Uses std::index_sequence to map variadic components to the function call.
	 * Operates directly on the memory addresses provided by the EnTT registry.
	 */
	template <size_t... Is>
	static _FORCE_INLINE_ void _execute_chunk(
			KernelRegistry &p_registry, 
			KernelFunc p_func, 
			uint64_t p_start, 
			uint64_t p_end, 
			std::index_sequence<Is...>) {
		
		// Capture raw pointers to the SoA streams to eliminate lookup overhead in the loop
		auto streams = std::make_tuple(&p_registry.get_stream<Components>()...);

		for (uint64_t i = p_start; i < p_end; i++) {
			// Construct the BigIntCore index for this entity to support galactic scaling
			BigIntCore current_index(static_cast<int64_t>(i));
			
			// Zero-Copy Call: Pass references to the data directly from the registry
			p_func(
				current_index,
				(std::get<Is>(streams)->operator[](i))...
			);
		}
	}
};

/**
 * Parallel Kernel Implementation for TIER_DETERMINISTIC
 * 
 * Provides a simplified interface for launching physics and material kernels.
 */
template <typename... Components>
_FORCE_INLINE_ void warp_launch(KernelRegistry &p_registry, void (*p_func)(const BigIntCore &, Components &...)) {
	WarpKernel<Components...>::launch(p_registry, p_func);
}

} // namespace UniversalSolver

#endif // WARP_KERNEL_H

--- END OF FILE core/math/warp_kernel.h ---
