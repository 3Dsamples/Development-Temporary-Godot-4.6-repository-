--- START OF FILE core/math/warp_kernel.h ---

#ifndef WARP_KERNEL_H
#define WARP_KERNEL_H

#include "core/math/kernel_registry.h"
#include "core/simulation/simulation_thread_pool.h"
#include <tuple>

/**
 * WarpKernel
 * 
 * The execution engine for batch-oriented mathematics.
 * Maps kernel functions to SoA component streams managed by EnTT.
 * Engineered for hardware-agnostic execution with bit-perfect FixedMathCore results.
 */
namespace UniversalSolver {

template <typename... Components>
class WarpKernel {
public:
	/**
	 * KernelFunc
	 * The signature for a mathematical kernel.
	 * p_index: The global simulation index (BigIntCore).
	 * args: References to the component data in the aligned SoA buffer.
	 */
	typedef void (*KernelFunc)(const BigIntCore &p_index, Components &...args);

	/**
	 * launch()
	 * Dispatches the kernel across the entire registry for the specified components.
	 * Divides the EnTT streams into cache-aligned chunks for parallel worker threads.
	 */
	static void launch(KernelRegistry &p_registry, KernelFunc p_func) {
		// Identify the primary stream to determine the total work size
		// In the Universal Solver, we assume synchronized SoA streams for these components
		auto &primary_stream = p_registry.get_stream<typename std::tuple_element<0, std::tuple<Components...>>::type>();
		uint64_t total_work = primary_stream.size();
		if (total_work == 0) return;

		// ETEngine Strategy: Divide work based on CPU hardware thread count
		uint32_t worker_count = SimulationThreadPool::get_singleton()->get_worker_count();
		uint64_t chunk_size = total_work / worker_count;

		for (uint32_t i = 0; i < worker_count; i++) {
			uint64_t start = i * chunk_size;
			uint64_t end = (i == worker_count - 1) ? total_work : (i + 1) * chunk_size;

			SimulationThreadPool::get_singleton()->enqueue_task([&p_registry, p_func, start, end]() {
				_execute_chunk(p_registry, p_func, start, end, std::make_index_sequence<sizeof...(Components)>{});
			}, SimulationThreadPool::PRIORITY_CRITICAL);
		}

		// Block until the simulation wave is complete to guarantee bit-perfect synchronization
		SimulationThreadPool::get_singleton()->wait_for_all();
	}

private:
	/**
	 * _execute_chunk()
	 * Unrolls the component tuple and executes the kernel on a range of entities.
	 * Optimized with ET_SIMD_INLINE to allow compiler auto-vectorization of the math.
	 */
	template <size_t... Is>
	static ET_SIMD_INLINE void _execute_chunk(KernelRegistry &p_registry, KernelFunc p_func, uint64_t p_start, uint64_t p_end, std::index_sequence<Is...>) {
		// Zero-Copy: Get raw pointers to the aligned SoA buffers
		auto streams = std::make_tuple(&p_registry.get_stream<Components>()...);

		for (uint64_t i = p_start; i < p_end; i++) {
			// BigIntCore indexing for infinite-scale entity mapping
			BigIntCore index_bi(static_cast<int64_t>(i));
			
			// Execute kernel with direct memory references to FixedMathCore/BigIntCore data
			p_func(
				index_bi,
				(std::get<Is>(streams)->operator[](i))...
			);
		}
	}
};

} // namespace UniversalSolver

#endif // WARP_KERNEL_H

--- END OF FILE core/math/warp_kernel.h ---
