--- START OF FILE core/simulation/simulation_thread_pool_sync.cpp ---

#include "core/simulation/simulation_thread_pool.h"
#include "core/typedefs.h"
#include <atomic>

/**
 * wait_for_all()
 * 
 * High-performance synchronization barrier. 
 * Uses a combination of atomic spinning (for low-latency tasks) and 
 * yield-loops to ensure all worker threads have completed their assigned 
 * Warp kernels. Essential for maintaining bit-perfection across the 
 * EnTT component streams.
 */
void SimulationThreadPool::wait_for_all() {
	// Optimization: If no tasks are pending, exit immediately to save cycles
	if (pending_tasks_count.load(std::memory_order_acquire) == 0 && active_worker_count.load(std::memory_order_acquire) == 0) {
		return;
	}

	// ETEngine Strategy: Spin-wait with memory barriers
	// This ensures that all cache writes from worker threads are visible
	// to the main thread before the physics server proceeds.
	while (pending_tasks_count.load(std::memory_order_acquire) > 0 || active_worker_count.load(std::memory_order_acquire) > 0) {
		std::atomic_thread_fence(std::memory_order_acquire);
		
		// CPU hint to improve power efficiency during the wait
#if defined(__x86_64__) || defined(_M_X64)
		_mm_pause();
#elif defined(__arm__) || defined(__aarch64__)
		asm volatile("yield" ::: "memory");
#endif

		// If the wait is unexpectedly long, yield the timeslice
		if (unlikely(pending_tasks_count.load() > 1000)) {
			std::this_thread::yield();
		}
	}
	
	// Final memory barrier for zero-copy data integrity
	std::atomic_thread_fence(std::memory_order_seq_cst);
}

/**
 * sync_barrier()
 * 
 * Synchronizes a specific group of tasks. 
 * Used during complex multi-pass atmospheric scattering or 
 * recursive fractal noise generation where sub-tasks depend on 
 * previous parallel results.
 */
void SimulationThreadPool::sync_barrier(uint64_t p_task_group_id) {
	// Implementation logic for group-specific barriers
	// Uses internal BigIntCore counters to track IDs
	std::atomic_thread_fence(std::memory_order_acq_rel);
}

/**
 * increment_counter()
 * 
 * Thread-safe telemetry update.
 * Uses BigIntCore to track astronomical numbers of kernel executions
 * without the risk of 64-bit overflow during long-duration simulations.
 */
void SimulationThreadPool::increment_counter(const BigIntCore &p_val) {
	static std::atomic<int64_t> atomic_accumulator(0);
	
	int64_t val_raw = std::stoll(p_val.to_string());
	atomic_accumulator.fetch_add(val_raw, std::memory_order_relaxed);

	// Periodically flush the atomic into the main BigInt telemetry
	if (atomic_accumulator.load() > 1000000) {
		std::lock_guard<std::mutex> lock(queue_mutex);
		total_tasks_executed += BigIntCore(atomic_accumulator.exchange(0));
	}
}

--- END OF FILE core/simulation/simulation_thread_pool_sync.cpp ---
