--- START OF FILE core/simulation/simulation_thread_pool_worker.h ---

#ifndef SIMULATION_THREAD_POOL_WORKER_H
#define SIMULATION_THREAD_POOL_WORKER_H

#include "core/typedefs.h"
#include "core/math/random_pcg.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"
#include <thread>
#include <atomic>

class SimulationThreadPool;

/**
 * SimulationThreadPoolWorker
 * 
 * Individual execution unit for the parallel simulation engine.
 * Encapsulates thread-local resources to ensure zero-contention
 * and bit-perfect mathematical determinism.
 */
class ET_ALIGN_32 SimulationThreadPoolWorker {
	// Reference to the parent pool for task fetching
	SimulationThreadPool *pool = nullptr;
	
	// Native thread handle
	std::thread thread;
	
	// Unique worker index for hardware affinity and data partitioning
	uint32_t worker_index = 0;
	
	// Thread-local deterministic entropy source
	// Essential for parallel procedural generation (Noise/WFC)
	RandomPCG local_pcg;

	// Atomic flags for state management
	std::atomic<bool> active;
	std::atomic<bool> processing;

	// Telemetry: Tasks processed by this specific core
	BigIntCore local_task_count;

public:
	// ------------------------------------------------------------------------
	// Execution API
	// ------------------------------------------------------------------------

	/**
	 * start()
	 * Spawns the worker thread and binds it to a physical CPU core.
	 */
	void start(SimulationThreadPool *p_pool, uint32_t p_index);

	/**
	 * stop()
	 * Triggers graceful termination of the worker thread.
	 */
	void stop();

	/**
	 * run()
	 * The main execution kernel. Continuously fetches and executes 
	 * Warp-style tasks or EnTT batch chunks.
	 */
	void run();

	// ------------------------------------------------------------------------
	// Telemetry & State
	// ------------------------------------------------------------------------

	_FORCE_INLINE_ uint32_t get_worker_index() const { return worker_index; }
	_FORCE_INLINE_ bool is_processing() const { return processing.load(std::memory_order_acquire); }
	
	/**
	 * get_local_task_count()
	 * Returns the number of kernels executed by this worker as BigIntCore.
	 */
	_FORCE_INLINE_ BigIntCore get_local_task_count() const { return local_task_count; }

	/**
	 * get_pcg()
	 * Provides access to the thread-local deterministic random generator.
	 */
	_FORCE_INLINE_ RandomPCG& get_pcg() { return local_pcg; }

	SimulationThreadPoolWorker();
	~SimulationThreadPoolWorker();
};

#endif // SIMULATION_THREAD_POOL_WORKER_H

--- END OF FILE core/simulation/simulation_thread_pool_worker.h ---
