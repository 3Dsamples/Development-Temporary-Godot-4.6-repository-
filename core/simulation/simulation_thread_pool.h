--- START OF FILE core/simulation/simulation_thread_pool.h ---

#ifndef SIMULATION_THREAD_POOL_H
#define SIMULATION_THREAD_POOL_H

#include "core/object/object.h"
#include "core/templates/vector.h"
#include "core/templates/list.h"
#include "core/os/memory.h"
#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

/**
 * SimulationThreadPool
 * 
 * The parallel execution backbone for the Universal Solver.
 * Manages worker threads with hardware affinity for Warp-style kernel sweeps.
 * Uses BigIntCore for infinite task tracking and FixedMathCore for load balancing.
 */
class SimulationThreadPool : public Object {
	GDCLASS(SimulationThreadPool, Object);

	static SimulationThreadPool *singleton;

public:
	// Priority levels for simulation tasks
	enum TaskPriority {
		PRIORITY_CRITICAL, // Deterministic Physics / CCD (Must finish before frame ends)
		PRIORITY_HIGH,     // Mesh Deformation / Vertex Integration
		PRIORITY_NORMAL,   // AI / Non-critical Logic
		PRIORITY_LOW       // Procedural Detail / Background Streaming
	};

	typedef std::function<void()> TaskFunc;

private:
	struct Task {
		TaskFunc func;
		TaskPriority priority;
		uint64_t id;
	};

	Vector<std::thread> workers;
	List<Task> task_queue[4];
	
	std::mutex queue_mutex;
	std::condition_variable condition;
	std::atomic<bool> stop_requested;
	std::atomic<uint32_t> active_worker_count;
	
	uint64_t next_task_id = 0;
	
	// ETEngine Telemetry
	BigIntCore total_tasks_executed;
	std::atomic<int64_t> pending_tasks_count;

	void _worker_loop(uint32_t p_core_index);

protected:
	static void _bind_methods();

public:
	static _FORCE_INLINE_ SimulationThreadPool *get_singleton() { return singleton; }

	/**
	 * enqueue_task()
	 * Injects a math kernel or logic task into the pool.
	 * Returns a unique task handle for synchronization.
	 */
	uint64_t enqueue_task(const TaskFunc &p_func, TaskPriority p_priority = PRIORITY_NORMAL);

	/**
	 * wait_for_all()
	 * Synchronization barrier. Blocks until the current simulation sweep is complete.
	 * Essential for maintaining bit-perfect deterministic state between physics ticks.
	 */
	void wait_for_all();

	// ------------------------------------------------------------------------
	// Hardware & Telemetry
	// ------------------------------------------------------------------------

	uint32_t get_worker_count() const { return workers.size(); }
	uint32_t get_active_worker_count() const { return active_worker_count.load(); }
	
	/**
	 * get_total_processed()
	 * Returns the absolute count of executed kernels since startup as BigIntCore.
	 */
	BigIntCore get_total_processed() const { return total_tasks_executed; }

	SimulationThreadPool();
	virtual ~SimulationThreadPool();
};

#endif // SIMULATION_THREAD_POOL_H

--- END OF FILE core/simulation/simulation_thread_pool.h ---
