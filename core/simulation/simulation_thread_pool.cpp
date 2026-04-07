--- START OF FILE core/simulation/simulation_thread_pool.cpp ---

#include "core/simulation/simulation_thread_pool.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "core/core_logger.h"

#ifdef WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

SimulationThreadPool *SimulationThreadPool::singleton = nullptr;

void SimulationThreadPool::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_worker_count"), &SimulationThreadPool::get_worker_count);
	ClassDB::bind_method(D_METHOD("get_active_worker_count"), &SimulationThreadPool::get_active_worker_count);
	ClassDB::bind_method(D_METHOD("get_total_processed"), &SimulationThreadPool::get_total_processed);
}

SimulationThreadPool::SimulationThreadPool() {
	singleton = this;
	stop_requested.store(false);
	active_worker_count.store(0);
	pending_tasks_count.store(0);
	total_tasks_executed = BigIntCore(0LL);

	// Determine hardware capacity for parallel Warp kernels
	uint32_t thread_count = std::thread::hardware_concurrency();
	if (thread_count < 2) thread_count = 2;

	for (uint32_t i = 0; i < thread_count; i++) {
		workers.push_back(std::thread(&SimulationThreadPool::_worker_loop, this, i));
	}
}

SimulationThreadPool::~SimulationThreadPool() {
	stop_requested.store(true);
	condition.notify_all();

	for (std::thread &t : workers) {
		if (t.joinable()) {
			t.join();
		}
	}

	singleton = nullptr;
}

/**
 * _worker_loop()
 * 
 * Internal execution kernel for worker threads.
 * Uses hardware affinity to maximize L1/L2 cache locality for EnTT SoA streams.
 */
void SimulationThreadPool::_worker_loop(uint32_t p_core_index) {
	// Set Thread Affinity to prevent context-switching jitter during 120 FPS simulation
#ifdef WIN32
	SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << p_core_index);
#elif defined(__linux__) || defined(__APPLE__)
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(p_core_index, &cpuset);
	pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif

	while (!stop_requested.load()) {
		Task task;
		bool found = false;

		{
			std::unique_lock<std::mutex> lock(queue_mutex);
			condition.wait(lock, [this] {
				if (stop_requested.load()) return true;
				for (int i = 0; i < 4; i++) {
					if (!task_queue[i].is_empty()) return true;
				}
				return false;
			});

			if (stop_requested.load()) break;

			// Priority Dispatch: CRITICAL -> HIGH -> NORMAL -> LOW
			for (int i = 0; i < 4; i++) {
				if (!task_queue[i].is_empty()) {
					task = task_queue[i].front()->get();
					task_queue[i].pop_front();
					found = true;
					break;
				}
			}
		}

		if (found) {
			active_worker_count.fetch_add(1);
			
			// Execute Warp Kernel or Simulation Logic
			task.func();

			// Telemetry Update using BigIntCore for infinite tracking
			// Using a local atomic to avoid contention, flushed periodically in a full build
			static thread_local uint64_t local_count = 0;
			local_count++;
			if (local_count >= 1000) {
				std::lock_guard<std::mutex> lock(queue_mutex);
				total_tasks_executed += BigIntCore(static_cast<int64_t>(local_count));
				local_count = 0;
			}

			pending_tasks_count.fetch_sub(1);
			active_worker_count.fetch_sub(1);
			condition.notify_all(); // Notify barrier
		}
	}
}

uint64_t SimulationThreadPool::enqueue_task(const TaskFunc &p_func, TaskPriority p_priority) {
	Task task;
	task.func = p_func;
	task.priority = p_priority;

	std::lock_guard<std::mutex> lock(queue_mutex);
	task.id = ++next_task_id;
	pending_tasks_count.fetch_add(1);
	task_queue[static_cast<int>(p_priority)].push_back(task);
	
	condition.notify_one();
	return task.id;
}

void SimulationThreadPool::wait_for_all() {
	// Synchronization barrier for deterministic frame completion
	while (pending_tasks_count.load() > 0) {
		std::this_thread::yield();
	}
}

--- END OF FILE core/simulation/simulation_thread_pool.cpp ---
