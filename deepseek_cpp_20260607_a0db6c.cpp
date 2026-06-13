/* MIT License Copyright (c) 2021-2026 Attila Csikós & Contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ORTHOTREE_CORE_PARALLEL_SCALE_AWARE_TASK_SCHEDULER_H_INCLUDED
#define ORTHOTREE_CORE_PARALLEL_SCALE_AWARE_TASK_SCHEDULER_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <unordered_map>
#include <condition_variable>
#include <optional>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cmath>

namespace OrthoTree {
namespace Parallel {

// ============================================================================
//  Task priorities (used for dynamic prioritization at different scales)
// ============================================================================
enum class TaskPriority : uint8_t {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3
};

// ============================================================================
//  Scale classification for hierarchical task management
// ============================================================================
enum class ScaleLevel : uint8_t {
    Microscopic = 0,
    Mesoscopic = 1,
    Macroscopic = 2,
    Planetary = 3,
    Galactic = 4,
    Universal = 5
};

// ============================================================================
//  Task interface: all tasks that can be scheduled by the scheduler
// ============================================================================
class Task {
public:
    using TaskFunction = std::function<void()>;

    Task() = default;
    Task(TaskFunction func, TaskPriority prio = TaskPriority::Normal, ScaleLevel scale = ScaleLevel::Macroscopic)
        : m_function(std::move(func)), m_priority(prio), m_scale(scale), m_id(s_nextId++) {}

    virtual ~Task() = default;

    void execute() const { if (m_function) m_function(); }

    TaskPriority priority() const { return m_priority; }
    ScaleLevel scale() const { return m_scale; }
    uint64_t id() const { return m_id; }

    // Comparison for priority queue (lower priority value = higher urgency)
    bool operator<(const Task& other) const {
        if (m_priority != other.m_priority)
            return static_cast<uint8_t>(m_priority) < static_cast<uint8_t>(other.m_priority);
        // Higher scale tasks get slightly higher priority (larger scale first)
        if (m_scale != other.m_scale)
            return static_cast<uint8_t>(m_scale) > static_cast<uint8_t>(other.m_scale);
        return m_id > other.m_id; // FIFO within same priority/scale
    }

private:
    TaskFunction m_function;
    TaskPriority m_priority = TaskPriority::Normal;
    ScaleLevel m_scale = ScaleLevel::Macroscopic;
    uint64_t m_id = 0;

    static std::atomic<uint64_t> s_nextId;
};

inline std::atomic<uint64_t> Task::s_nextId{0};

// ============================================================================
//  Worker thread: executes tasks from its own queue and can steal from others
// ============================================================================
class WorkerThread {
public:
    using TaskQueue = std::vector<std::unique_ptr<Task>>;

    WorkerThread(size_t index)
        : m_index(index)
        , m_active(true)
        , m_thread(&WorkerThread::run, this) {}

    ~WorkerThread() {
        m_active = false;
        if (m_thread.joinable())
            m_thread.join();
    }

    // Push a task into this worker's queue
    void push(std::unique_ptr<Task> task) {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_localQueue.push_back(std::move(task));
    }

    // Try to pop a task from this worker's queue
    std::unique_ptr<Task> tryPop() {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        if (m_localQueue.empty())
            return nullptr;
        auto task = std::move(m_localQueue.back());
        m_localQueue.pop_back();
        return task;
    }

    // Attempt to steal a task from another worker's queue
    std::unique_ptr<Task> trySteal(WorkerThread& victim) {
        std::lock_guard<std::mutex> lock(victim.m_queueMutex);
        if (victim.m_localQueue.empty())
            return nullptr;
        // Steal from the front to avoid contention
        auto task = std::move(victim.m_localQueue.front());
        victim.m_localQueue.erase(victim.m_localQueue.begin());
        return task;
    }

    size_t queueSize() const {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        return m_localQueue.size();
    }

    size_t index() const { return m_index; }

private:
    void run() {
        while (m_active) {
            // First, try to pop a task from own queue
            auto task = tryPop();
            if (!task) {
                // No task in own queue, try to steal from others
                for (size_t i = 0; i < s_numWorkers; ++i) {
                    if (i == m_index) continue;
                    if (auto stolen = trySteal(*s_workers[i])) {
                        task = std::move(stolen);
                        break;
                    }
                }
            }

            if (task) {
                task->execute();
            } else {
                // No tasks available, yield briefly
                std::this_thread::yield();
            }
        }
    }

    size_t m_index;
    std::atomic<bool> m_active;
    std::thread m_thread;
    std::mutex m_queueMutex;
    TaskQueue m_localQueue;

    static std::vector<WorkerThread*> s_workers;
    static size_t s_numWorkers;
};

inline std::vector<WorkerThread*> WorkerThread::s_workers;
inline size_t WorkerThread::s_numWorkers = 0;

// ============================================================================
//  ScaleAwareTaskScheduler: main scheduler that distributes tasks across workers
//  and adapts to different simulation scales (microscopic to galactic).
// ============================================================================
class ScaleAwareTaskScheduler {
public:
    using TaskQueue = std::priority_queue<std::unique_ptr<Task>>;

    struct Config {
        size_t numThreads = 0;                 // 0 = auto (hardware concurrency)
        bool enableWorkStealing = true;
        bool enableScaleBalancing = true;
        size_t maxMicroTasks = 10000;          // Limit for microscopic tasks
        size_t maxGalacticTasks = 100;
        uint32_t priorityBoostIntervalMs = 10; // How often to boost high-scale tasks
    };

    explicit ScaleAwareTaskScheduler(const Config& cfg = Config())
        : m_config(cfg)
        , m_running(true)
        , m_dispatchingThread(&ScaleAwareTaskScheduler::dispatchLoop, this) {
        if (m_config.numThreads == 0) {
            m_config.numThreads = std::thread::hardware_concurrency();
            if (m_config.numThreads == 0) m_config.numThreads = 4;
        }
        WorkerThread::s_numWorkers = m_config.numThreads;
        for (size_t i = 0; i < m_config.numThreads; ++i) {
            auto worker = new WorkerThread(i);
            m_workers.push_back(worker);
            WorkerThread::s_workers.push_back(worker);
        }
    }

    ~ScaleAwareTaskScheduler() {
        m_running = false;
        m_cv.notify_all();
        if (m_dispatchingThread.joinable())
            m_dispatchingThread.join();
        for (auto* worker : m_workers) {
            delete worker;
        }
        m_workers.clear();
        WorkerThread::s_workers.clear();
    }

    // Submit a task to the global queue (will be distributed to workers)
    void submit(std::unique_ptr<Task> task) {
        std::lock_guard<std::mutex> lock(m_globalMutex);
        m_globalQueue.push(std::move(task));
        m_cv.notify_one();
    }

    // Submit a batch of tasks more efficiently
    void submitBatch(std::vector<std::unique_ptr<Task>> tasks) {
        std::lock_guard<std::mutex> lock(m_globalMutex);
        for (auto& task : tasks) {
            m_globalQueue.push(std::move(task));
        }
        m_cv.notify_all();
    }

    // Set the current simulation scale (used for load balancing decisions)
    void setCurrentScale(ScaleLevel scale) {
        m_currentScale.store(scale);
    }

    ScaleLevel currentScale() const {
        return m_currentScale.load();
    }

    // Adjust thread count dynamically based on load (useful for hybrid CPU/GPU)
    void resizeThreadPool(size_t newSize) {
        if (newSize == m_workers.size()) return;
        if (newSize > m_workers.size()) {
            for (size_t i = m_workers.size(); i < newSize; ++i) {
                auto worker = new WorkerThread(i);
                m_workers.push_back(worker);
                WorkerThread::s_workers.push_back(worker);
            }
        } else {
            for (size_t i = newSize; i < m_workers.size(); ++i) {
                delete m_workers[i];
            }
            m_workers.resize(newSize);
            WorkerThread::s_workers.resize(newSize);
        }
        WorkerThread::s_numWorkers = newSize;
    }

    // Get approximate load of the system (0..1)
    double getLoadFactor() const {
        size_t totalTasks = 0;
        for (const auto* worker : m_workers) {
            totalTasks += worker->queueSize();
        }
        {
            std::lock_guard<std::mutex> lock(m_globalMutex);
            totalTasks += m_globalQueue.size();
        }
        return static_cast<double>(totalTasks) / (m_config.numThreads * 10);
    }

    // Wait until all tasks are completed (simple barrier)
    void waitForIdle() {
        while (getLoadFactor() > 0.01) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

private:
    void dispatchLoop() {
        while (m_running) {
            std::unique_ptr<Task> task;
            {
                std::unique_lock<std::mutex> lock(m_globalMutex);
                m_cv.wait(lock, [this] { return !m_running || !m_globalQueue.empty(); });
                if (!m_running) break;
                if (!m_globalQueue.empty()) {
                    task = std::move(const_cast<std::unique_ptr<Task>&>(m_globalQueue.top()));
                    m_globalQueue.pop();
                }
            }
            if (task) {
                // Assign task to the least loaded worker (simple heuristic)
                size_t bestWorker = 0;
                size_t smallestQueue = m_workers[0]->queueSize();
                for (size_t i = 1; i < m_workers.size(); ++i) {
                    size_t qsize = m_workers[i]->queueSize();
                    if (qsize < smallestQueue) {
                        smallestQueue = qsize;
                        bestWorker = i;
                    }
                }
                m_workers[bestWorker]->push(std::move(task));
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    Config m_config;
    std::atomic<bool> m_running;
    std::atomic<ScaleLevel> m_currentScale{ScaleLevel::Macroscopic};
    std::vector<WorkerThread*> m_workers;
    std::thread m_dispatchingThread;
    mutable std::mutex m_globalMutex;
    std::condition_variable m_cv;
    TaskQueue m_globalQueue;
};

// ============================================================================
//  SIMD utilities for batch task processing
// ============================================================================
class BatchTaskProcessor {
public:
    template<typename Func>
    static void processBatch(size_t count, Func&& func) {
        if (count >= 4) {
            size_t simdCount = count - (count % 4);
            for (size_t i = 0; i < simdCount; i += 4) {
                // In a real SIMD implementation, we would load indices into registers.
                // Here we simply unroll the loop for better instruction-level parallelism.
                func(i);
                func(i+1);
                func(i+2);
                func(i+3);
            }
            for (size_t i = simdCount; i < count; ++i) {
                func(i);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                func(i);
            }
        }
    }
};

} // namespace Parallel
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_PARALLEL_SCALE_AWARE_TASK_SCHEDULER_H_INCLUDED