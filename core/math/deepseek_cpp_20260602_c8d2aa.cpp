//File 0107 : core/parallel/task_scheduler.h
//Work‑stealing task scheduler with a global thread pool, per‑worker lock‑free deques, and unified interface for submitting tasks and waiting for completion.
#ifndef CORE_PARALLEL_TASK_SCHEDULER_H
#define CORE_PARALLEL_TASK_SCHEDULER_H

#include <functional>
#include <future>
#include <atomic>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <deque>
#include <random>
#include <type_traits>
#include <cstdint>

namespace SimulationMath {
namespace parallel {

// -----------------------------------------------------------------------------
// 1. Lock‑free work‑stealing deque (Chase‑Lev) for task storage
// -----------------------------------------------------------------------------
class WorkStealingDeque {
    static constexpr size_t INITIAL_CAPACITY = 64;
public:
    WorkStealingDeque() : capacity_(INITIAL_CAPACITY),
                          top_(0), bottom_(0) {
        data_ = new std::function<void()>[capacity_];
    }
    ~WorkStealingDeque() {
        delete[] data_;
    }

    void push(std::function<void()> task) noexcept {
        size_t b = bottom_.load(std::memory_order_acquire);
        size_t t = top_.load(std::memory_order_acquire);
        if (b - t >= capacity_) {
            grow();
            b = bottom_.load(std::memory_order_acquire);
        }
        data_[b % capacity_] = std::move(task);
        bottom_.store(b + 1, std::memory_order_release);
    }

    std::function<void()> pop() noexcept {
        size_t b = bottom_.load(std::memory_order_acquire);
        size_t t = top_.load(std::memory_order_acquire);
        if (b <= t) return {};
        b--;
        bottom_.store(b, std::memory_order_release);
        std::function<void()> task = std::move(data_[b % capacity_]);
        if (b == t) {
            // Last element, need to avoid race with steal
            if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_release, std::memory_order_relaxed)) {
                task = {}; // stolen, discard
            }
        }
        return task;
    }

    std::function<void()> steal() noexcept {
        size_t t = top_.load(std::memory_order_acquire);
        size_t b = bottom_.load(std::memory_order_acquire);
        if (t >= b) return {};
        std::function<void()> task = std::move(data_[t % capacity_]);
        if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_release, std::memory_order_relaxed)) {
            return {}; // failed, someone else stole or popped
        }
        return task;
    }

    size_t size() const noexcept {
        size_t b = bottom_.load(std::memory_order_acquire);
        size_t t = top_.load(std::memory_order_acquire);
        return (b >= t) ? (b - t) : 0;
    }

    bool empty() const noexcept { return size() == 0; }

private:
    void grow() {
        size_t b = bottom_.load(std::memory_order_acquire);
        size_t t = top_.load(std::memory_order_acquire);
        size_t new_capacity = capacity_ * 2;
        auto new_data = new std::function<void()>[new_capacity];
        for (size_t i = t; i < b; ++i)
            new_data[i % new_capacity] = std::move(data_[i % capacity_]);
        delete[] data_;
        data_ = new_data;
        capacity_ = new_capacity;
    }

    alignas(64) std::atomic<size_t> top_;
    alignas(64) std::atomic<size_t> bottom_;
    std::function<void()>* data_;
    size_t capacity_;
};

// -----------------------------------------------------------------------------
// 2. Global task scheduler with work‑stealing thread pool
// -----------------------------------------------------------------------------
class TaskScheduler {
public:
    // Singleton access
    static TaskScheduler& instance() noexcept {
        static TaskScheduler scheduler;
        return scheduler;
    }

    // Initialise the thread pool (must be called before any task is submitted)
    void initialize(unsigned num_threads = 0) noexcept {
        if (initialized_.load(std::memory_order_acquire)) return;
        if (num_threads == 0)
            num_threads = std::thread::hardware_concurrency();
        if (num_threads < 1) num_threads = 1;
        num_workers_ = num_threads;

        deques_.resize(num_workers_);
        workers_.reserve(num_workers_);
        stop_.store(false, std::memory_order_release);

        for (unsigned i = 0; i < num_workers_; ++i) {
            workers_.emplace_back([this, i]() { worker_loop(i); });
        }
        initialized_.store(true, std::memory_order_release);
    }

    // Shutdown the thread pool
    void shutdown() noexcept {
        if (!initialized_.load(std::memory_order_acquire)) return;
        stop_.store(true, std::memory_order_release);
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
        workers_.clear();
        deques_.clear();
        initialized_.store(false, std::memory_order_release);
    }

    // Submit a void() task (fire‑and‑forget)
    void submit(std::function<void()> task) noexcept {
        // Push to a random worker's deque (or to the local deque if called from a worker)
        if (is_worker_thread()) {
            deques_[current_worker_id()].push(std::move(task));
        } else {
            unsigned target = rng_() % num_workers_;
            deques_[target].push(std::move(task));
        }
    }

    // Submit a task and return a future
    template <typename F, typename... Args>
    auto submit_future(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using result_type = typename std::result_of<F(Args...)>::type;
        auto promise = std::make_shared<std::promise<result_type>>();
        auto future = promise->get_future();
        auto task = [p = std::move(promise), f = std::forward<F>(f), ...args = std::forward<Args>(args)]() mutable {
            p->set_value(f(args...));
        };
        submit(std::move(task));
        return future;
    }

    // Wait until all tasks currently in all deques are finished (simplistic: drain until all deques empty)
    void wait_all() noexcept {
        // This is a naive busy‑wait; the calling thread also helps by stealing and executing tasks.
        // To implement properly, we need a counter of outstanding tasks. For simplicity we just loop until all deques empty.
        bool work_done = true;
        do {
            work_done = true;
            for (auto& dq : deques_)
                if (!dq.empty()) { work_done = false; break; }
            if (!work_done) {
                std::function<void()> task = steal_task();
                if (task) task();
                else std::this_thread::yield();
            }
        } while (!work_done);
    }

    // Number of worker threads
    unsigned num_workers() const noexcept { return num_workers_; }

private:
    TaskScheduler() = default;
    ~TaskScheduler() { shutdown(); }

    unsigned num_workers_ = 0;
    std::vector<std::thread> workers_;
    std::vector<WorkStealingDeque> deques_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> initialized_{false};
    std::mt19937 rng_{std::random_device{}()};
    // Thread‑local storage for worker ID (not portable thread_local static, but we use a static variable inside worker function)

    // Worker ID storage – we'll use a thread_local variable set inside worker_loop.
    static thread_local unsigned worker_id_;
    static thread_local bool is_worker_;

    void worker_loop(unsigned id) {
        worker_id_ = id;
        is_worker_ = true;
        while (!stop_.load(std::memory_order_acquire)) {
            // Try to pop a task from own deque
            std::function<void()> task = deques_[id].pop();
            if (!task) {
                // Steal from another worker
                unsigned victim = (id + 1 + rng_() % (num_workers_ - 1)) % num_workers_;
                task = deques_[victim].steal();
            }
            if (task) {
                task();
            } else {
                // No task available; yield to avoid busy‑wait
                std::this_thread::yield();
            }
        }
        is_worker_ = false;
    }

    static bool is_worker_thread() noexcept { return is_worker_; }
    static unsigned current_worker_id() noexcept { return worker_id_; }

    // Try to steal a task from any worker (used by external thread during wait)
    std::function<void()> steal_task() noexcept {
        unsigned start = rng_() % num_workers_;
        for (unsigned i = 0; i < num_workers_; ++i) {
            auto task = deques_[(start + i) % num_workers_].steal();
            if (task) return task;
        }
        return {};
    }
};

// Thread‑local storage definitions
thread_local unsigned TaskScheduler::worker_id_ = 0;
thread_local bool TaskScheduler::is_worker_ = false;

} // namespace parallel
} // namespace SimulationMath

#endif // CORE_PARALLEL_TASK_SCHEDULER_H