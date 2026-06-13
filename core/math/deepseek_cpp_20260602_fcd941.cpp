//File 0115 : core/parallel/atomic_operations.h
//Portable atomic operations and memory‑order helpers for lock‑free programming: atomic fetch‑and‑add/sub/and/or/xor, CAS loops, exponential backoff spin‑lock, and hazard pointers (basic).
#ifndef CORE_PARALLEL_ATOMIC_OPERATIONS_H
#define CORE_PARALLEL_ATOMIC_OPERATIONS_H

#include <atomic>
#include <cstdint>
#include <type_traits>
#include <thread>
#include <chrono>
#include <functional>

namespace SimulationMath {
namespace parallel {

// -----------------------------------------------------------------------------
// 1. Generic Atomic class wrapping std::atomic with additional convenience methods
// -----------------------------------------------------------------------------
template <typename T>
class Atomic {
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
    std::atomic<T> val_;
public:
    Atomic() noexcept = default;
    explicit Atomic(T init) noexcept : val_(init) {}

    T load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return val_.load(order);
    }
    void store(T desired, std::memory_order order = std::memory_order_seq_cst) noexcept {
        val_.store(desired, order);
    }
    T exchange(T desired, std::memory_order order = std::memory_order_seq_cst) noexcept {
        return val_.exchange(desired, order);
    }
    bool compare_exchange_weak(T& expected, T desired,
                              std::memory_order success = std::memory_order_seq_cst,
                              std::memory_order failure = std::memory_order_seq_cst) noexcept {
        return val_.compare_exchange_weak(expected, desired, success, failure);
    }
    bool compare_exchange_strong(T& expected, T desired,
                                std::memory_order success = std::memory_order_seq_cst,
                                std::memory_order failure = std::memory_order_seq_cst) noexcept {
        return val_.compare_exchange_strong(expected, desired, success, failure);
    }

    // Convenience arithmetic
    T fetch_add(T arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
        return val_.fetch_add(arg, order);
    }
    T fetch_sub(T arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
        return val_.fetch_sub(arg, order);
    }
    T fetch_and(T arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
        return val_.fetch_and(arg, order);
    }
    T fetch_or(T arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
        return val_.fetch_or(arg, order);
    }
    T fetch_xor(T arg, std::memory_order order = std::memory_order_seq_cst) noexcept {
        return val_.fetch_xor(arg, order);
    }

    T operator++() noexcept { return fetch_add(1) + 1; }
    T operator++(int) noexcept { return fetch_add(1); }
    T operator--() noexcept { return fetch_sub(1) - 1; }
    T operator--(int) noexcept { return fetch_sub(1); }

    T operator+=(T arg) noexcept { return fetch_add(arg) + arg; }
    T operator-=(T arg) noexcept { return fetch_sub(arg) - arg; }
};

// -----------------------------------------------------------------------------
// 2. Exponential backoff spin‑lock (higher‑performance than simple flag)
// -----------------------------------------------------------------------------
class ExponentialBackoff {
    int min_delay_;
    int max_delay_;
    int current_delay_;

public:
    explicit ExponentialBackoff(int min_delay_us = 4, int max_delay_us = 1024) noexcept
        : min_delay_(min_delay_us), max_delay_(max_delay_us), current_delay_(min_delay_us) {}

    // Call this while the condition is not met; yields the CPU with increasing delays.
    void wait() noexcept {
        if (current_delay_ <= max_delay_) {
            std::this_thread::sleep_for(std::chrono::microseconds(current_delay_));
            current_delay_ *= 2;
        } else {
            std::this_thread::yield();
        }
    }

    // Reset the delay for the next operation
    void reset() noexcept { current_delay_ = min_delay_; }
};

// -----------------------------------------------------------------------------
// 3. Adaptive spin‑lock with exponential backoff and yielding
// -----------------------------------------------------------------------------
class AdaptiveSpinLock {
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
public:
    void lock() noexcept {
        ExponentialBackoff backoff;
        while (flag_.test_and_set(std::memory_order_acquire)) {
            backoff.wait();
        }
    }
    void unlock() noexcept {
        flag_.clear(std::memory_order_release);
    }
    bool try_lock() noexcept {
        return !flag_.test_and_set(std::memory_order_acquire);
    }
};

// -----------------------------------------------------------------------------
// 4. CAS‑based lock‑free stack (Treiber stack)
// -----------------------------------------------------------------------------
template <typename T>
class LockFreeStack {
    struct Node {
        T value;
        Node* next;
        Node(const T& val) noexcept : value(val), next(nullptr) {}
    };
    std::atomic<Node*> head_{nullptr};

public:
    ~LockFreeStack() {
        while (pop()) {}
    }

    void push(const T& val) noexcept {
        Node* node = new Node(val);
        node->next = head_.load(std::memory_order_relaxed);
        while (!head_.compare_exchange_weak(node->next, node,
                                            std::memory_order_release,
                                            std::memory_order_relaxed)) {}
    }

    bool pop(T& out) noexcept {
        Node* old_head = head_.load(std::memory_order_relaxed);
        while (old_head) {
            if (head_.compare_exchange_weak(old_head, old_head->next,
                                            std::memory_order_acquire,
                                            std::memory_order_relaxed)) {
                out = std::move(old_head->value);
                delete old_head;
                return true;
            }
        }
        return false;
    }

    bool pop() noexcept {
        T dummy;
        return pop(dummy);
    }

    bool empty() const noexcept {
        return head_.load(std::memory_order_relaxed) == nullptr;
    }
};

// -----------------------------------------------------------------------------
// 5. Simple hazard pointer – for safe memory reclamation (single guard)
// -----------------------------------------------------------------------------
template <typename T>
class HazardPointer {
    std::atomic<T*> ptr_{nullptr};
public:
    // Protect a pointer (store it)
    void protect(T* p) noexcept {
        ptr_.store(p, std::memory_order_release);
    }
    // Unprotect
    void clear() noexcept {
        ptr_.store(nullptr, std::memory_order_release);
    }
    // Get the currently protected pointer
    T* load() const noexcept {
        return ptr_.load(std::memory_order_acquire);
    }
};

// -----------------------------------------------------------------------------
// 6. Utility: atomic min / max (CAS loop)
// -----------------------------------------------------------------------------
template <typename T>
T atomic_fetch_min(std::atomic<T>& atomic, T val) noexcept {
    T prev = atomic.load(std::memory_order_relaxed);
    while (prev > val && !atomic.compare_exchange_weak(prev, val,
                                                       std::memory_order_release,
                                                       std::memory_order_relaxed)) {}
    return prev;
}

template <typename T>
T atomic_fetch_max(std::atomic<T>& atomic, T val) noexcept {
    T prev = atomic.load(std::memory_order_relaxed);
    while (prev < val && !atomic.compare_exchange_weak(prev, val,
                                                       std::memory_order_release,
                                                       std::memory_order_relaxed)) {}
    return prev;
}

} // namespace parallel
} // namespace SimulationMath

#endif // CORE_PARALLEL_ATOMIC_OPERATIONS_H