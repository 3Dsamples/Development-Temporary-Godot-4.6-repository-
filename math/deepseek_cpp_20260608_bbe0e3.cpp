// File 376: modules/gaia/src/parallelization/spsc_queue.h
// Lock-free single-producer single-consumer (SPSC) queue for physics tasks.
// Used to pass work items (e.g., island indices, particle batches) between
// the main thread and worker threads without mutex contention.  The ring
// buffer is cache-line-padded to prevent false sharing.  All operations are
// wait‑free and use relaxed memory ordering for maximum throughput.

#ifndef GAIA_PARALLEL_SPSC_QUEUE_H
#define GAIA_PARALLEL_SPSC_QUEUE_H

#include "core/typedefs.h"
#include <atomic>

namespace gaia::parallel {

template <typename T, int64_t CAPACITY = 1024>
class SPSCQueue {
    static_assert((CAPACITY & (CAPACITY - 1)) == 0, "Capacity must be power of two");

    // Cache-line padding (64 bytes on most architectures)
    static constexpr int64_t CACHE_LINE = 64;

    alignas(CACHE_LINE) T buffer[CAPACITY];
    alignas(CACHE_LINE) std::atomic<int64_t> write_pos { 0 };
    alignas(CACHE_LINE) std::atomic<int64_t> read_pos  { 0 };

public:
    SPSCQueue() {}

    // Push an item (called by the producer). Returns false if queue is full.
    inline bool try_push(const T &p_item) {
        int64_t w = write_pos.load(std::memory_order_relaxed);
        int64_t r = read_pos.load(std::memory_order_acquire);
        if (w - r >= CAPACITY) return false; // full
        buffer[w & (CAPACITY - 1)] = p_item;
        write_pos.store(w + 1, std::memory_order_release);
        return true;
    }

    // Pop an item (called by the consumer). Returns false if queue is empty.
    inline bool try_pop(T &r_item) {
        int64_t r = read_pos.load(std::memory_order_relaxed);
        int64_t w = write_pos.load(std::memory_order_acquire);
        if (r >= w) return false; // empty
        r_item = buffer[r & (CAPACITY - 1)];
        read_pos.store(r + 1, std::memory_order_release);
        return true;
    }

    // Check if queue is empty (consumer side).
    inline bool is_empty() const {
        return read_pos.load(std::memory_order_acquire) >= write_pos.load(std::memory_order_acquire);
    }

    // Check if queue is full (producer side).
    inline bool is_full() const {
        return write_pos.load(std::memory_order_relaxed) - read_pos.load(std::memory_order_acquire) >= CAPACITY;
    }

    // Number of items currently in the queue.
    inline int64_t size() const {
        int64_t w = write_pos.load(std::memory_order_acquire);
        int64_t r = read_pos.load(std::memory_order_acquire);
        return MAX(w - r, 0);
    }

    // Reset the queue to empty (only safe when both threads are idle).
    void reset() {
        write_pos.store(0, std::memory_order_relaxed);
        read_pos.store(0, std::memory_order_relaxed);
    }
};

} // namespace gaia::parallel

#endif // GAIA_PARALLEL_SPSC_QUEUE_H