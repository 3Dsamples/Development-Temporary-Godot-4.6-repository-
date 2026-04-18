// include/xtu/godot/xdata_structures.hpp
// xtensor-unified - High-performance data structures for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XDATA_STRUCTURES_HPP
#define XTU_GODOT_XDATA_STRUCTURES_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <type_traits>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace core {

// #############################################################################
// PoolAllocator - Fast fixed-size object pool
// #############################################################################
template <typename T, size_t BlockSize = 64>
class PoolAllocator {
private:
    struct Block {
        alignas(alignof(T)) char data[BlockSize * sizeof(T)];
        Block* next = nullptr;
    };

    struct FreeNode {
        FreeNode* next = nullptr;
    };

    Block* m_first_block = nullptr;
    FreeNode* m_free_list = nullptr;
    size_t m_capacity = 0;
    size_t m_size = 0;
    mutable std::mutex m_mutex;

    void allocate_block() {
        Block* block = static_cast<Block*>(std::aligned_alloc(alignof(Block), sizeof(Block)));
        block->next = m_first_block;
        m_first_block = block;

        // Add all slots to free list
        T* items = reinterpret_cast<T*>(block->data);
        for (size_t i = 0; i < BlockSize; ++i) {
            FreeNode* node = reinterpret_cast<FreeNode*>(items + i);
            node->next = m_free_list;
            m_free_list = node;
        }
        m_capacity += BlockSize;
    }

public:
    PoolAllocator() = default;

    ~PoolAllocator() {
        clear();
        while (m_first_block) {
            Block* next = m_first_block->next;
            std::free(m_first_block);
            m_first_block = next;
        }
    }

    template <typename... Args>
    T* allocate(Args&&... args) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_free_list) {
            allocate_block();
        }

        FreeNode* node = m_free_list;
        m_free_list = node->next;
        ++m_size;

        T* obj = reinterpret_cast<T*>(node);
        new (obj) T(std::forward<Args>(args)...);
        return obj;
    }

    void deallocate(T* obj) {
        if (!obj) return;

        std::lock_guard<std::mutex> lock(m_mutex);
        obj->~T();
        FreeNode* node = reinterpret_cast<FreeNode*>(obj);
        node->next = m_free_list;
        m_free_list = node;
        --m_size;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        // Call destructors on all allocated objects
        for (Block* block = m_first_block; block; block = block->next) {
            T* items = reinterpret_cast<T*>(block->data);
            for (size_t i = 0; i < BlockSize; ++i) {
                items[i].~T();
            }
        }
        // Rebuild free list
        m_free_list = nullptr;
        for (Block* block = m_first_block; block; block = block->next) {
            T* items = reinterpret_cast<T*>(block->data);
            for (size_t i = 0; i < BlockSize; ++i) {
                FreeNode* node = reinterpret_cast<FreeNode*>(items + i);
                node->next = m_free_list;
                m_free_list = node;
            }
        }
        m_size = 0;
    }

    size_t size() const { return m_size; }
    size_t capacity() const { return m_capacity; }
};

// #############################################################################
// RingBuffer - Lock-free SPSC ring buffer
// #############################################################################
template <typename T, size_t Capacity>
class RingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of two");

private:
    alignas(64) std::atomic<size_t> m_read_index{0};
    alignas(64) std::atomic<size_t> m_write_index{0};
    alignas(64) T m_buffer[Capacity];

    static constexpr size_t MASK = Capacity - 1;

public:
    RingBuffer() = default;

    bool push(const T& value) {
        size_t write = m_write_index.load(std::memory_order_relaxed);
        size_t next = (write + 1) & MASK;

        if (next == m_read_index.load(std::memory_order_acquire)) {
            return false; // Full
        }

        new (&m_buffer[write]) T(value);
        m_write_index.store(next, std::memory_order_release);
        return true;
    }

    bool push(T&& value) {
        size_t write = m_write_index.load(std::memory_order_relaxed);
        size_t next = (write + 1) & MASK;

        if (next == m_read_index.load(std::memory_order_acquire)) {
            return false;
        }

        new (&m_buffer[write]) T(std::move(value));
        m_write_index.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T& out) {
        size_t read = m_read_index.load(std::memory_order_relaxed);
        if (read == m_write_index.load(std::memory_order_acquire)) {
            return false; // Empty
        }

        out = std::move(m_buffer[read]);
        m_buffer[read].~T();
        m_read_index.store((read + 1) & MASK, std::memory_order_release);
        return true;
    }

    bool peek(T& out) const {
        size_t read = m_read_index.load(std::memory_order_relaxed);
        if (read == m_write_index.load(std::memory_order_acquire)) {
            return false;
        }
        out = m_buffer[read];
        return true;
    }

    bool empty() const {
        return m_read_index.load(std::memory_order_acquire) ==
               m_write_index.load(std::memory_order_acquire);
    }

    bool full() const {
        size_t write = m_write_index.load(std::memory_order_relaxed);
        return ((write + 1) & MASK) == m_read_index.load(std::memory_order_acquire);
    }

    size_t size() const {
        size_t write = m_write_index.load(std::memory_order_acquire);
        size_t read = m_read_index.load(std::memory_order_acquire);
        if (write >= read) {
            return write - read;
        }
        return Capacity - read + write;
    }

    size_t capacity() const { return Capacity - 1; }

    void clear() {
        T dummy;
        while (pop(dummy)) {}
    }
};

// #############################################################################
// CommandQueue - Thread-safe command queue for async operations
// #############################################################################
template <typename Command>
class CommandQueue {
private:
    std::queue<Command> m_queue;
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_flushing = false;

public:
    void push(const Command& cmd) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push(cmd);
        m_cv.notify_one();
    }

    void push(Command&& cmd) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push(std::move(cmd));
        m_cv.notify_one();
    }

    bool pop(Command& out) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [this]() { return !m_queue.empty() || m_flushing; });
        if (m_queue.empty()) return false;
        out = std::move(m_queue.front());
        m_queue.pop();
        return true;
    }

    bool try_pop(Command& out) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_queue.empty()) return false;
        out = std::move(m_queue.front());
        m_queue.pop();
        return true;
    }

    void flush() {
        std::lock_guard<std::mutex> lock(m_mutex);
        while (!m_queue.empty()) m_queue.pop();
    }

    void notify_all() {
        m_cv.notify_all();
    }

    void set_flushing(bool flushing) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_flushing = flushing;
        if (flushing) m_cv.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.empty();
    }
};

// #############################################################################
// PagedAllocator - Efficient allocator for many small objects
// #############################################################################
class PagedAllocator {
private:
    struct Page {
        Page* next = nullptr;
        size_t used = 0;
        alignas(16) uint8_t data[1];
    };

    size_t m_page_size;
    size_t m_total_allocated = 0;
    size_t m_peak_allocated = 0;
    Page* m_pages = nullptr;
    mutable std::mutex m_mutex;

    Page* allocate_page() {
        size_t alloc_size = sizeof(Page) + m_page_size - 1;
        Page* page = static_cast<Page*>(std::aligned_alloc(16, alloc_size));
        page->next = m_pages;
        page->used = 0;
        m_pages = page;
        m_total_allocated += m_page_size;
        if (m_total_allocated > m_peak_allocated) {
            m_peak_allocated = m_total_allocated;
        }
        return page;
    }

public:
    explicit PagedAllocator(size_t page_size = 65536) : m_page_size(page_size) {}

    ~PagedAllocator() {
        while (m_pages) {
            Page* next = m_pages->next;
            std::free(m_pages);
            m_pages = next;
        }
    }

    void* allocate(size_t size, size_t alignment = 8) {
        std::lock_guard<std::mutex> lock(m_mutex);

        // Align size
        size = (size + alignment - 1) & ~(alignment - 1);

        if (size > m_page_size) {
            // Large allocation - use separate page
            Page* page = allocate_page();
            page->used = m_page_size;
            return page->data;
        }

        // Find page with enough space
        Page* page = m_pages;
        while (page) {
            if (m_page_size - page->used >= size) {
                void* ptr = page->data + page->used;
                page->used += size;
                return ptr;
            }
            page = page->next;
        }

        // Allocate new page
        page = allocate_page();
        void* ptr = page->data;
        page->used = size;
        return ptr;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (Page* page = m_pages; page; page = page->next) {
            page->used = 0;
        }
        m_total_allocated = 0;
    }

    size_t get_total_allocated() const { return m_total_allocated; }
    size_t get_peak_allocated() const { return m_peak_allocated; }
    size_t get_page_size() const { return m_page_size; }
};

// #############################################################################
// ThreadWorkPool - Work-stealing thread pool
// #############################################################################
class ThreadWorkPool {
public:
    using WorkItem = std::function<void()>;

private:
    struct Worker {
        std::thread thread;
        std::vector<WorkItem> local_queue;
        std::mutex mutex;
        std::atomic<bool> running{true};
        std::atomic<size_t> active_tasks{0};
        ThreadWorkPool* pool = nullptr;
        size_t index = 0;
    };

    std::vector<std::unique_ptr<Worker>> m_workers;
    std::atomic<size_t> m_next_worker{0};
    std::atomic<size_t> m_pending_tasks{0};
    std::mutex m_global_mutex;
    std::condition_variable m_cv;
    bool m_shutdown = false;

public:
    explicit ThreadWorkPool(size_t num_threads = 0) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
        }
        num_threads = std::max(size_t(1), num_threads);

        m_workers.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            auto worker = std::make_unique<Worker>();
            worker->pool = this;
            worker->index = i;
            worker->thread = std::thread(&ThreadWorkPool::worker_loop, worker.get());
            m_workers.push_back(std::move(worker));
        }
    }

    ~ThreadWorkPool() {
        {
            std::lock_guard<std::mutex> lock(m_global_mutex);
            m_shutdown = true;
        }
        m_cv.notify_all();

        for (auto& worker : m_workers) {
            worker->running = false;
            if (worker->thread.joinable()) {
                worker->thread.join();
            }
        }
    }

    void push(WorkItem item) {
        m_pending_tasks.fetch_add(1);

        // Try to find a worker with empty queue
        size_t start = m_next_worker.fetch_add(1) % m_workers.size();
        for (size_t i = 0; i < m_workers.size(); ++i) {
            size_t idx = (start + i) % m_workers.size();
            Worker& worker = *m_workers[idx];

            std::unique_lock<std::mutex> lock(worker.mutex, std::try_to_lock);
            if (lock.owns_lock()) {
                worker.local_queue.push_back(std::move(item));
                worker.active_tasks.fetch_add(1);
                m_cv.notify_one();
                return;
            }
        }

        // All workers busy - pick one randomly
        size_t idx = start;
        Worker& worker = *m_workers[idx];
        {
            std::lock_guard<std::mutex> lock(worker.mutex);
            worker.local_queue.push_back(std::move(item));
            worker.active_tasks.fetch_add(1);
        }
        m_cv.notify_one();
    }

    void wait_all() {
        while (m_pending_tasks.load() > 0) {
            std::this_thread::yield();
        }
    }

    size_t get_pending_count() const {
        return m_pending_tasks.load();
    }

    size_t get_thread_count() const {
        return m_workers.size();
    }

private:
    static void worker_loop(Worker* worker) {
        ThreadWorkPool* pool = worker->pool;

        while (worker->running) {
            WorkItem item;

            // Try to get work from local queue
            {
                std::unique_lock<std::mutex> lock(worker->mutex);
                pool->m_cv.wait_for(lock, std::chrono::milliseconds(10),
                    [&]() { return !worker->local_queue.empty() || pool->m_shutdown; });

                if (pool->m_shutdown) break;

                if (!worker->local_queue.empty()) {
                    item = std::move(worker->local_queue.back());
                    worker->local_queue.pop_back();
                }
            }

            // If no local work, try to steal
            if (!item) {
                for (size_t i = 0; i < pool->m_workers.size(); ++i) {
                    size_t victim_idx = (worker->index + i + 1) % pool->m_workers.size();
                    Worker* victim = pool->m_workers[victim_idx].get();

                    std::unique_lock<std::mutex> lock(victim->mutex, std::try_to_lock);
                    if (lock.owns_lock() && !victim->local_queue.empty()) {
                        item = std::move(victim->local_queue.front());
                        victim->local_queue.erase(victim->local_queue.begin());
                        worker->active_tasks.fetch_add(1);
                        break;
                    }
                }
            }

            if (item) {
                item();
                worker->active_tasks.fetch_sub(1);
                pool->m_pending_tasks.fetch_sub(1);
            }
        }
    }
};

// #############################################################################
// LocalVector - Stack-allocated vector with fallback to heap
// #############################################################################
template <typename T, size_t StackCapacity = 16>
class LocalVector {
private:
    alignas(T) uint8_t m_stack_data[StackCapacity * sizeof(T)];
    T* m_data = nullptr;
    size_t m_size = 0;
    size_t m_capacity = StackCapacity;
    bool m_using_heap = false;

    void grow(size_t new_capacity) {
        T* new_data = static_cast<T*>(std::aligned_alloc(alignof(T), new_capacity * sizeof(T)));

        // Move existing elements
        for (size_t i = 0; i < m_size; ++i) {
            new (new_data + i) T(std::move(m_data[i]));
            m_data[i].~T();
        }

        if (m_using_heap) {
            std::free(m_data);
        }

        m_data = new_data;
        m_capacity = new_capacity;
        m_using_heap = true;
    }

public:
    LocalVector() {
        m_data = reinterpret_cast<T*>(m_stack_data);
    }

    ~LocalVector() {
        clear();
        if (m_using_heap) {
            std::free(m_data);
        }
    }

    LocalVector(const LocalVector&) = delete;
    LocalVector& operator=(const LocalVector&) = delete;

    LocalVector(LocalVector&& other) noexcept
        : m_size(other.m_size)
        , m_capacity(other.m_capacity)
        , m_using_heap(other.m_using_heap) {
        if (m_using_heap) {
            m_data = other.m_data;
            other.m_data = nullptr;
            other.m_size = 0;
            other.m_capacity = 0;
            other.m_using_heap = false;
        } else {
            m_data = reinterpret_cast<T*>(m_stack_data);
            for (size_t i = 0; i < m_size; ++i) {
                new (m_data + i) T(std::move(other.m_data[i]));
                other.m_data[i].~T();
            }
            other.m_size = 0;
        }
    }

    void push_back(const T& value) {
        if (m_size == m_capacity) {
            grow(m_capacity * 2);
        }
        new (m_data + m_size) T(value);
        ++m_size;
    }

    void push_back(T&& value) {
        if (m_size == m_capacity) {
            grow(m_capacity * 2);
        }
        new (m_data + m_size) T(std::move(value));
        ++m_size;
    }

    template <typename... Args>
    void emplace_back(Args&&... args) {
        if (m_size == m_capacity) {
            grow(m_capacity * 2);
        }
        new (m_data + m_size) T(std::forward<Args>(args)...);
        ++m_size;
    }

    void pop_back() {
        if (m_size > 0) {
            --m_size;
            m_data[m_size].~T();
        }
    }

    void clear() {
        for (size_t i = 0; i < m_size; ++i) {
            m_data[i].~T();
        }
        m_size = 0;
    }

    void resize(size_t new_size) {
        if (new_size > m_capacity) {
            grow(new_size);
        }
        for (size_t i = m_size; i < new_size; ++i) {
            new (m_data + i) T();
        }
        for (size_t i = new_size; i < m_size; ++i) {
            m_data[i].~T();
        }
        m_size = new_size;
    }

    void reserve(size_t new_capacity) {
        if (new_capacity > m_capacity) {
            grow(new_capacity);
        }
    }

    T& operator[](size_t index) { return m_data[index]; }
    const T& operator[](size_t index) const { return m_data[index]; }

    T* data() { return m_data; }
    const T* data() const { return m_data; }

    size_t size() const { return m_size; }
    size_t capacity() const { return m_capacity; }
    bool empty() const { return m_size == 0; }

    T* begin() { return m_data; }
    T* end() { return m_data + m_size; }
    const T* begin() const { return m_data; }
    const T* end() const { return m_data + m_size; }
};

// #############################################################################
// ObjectPool - Pool for RefCounted objects with automatic recycling
// #############################################################################
template <typename T>
class ObjectPool {
    static_assert(std::is_base_of_v<RefCounted, T>, "T must derive from RefCounted");

private:
    std::vector<Ref<T>> m_pool;
    size_t m_max_size = 64;
    mutable std::mutex m_mutex;

public:
    explicit ObjectPool(size_t max_size = 64) : m_max_size(max_size) {
        m_pool.reserve(max_size);
    }

    Ref<T> acquire() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_pool.empty()) {
            Ref<T> obj = std::move(m_pool.back());
            m_pool.pop_back();
            obj->reset();
            return obj;
        }
        Ref<T> obj;
        obj.instance();
        return obj;
    }

    void release(Ref<T> obj) {
        if (!obj.is_valid()) return;
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_pool.size() < m_max_size) {
            m_pool.push_back(std::move(obj));
        }
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_pool.clear();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_pool.size();
    }

    size_t capacity() const { return m_max_size; }
};

} // namespace core

// Bring into main namespace
using core::PoolAllocator;
using core::RingBuffer;
using core::CommandQueue;
using core::PagedAllocator;
using core::ThreadWorkPool;
using core::LocalVector;
using core::ObjectPool;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XDATA_STRUCTURES_HPP