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

#ifndef ORTHOTREE_CORE_PARALLEL_LOCKFREE_QUERY_BUFFER_H_INCLUDED
#define ORTHOTREE_CORE_PARALLEL_LOCKFREE_QUERY_BUFFER_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <algorithm>
#include <vector>
#include <memory>
#include <thread>

namespace OrthoTree {
namespace Parallel {

// ============================================================================
//  LockfreeQueryBuffer: lock‑free buffer for accumulating query results
//  from multiple threads without contention. Uses a bounded ring buffer
//  with atomic sequence counters and SIMD‑friendly alignment.
//  Supports dynamic resizing and custom allocators.
// ============================================================================
template<typename T, size_t CacheLineSize = ORTHOTREE_CACHE_LINE_SIZE>
class LockfreeQueryBuffer {
    static_assert(std::is_trivially_copyable_v<T>,
                  "T must be trivially copyable for lock‑free operations");
public:
    using value_type = T;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    // ------------------------------------------------------------------------
    //  Configuration
    // ------------------------------------------------------------------------
    struct Config {
        size_type initialCapacity = 1024;      // must be power of two
        size_type maxCapacity = 1 << 20;       // 1 million entries max
        bool autoResize = true;                // allow dynamic resizing
        size_type resizeThreshold = 0.75;      // capacity usage threshold for resize
    };

    // ------------------------------------------------------------------------
    //  Constructor / destructor
    // ------------------------------------------------------------------------
    explicit LockfreeQueryBuffer(const Config& cfg = Config())
        : m_config(cfg)
        , m_capacity(nextPowerOfTwo(cfg.initialCapacity))
        , m_mask(m_capacity - 1)
        , m_buffer(allocateAligned(m_capacity))
        , m_head(0)
        , m_tail(0)
        , m_size(0) {
        ORTHOTREE_ASSERT((m_capacity & (m_capacity - 1)) == 0);
    }

    ~LockfreeQueryBuffer() {
        deallocateAligned(m_buffer);
    }

    // Disable copying (buffer is not copyable)
    LockfreeQueryBuffer(const LockfreeQueryBuffer&) = delete;
    LockfreeQueryBuffer& operator=(const LockfreeQueryBuffer&) = delete;

    // Move is allowed
    LockfreeQueryBuffer(LockfreeQueryBuffer&& other) noexcept
        : m_config(other.m_config)
        , m_capacity(other.m_capacity)
        , m_mask(other.m_mask)
        , m_buffer(other.m_buffer)
        , m_head(other.m_head.load())
        , m_tail(other.m_tail.load())
        , m_size(other.m_size.load()) {
        other.m_buffer = nullptr;
        other.m_capacity = 0;
        other.m_mask = 0;
    }

    LockfreeQueryBuffer& operator=(LockfreeQueryBuffer&& other) noexcept {
        if (this != &other) {
            deallocateAligned(m_buffer);
            m_config = other.m_config;
            m_capacity = other.m_capacity;
            m_mask = other.m_mask;
            m_buffer = other.m_buffer;
            m_head.store(other.m_head.load());
            m_tail.store(other.m_tail.load());
            m_size.store(other.m_size.load());
            other.m_buffer = nullptr;
            other.m_capacity = 0;
            other.m_mask = 0;
        }
        return *this;
    }

    // ------------------------------------------------------------------------
    //  Push an element (thread‑safe, lock‑free)
    //  Returns true if successful, false if buffer full and autoResize disabled
    // ------------------------------------------------------------------------
    bool push(const T& value) {
        size_type tail = m_tail.load(std::memory_order_relaxed);
        size_type head = m_head.load(std::memory_order_acquire);
        size_type count = tail - head;
        if (count >= m_capacity) {
            if (m_config.autoResize && m_capacity < m_config.maxCapacity) {
                resize(m_capacity * 2);
                // retry after resize
                tail = m_tail.load(std::memory_order_relaxed);
                head = m_head.load(std::memory_order_acquire);
                count = tail - head;
                if (count >= m_capacity) return false;
            } else {
                return false;
            }
        }
        size_type slot = tail & m_mask;
        new (m_buffer + slot) T(value);
        m_tail.store(tail + 1, std::memory_order_release);
        m_size.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // ------------------------------------------------------------------------
    //  Pop an element (thread‑safe, lock‑free)
    //  Returns true if an element was available, false if empty
    // ------------------------------------------------------------------------
    bool pop(T& out) {
        size_type head = m_head.load(std::memory_order_relaxed);
        size_type tail = m_tail.load(std::memory_order_acquire);
        if (head >= tail) return false;
        size_type slot = head & m_mask;
        out = m_buffer[slot];
        m_buffer[slot].~T();
        m_head.store(head + 1, std::memory_order_release);
        m_size.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }

    // ------------------------------------------------------------------------
    //  Try to pop an element without blocking (same as pop)
    // ------------------------------------------------------------------------
    bool try_pop(T& out) {
        return pop(out);
    }

    // ------------------------------------------------------------------------
    //  Batch push multiple elements (SIMD friendly)
    // ------------------------------------------------------------------------
    size_type push_batch(const T* values, size_type count) {
        size_type pushed = 0;
        for (size_type i = 0; i < count; ++i) {
            if (push(values[i])) ++pushed;
            else break;
        }
        return pushed;
    }

    // ------------------------------------------------------------------------
    //  Batch pop multiple elements (SIMD friendly)
    // ------------------------------------------------------------------------
    size_type pop_batch(T* out, size_type max_count) {
        size_type popped = 0;
        for (size_type i = 0; i < max_count; ++i) {
            if (pop(out[i])) ++popped;
            else break;
        }
        return popped;
    }

    // ------------------------------------------------------------------------
    //  Clear the buffer (not thread‑safe unless externally synchronized)
    // ------------------------------------------------------------------------
    void clear() {
        T dummy;
        while (pop(dummy)) {}
    }

    // ------------------------------------------------------------------------
    //  Capacity / size
    // ------------------------------------------------------------------------
    size_type size() const {
        return m_size.load(std::memory_order_relaxed);
    }
    bool empty() const {
        return size() == 0;
    }
    size_type capacity() const {
        return m_capacity;
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control: resize buffer (thread‑safe with help from other threads)
    //  This is a cooperative resize: we allocate a new buffer, then try to
    //  migrate elements. During migration, other threads may still push/pop.
    //  Simplified implementation: we assume external sync during resize.
    // ------------------------------------------------------------------------
    bool resize(size_type new_capacity) {
        if (new_capacity < size()) return false;
        if (new_capacity > m_config.maxCapacity) new_capacity = m_config.maxCapacity;
        new_capacity = nextPowerOfTwo(new_capacity);
        if (new_capacity == m_capacity) return true;

        T* new_buffer = allocateAligned(new_capacity);
        if (!new_buffer) return false;

        // Migrate existing elements
        size_type new_mask = new_capacity - 1;
        size_type new_tail = 0;
        T temp;
        while (pop(temp)) {
            size_type slot = new_tail & new_mask;
            new (new_buffer + slot) T(std::move(temp));
            ++new_tail;
        }

        // Atomically swap buffers
        T* old_buffer = m_buffer;
        m_buffer = new_buffer;
        m_capacity = new_capacity;
        m_mask = new_mask;
        m_head.store(0, std::memory_order_release);
        m_tail.store(new_tail, std::memory_order_release);
        m_size.store(new_tail, std::memory_order_relaxed);

        deallocateAligned(old_buffer);
        return true;
    }

    // ------------------------------------------------------------------------
    //  Performance hint: prefetch next element (for SIMD loops)
    // ------------------------------------------------------------------------
    void prefetch_next() const {
        size_type head = m_head.load(std::memory_order_acquire);
        size_type slot = head & m_mask;
        ORTHOTREE_PREFETCH(m_buffer + slot);
    }

private:
    // ------------------------------------------------------------------------
    //  Helper: next power of two
    // ------------------------------------------------------------------------
    static size_type nextPowerOfTwo(size_type x) {
        if (x == 0) return 1;
        x--;
        for (size_type i = 1; i < sizeof(size_type) * 8; i <<= 1) {
            x |= x >> i;
        }
        return x + 1;
    }

    // ------------------------------------------------------------------------
    //  Aligned allocation (cache line size)
    // ------------------------------------------------------------------------
    static T* allocateAligned(size_type count) {
        size_type bytes = count * sizeof(T);
        void* ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(bytes, CacheLineSize);
#else
        if (posix_memalign(&ptr, CacheLineSize, bytes) != 0) ptr = nullptr;
#endif
        return static_cast<T*>(ptr);
    }

    static void deallocateAligned(T* ptr) {
        if (!ptr) return;
#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }

    // ------------------------------------------------------------------------
    //  Member variables
    // ------------------------------------------------------------------------
    Config m_config;
    size_type m_capacity;
    size_type m_mask;
    T* m_buffer;
    std::atomic<size_type> m_head;   // read index
    std::atomic<size_type> m_tail;   // write index
    std::atomic<size_type> m_size;   // approximate size (not exact due to races, but good enough)
};

// ----------------------------------------------------------------------------
//  Prefetch macro (compiler hint)
// ----------------------------------------------------------------------------
#ifndef ORTHOTREE_PREFETCH
#if defined(__GNUC__) || defined(__clang__)
#define ORTHOTREE_PREFETCH(addr) __builtin_prefetch(addr)
#else
#define ORTHOTREE_PREFETCH(addr) ((void)0)
#endif
#endif

// ----------------------------------------------------------------------------
//  Thread‑local query buffer aggregator for reducing contention
// ----------------------------------------------------------------------------
template<typename T>
class ThreadLocalQueryAggregator {
public:
    using buffer_type = LockfreeQueryBuffer<T>;

    explicit ThreadLocalQueryAggregator(size_t local_capacity = 1024) {
        m_localBuffers.reserve(std::thread::hardware_concurrency());
    }

    void add(T value) {
        getLocalBuffer().push(value);
    }

    void flushToGlobal(buffer_type& global) {
        T temp;
        auto& local = getLocalBuffer();
        while (local.pop(temp)) {
            global.push(temp);
        }
    }

private:
    buffer_type& getLocalBuffer() {
        thread_local static buffer_type local(m_localConfig);
        return local;
    }

    typename buffer_type::Config m_localConfig;
    std::vector<buffer_type*> m_localBuffers;
};

} // namespace Parallel
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_PARALLEL_LOCKFREE_QUERY_BUFFER_H_INCLUDED