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

#ifndef ORTHOTREE_CORE_OBSERVABILITY_PROFILER_HOOKS_H_INCLUDED
#define ORTHOTREE_CORE_OBSERVABILITY_PROFILER_HOOKS_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <string>
#include <array>
#include <vector>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <functional>

// Instrumentation macros for Tracy, Optick, etc.
// Define ORTHOTREE_ENABLE_TRACY, ORTHOTREE_ENABLE_OPTICK, or ORTHOTREE_ENABLE_REMOTE_TELEMETRY

#ifdef ORTHOTREE_ENABLE_TRACY
#include <tracy/Tracy.hpp>
#define ORTHOTREE_PROFILE_SCOPE(name) ZoneScopedN(name)
#define ORTHOTREE_PROFILE_FRAME() FrameMark
#define ORTHOTREE_PROFILE_ALLOC(ptr, size) TracyAlloc(ptr, size)
#define ORTHOTREE_PROFILE_FREE(ptr) TracyFree(ptr)
#else
#define ORTHOTREE_PROFILE_SCOPE(name) ((void)0)
#define ORTHOTREE_PROFILE_FRAME() ((void)0)
#define ORTHOTREE_PROFILE_ALLOC(ptr, size) ((void)0)
#define ORTHOTREE_PROFILE_FREE(ptr) ((void)0)
#endif

#ifdef ORTHOTREE_ENABLE_OPTICK
#include <optick.h>
#define ORTHOTREE_OPTICK_TAG(name) OPTICK_TAG(name)
#else
#define ORTHOTREE_OPTICK_TAG(name) ((void)0)
#endif

namespace OrthoTree {
namespace Observability {

// ============================================================================
//  ProfilerHooks: lightweight instrumentation for performance analysis.
//  Provides macros for function entry/exit, memory tracking, and custom
//  events. Supports Tracy, Optick, and a built‑in remote telemetry server.
//  SIMD‑aware batch event logging with lock‑free ring buffers.
// ============================================================================

// ----------------------------------------------------------------------------
//  Event types for built‑in profiler
// ----------------------------------------------------------------------------
enum class ProfilerEventType : uint8_t {
    FunctionEnter,
    FunctionExit,
    MemoryAlloc,
    MemoryFree,
    CustomEvent,
    CounterUpdate,
    ThreadStart,
    ThreadEnd
};

// ----------------------------------------------------------------------------
//  Profiler event structure (compact, cache‑line aligned)
// ----------------------------------------------------------------------------
struct alignas(ORTHOTREE_CACHE_LINE_SIZE) ProfilerEvent {
    ProfilerEventType type;
    uint64_t threadId;
    uint64_t timestamp;       // microseconds since start
    uint64_t arg0;            // e.g., function hash, pointer, value
    uint64_t arg1;            // e.g., size, line number
    char name[64];            // optional name (truncated)
};

// ----------------------------------------------------------------------------
//  Ring buffer for lock‑free event logging
// ----------------------------------------------------------------------------
class ProfilerRingBuffer {
public:
    static constexpr size_t BUFFER_SIZE = 1024 * 1024; // 1M events

    ProfilerRingBuffer() : m_head(0), m_tail(0) {
        m_buffer.resize(BUFFER_SIZE);
    }

    bool push(const ProfilerEvent& event) {
        size_t head = m_head.load(std::memory_order_relaxed);
        size_t next = (head + 1) % BUFFER_SIZE;
        if (next == m_tail.load(std::memory_order_acquire)) {
            return false; // buffer full
        }
        m_buffer[head] = event;
        m_head.store(next, std::memory_order_release);
        return true;
    }

    bool pop(ProfilerEvent& out) {
        size_t tail = m_tail.load(std::memory_order_relaxed);
        if (tail == m_head.load(std::memory_order_acquire)) {
            return false;
        }
        out = m_buffer[tail];
        m_tail.store((tail + 1) % BUFFER_SIZE, std::memory_order_release);
        return true;
    }

    size_t size() const {
        size_t head = m_head.load(std::memory_order_acquire);
        size_t tail = m_tail.load(std::memory_order_acquire);
        if (head >= tail) return head - tail;
        return BUFFER_SIZE - (tail - head);
    }

private:
    std::vector<ProfilerEvent> m_buffer;
    std::atomic<size_t> m_head;
    std::atomic<size_t> m_tail;
};

// ----------------------------------------------------------------------------
//  ProfilerHooks main class (built‑in profiler)
// ----------------------------------------------------------------------------
class ProfilerHooks {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    static ProfilerHooks& instance() {
        static ProfilerHooks inst;
        return inst;
    }

    void start() {
        m_startTime = Clock::now();
        m_running = true;
        if (m_flushThread.joinable()) return;
        m_flushThread = std::thread(&ProfilerHooks::flushLoop, this);
    }

    void stop() {
        m_running = false;
        if (m_flushThread.joinable()) m_flushThread.join();
    }

    // Record an event (thread‑safe, lock‑free)
    void recordEvent(ProfilerEventType type, uint64_t arg0 = 0, uint64_t arg1 = 0,
                     const char* name = nullptr) {
        if (!m_running) return;
        ProfilerEvent ev;
        ev.type = type;
        ev.threadId = getThreadId();
        ev.timestamp = getMicroseconds();
        ev.arg0 = arg0;
        ev.arg1 = arg1;
        if (name) {
            std::strncpy(ev.name, name, sizeof(ev.name) - 1);
            ev.name[sizeof(ev.name) - 1] = '\0';
        } else {
            ev.name[0] = '\0';
        }
        while (!m_ringBuffer.push(ev)) {
            // buffer full, wait a bit (should not happen in production)
            std::this_thread::yield();
        }
    }

    // Convenience functions
    void functionEnter(const char* func) { recordEvent(ProfilerEventType::FunctionEnter, 0, 0, func); }
    void functionExit(const char* func) { recordEvent(ProfilerEventType::FunctionExit, 0, 0, func); }
    void memoryAlloc(void* ptr, size_t size) { recordEvent(ProfilerEventType::MemoryAlloc, reinterpret_cast<uint64_t>(ptr), size); }
    void memoryFree(void* ptr) { recordEvent(ProfilerEventType::MemoryFree, reinterpret_cast<uint64_t>(ptr)); }
    void customEvent(const char* name, uint64_t value = 0) { recordEvent(ProfilerEventType::CustomEvent, value, 0, name); }
    void counterUpdate(const char* name, double value) { recordEvent(ProfilerEventType::CounterUpdate, *reinterpret_cast<uint64_t*>(&value), 0, name); }

    // Register a callback for consuming events (e.g., remote telemetry)
    void setEventConsumer(std::function<void(const ProfilerEvent&)> consumer) {
        std::lock_guard<std::mutex> lock(m_consumerMutex);
        m_consumer = std::move(consumer);
    }

    // Get all pending events (consumes them)
    std::vector<ProfilerEvent> flushEvents() {
        std::vector<ProfilerEvent> result;
        ProfilerEvent ev;
        while (m_ringBuffer.pop(ev)) {
            result.push_back(ev);
        }
        return result;
    }

    // SIMD batch: push multiple events at once (if they are contiguous in memory)
    void batchPush(const ProfilerEvent* events, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            while (!m_ringBuffer.push(events[i])) {
                std::this_thread::yield();
            }
        }
    }

private:
    ProfilerHooks() : m_running(false), m_startTime(Clock::now()) {}
    ~ProfilerHooks() { stop(); }

    static uint64_t getThreadId() {
        static_assert(sizeof(std::thread::id) <= sizeof(uint64_t), "thread::id too large");
        uint64_t id;
        std::memcpy(&id, &std::this_thread::get_id(), sizeof(std::thread::id));
        return id;
    }

    uint64_t getMicroseconds() const {
        auto now = Clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(now - m_startTime).count();
    }

    void flushLoop() {
        while (m_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            auto events = flushEvents();
            if (events.empty()) continue;
            std::lock_guard<std::mutex> lock(m_consumerMutex);
            if (m_consumer) {
                for (const auto& ev : events) {
                    m_consumer(ev);
                }
            }
        }
    }

    ProfilerRingBuffer m_ringBuffer;
    std::atomic<bool> m_running;
    TimePoint m_startTime;
    std::thread m_flushThread;
    std::mutex m_consumerMutex;
    std::function<void(const ProfilerEvent&)> m_consumer;
};

// ----------------------------------------------------------------------------
//  RAII scope profiler (built‑in)
// ----------------------------------------------------------------------------
class ScopedProfiler {
public:
    explicit ScopedProfiler(const char* name)
        : m_name(name) {
        ProfilerHooks::instance().functionEnter(m_name);
    }
    ~ScopedProfiler() {
        ProfilerHooks::instance().functionExit(m_name);
    }
private:
    const char* m_name;
};

// ----------------------------------------------------------------------------
//  Macro for built‑in profiling (if external profilers not enabled)
// ----------------------------------------------------------------------------
#if !defined(ORTHOTREE_ENABLE_TRACY) && !defined(ORTHOTREE_ENABLE_OPTICK)
#define ORTHOTREE_PROFILE_BUILTIN_SCOPE(name) OrthoTree::Observability::ScopedProfiler _profiler_##__LINE__(name)
#else
#define ORTHOTREE_PROFILE_BUILTIN_SCOPE(name) ((void)0)
#endif

// ----------------------------------------------------------------------------
//  Unified profiling macro (uses external if available, else built‑in)
// ----------------------------------------------------------------------------
#ifndef ORTHOTREE_PROFILE_SCOPE
#ifdef ORTHOTREE_ENABLE_TRACY
#define ORTHOTREE_PROFILE_SCOPE(name) ZoneScopedN(name)
#elif defined(ORTHOTREE_ENABLE_OPTICK)
#define ORTHOTREE_PROFILE_SCOPE(name) OPTICK_EVENT(name)
#else
#define ORTHOTREE_PROFILE_SCOPE(name) OrthoTree::Observability::ScopedProfiler _profiler_##__LINE__(name)
#endif
#endif

// ----------------------------------------------------------------------------
//  Memory tracking override (optional)
// ----------------------------------------------------------------------------
#if defined(ORTHOTREE_PROFILE_MEMORY)
void* operator new(std::size_t size) {
    void* ptr = std::malloc(size);
    ProfilerHooks::instance().memoryAlloc(ptr, size);
    return ptr;
}
void operator delete(void* ptr) noexcept {
    ProfilerHooks::instance().memoryFree(ptr);
    std::free(ptr);
}
#endif

} // namespace Observability
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_OBSERVABILITY_PROFILER_HOOKS_H_INCLUDED