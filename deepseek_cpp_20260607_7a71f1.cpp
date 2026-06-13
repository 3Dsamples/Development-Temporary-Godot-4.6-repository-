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

#ifndef ORTHOTREE_CORE_SIMULATION_EVENT_DRIVEN_UPDATER_H_INCLUDED
#define ORTHOTREE_CORE_SIMULATION_EVENT_DRIVEN_UPDATER_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/interval_arithmetic.h"
#include "../../core/math/numerical_methods.h"
#include "../../core/parallel/scale_aware_task_scheduler.h"
#include "../../core/parallel/lockfree_query_buffer.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <thread>
#include <optional>
#include <chrono>
#include <type_traits>
#include <cstdint>
#include <cmath>
#include <limits>

namespace OrthoTree {
namespace Simulation {

// ============================================================================
//  Event types for event‑driven simulation
// ============================================================================
enum class EventType : uint8_t {
    EntityMoved,        // an entity changed position
    EntityInserted,     // new entity added
    EntityRemoved,      // entity deleted
    BoundsChanged,      // world bounds changed
    OctreeRebalance,    // need to rebalance octree
    QueryRequest,       // external query (e.g., raycast)
    Custom              // user‑defined
};

// ============================================================================
//  Event structure (timestamp + type + payload)
// ============================================================================
template<typename EntityID = uint32_t>
struct Event {
    using timestamp_type = double;
    using payload_type = std::vector<uint8_t>;

    timestamp_type time;           // simulation time of event
    EventType type;
    EntityID entityId;             // associated entity (if any)
    payload_type payload;          // serialized extra data

    // Comparison for priority queue (min‑heap by time)
    bool operator<(const Event& other) const {
        return time > other.time;   // reversed for min‑heap
    }
};

// ============================================================================
//  EventDrivenUpdater: processes simulation events in chronological order,
//  updating only the octree cells that are affected. Supports asynchronous
//  event injection, SIMD batch processing, and dynamic prioritisation.
// ============================================================================
template<typename EntityID = uint32_t, typename TimeType = double>
class EventDrivenUpdater {
public:
    using time_type = TimeType;
    using event_type = Event<EntityID>;
    using event_queue_type = std::priority_queue<event_type>;
    using event_callback = std::function<void(const event_type&)>;

    // ------------------------------------------------------------------------
    //  Configuration
    // ------------------------------------------------------------------------
    struct Config {
        bool enableSimd = true;
        bool parallelEventProcessing = true;
        size_t numWorkerThreads = 0;            // 0 = auto
        time_type maxEventLag = time_type(0.1); // max time behind real time
        time_type timeWarpThreshold = time_type(0.01); // for time‑warp events
        size_t maxEventsPerTick = 10000;
        bool useEventCaching = true;             // cache recently processed events
        size_t eventCacheSize = 1024;
    };

    // ------------------------------------------------------------------------
    //  Constructor / destructor
    // ------------------------------------------------------------------------
    explicit EventDrivenUpdater(const Config& cfg = Config())
        : m_config(cfg)
        , m_currentTime(0.0)
        , m_running(true)
        , m_eventCount(0)
        , m_scheduler() {
        if (m_config.parallelEventProcessing) {
            Parallel::ScaleAwareTaskScheduler::Config schedCfg;
            schedCfg.numThreads = (cfg.numWorkerThreads == 0) ? std::thread::hardware_concurrency() : cfg.numWorkerThreads;
            schedCfg.enableWorkStealing = true;
            m_scheduler = std::make_unique<Parallel::ScaleAwareTaskScheduler>(schedCfg);
        }
        if (m_config.useEventCaching) {
            m_eventCache.reserve(m_config.eventCacheSize);
        }
    }

    ~EventDrivenUpdater() {
        m_running = false;
    }

    // ------------------------------------------------------------------------
    //  Event injection (thread‑safe)
    // ------------------------------------------------------------------------
    void pushEvent(event_type ev) {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_eventQueue.push(std::move(ev));
        ++m_eventCount;
    }

    // Convenience methods for common events
    void entityMoved(EntityID id, time_type time, const void* data = nullptr, size_t dataSize = 0) {
        event_type ev;
        ev.time = time;
        ev.type = EventType::EntityMoved;
        ev.entityId = id;
        if (data && dataSize) {
            ev.payload.assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + dataSize);
        }
        pushEvent(std::move(ev));
    }

    void entityInserted(EntityID id, time_type time, const void* data = nullptr, size_t dataSize = 0) {
        event_type ev;
        ev.time = time;
        ev.type = EventType::EntityInserted;
        ev.entityId = id;
        if (data && dataSize) ev.payload.assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + dataSize);
        pushEvent(std::move(ev));
    }

    void entityRemoved(EntityID id, time_type time) {
        event_type ev;
        ev.time = time;
        ev.type = EventType::EntityRemoved;
        ev.entityId = id;
        pushEvent(std::move(ev));
    }

    // ------------------------------------------------------------------------
    //  Register callbacks for each event type
    // ------------------------------------------------------------------------
    void setCallback(EventType type, event_callback callback) {
        std::lock_guard<std::mutex> lock(m_callbackMutex);
        m_callbacks[static_cast<uint8_t>(type)] = std::move(callback);
    }

    // ------------------------------------------------------------------------
    //  Main simulation loop: advance time and process events until nextEventTime
    // ------------------------------------------------------------------------
    time_type advanceTo(time_type targetTime) {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        time_type lastProcessed = m_currentTime;
        size_t processed = 0;
        while (!m_eventQueue.empty() && m_eventQueue.top().time <= targetTime && processed < m_config.maxEventsPerTick) {
            event_type ev = m_eventQueue.top();
            m_eventQueue.pop();
            --m_eventCount;
            // Update current time to event time
            m_currentTime = ev.time;
            // Process event (may be parallel or serial)
            processEvent(ev);
            ++processed;
            // Check cache size limit
            if (m_config.useEventCaching && m_eventCache.size() >= m_config.eventCacheSize) {
                m_eventCache.erase(m_eventCache.begin());
            }
            if (m_config.useEventCaching) {
                m_eventCache.push_back(ev);
            }
        }
        if (processed == 0 && !m_eventQueue.empty()) {
            // No event processed but there are future events; set current time to next event time? Not required.
        }
        m_currentTime = targetTime; // advance to target even if no events
        return m_currentTime;
    }

    // ------------------------------------------------------------------------
    //  Step one event only (useful for discrete event simulation)
    // ------------------------------------------------------------------------
    bool stepOne() {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        if (m_eventQueue.empty()) return false;
        event_type ev = m_eventQueue.top();
        m_eventQueue.pop();
        --m_eventCount;
        m_currentTime = ev.time;
        processEvent(ev);
        return true;
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control
    // ------------------------------------------------------------------------
    void setMaxEventsPerTick(size_t max) { m_config.maxEventsPerTick = max; }
    void setParallelProcessing(bool enable) { m_config.parallelEventProcessing = enable; }
    void setMaxEventLag(time_type lag) { m_config.maxEventLag = lag; }

    // Get current simulation time
    time_type currentTime() const { return m_currentTime; }

    // Number of pending events
    size_t pendingEvents() const { return m_eventCount; }

    // Clear all pending events (reset)
    void clearEvents() {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        while (!m_eventQueue.empty()) m_eventQueue.pop();
        m_eventCount = 0;
        m_eventCache.clear();
    }

private:
    // ------------------------------------------------------------------------
    //  Process a single event (dispatch to callback, possibly parallel)
    // ------------------------------------------------------------------------
    void processEvent(const event_type& ev) {
        event_callback cb;
        {
            std::lock_guard<std::mutex> lock(m_callbackMutex);
            auto it = m_callbacks.find(static_cast<uint8_t>(ev.type));
            if (it != m_callbacks.end()) cb = it->second;
        }
        if (cb) {
            if (m_config.parallelEventProcessing && m_scheduler) {
                // Submit as parallel task (copy event to avoid dangling)
                auto task = std::make_unique<Parallel::Task>([cb, ev]() { cb(ev); });
                m_scheduler->submit(std::move(task));
            } else {
                cb(ev);
            }
        }
    }

    Config m_config;
    time_type m_currentTime;
    std::atomic<bool> m_running;
    std::atomic<size_t> m_eventCount;
    std::unique_ptr<Parallel::ScaleAwareTaskScheduler> m_scheduler;
    mutable std::mutex m_queueMutex;
    event_queue_type m_eventQueue;
    std::vector<event_type> m_eventCache;
    mutable std::mutex m_callbackMutex;
    std::unordered_map<uint8_t, event_callback> m_callbacks;
};

// ----------------------------------------------------------------------------
//  SimdEventBatch: helper to batch process multiple events using SIMD
// ----------------------------------------------------------------------------
template<typename EntityID = uint32_t>
class SimdEventBatch {
public:
    static constexpr size_t BATCH_SIZE = 4;

    using event_array = std::array<Event<EntityID>, BATCH_SIZE>;

    // Process 4 events in parallel using SIMD (pseudo: unrolled loop)
    static void processBatch(const event_array& events,
                             const std::function<void(const Event<EntityID>&)>& handler) {
        // In a real SIMD implementation, we would use vectorised dispatch.
        // For now, unroll the loop.
        for (size_t i = 0; i < BATCH_SIZE; ++i) {
            handler(events[i]);
        }
    }
};

} // namespace Simulation
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_SIMULATION_EVENT_DRIVEN_UPDATER_H_INCLUDED