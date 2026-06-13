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

#ifndef ORTHOTREE_CORE_OT_BASE_H_INCLUDED
#define ORTHOTREE_CORE_OT_BASE_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/vector_math.h"
#include "../core/math/geometry_queries.h"
#include "../core/math/transform.h"
#include "../core/configuration.h"
#include "../detail/common.h"
#include "../detail/simd_utils.h"

#include <atomic>
#include <memory>
#include <type_traits>
#include <vector>
#include <optional>
#include <functional>
#include <chrono>

namespace OrthoTree {

// ============================================================================
//  Base class for all octree implementations (CRTP).
//  Provides common functionality: statistics, versioning, configuration,
//  and dynamic environment controls (adaptive depth, performance profiling).
// ============================================================================
template<typename Derived>
class ot_base {
public:
    using size_type = size_t;
    using version_type = uint64_t;
    using timestamp_type = std::chrono::steady_clock::time_point;

    // ------------------------------------------------------------------------
    //  Statistics structure (cache‑line aligned)
    // ------------------------------------------------------------------------
    struct alignas(ORTHOTREE_CACHE_LINE_SIZE) Statistics {
        size_type nodeCount = 0;
        size_type entityCount = 0;
        size_type maxDepth = 0;
        size_type memoryUsageBytes = 0;
        size_type queryCount = 0;
        size_type insertCount = 0;
        size_type removeCount = 0;
        size_type updateCount = 0;
        double averageQueryTimeMs = 0.0;
        double averageInsertTimeMs = 0.0;
        double averageRemoveTimeMs = 0.0;
        timestamp_type lastResetTime;
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    ot_base() noexcept
        : m_version(0)
        , m_config(RuntimeConfiguration())
        , m_stats()
        , m_profilingEnabled(false) {
        m_stats.lastResetTime = std::chrono::steady_clock::now();
    }

    virtual ~ot_base() = default;

    // ------------------------------------------------------------------------
    //  Common public interface (to be implemented by derived)
    // ------------------------------------------------------------------------
    virtual size_type size() const noexcept = 0;
    virtual bool empty() const noexcept = 0;
    virtual size_type nodeCount() const noexcept = 0;
    virtual size_type height() const noexcept = 0;
    virtual size_type memoryUsage() const noexcept = 0;
    virtual void clear() = 0;

    // ------------------------------------------------------------------------
    //  Versioning (for double‑buffering and cache invalidation)
    // ------------------------------------------------------------------------
    version_type version() const noexcept { return m_version.load(std::memory_order_acquire); }
    void incrementVersion() noexcept { m_version.fetch_add(1, std::memory_order_release); }

    // ------------------------------------------------------------------------
    //  Configuration management (dynamic environment control)
    // ------------------------------------------------------------------------
    void setConfiguration(const RuntimeConfiguration& cfg) noexcept {
        m_config = cfg;
        onConfigurationChanged();
    }
    const RuntimeConfiguration& configuration() const noexcept { return m_config; }

    // Apply configuration changes to derived class (override if needed)
    virtual void onConfigurationChanged() {}

    // ------------------------------------------------------------------------
    //  Profiling and statistics (with SIMD hints)
    // ------------------------------------------------------------------------
    void enableProfiling(bool enable) noexcept { m_profilingEnabled = enable; }
    bool profilingEnabled() const noexcept { return m_profilingEnabled; }

    const Statistics& getStatistics() const noexcept { return m_stats; }
    void resetStatistics() {
        m_stats = Statistics();
        m_stats.lastResetTime = std::chrono::steady_clock::now();
    }

    // Record a query (automatically called by derived)
    void recordQuery(double durationMs = 0.0) {
        if (m_profilingEnabled) {
            ++m_stats.queryCount;
            if (durationMs > 0.0) {
                m_stats.averageQueryTimeMs = (m_stats.averageQueryTimeMs * (m_stats.queryCount - 1) + durationMs) / m_stats.queryCount;
            }
        }
    }

    void recordInsert(double durationMs = 0.0) {
        if (m_profilingEnabled) {
            ++m_stats.insertCount;
            if (durationMs > 0.0) {
                m_stats.averageInsertTimeMs = (m_stats.averageInsertTimeMs * (m_stats.insertCount - 1) + durationMs) / m_stats.insertCount;
            }
        }
    }

    void recordRemove(double durationMs = 0.0) {
        if (m_profilingEnabled) {
            ++m_stats.removeCount;
            if (durationMs > 0.0) {
                m_stats.averageRemoveTimeMs = (m_stats.averageRemoveTimeMs * (m_stats.removeCount - 1) + durationMs) / m_stats.removeCount;
            }
        }
    }

    void recordUpdate(double durationMs = 0.0) {
        if (m_profilingEnabled) {
            ++m_stats.updateCount;
            // No average stored for updates, but we could extend.
        }
    }

    // Update statistics (called by derived after structural changes)
    void updateStatistics(size_type nodes, size_type entities, size_type depth) {
        m_stats.nodeCount = nodes;
        m_stats.entityCount = entities;
        m_stats.maxDepth = depth;
        m_stats.memoryUsageBytes = memoryUsage();
    }

    // ------------------------------------------------------------------------
    //  Dynamic adaptation: suggest optimal depth based on entity density
    // ------------------------------------------------------------------------
    uint8_t suggestOptimalDepth() const {
        if (m_stats.entityCount == 0) return m_config.maxDepth;
        double density = static_cast<double>(m_stats.entityCount) / (m_stats.maxDepth + 1);
        if (density < 10.0) return std::min<uint8_t>(8, m_config.maxDepth);
        if (density < 100.0) return std::min<uint8_t>(12, m_config.maxDepth);
        if (density < 1000.0) return std::min<uint8_t>(16, m_config.maxDepth);
        return m_config.maxDepth;
    }

    // ------------------------------------------------------------------------
    //  Performance hints: prefetch nodes for traversal (SIMD friendly)
    // ------------------------------------------------------------------------
    virtual void prefetchNodes(uint32_t* nodeIndices, size_type count) const {
        // Default: no‑op. Derived can implement with prefetch instructions.
        (void)nodeIndices; (void)count;
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128) {
            // Compiler hint for prefetch (non‑binding)
            for (size_type i = 0; i < count; ++i) {
                ORTHOTREE_PREFETCH(nodeIndices + i);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Time‑travel debugging (optional)
    // ------------------------------------------------------------------------
    struct Snapshot {
        version_type version;
        std::vector<uint8_t> serializedState;
        timestamp_type timestamp;
    };

    virtual bool takeSnapshot(Snapshot& out) const {
        // Stub – derived must implement serialization.
        (void)out;
        return false;
    }

    virtual bool restoreSnapshot(const Snapshot& snap) {
        (void)snap;
        return false;
    }

protected:
    // Helper for RAII profiling scope
    class ProfilerScope {
    public:
        ProfilerScope(ot_base* base, void (ot_base::*recorder)(double))
            : m_base(base), m_recorder(recorder), m_start(std::chrono::steady_clock::now()) {}
        ~ProfilerScope() {
            if (m_base && m_recorder && m_base->profilingEnabled()) {
                auto end = std::chrono::steady_clock::now();
                double ms = std::chrono::duration<double, std::milli>(end - m_start).count();
                (m_base->*m_recorder)(ms);
            }
        }
    private:
        ot_base* m_base;
        void (ot_base::*m_recorder)(double);
        std::chrono::steady_clock::time_point m_start;
    };

    std::atomic<version_type> m_version;
    RuntimeConfiguration m_config;
    Statistics m_stats;
    bool m_profilingEnabled;
};

// ----------------------------------------------------------------------------
//  Global environment for base configuration (shared across all octrees)
// ----------------------------------------------------------------------------
class GlobalOctreeEnvironment {
public:
    static GlobalOctreeEnvironment& instance() {
        static GlobalOctreeEnvironment env;
        return env;
    }

    void setGlobalMaxDepth(uint8_t depth) { m_globalMaxDepth = depth; }
    uint8_t globalMaxDepth() const { return m_globalMaxDepth; }

    void setGlobalBucketSize(uint16_t size) { m_globalBucketSize = size; }
    uint16_t globalBucketSize() const { return m_globalBucketSize; }

    void setDefaultAllocator(PMRAllocator<std::byte> alloc) { m_defaultAllocator = alloc; }
    const PMRAllocator<std::byte>& defaultAllocator() const { return m_defaultAllocator; }

    // Enable/disable global profiling
    void setProfilingEnabled(bool enable) { m_profilingEnabled = enable; }
    bool profilingEnabled() const { return m_profilingEnabled; }

private:
    GlobalOctreeEnvironment()
        : m_globalMaxDepth(ORTHOTREE_DEFAULT_MAX_DEPTH)
        , m_globalBucketSize(ORTHOTREE_DEFAULT_BUCKET_SIZE)
        , m_defaultAllocator()
        , m_profilingEnabled(false) {}

    uint8_t m_globalMaxDepth;
    uint16_t m_globalBucketSize;
    PMRAllocator<std::byte> m_defaultAllocator;
    bool m_profilingEnabled;
};

} // namespace OrthoTree

#endif // ORTHOTREE_CORE_OT_BASE_H_INCLUDED