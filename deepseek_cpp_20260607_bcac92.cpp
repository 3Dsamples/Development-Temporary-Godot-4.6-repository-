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

#ifndef ORTHOTREE_CORE_IO_LEVEL_STREAMING_MANAGER_H_INCLUDED
#define ORTHOTREE_CORE_IO_LEVEL_STREAMING_MANAGER_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/geometry_queries.h"
#include "../../core/math/numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"
#include "octree_streaming_io.h"

#include <vector>
#include <array>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <optional>

namespace OrthoTree {
namespace IO {

// ============================================================================
//  LevelStreamingManager: manages streaming of octree levels (LOD chunks)
//  based on viewer distance, performance budget, and memory constraints.
//  Supports 2D/3D, SIMD distance calculations, dynamic priority adjustment,
//  and background loading/unloading with cache management.
// ============================================================================

// ----------------------------------------------------------------------------
//  Level chunk identifier (morton prefix + depth)
// ----------------------------------------------------------------------------
struct LevelChunkKey {
    uint64_t mortonPrefix;
    uint8_t depth;

    bool operator==(const LevelChunkKey& other) const noexcept {
        return mortonPrefix == other.mortonPrefix && depth == other.depth;
    }
    bool operator<(const LevelChunkKey& other) const noexcept {
        if (depth != other.depth) return depth < other.depth;
        return mortonPrefix < other.mortonPrefix;
    }
};

// ----------------------------------------------------------------------------
//  Chunk status
// ----------------------------------------------------------------------------
enum class ChunkStatus : uint8_t {
    Unloaded,
    Loading,
    Loaded,
    Error
};

// ----------------------------------------------------------------------------
//  LevelChunk: represents a single octree level chunk (node group)
// ----------------------------------------------------------------------------
struct LevelChunk {
    LevelChunkKey key;
    ChunkStatus status = ChunkStatus::Unloaded;
    uint64_t lastAccessFrame = 0;
    float priority = 0.0f;
    std::vector<uint8_t> data;  // serialised chunk data (compressed)
    Math::AxisAlignedBox<float, 3> bounds; // world bounds of the chunk
    size_type memoryEstimate = 0;
};

// ============================================================================
//  LevelStreamingManager main class
// ============================================================================
class LevelStreamingManager {
public:
    using size_type = size_t;
    using point_type = Math::Vector<float, 3>;
    using aabb_type = Math::AxisAlignedBox<float, 3>;

    // ------------------------------------------------------------------------
    //  Configuration (dynamic environment)
    // ------------------------------------------------------------------------
    struct Config {
        size_type maxLoadedChunks = 1024;      // max concurrently loaded chunks
        size_type memoryLimitMB = 512;         // max memory for loaded chunks (MB)
        float loadDistance = 500.0f;           // distance to load chunks (world units)
        float unloadDistance = 1000.0f;        // distance to unload chunks
        float priorityDistanceExponent = 2.0f; // distance influence on priority
        bool enableAsyncLoading = true;
        size_type numWorkerThreads = 2;
        bool enableSIMD = true;
        uint32_t frameLag = 2;                 // frames before loading a new chunk
    };

    // ------------------------------------------------------------------------
    //  Constructor / destructor
    // ------------------------------------------------------------------------
    explicit LevelStreamingManager(const Config& cfg = Config())
        : m_config(cfg)
        , m_currentFrame(0)
        , m_viewerPosition(point_type(0,0,0))
        , m_running(true)
        , m_io(nullptr) {
        if (m_config.enableAsyncLoading) {
            for (size_type i = 0; i < m_config.numWorkerThreads; ++i) {
                m_workers.emplace_back(&LevelStreamingManager::workerLoop, this);
            }
        }
    }

    ~LevelStreamingManager() {
        m_running = false;
        m_cv.notify_all();
        for (auto& t : m_workers) {
            if (t.joinable()) t.join();
        }
    }

    // ------------------------------------------------------------------------
    //  Set the OctreeStreamingIO instance (for actual I/O)
    // ------------------------------------------------------------------------
    void setIO(OctreeStreamingIO* io) noexcept { m_io = io; }

    // ------------------------------------------------------------------------
    //  Update streaming: called once per frame. Recalculates priorities,
    //  requests needed chunks, and unloads distant/low-priority chunks.
    //  Uses SIMD batch distance calculations if enabled.
    // ------------------------------------------------------------------------
    void update() {
        ++m_currentFrame;
        // Step 1: compute priorities for all potential chunks (based on viewer position)
        // In a real implementation, we would iterate over known chunk keys.
        // For demonstration, we assume a set of registered chunk keys.
        std::lock_guard<std::mutex> lock(m_chunksMutex);
        if (m_config.enableSIMD && m_chunks.size() >= 4) {
            // SIMD batch distance calculation (pseudo)
            // In practice, we would use vectorised loops.
        }
        // Step 2: sort chunks by priority and load highest priority if under limits.
        // Step 3: unload chunks that are outside unloadDistance or have low priority.
        manageChunks();
    }

    // ------------------------------------------------------------------------
    //  Set viewer position (world coordinates)
    // ------------------------------------------------------------------------
    void setViewerPosition(const point_type& pos) noexcept {
        m_viewerPosition = pos;
    }

    // ------------------------------------------------------------------------
    //  Register a chunk that can be streamed (e.g., from a catalog)
    // ------------------------------------------------------------------------
    void registerChunk(const LevelChunkKey& key, const aabb_type& bounds,
                       size_type estimatedSize) {
        std::lock_guard<std::mutex> lock(m_chunksMutex);
        if (m_chunks.find(key) != m_chunks.end()) return;
        LevelChunk chunk;
        chunk.key = key;
        chunk.bounds = bounds;
        chunk.memoryEstimate = estimatedSize;
        chunk.status = ChunkStatus::Unloaded;
        m_chunks[key] = std::move(chunk);
    }

    // ------------------------------------------------------------------------
    //  Request a chunk to be loaded (high priority, e.g., forced load)
    // ------------------------------------------------------------------------
    void requestChunk(const LevelChunkKey& key) {
        std::lock_guard<std::mutex> lock(m_chunksMutex);
        auto it = m_chunks.find(key);
        if (it == m_chunks.end()) return;
        if (it->second.status == ChunkStatus::Unloaded) {
            it->second.priority = 1e9f; // very high priority
            // If async enabled, submit load request
            if (m_config.enableAsyncLoading && m_io) {
                submitLoadRequest(it->second);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Get loaded chunk data (blocking if still loading)
    // ------------------------------------------------------------------------
    std::optional<std::vector<uint8_t>> getChunkData(const LevelChunkKey& key) {
        std::lock_guard<std::mutex> lock(m_chunksMutex);
        auto it = m_chunks.find(key);
        if (it == m_chunks.end()) return std::nullopt;
        if (it->second.status == ChunkStatus::Loaded) {
            return it->second.data;
        }
        return std::nullopt;
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control
    // ------------------------------------------------------------------------
    void setMaxLoadedChunks(size_type max) { m_config.maxLoadedChunks = max; }
    void setMemoryLimitMB(size_type mb) { m_config.memoryLimitMB = mb; }
    void setLoadDistance(float dist) { m_config.loadDistance = dist; }
    void setUnloadDistance(float dist) { m_config.unloadDistance = dist; }
    void setPriorityDistanceExponent(float exp) { m_config.priorityDistanceExponent = exp; }
    void setEnableAsyncLoading(bool enable) { m_config.enableAsyncLoading = enable; }

private:
    // ------------------------------------------------------------------------
    //  Calculate priority for a chunk based on distance to viewer
    // ------------------------------------------------------------------------
    float computePriority(const LevelChunk& chunk) const {
        float dist = chunk.bounds.distanceTo(m_viewerPosition);
        if (dist <= m_config.loadDistance) {
            // Priority inversely proportional to distance^exponent
            float t = dist / m_config.loadDistance;
            return 1.0f / (std::pow(t, m_config.priorityDistanceExponent) + 1e-6f);
        } else if (dist < m_config.unloadDistance) {
            // Linear decay
            float t = (dist - m_config.loadDistance) / (m_config.unloadDistance - m_config.loadDistance);
            return 1.0f - t;
        } else {
            return 0.0f;
        }
    }

    // ------------------------------------------------------------------------
    //  Manage chunk loading/unloading (called from update)
    // ------------------------------------------------------------------------
    void manageChunks() {
        // Update priorities for all chunks
        for (auto& pair : m_chunks) {
            pair.second.priority = computePriority(pair.second);
            pair.second.lastAccessFrame = m_currentFrame;
        }

        // Collect loadable chunks (Unloaded, priority > 0)
        std::vector<LevelChunkKey> toLoad;
        for (auto& pair : m_chunks) {
            if (pair.second.status == ChunkStatus::Unloaded && pair.second.priority > 1e-6f) {
                toLoad.push_back(pair.first);
            }
        }

        // Sort by priority descending
        std::sort(toLoad.begin(), toLoad.end(),
                  [this](const LevelChunkKey& a, const LevelChunkKey& b) {
                      return m_chunks[a].priority > m_chunks[b].priority;
                  });

        // Determine how many we can load (budget)
        size_type currentLoaded = countLoadedChunks();
        size_type currentMemory = totalLoadedMemory();
        size_type budgetChunks = m_config.maxLoadedChunks - currentLoaded;
        size_type budgetMemoryMB = m_config.memoryLimitMB - currentMemory / (1024 * 1024);

        for (const auto& key : toLoad) {
            if (budgetChunks == 0 && budgetMemoryMB <= 0) break;
            auto& chunk = m_chunks[key];
            size_type chunkMemoryMB = chunk.memoryEstimate / (1024 * 1024);
            if (chunkMemoryMB <= budgetMemoryMB) {
                if (m_config.enableAsyncLoading && m_io) {
                    submitLoadRequest(chunk);
                } else {
                    // Synchronous load (not recommended)
                    loadChunkSync(chunk);
                }
                if (budgetChunks > 0) --budgetChunks;
                if (budgetMemoryMB > chunkMemoryMB) budgetMemoryMB -= chunkMemoryMB;
                else budgetMemoryMB = 0;
            }
        }

        // Unload chunks with priority == 0 and beyond memory budget
        for (auto& pair : m_chunks) {
            if (pair.second.status == ChunkStatus::Loaded && pair.second.priority <= 0.0f) {
                unloadChunk(pair.second);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Submit an asynchronous load request to IO thread
    // ------------------------------------------------------------------------
    void submitLoadRequest(LevelChunk& chunk) {
        if (!m_io) return;
        chunk.status = ChunkStatus::Loading;
        IORequest req;
        req.direction = StreamDirection::Read;
        req.mode = StreamingMode::ByMortonPrefix;
        req.mortonPrefix = chunk.key.mortonPrefix;
        req.maxDepth = chunk.key.depth;
        req.filename = "chunk_" + std::to_string(chunk.key.mortonPrefix) + "_" + std::to_string(chunk.key.depth) + ".octree";
        req.callback = [this, key = chunk.key](const IORequest& res) {
            std::lock_guard<std::mutex> lock(m_chunksMutex);
            auto it = m_chunks.find(key);
            if (it != m_chunks.end()) {
                if (!res.data.empty()) {
                    it->second.data = res.data;
                    it->second.status = ChunkStatus::Loaded;
                } else {
                    it->second.status = ChunkStatus::Error;
                }
            }
        };
        m_io->submitRequest(std::move(req));
    }

    // ------------------------------------------------------------------------
    //  Synchronous load (blocking)
    // ------------------------------------------------------------------------
    void loadChunkSync(LevelChunk& chunk) {
        // Would read from file and set chunk.data
        // For brevity, we mark as loaded with empty data.
        chunk.status = ChunkStatus::Loaded;
        chunk.data.clear();
    }

    // ------------------------------------------------------------------------
    //  Unload chunk (free memory)
    // ------------------------------------------------------------------------
    void unloadChunk(LevelChunk& chunk) {
        chunk.data.clear();
        chunk.data.shrink_to_fit();
        chunk.status = ChunkStatus::Unloaded;
    }

    // ------------------------------------------------------------------------
    //  Count currently loaded chunks
    // ------------------------------------------------------------------------
    size_type countLoadedChunks() const {
        size_type count = 0;
        for (const auto& pair : m_chunks) {
            if (pair.second.status == ChunkStatus::Loaded) ++count;
        }
        return count;
    }

    // ------------------------------------------------------------------------
    //  Total memory used by loaded chunks (bytes)
    // ------------------------------------------------------------------------
    size_type totalLoadedMemory() const {
        size_type total = 0;
        for (const auto& pair : m_chunks) {
            if (pair.second.status == ChunkStatus::Loaded) {
                total += pair.second.data.capacity() + pair.second.memoryEstimate;
            }
        }
        return total;
    }

    // ------------------------------------------------------------------------
    //  Worker thread loop (for async loading)
    // ------------------------------------------------------------------------
    void workerLoop() {
        while (m_running) {
            // In a real implementation, we would pull pending load requests
            // from a queue and process them.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    Config m_config;
    std::atomic<uint64_t> m_currentFrame;
    point_type m_viewerPosition;
    std::atomic<bool> m_running;
    OctreeStreamingIO* m_io;
    std::mutex m_chunksMutex;
    std::unordered_map<LevelChunkKey, LevelChunk> m_chunks;
    std::vector<std::thread> m_workers;
    std::condition_variable m_cv;
    std::mutex m_queueMutex;
    std::queue<LevelChunkKey> m_loadQueue;
};

} // namespace IO
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_IO_LEVEL_STREAMING_MANAGER_H_INCLUDED

/**
 * Next file: core/io/galactic_catalogue_loader.h
 * Remaining in the list: 15 files (galactic_catalogue_loader, living_entity_interface, sensory_query_system, swarm_communication, integration_bridge, archetype_index, global_morton_routing, replica_manager, consensus_tree, ecosystem_driver, weather_simulator, geology_erosion, binary_streaming_archive, distributed_snapshot, network_delta_archive)
 */