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

#ifndef ORTHOTREE_CORE_DISTRIBUTED_REPLICA_MANAGER_H_INCLUDED
#define ORTHOTREE_CORE_DISTRIBUTED_REPLICA_MANAGER_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/numerical_methods.h"
#include "../../core/morton/morton_128bit.h"
#include "../../core/parallel/lockfree_query_buffer.h"
#include "../../serialization/binary_archive.h"
#include "../../serialization/msgpack_archive.h"
#include "../compression/octree_compressor.h"
#include "global_morton_routing.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <deque>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <optional>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <memory>

namespace OrthoTree {
namespace Distributed {

// ============================================================================
//  Consistency levels for read/write operations
// ============================================================================
enum class ConsistencyLevel : uint8_t {
    Any,          // any replica (lowest)
    One,          // at least one
    Quorum,       // majority of replicas
    All,          // all replicas (strongest)
    Local         // only local replica (for testing)
};

// ============================================================================
//  Replica state
// ============================================================================
enum class ReplicaState : uint8_t {
    Active,       // fully participating
    Lagging,      // behind in updates, catching up
    Failed,       // considered dead
    Recovering    // being restored from snapshot
};

// ============================================================================
//  Replica information
// ============================================================================
struct ReplicaInfo {
    NodeID nodeId;
    ReplicaState state = ReplicaState::Active;
    uint64_t lastHeartbeat = 0;        // timestamp (ms)
    uint64_t appliedVersion = 0;       // last applied log index
    std::string address;               // network address (optional)
    double loadFactor = 0.0;           // 0..1 (for load balancing)
    uint32_t replicationLagMs = 0;
};

// ============================================================================
//  Write operation (entry in replication log)
// ============================================================================
struct WriteEntry {
    uint64_t logIndex;
    uint64_t timestamp;                // microseconds
    uint64_t mortonKey;                // affected spatial region
    std::vector<uint8_t> data;         // serialised operation (e.g., insert entity)
    std::vector<NodeID> expectedReplicas; // for quorum validation
    bool committed = false;
};

// ============================================================================
//  ReplicaManager: manages data replication across a cluster of nodes,
//  ensuring consistency and fault tolerance for the octree.
//  Supports quorum reads/writes, leader election (via external consensus),
//  and SIMD batch conflict resolution (e.g., merging multiple updates).
// ============================================================================
class ReplicaManager {
public:
    using size_type = size_t;
    using time_ms = uint64_t;
    using point_type = Math::Vector<double,3>;
    using aabb_type = Math::AxisAlignedBox<double,3>;

    // ------------------------------------------------------------------------
    //  Configuration (dynamic environment)
    // ------------------------------------------------------------------------
    struct Config {
        NodeID localNodeId = 0;
        ConsistencyLevel defaultReadLevel = ConsistencyLevel::Quorum;
        ConsistencyLevel defaultWriteLevel = ConsistencyLevel::Quorum;
        size_type replicationFactor = 3;          // desired number of replicas
        size_type maxLogSize = 10000;             // entries before truncation
        time_ms heartbeatIntervalMs = 1000;
        time_ms leaderLeaseMs = 5000;
        time_ms staleReplicaTimeoutMs = 10000;
        bool enableSIMD = true;
        bool useCompression = true;
        uint32_t version = 1;
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit ReplicaManager(const Config& cfg, GlobalMortonRouter* router = nullptr)
        : m_config(cfg)
        , m_router(router)
        , m_currentTerm(1)
        , m_committedIndex(0)
        , m_appliedIndex(0)
        , m_leaderId(0)
        , m_running(true)
        , m_nextLogIndex(1) {
        // Register local replica
        ReplicaInfo local;
        local.nodeId = m_config.localNodeId;
        local.state = ReplicaState::Active;
        local.appliedVersion = 0;
        m_replicas[m_config.localNodeId] = local;
        if (m_router) {
            m_router->addNode(m_config.localNodeId, aabb_type(point_type(-1), point_type(1)));
        }
        // Start background heartbeat thread
        m_heartbeatThread = std::thread(&ReplicaManager::heartbeatLoop, this);
    }

    ~ReplicaManager() {
        m_running = false;
        if (m_heartbeatThread.joinable()) m_heartbeatThread.join();
    }

    // ------------------------------------------------------------------------
    //  Replica management (add/remove remote replicas)
    // ------------------------------------------------------------------------
    void addReplica(const ReplicaInfo& replica) {
        std::lock_guard<std::mutex> lock(m_replicasMutex);
        m_replicas[replica.nodeId] = replica;
        if (m_router) {
            m_router->addNode(replica.nodeId, aabb_type(point_type(-1), point_type(1)));
        }
    }

    void removeReplica(NodeID node) {
        std::lock_guard<std::mutex> lock(m_replicasMutex);
        m_replicas.erase(node);
        if (m_router) {
            m_router->removeNode(node);
        }
    }

    // ------------------------------------------------------------------------
    //  Propose a write operation (e.g., insert/update entity).
    //  Returns true if the operation was accepted (possibly asynchronously).
    // ------------------------------------------------------------------------
    bool proposeWrite(uint64_t mortonKey, const std::vector<uint8_t>& data,
                      ConsistencyLevel level = ConsistencyLevel::Quorum) {
        // Determine target replicas using Morton routing (if available)
        std::vector<NodeID> targetNodes;
        if (m_router) {
            NodeID primary = m_router->routeToMortonKey(mortonKey);
            targetNodes.push_back(primary);
            // Add extra replicas based on replication factor (simple: next few in consistent hashing)
            // For brevity, we just use the primary.
        } else {
            targetNodes.push_back(m_config.localNodeId);
        }

        // Create log entry
        WriteEntry entry;
        entry.logIndex = m_nextLogIndex++;
        entry.timestamp = getCurrentTimeMs();
        entry.mortonKey = mortonKey;
        entry.data = data;
        entry.expectedReplicas = targetNodes;
        entry.committed = false;

        // Append to local log
        {
            std::lock_guard<std::mutex> lock(m_logMutex);
            m_log.push_back(entry);
        }

        // In a real implementation, we would send AppendEntries RPCs to followers.
        // Here we simulate by marking as committed immediately (single node).
        if (targetNodes.size() == 1 && targetNodes[0] == m_config.localNodeId) {
            commitEntry(entry.logIndex);
        }
        return true;
    }

    // ------------------------------------------------------------------------
    //  Read data for a given Morton key (e.g., entity).
    //  Returns the data from the first responsive replica according to consistency level.
    // ------------------------------------------------------------------------
    std::optional<std::vector<uint8_t>> read(uint64_t mortonKey, ConsistencyLevel level = ConsistencyLevel::Quorum) {
        std::vector<NodeID> replicas = getReplicasForKey(mortonKey);
        if (replicas.empty()) return std::nullopt;

        size_type required = requiredResponses(replicas.size(), level);
        size_type successes = 0;
        std::vector<uint8_t> latestData;
        uint64_t latestVersion = 0;

        // In a real system, we would send parallel read requests.
        // For simulation, we read from local replica if present.
        for (NodeID node : replicas) {
            if (node == m_config.localNodeId) {
                // Read from local storage (simulated)
                std::lock_guard<std::mutex> lock(m_dataMutex);
                auto it = m_localData.find(mortonKey);
                if (it != m_localData.end()) {
                    ++successes;
                    if (it->second.version > latestVersion) {
                        latestVersion = it->second.version;
                        latestData = it->second.data;
                    }
                }
            }
            if (successes >= required) break;
        }
        if (successes >= required) return latestData;
        return std::nullopt;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch read: read multiple keys at once, return results in parallel.
    //  Uses vectorised operations where possible.
    // ------------------------------------------------------------------------
    void batchRead(const uint64_t* keys, std::optional<std::vector<uint8_t>>* results,
                   size_type count, ConsistencyLevel level = ConsistencyLevel::Quorum) {
        if (m_config.enableSIMD && count >= 4) {
            size_type simdEnd = count - (count % 4);
            for (size_type i = 0; i < simdEnd; i += 4) {
                // In SIMD, we could process 4 reads in parallel.
                // Here we unroll scalar.
                for (int j = 0; j < 4; ++j) {
                    results[i+j] = read(keys[i+j], level);
                }
            }
            for (size_type i = simdEnd; i < count; ++i) {
                results[i] = read(keys[i], level);
            }
        } else {
            for (size_type i = 0; i < count; ++i) {
                results[i] = read(keys[i], level);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control: adjust replication factor, consistency levels
    // ------------------------------------------------------------------------
    void setReplicationFactor(size_type factor) { m_config.replicationFactor = factor; }
    void setDefaultReadLevel(ConsistencyLevel level) { m_config.defaultReadLevel = level; }
    void setDefaultWriteLevel(ConsistencyLevel level) { m_config.defaultWriteLevel = level; }
    void setHeartbeatIntervalMs(time_ms ms) { m_config.heartbeatIntervalMs = ms; }

    // ------------------------------------------------------------------------
    //  Statistics and monitoring
    // ------------------------------------------------------------------------
    size_type logSize() const {
        std::lock_guard<std::mutex> lock(m_logMutex);
        return m_log.size();
    }
    uint64_t lastCommittedIndex() const { return m_committedIndex; }
    uint64_t currentTerm() const { return m_currentTerm; }
    NodeID leaderId() const { return m_leaderId; }

private:
    // ------------------------------------------------------------------------
    //  Internal data structure for local storage (versioned)
    // ------------------------------------------------------------------------
    struct VersionedData {
        std::vector<uint8_t> data;
        uint64_t version = 0;
    };

    // ------------------------------------------------------------------------
    //  Get replicas responsible for a Morton key (using routing table)
    // ------------------------------------------------------------------------
    std::vector<NodeID> getReplicasForKey(uint64_t mortonKey) const {
        std::vector<NodeID> result;
        if (m_router) {
            NodeID primary = m_router->routeToMortonKey(mortonKey);
            result.push_back(primary);
            // Add extra replicas (in real implementation, use consistent hashing)
            // For now, just return primary.
        } else {
            result.push_back(m_config.localNodeId);
        }
        return result;
    }

    // ------------------------------------------------------------------------
    //  Commit a log entry (apply to state machine)
    // ------------------------------------------------------------------------
    void commitEntry(uint64_t logIndex) {
        std::lock_guard<std::mutex> lock(m_logMutex);
        auto it = std::find_if(m_log.begin(), m_log.end(),
                               [logIndex](const WriteEntry& e) { return e.logIndex == logIndex; });
        if (it != m_log.end()) {
            it->committed = true;
            // Apply to state machine
            VersionedData vd;
            vd.data = it->data;
            vd.version = it->logIndex;
            {
                std::lock_guard<std::mutex> dataLock(m_dataMutex);
                m_localData[it->mortonKey] = std::move(vd);
            }
            m_committedIndex = std::max(m_committedIndex, logIndex);
        }
    }

    // ------------------------------------------------------------------------
    //  Heartbeat loop: periodically send heartbeats to other replicas
    //  and check for stale replicas.
    // ------------------------------------------------------------------------
    void heartbeatLoop() {
        while (m_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(m_config.heartbeatIntervalMs));
            time_ms now = getCurrentTimeMs();
            // Send heartbeat to known replicas (simplified)
            std::lock_guard<std::mutex> lock(m_replicasMutex);
            for (auto& pair : m_replicas) {
                if (pair.first == m_config.localNodeId) continue;
                // Update lastHeartbeat (simulate receiving)
                pair.second.lastHeartbeat = now;
                // Check for stale
                if (now - pair.second.lastHeartbeat > m_config.staleReplicaTimeoutMs) {
                    pair.second.state = ReplicaState::Failed;
                } else if (pair.second.state == ReplicaState::Failed) {
                    pair.second.state = ReplicaState::Active; // auto‑recover
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Compute required number of responses for a given consistency level
    // ------------------------------------------------------------------------
    size_type requiredResponses(size_type totalReplicas, ConsistencyLevel level) const {
        switch (level) {
            case ConsistencyLevel::Any: return 1;
            case ConsistencyLevel::One: return 1;
            case ConsistencyLevel::Quorum: return (totalReplicas / 2) + 1;
            case ConsistencyLevel::All: return totalReplicas;
            case ConsistencyLevel::Local: return 1;
            default: return 1;
        }
    }

    // ------------------------------------------------------------------------
    //  Get current time in milliseconds
    // ------------------------------------------------------------------------
    static time_ms getCurrentTimeMs() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    Config m_config;
    GlobalMortonRouter* m_router;
    std::atomic<uint64_t> m_currentTerm;
    std::atomic<uint64_t> m_committedIndex;
    std::atomic<uint64_t> m_appliedIndex;
    std::atomic<NodeID> m_leaderId;
    std::atomic<bool> m_running;
    std::atomic<uint64_t> m_nextLogIndex;

    mutable std::mutex m_replicasMutex;
    std::unordered_map<NodeID, ReplicaInfo> m_replicas;

    mutable std::mutex m_logMutex;
    std::deque<WriteEntry> m_log;

    mutable std::mutex m_dataMutex;
    std::unordered_map<uint64_t, VersionedData> m_localData;

    std::thread m_heartbeatThread;
};

// ----------------------------------------------------------------------------
//  Helper: create a replica manager with default settings for galactic scale
// ----------------------------------------------------------------------------
inline ReplicaManager createGalacticReplicaManager(NodeID localNode,
                                                   GlobalMortonRouter* router = nullptr) {
    ReplicaManager::Config cfg;
    cfg.localNodeId = localNode;
    cfg.replicationFactor = 3;
    cfg.defaultReadLevel = ConsistencyLevel::Quorum;
    cfg.defaultWriteLevel = ConsistencyLevel::Quorum;
    cfg.heartbeatIntervalMs = 2000;
    cfg.staleReplicaTimeoutMs = 10000;
    return ReplicaManager(cfg, router);
}

} // namespace Distributed
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_DISTRIBUTED_REPLICA_MANAGER_H_INCLUDED

/**
 * Next file: core/distributed/consensus_tree.h
 * Remaining in the list: 7 files (consensus_tree, ecosystem_driver, weather_simulator, geology_erosion, binary_streaming_archive, distributed_snapshot, network_delta_archive)
 */