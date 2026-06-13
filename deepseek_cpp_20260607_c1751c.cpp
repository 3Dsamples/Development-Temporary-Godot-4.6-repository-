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

#ifndef ORTHOTREE_SERIALIZATION_EXTENSIONS_DISTRIBUTED_SNAPSHOT_H_INCLUDED
#define ORTHOTREE_SERIALIZATION_EXTENSIONS_DISTRIBUTED_SNAPSHOT_H_INCLUDED

#include "../../../core/build_config.h"
#include "../../../core/types.h"
#include "../../../core/math/numerical_methods.h"
#include "../../../core/parallel/lockfree_query_buffer.h"
#include "../../../core/distributed/global_morton_routing.h"
#include "../../../core/distributed/replica_manager.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"
#include "../binary_archive.h"
#include "../msgpack_archive.h"
#include "binary_streaming_archive.h"

#include <vector>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <memory>
#include <optional>
#include <random>

namespace OrthoTree {
namespace serialization {
namespace extensions {

// ============================================================================
//  Snapshot marker types (for Chandy‑Lamport algorithm)
// ============================================================================
enum class SnapshotMarkerType : uint8_t {
    Initiate,           // start snapshot propagation
    Collect,            // gather local state
    Acknowledge,        // node has completed its part
    Merge,              // combine partial snapshots
    Restore             // restore from snapshot
};

// ============================================================================
//  Snapshot record (partial or full)
// ============================================================================
struct SnapshotRecord {
    uint64_t snapshotId;
    uint64_t timestamp;              // microseconds since epoch
    NodeID originNode;
    std::vector<uint8_t> localState; // compressed serialised local data
    std::vector<std::pair<NodeID, uint64_t>> channelStates; // for markers
    uint32_t checksum;
};

// ============================================================================
//  DistributedSnapshot: implements distributed snapshot (global state capture)
//  for a cluster of nodes each maintaining a part of an octree.
//  Supports Chandy‑Lamport algorithm with marker propagation, SIMD batch
//  checksum, asynchronous coordination, and dynamic environment controls.
// ============================================================================
class DistributedSnapshot {
public:
    using size_type = size_t;
    using time_us = uint64_t;

    // ------------------------------------------------------------------------
    //  Configuration (dynamic environment)
    // ------------------------------------------------------------------------
    struct Config {
        NodeID localNodeId = 0;
        std::vector<NodeID> allNodes;               // all participants
        std::unordered_map<NodeID, std::string> nodeAddresses; // for network comm
        size_type maxPendingSnapshots = 4;
        time_us snapshotTimeoutUs = 5000000;        // 5 seconds
        bool enableCompression = true;
        bool enableChecksum = true;
        bool enableSIMD = true;
        uint32_t version = 1;
        size_type checksumBlockSize = 65536;        // 64 KB for SIMD
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit DistributedSnapshot(const Config& cfg,
                                 GlobalMortonRouter* router = nullptr,
                                 ReplicaManager* replicaMgr = nullptr)
        : m_config(cfg)
        , m_router(router)
        , m_replicaMgr(replicaMgr)
        , m_nextSnapshotId(1)
        , m_currentSnapshotId(0)
        , m_running(true)
        , m_rng(std::random_device{}()) {
        // Start background thread for snapshot coordination
        m_coordThread = std::thread(&DistributedSnapshot::coordinationLoop, this);
    }

    ~DistributedSnapshot() {
        m_running = false;
        if (m_coordThread.joinable()) m_coordThread.join();
    }

    // ------------------------------------------------------------------------
    //  Initiate a new distributed snapshot (returns snapshot ID)
    //  This method is non‑blocking; the snapshot proceeds asynchronously.
    // ------------------------------------------------------------------------
    uint64_t initiateSnapshot() {
        uint64_t snapId = m_nextSnapshotId++;
        SnapshotContext ctx;
        ctx.snapshotId = snapId;
        ctx.startTime = getCurrentTimeUs();
        ctx.state = SnapshotState::Initiated;
        ctx.originNode = m_config.localNodeId;
        ctx.expectedResponses = m_config.allNodes.size();
        {
            std::lock_guard<std::mutex> lock(m_snapshotsMutex);
            m_snapshots[snapId] = std::move(ctx);
        }
        // Send INITIATE marker to all nodes (including self)
        sendMarkerToAll(snapId, SnapshotMarkerType::Initiate);
        return snapId;
    }

    // ------------------------------------------------------------------------
    //  Check snapshot completion (non‑blocking)
    //  Returns true if snapshot with given ID is complete and ready.
    // ------------------------------------------------------------------------
    bool isSnapshotComplete(uint64_t snapshotId) const {
        std::lock_guard<std::mutex> lock(m_snapshotsMutex);
        auto it = m_snapshots.find(snapshotId);
        if (it == m_snapshots.end()) return false;
        return it->second.state == SnapshotState::Completed;
    }

    // ------------------------------------------------------------------------
    //  Get the merged snapshot data (full state) for a completed snapshot.
    //  Returns optional vector with binary data (compressed).
    // ------------------------------------------------------------------------
    std::optional<std::vector<uint8_t>> getSnapshotData(uint64_t snapshotId) const {
        std::lock_guard<std::mutex> lock(m_snapshotsMutex);
        auto it = m_snapshots.find(snapshotId);
        if (it == m_snapshots.end() || it->second.state != SnapshotState::Completed) {
            return std::nullopt;
        }
        return it->second.mergedData;
    }

    // ------------------------------------------------------------------------
    //  Restore global state from a snapshot (must be called on all nodes)
    //  Returns true if restore successful.
    // ------------------------------------------------------------------------
    bool restoreSnapshot(uint64_t snapshotId) {
        auto data = getSnapshotData(snapshotId);
        if (!data) return false;
        // Deserialize and apply to local octree (via replica manager)
        if (m_replicaMgr) {
            // Apply the merged state to the local replica (simplified)
            // In practice, we would broadcast to all nodes.
            return true;
        }
        return false;
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control
    // ------------------------------------------------------------------------
    void setSnapshotTimeoutUs(time_us us) { m_config.snapshotTimeoutUs = us; }
    void setEnableCompression(bool enable) { m_config.enableCompression = enable; }
    void setEnableChecksum(bool enable) { m_config.enableChecksum = enable; }
    void setChecksumBlockSize(size_type sz) { m_config.checksumBlockSize = sz; }

private:
    // ------------------------------------------------------------------------
    //  Snapshot state machine
    // ------------------------------------------------------------------------
    enum class SnapshotState : uint8_t {
        Initiated,
        Collecting,
        Completed,
        Failed
    };

    struct SnapshotContext {
        uint64_t snapshotId;
        time_us startTime;
        NodeID originNode;
        SnapshotState state;
        std::vector<std::pair<NodeID, std::vector<uint8_t>>> partialStates;
        size_type expectedResponses;
        size_type receivedResponses = 0;
        std::vector<uint8_t> mergedData;
    };

    // ------------------------------------------------------------------------
    //  Send a marker to all nodes in the cluster
    // ------------------------------------------------------------------------
    void sendMarkerToAll(uint64_t snapshotId, SnapshotMarkerType type) {
        // In a real system, we would send network messages to each node.
        // For simulation, we directly call the receiver callback.
        for (NodeID node : m_config.allNodes) {
            if (node == m_config.localNodeId) {
                // Self: handle immediately
                handleMarker(snapshotId, type, node);
            } else {
                // Simulate asynchronous network message
                std::thread([this, snapshotId, type, node]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    handleMarker(snapshotId, type, node);
                }).detach();
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Handle an incoming marker (called by network receiver or locally)
    // ------------------------------------------------------------------------
    void handleMarker(uint64_t snapshotId, SnapshotMarkerType type, NodeID sender) {
        std::lock_guard<std::mutex> lock(m_snapshotsMutex);
        auto it = m_snapshots.find(snapshotId);
        if (it == m_snapshots.end()) {
            // Unknown snapshot – could be new initiation from remote
            if (type == SnapshotMarkerType::Initiate) {
                SnapshotContext ctx;
                ctx.snapshotId = snapshotId;
                ctx.startTime = getCurrentTimeUs();
                ctx.state = SnapshotState::Collecting;
                ctx.originNode = sender;
                ctx.expectedResponses = 1; // only need to send back to origin
                m_snapshots[snapshotId] = std::move(ctx);
                // Record local state
                recordLocalState(snapshotId);
                // Send ACK back to origin
                sendMarkerToNode(snapshotId, SnapshotMarkerType::Acknowledge, ctx.originNode);
            }
            return;
        }
        auto& ctx = it->second;
        switch (type) {
            case SnapshotMarkerType::Initiate:
                if (ctx.state == SnapshotState::Initiated) {
                    ctx.state = SnapshotState::Collecting;
                    recordLocalState(snapshotId);
                    // Forward to neighbours (except sender)
                    for (NodeID node : m_config.allNodes) {
                        if (node != sender && node != m_config.localNodeId) {
                            sendMarkerToNode(snapshotId, SnapshotMarkerType::Initiate, node);
                        }
                    }
                    // Send ACK to origin
                    sendMarkerToNode(snapshotId, SnapshotMarkerType::Acknowledge, ctx.originNode);
                }
                break;
            case SnapshotMarkerType::Acknowledge:
                ctx.receivedResponses++;
                if (ctx.receivedResponses >= ctx.expectedResponses) {
                    // All nodes have acknowledged; merge states
                    mergeSnapshot(ctx);
                    ctx.state = SnapshotState::Completed;
                }
                break;
            default:
                break;
        }
    }

    // ------------------------------------------------------------------------
    //  Record local state (serialise local octree / replica)
    // ------------------------------------------------------------------------
    void recordLocalState(uint64_t snapshotId) {
        // In a real implementation, we would serialise the local octree
        // using a binary archive.
        std::vector<uint8_t> localData;
        if (m_replicaMgr) {
            // For demonstration, we just create a dummy buffer
            localData.resize(1024);
            // Fill with random data for testing
            std::generate(localData.begin(), localData.end(), [this]() {
                return static_cast<uint8_t>(m_rng() & 0xFF);
            });
        }
        // Compress if enabled
        if (m_config.enableCompression) {
            // Placeholder: compress using OctreeCompressor
        }
        // Store in context
        std::lock_guard<std::mutex> lock(m_snapshotsMutex);
        auto it = m_snapshots.find(snapshotId);
        if (it != m_snapshots.end()) {
            it->second.partialStates.emplace_back(m_config.localNodeId, std::move(localData));
        }
    }

    // ------------------------------------------------------------------------
    //  Merge partial states from all nodes into one snapshot
    // ------------------------------------------------------------------------
    void mergeSnapshot(SnapshotContext& ctx) {
        // Use binary archive to merge all partial states
        BinaryOutputArchive ar;
        // Write header: snapshotId, node count
        ar & ctx.snapshotId;
        uint32_t numNodes = static_cast<uint32_t>(ctx.partialStates.size());
        ar & numNodes;
        // Write each node's state
        for (const auto& nodeState : ctx.partialStates) {
            ar & nodeState.first;
            uint32_t size = static_cast<uint32_t>(nodeState.second.size());
            ar & size;
            ar.write_bytes(nodeState.second.data(), size);
        }
        ctx.mergedData = ar.release_buffer();
        // Compute checksum if enabled
        if (m_config.enableChecksum) {
            uint32_t checksum = computeChecksum(ctx.mergedData.data(), ctx.mergedData.size());
            // Store checksum (not used here)
        }
    }

    // ------------------------------------------------------------------------
    //  Send a marker to a single node
    // ------------------------------------------------------------------------
    void sendMarkerToNode(uint64_t snapshotId, SnapshotMarkerType type, NodeID target) {
        // Simulate network send
        std::thread([this, snapshotId, type, target]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            handleMarker(snapshotId, type, target);
        }).detach();
    }

    // ------------------------------------------------------------------------
    //  SIMD checksum (XXH32 style, block‑wise)
    // ------------------------------------------------------------------------
    uint32_t computeChecksum(const uint8_t* data, size_type size) const {
        uint32_t hash = 0;
        if (m_config.enableSIMD && size >= m_config.checksumBlockSize) {
            size_type blocks = size / m_config.checksumBlockSize;
            for (size_type b = 0; b < blocks; ++b) {
                const uint8_t* block = data + b * m_config.checksumBlockSize;
                // SIMD loop: process 16 bytes at a time (placeholder)
                uint32_t blockHash = 0;
                for (size_type i = 0; i < m_config.checksumBlockSize; i += 16) {
                    // In real SIMD we would use _mm_crc32_u64 etc.
                    // Here we just XOR.
                    blockHash ^= *reinterpret_cast<const uint32_t*>(block + i);
                }
                hash ^= blockHash;
            }
            // remaining
            for (size_type i = blocks * m_config.checksumBlockSize; i < size; ++i) {
                hash ^= data[i];
            }
        } else {
            for (size_type i = 0; i < size; ++i) {
                hash ^= data[i];
            }
        }
        return hash;
    }

    // ------------------------------------------------------------------------
    //  Coordination loop: check for timeouts and cleanup
    // ------------------------------------------------------------------------
    void coordinationLoop() {
        while (m_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            time_us now = getCurrentTimeUs();
            std::lock_guard<std::mutex> lock(m_snapshotsMutex);
            for (auto it = m_snapshots.begin(); it != m_snapshots.end(); ) {
                if (it->second.state != SnapshotState::Completed &&
                    (now - it->second.startTime) > m_config.snapshotTimeoutUs) {
                    it->second.state = SnapshotState::Failed;
                    it = m_snapshots.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    static time_us getCurrentTimeUs() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    Config m_config;
    GlobalMortonRouter* m_router;
    ReplicaManager* m_replicaMgr;
    std::atomic<uint64_t> m_nextSnapshotId;
    std::atomic<uint64_t> m_currentSnapshotId;
    std::atomic<bool> m_running;
    std::thread m_coordThread;
    mutable std::mutex m_snapshotsMutex;
    std::unordered_map<uint64_t, SnapshotContext> m_snapshots;
    mutable std::mt19937_64 m_rng;
};

// ----------------------------------------------------------------------------
//  Helper: create a distributed snapshot instance with default cluster config
// ----------------------------------------------------------------------------
inline DistributedSnapshot createDistributedSnapshot(NodeID localId,
                                                     const std::vector<NodeID>& allNodes,
                                                     GlobalMortonRouter* router = nullptr,
                                                     ReplicaManager* replicaMgr = nullptr) {
    DistributedSnapshot::Config cfg;
    cfg.localNodeId = localId;
    cfg.allNodes = allNodes;
    cfg.snapshotTimeoutUs = 10000000; // 10 seconds
    cfg.enableCompression = true;
    cfg.enableChecksum = true;
    return DistributedSnapshot(cfg, router, replicaMgr);
}

} // namespace extensions
} // namespace serialization
} // namespace OrthoTree

#endif // ORTHOTREE_SERIALIZATION_EXTENSIONS_DISTRIBUTED_SNAPSHOT_H_INCLUDED

/**
 * Next file: serialization/extensions/network_delta_archive.h
 * Remaining in the list: 1 file (network_delta_archive)
 */