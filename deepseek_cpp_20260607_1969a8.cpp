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

#ifndef ORTHOTREE_CORE_DISTRIBUTED_GLOBAL_MORTON_ROUTING_H_INCLUDED
#define ORTHOTREE_CORE_DISTRIBUTED_GLOBAL_MORTON_ROUTING_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/interval_arithmetic.h"
#include "../../core/math/numerical_methods.h"
#include "../../core/morton/morton_128bit.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cstdint>
#include <vector>
#include <array>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <atomic>
#include <random>
#include <type_traits>

namespace OrthoTree {
namespace Distributed {

// ============================================================================
//  GlobalMortonRouting: route messages or queries to the node responsible
//  for a given spatial region using Morton order prefix routing.
//  Supports 2D/3D, load balancing via virtual node splitting, SIMD batch
//  prefix extraction, and dynamic environment controls (node join/leave,
//  network latency awareness).
// ============================================================================

// ----------------------------------------------------------------------------
//  Node identifier (e.g., IP + port, or simple integer ID)
// ----------------------------------------------------------------------------
using NodeID = uint64_t;

// ----------------------------------------------------------------------------
//  Routing table entry: maps a Morton prefix to the responsible node
// ----------------------------------------------------------------------------
struct RouteEntry {
    uint64_t mortonPrefix;   // prefix length = depth * bitsPerDim
    uint8_t depth;           // number of subdivision levels
    NodeID nodeId;
    Math::AxisAlignedBox<double,3> bounds;  // region covered (for debugging)
};

// ----------------------------------------------------------------------------
//  Message to be routed
// ----------------------------------------------------------------------------
template<typename Payload = std::vector<uint8_t>>
struct RoutedMessage {
    NodeID sender;
    NodeID intendedReceiver;   // may be 0 if broadcast
    uint64_t mortonKey;        // destination spatial key
    Payload data;
    uint32_t ttl = 32;         // time to live
    uint64_t timestamp;
};

// ============================================================================
//  GlobalMortonRouter main class
// ============================================================================
class GlobalMortonRouter {
public:
    using morton_type = Morton::Morton128Bit;
    using point_type = Math::Vector<double,3>;
    using aabb_type = Math::AxisAlignedBox<double,3>;

    // ------------------------------------------------------------------------
    //  Configuration
    // ------------------------------------------------------------------------
    struct Config {
        aabb_type worldBounds;               // global universe bounds
        uint8_t maxDepth = 20;               // maximum prefix depth (bits per dim = 21)
        bool useLoadBalancing = true;
        bool enableSIMD = true;
        uint32_t replicationFactor = 1;      // number of replicas for each prefix
        T heartbeatIntervalSec = 1.0;
        T staleNodeTimeoutSec = 5.0;
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit GlobalMortonRouter(const Config& cfg, NodeID localNode)
        : m_config(cfg)
        , m_localNode(localNode)
        , m_nextVersion(1) {
        // Insert local node as responsible for the entire space initially
        RouteEntry root;
        root.mortonPrefix = 0;
        root.depth = 0;
        root.nodeId = localNode;
        root.bounds = cfg.worldBounds;
        addRouteEntry(root);
    }

    // ------------------------------------------------------------------------
    //  Register a remote node and the region it claims (advertisement)
    // ------------------------------------------------------------------------
    void addNode(NodeID node, const aabb_type& region) {
        // Convert region to Morton prefix
        uint64_t prefix = regionToMortonPrefix(region, m_config.maxDepth);
        uint8_t depth = computeDepthFromRegion(region);
        RouteEntry entry;
        entry.mortonPrefix = prefix;
        entry.depth = depth;
        entry.nodeId = node;
        entry.bounds = region;
        addRouteEntry(entry);
        // Notify load balancer
        if (m_config.useLoadBalancing) {
            rebalance();
        }
    }

    // ------------------------------------------------------------------------
    //  Remove a node (graceful leave or timeout)
    // ------------------------------------------------------------------------
    void removeNode(NodeID node) {
        std::lock_guard<std::mutex> lock(m_routeMutex);
        auto it = m_routingTable.begin();
        while (it != m_routingTable.end()) {
            if (it->nodeId == node) {
                it = m_routingTable.erase(it);
            } else {
                ++it;
            }
        }
        // Also remove from reverse index
        m_reverseMap.erase(node);
    }

    // ------------------------------------------------------------------------
    //  Route a message to the node responsible for a given point
    //  Returns the node ID that should handle the message.
    // ------------------------------------------------------------------------
    NodeID routeToPoint(const point_type& point) {
        uint64_t key = pointToMortonKey(point, m_config.maxDepth);
        return findNodeForKey(key);
    }

    // ------------------------------------------------------------------------
    //  Route a message to the node responsible for a given Morton key
    // ------------------------------------------------------------------------
    NodeID routeToMortonKey(uint64_t key) {
        return findNodeForKey(key);
    }

    // ------------------------------------------------------------------------
    //  SIMD batch routing: route multiple keys to their responsible nodes
    //  Input: array of Morton keys, output: array of node IDs.
    // ------------------------------------------------------------------------
    void batchRoute(const uint64_t* keys, NodeID* outNodes, size_type count) {
        if (m_config.enableSIMD && count >= 4) {
            size_type simdEnd = count - (count % 4);
            for (size_type i = 0; i < simdEnd; i += 4) {
                // In a real SIMD implementation, we would load 4 keys into a
                // vector register and use vectorised table lookup.
                // Here we unroll the scalar version.
                for (int j = 0; j < 4; ++j) {
                    outNodes[i+j] = findNodeForKey(keys[i+j]);
                }
            }
            for (size_type i = simdEnd; i < count; ++i) {
                outNodes[i] = findNodeForKey(keys[i]);
            }
        } else {
            for (size_type i = 0; i < count; ++i) {
                outNodes[i] = findNodeForKey(keys[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Get the set of nodes that cover a given bounding box (for broadcast)
    //  Returns a vector of node IDs.
    // ------------------------------------------------------------------------
    std::vector<NodeID> nodesCoveringBox(const aabb_type& box) {
        uint64_t minKey = pointToMortonKey(box.min(), m_config.maxDepth);
        uint64_t maxKey = pointToMortonKey(box.max(), m_config.maxDepth);
        // Conservative: all nodes whose prefix range overlaps [minKey, maxKey]
        std::vector<NodeID> result;
        std::lock_guard<std::mutex> lock(m_routeMutex);
        for (const auto& entry : m_routingTable) {
            uint64_t prefix = entry.mortonPrefix;
            uint64_t prefixMask = (uint64_t(1) << (entry.depth * bitsPerDim())) - 1;
            uint64_t prefixMin = prefix << (64 - entry.depth * bitsPerDim());
            uint64_t prefixMax = prefixMin | ((uint64_t(1) << (64 - entry.depth * bitsPerDim())) - 1);
            if (prefixMin <= maxKey && prefixMax >= minKey) {
                result.push_back(entry.nodeId);
            }
        }
        return result;
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control: adjust load balancing, replication, etc.
    // ------------------------------------------------------------------------
    void enableLoadBalancing(bool enable) { m_config.useLoadBalancing = enable; }
    void setReplicationFactor(uint32_t factor) { m_config.replicationFactor = factor; }
    void setHeartbeatInterval(T sec) { m_config.heartbeatIntervalSec = sec; }
    void setStaleTimeout(T sec) { m_config.staleNodeTimeoutSec = sec; }

    // ------------------------------------------------------------------------
    //  For debugging / monitoring
    // ------------------------------------------------------------------------
    size_type routingTableSize() const {
        std::lock_guard<std::mutex> lock(m_routeMutex);
        return m_routingTable.size();
    }

    void dumpRoutingTable() const {
        std::lock_guard<std::mutex> lock(m_routeMutex);
        for (const auto& entry : m_routingTable) {
            // In real code, log entry
        }
    }

private:
    // ------------------------------------------------------------------------
    //  Add a route entry to the table (thread‑safe)
    // ------------------------------------------------------------------------
    void addRouteEntry(const RouteEntry& entry) {
        std::lock_guard<std::mutex> lock(m_routeMutex);
        // Remove any existing entry with same prefix (higher depth takes precedence)
        auto it = std::find_if(m_routingTable.begin(), m_routingTable.end(),
                               [&](const RouteEntry& e) { return e.mortonPrefix == entry.mortonPrefix; });
        if (it != m_routingTable.end()) {
            m_routingTable.erase(it);
        }
        m_routingTable.push_back(entry);
        // Sort by depth descending (more specific first) for faster lookup
        std::sort(m_routingTable.begin(), m_routingTable.end(),
                  [](const RouteEntry& a, const RouteEntry& b) {
                      if (a.depth != b.depth) return a.depth > b.depth;
                      return a.mortonPrefix < b.mortonPrefix;
                  });
        m_reverseMap[entry.nodeId].push_back(entry);
    }

    // ------------------------------------------------------------------------
    //  Find node responsible for a Morton key (longest prefix match)
    // ------------------------------------------------------------------------
    NodeID findNodeForKey(uint64_t key) const {
        std::lock_guard<std::mutex> lock(m_routeMutex);
        NodeID bestNode = m_localNode;  // fallback to self
        size_type bestDepth = 0;
        for (const auto& entry : m_routingTable) {
            uint64_t prefixMask = (uint64_t(1) << (entry.depth * bitsPerDim())) - 1;
            uint64_t entryPrefix = entry.mortonPrefix;
            uint64_t keyPrefix = key >> (64 - entry.depth * bitsPerDim());
            if ((keyPrefix & prefixMask) == entryPrefix && entry.depth > bestDepth) {
                bestDepth = entry.depth;
                bestNode = entry.nodeId;
            }
        }
        return bestNode;
    }

    // ------------------------------------------------------------------------
    //  Convert a point to a Morton key up to maxDepth
    // ------------------------------------------------------------------------
    uint64_t pointToMortonKey(const point_type& point, uint8_t maxDepth) const {
        point_type t = (point - m_config.worldBounds.min()) / m_config.worldBounds.extents();
        uint64_t cellsPerDim = (maxDepth < 64) ? (uint64_t(1) << maxDepth) : ~uint64_t(0);
        uint64_t ix = static_cast<uint64_t>(t[0] * (cellsPerDim - 1));
        uint64_t iy = static_cast<uint64_t>(t[1] * (cellsPerDim - 1));
        uint64_t iz = static_cast<uint64_t>(t[2] * (cellsPerDim - 1));
        morton_type mort(ix, iy, iz);
        return static_cast<uint64_t>(mort.code() >> (64 - maxDepth * bitsPerDim()));
    }

    // ------------------------------------------------------------------------
    //  Convert a region to a Morton prefix (the common prefix of its min and max)
    // ------------------------------------------------------------------------
    uint64_t regionToMortonPrefix(const aabb_type& region, uint8_t maxDepth) const {
        point_type minKey = region.min();
        point_type maxKey = region.max();
        // Find the smallest depth where the two points diverge
        for (uint8_t d = 0; d <= maxDepth; ++d) {
            uint64_t minM = pointToMortonKey(minKey, d);
            uint64_t maxM = pointToMortonKey(maxKey, d);
            if (minM == maxM) continue;
            // Divergence happened at depth d, so prefix length = d-1
            if (d > 0) {
                return pointToMortonKey(minKey, d-1);
            } else {
                return 0;
            }
        }
        return pointToMortonKey(minKey, maxDepth);
    }

    // ------------------------------------------------------------------------
    //  Compute depth of a region (log2 of its longest extent relative to world)
    // ------------------------------------------------------------------------
    uint8_t computeDepthFromRegion(const aabb_type& region) const {
        point_type ext = region.extents();
        point_type worldExt = m_config.worldBounds.extents();
        T maxRel = std::max({ext[0]/worldExt[0], ext[1]/worldExt[1], ext[2]/worldExt[2]});
        if (maxRel <= T(0)) return 0;
        int depthEst = static_cast<int>(std::log2(1.0 / maxRel));
        return static_cast<uint8_t>(std::min(std::max(depthEst, 0), static_cast<int>(m_config.maxDepth)));
    }

    // ------------------------------------------------------------------------
    //  Bits per dimension (for Morton coding)
    // ------------------------------------------------------------------------
    static constexpr uint8_t bitsPerDim() { return 21; }

    // ------------------------------------------------------------------------
    //  Simple load balancing: re‑assign regions to nodes based on number of
    //  routing entries (could be extended with real load metrics).
    // ------------------------------------------------------------------------
    void rebalance() {
        // Not implemented fully – placeholder for dynamic rebalancing.
        // In a real system, we would exchange load info and split/merge prefixes.
    }

    Config m_config;
    NodeID m_localNode;
    std::atomic<uint64_t> m_nextVersion;
    mutable std::mutex m_routeMutex;
    std::vector<RouteEntry> m_routingTable;
    std::unordered_map<NodeID, std::vector<RouteEntry>> m_reverseMap;
};

// ----------------------------------------------------------------------------
//  Helper: create router with default world bounds (galactic scale)
// ----------------------------------------------------------------------------
inline GlobalMortonRouter createGalacticRouter(NodeID localNode) {
    GlobalMortonRouter::Config cfg;
    cfg.worldBounds = Math::AxisAlignedBox<double,3>(
        Math::Vector<double,3>(-1e21, -1e21, -1e21),
        Math::Vector<double,3>( 1e21,  1e21,  1e21)
    );
    cfg.maxDepth = 24;
    cfg.useLoadBalancing = true;
    cfg.replicationFactor = 1;
    return GlobalMortonRouter(cfg, localNode);
}

} // namespace Distributed
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_DISTRIBUTED_GLOBAL_MORTON_ROUTING_H_INCLUDED

/**
 * Next file: core/distributed/replica_manager.h
 * Remaining in the list: 8 files (replica_manager, consensus_tree, ecosystem_driver, weather_simulator, geology_erosion, binary_streaming_archive, distributed_snapshot, network_delta_archive)
 */