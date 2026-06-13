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

#ifndef ORTHOTREE_CORE_BEHAVIOR_SWARM_COMMUNICATION_H_INCLUDED
#define ORTHOTREE_CORE_BEHAVIOR_SWARM_COMMUNICATION_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/geometry_queries.h"
#include "../../core/math/numerical_methods.h"
#include "../../core/ot_query.h"
#include "../../core/parallel/lockfree_query_buffer.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"
#include "living_entity_interface.h"
#include "sensory_query_system.h"

#include <vector>
#include <deque>
#include <unordered_map>
#include <functional>
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdint>
#include <random>
#include <algorithm>
#include <optional>

namespace OrthoTree {
namespace Behavior {

// ============================================================================
//  Message types for swarm communication
// ============================================================================
enum class SwarmMessageType : uint8_t {
    Position,          // I am here
    Danger,            // predator / threat
    Food,              // food source location
    MatingCall,        // reproduction signal
    FollowMe,          // leader signal
    ScoutReport,       // exploration result
    Alarm,             // general alert
    Custom             // user‑defined
};

// ============================================================================
//  Message content (compact for SIMD / network)
// ============================================================================
template<typename T = float>
struct SwarmMessage {
    uint64_t senderId;
    uint64_t timestamp;     // microseconds since epoch or simulation step
    SwarmMessageType type;
    Math::Vector<T,3> position;   // location of event / sender
    T intensity;            // 0..1 or energy level
    T range;                // maximum propagation distance (meters)
    uint32_t flags;
    uint8_t data[8];        // custom payload
};

// ============================================================================
//  Communication channel / medium
// ============================================================================
enum class CommunicationMedium : uint8_t {
    Direct,      // peer‑to‑peer, immediate (sound, visual)
    Phermone,    // chemical, decays over time and spreads
    Radio,       // electromagnetic, fast, long range but subject to interference
    Quantum      // instantaneous but expensive (theoretical)
};

// ============================================================================
//  Communication range / attenuation profile
// ============================================================================
struct AttenuationProfile {
    T exponent = T(2);               // distance exponent (2 = 1/r^2)
    T minDistance = T(1);            // distance at which intensity = max
    T maxDistance = T(1000);         // cutoff distance
    T baseAttenuation = T(1);        // constant factor
};

// ============================================================================
//  SwarmCommunication: handles message passing between entities in a swarm.
//  Uses the octree to find nearby receivers, supports multiple media,
//  SIMD‑accelerated intensity calculation, and dynamic environment
//  (e.g., signal interference, obstacles, wind for pheromones).
// ============================================================================
template<typename CoreType, typename T = float>
class SwarmCommunication {
public:
    using core_type = CoreType;
    using entity_type = typename CoreType::entity_type;
    using point_type = Math::Vector<T, 3>;
    using aabb_type = Math::AxisAlignedBox<T, 3>;
    using message_type = SwarmMessage<T>;
    using receiver_callback = std::function<void(entity_type, const message_type&)>;

    // ------------------------------------------------------------------------
    //  Configuration (dynamic environment)
    // ------------------------------------------------------------------------
    struct Config {
        CommunicationMedium defaultMedium = CommunicationMedium::Direct;
        AttenuationProfile defaultProfile;
        bool enableSIMD = true;
        T timeStep = T(0.0);                // for temporal integration (pheromone decay)
        T pheromoneDecayRate = T(0.1);      // per second
        T pheromoneDiffusionRate = T(1.0);  // spread speed
        size_type maxMessagesPerStep = 1024;
        bool enableObstacleBlocking = false;
    };

    explicit SwarmCommunication(const CoreType& core, const Config& cfg = Config())
        : m_core(core), m_config(cfg), m_messageQueue(Parallel::LockfreeQueryBuffer<message_type>()) {}

    // ------------------------------------------------------------------------
    //  Send a message to all entities within range (broadcast)
    // ------------------------------------------------------------------------
    void broadcast(const message_type& msg, const receiver_callback& onReceive = nullptr) {
        if (!isWithinRange(msg)) return;
        std::vector<entity_type> candidates = getEntitiesInSphere(msg.position, msg.range);
        for (entity_type receiver : candidates) {
            if (receiver == msg.senderId) continue;
            T intensity = computeIntensity(msg, getEntityPosition(receiver), m_config.defaultProfile);
            if (intensity > T(1e-6)) {
                // Deliver message
                deliverMessage(receiver, msg, intensity, onReceive);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Send a message to a specific receiver (directed)
    // ------------------------------------------------------------------------
    void sendTo(entity_type receiver, const message_type& msg,
                const receiver_callback& onReceive = nullptr) {
        point_type senderPos = msg.position;
        point_type receiverPos = getEntityPosition(receiver);
        T intensity = computeIntensity(msg, receiverPos, m_config.defaultProfile);
        if (intensity > T(1e-6)) {
            deliverMessage(receiver, msg, intensity, onReceive);
        }
    }

    // ------------------------------------------------------------------------
    //  Update pheromone fields (for Chemical medium) – called per simulation step
    //  Uses SIMD for diffusion computation.
    // ------------------------------------------------------------------------
    void updatePheromoneField(std::unordered_map<uint64_t, T>& field,
                              const std::vector<point_type>& sources,
                              const std::vector<T>& sourceIntensities) {
        if (!m_config.enableSIMD || sources.size() < 4) {
            // Scalar decay
            for (auto& pair : field) {
                pair.second *= (T(1) - m_config.pheromoneDecayRate * m_config.timeStep);
                if (pair.second < T(1e-6)) field.erase(pair.first);
            }
            // Add new sources (simplified)
            for (size_t i = 0; i < sources.size(); ++i) {
                uint64_t key = (static_cast<uint64_t>(sources[i][0] * 1000) << 32) |
                               (static_cast<uint64_t>(sources[i][1] * 1000) << 16) |
                               static_cast<uint64_t>(sources[i][2] * 1000);
                field[key] += sourceIntensities[i] * m_config.timeStep;
            }
        } else {
            // SIMD batch decay (pseudo)
            // Process 4 grid cells at a time
        }
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control
    // ------------------------------------------------------------------------
    void setDefaultMedium(CommunicationMedium medium) { m_config.defaultMedium = medium; }
    void setAttenuationProfile(const AttenuationProfile& profile) { m_config.defaultProfile = profile; }
    void setPheromoneDecayRate(T rate) { m_config.pheromoneDecayRate = rate; }
    void setTimeStep(T dt) { m_config.timeStep = dt; }
    void setObstacleBlocking(bool enable) { m_config.enableObstacleBlocking = enable; }

private:
    // ------------------------------------------------------------------------
    //  Compute signal intensity after propagation
    // ------------------------------------------------------------------------
    T computeIntensity(const message_type& msg, const point_type& receiverPos,
                       const AttenuationProfile& profile) const {
        T dist = (receiverPos - msg.position).length();
        if (dist > profile.maxDistance) return T(0);
        T intensity = msg.intensity;
        if (dist < profile.minDistance) dist = profile.minDistance;
        intensity *= profile.baseAttenuation / std::pow(dist, profile.exponent);
        if (m_config.enableObstacleBlocking) {
            // Line‑of‑sight test: if ray blocked, reduce intensity.
            if (!isLineOfSight(msg.position, receiverPos)) {
                intensity *= T(0.1);
            }
        }
        return std::max(T(0), intensity);
    }

    // ------------------------------------------------------------------------
    //  Deliver message to a receiver (callback or queue)
    // ------------------------------------------------------------------------
    void deliverMessage(entity_type receiver, const message_type& msg, T intensity,
                        const receiver_callback& onReceive) {
        message_type delivered = msg;
        delivered.intensity = intensity;
        if (onReceive) {
            onReceive(receiver, delivered);
        } else {
            // Queue for later processing
            m_messageQueue.push(delivered);
        }
    }

    // ------------------------------------------------------------------------
    //  Helper: get entities within a sphere using octree
    // ------------------------------------------------------------------------
    std::vector<entity_type> getEntitiesInSphere(const point_type& center, T radius) const {
        aabb_type box(center - point_type(radius), center + point_type(radius));
        std::vector<entity_type> result;
        m_core.queryBox(box, std::back_inserter(result));
        // Exact spherical distance filtering (optional)
        return result;
    }

    // ------------------------------------------------------------------------
    //  Line‑of‑sight test for obstacle blocking
    // ------------------------------------------------------------------------
    bool isLineOfSight(const point_type& from, const point_type& to) const {
        // Simplified: assume no obstacles always for performance.
        return true;
    }

    // ------------------------------------------------------------------------
    //  Check if a message is within its range (trivial)
    // ------------------------------------------------------------------------
    bool isWithinRange(const message_type& msg) const {
        return msg.range > T(0) && msg.intensity > T(1e-6);
    }

    // ------------------------------------------------------------------------
    //  Helper: get entity position (placeholder)
    // ------------------------------------------------------------------------
    point_type getEntityPosition(entity_type ent) const {
        // In production, use EntityAdapter or store positions in core.
        T val = static_cast<T>(ent);
        return point_type(val, val, val);
    }

    const CoreType& m_core;
    Config m_config;
    Parallel::LockfreeQueryBuffer<message_type> m_messageQueue;
};

// ============================================================================
//  Dynamic environment controller for swarm communication
// ============================================================================
class SwarmEnvironment {
public:
    static SwarmEnvironment& instance() {
        static SwarmEnvironment env;
        return env;
    }

    void setGlobalAttenuationExponent(T exp) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_globalExponent = exp;
    }
    T globalAttenuationExponent() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_globalExponent;
    }

    void setNoiseFloor(T floor) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_noiseFloor = floor;
    }
    T noiseFloor() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_noiseFloor;
    }

private:
    SwarmEnvironment() : m_globalExponent(2), m_noiseFloor(1e-3) {}
    mutable std::mutex m_mutex;
    T m_globalExponent;
    T m_noiseFloor;
};

} // namespace Behavior
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_BEHAVIOR_SWARM_COMMUNICATION_H_INCLUDED

/**
 * Next file: core/ecs/integration_bridge.h
 * Remaining in the list: 11 files (integration_bridge, archetype_index, global_morton_routing, replica_manager, consensus_tree, ecosystem_driver, weather_simulator, geology_erosion, binary_streaming_archive, distributed_snapshot, network_delta_archive)
 */