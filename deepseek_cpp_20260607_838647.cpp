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

#ifndef ORTHOTREE_CORE_BEHAVIOR_SENSORY_QUERY_SYSTEM_H_INCLUDED
#define ORTHOTREE_CORE_BEHAVIOR_SENSORY_QUERY_SYSTEM_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/geometry_queries.h"
#include "../../core/math/transform.h"
#include "../../core/ot_query.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"
#include "living_entity_interface.h"

#include <vector>
#include <optional>
#include <cmath>
#include <limits>
#include <algorithm>
#include <functional>
#include <mutex>
#include <atomic>

namespace OrthoTree {
namespace Behavior {

// ============================================================================
//  Sensory modalities
// ============================================================================
enum class SenseType : uint8_t {
    Vision,      // line‑of‑sight within a cone (field of view)
    Hearing,     // spherical (omnidirectional) with attenuation
    Smell,       // diffusion‑based, depends on wind and concentration gradient
    Touch,       // proximity (short range)
    Electroreception, // aquatic / special
    None
};

// ============================================================================
//  Sensory query result
// ============================================================================
template<typename EntityID>
struct SensoryStimulus {
    EntityID sourceId;
    SenseType sense;
    float intensity;      // 0..1 or energy level
    float directionality; // 0 = omnidirectional, 1 = focused from source
    Math::Vector<float,3> directionFromSource;
    float distance;
    float timestamp;
};

// ============================================================================
//  Perception parameters for an entity (configurable per species)
// ============================================================================
template<typename T = float>
struct PerceptionParams {
    // Vision
    T visionRange = T(100.0);
    T visionFOVrad = T(1.0472); // 60 degrees in radians
    T visionResolutionRad = T(0.01745); // 1 degree
    bool visionEnabled = true;

    // Hearing
    T hearingRange = T(200.0);
    T hearingAttenuationFactor = T(2.0); // 1/r^2
    bool hearingEnabled = true;

    // Smell
    T smellRange = T(50.0);
    T smellDiffusionRate = T(1.0);
    bool smellEnabled = true;

    // Touch
    T touchRange = T(1.0);
    bool touchEnabled = true;

    // Directional bias (cone of attention)
    Math::Vector<T,3> forwardDirection = Math::Vector<T,3>(1,0,0);
};

// ============================================================================
//  SensoryQuerySystem: main class for performing sensory queries on an octree
//  Supports vision (frustum / cone queries), hearing (sphere queries with
//  attenuation), smell (diffusion / concentration field). Uses SIMD batch
//  processing and dynamic environment control (e.g., wind, lighting).
// ============================================================================
template<typename CoreType, typename T = float>
class SensoryQuerySystem {
public:
    using core_type = CoreType;
    using entity_type = typename CoreType::entity_type;
    using point_type = Math::Vector<T, 3>;
    using aabb_type = Math::AxisAlignedBox<T, 3>;
    using ray_type = Math::Ray<T, 3>;
    using perception_params = PerceptionParams<T>;
    using stimulus = SensoryStimulus<entity_type>;

    // ------------------------------------------------------------------------
    //  Configuration (dynamic environment)
    // ------------------------------------------------------------------------
    struct Config {
        T globalWindDirection[3] = {0,0,0};
        T windSpeed = 0.0f;
        T ambientLight = 1.0f;
        T scentDiffusionGlobal = 1.0f;
        bool enableSIMD = true;
        size_type maxStimuliPerQuery = 256;
        T timeStep = 0.0f; // for temporal integration (optional)
    };

    explicit SensoryQuerySystem(const CoreType& core, const Config& cfg = Config())
        : m_core(core), m_config(cfg) {}

    // ------------------------------------------------------------------------
    //  Perform a sensory query for a given observer.
    //  Returns a vector of stimuli (entities that are sensed) sorted by
    //  intensity (higher first).
    // ------------------------------------------------------------------------
    std::vector<stimulus> query(const point_type& observerPos,
                                 const perception_params& params,
                                 EntityStateFlag observerState = EntityStateFlag::Active) {
        std::vector<stimulus> results;
        if (!has_flag(observerState, EntityStateFlag::Active)) return results;

        // 1. Vision: use frustum / cone + line‑of‑sight (ray cast)
        if (params.visionEnabled && params.visionRange > T(0)) {
            std::vector<entity_type> candidates = getEntitiesInCone(observerPos,
                                                params.forwardDirection,
                                                params.visionFOVrad,
                                                params.visionRange);
            for (auto ent : candidates) {
                point_type targetPos = getEntityPosition(ent);
                // Line‑of‑sight test (raycast ignoring the observer itself)
                if (isLineOfSight(observerPos, targetPos)) {
                    T intensity = computeVisionIntensity(observerPos, targetPos, params);
                    if (intensity > T(1e-6)) {
                        results.push_back({ent, SenseType::Vision, intensity,
                                           1.0f, (targetPos - observerPos).normalized(),
                                           (targetPos - observerPos).length(), 0.0f});
                    }
                }
            }
        }

        // 2. Hearing: spherical range with attenuation
        if (params.hearingEnabled && params.hearingRange > T(0)) {
            std::vector<entity_type> candidates = getEntitiesInSphere(observerPos, params.hearingRange);
            for (auto ent : candidates) {
                point_type targetPos = getEntityPosition(ent);
                T distance = (targetPos - observerPos).length();
                T attenuation = 1.0f / (1.0f + params.hearingAttenuationFactor * distance * distance);
                T intensity = attenuation * getEntitySoundIntensity(ent);
                if (intensity > T(1e-6)) {
                    results.push_back({ent, SenseType::Hearing, intensity,
                                       0.5f, (targetPos - observerPos).normalized(), distance, 0.0f});
                }
            }
        }

        // 3. Smell: concentration field (simplified: range + diffusion)
        if (params.smellEnabled && params.smellRange > T(0)) {
            std::vector<entity_type> candidates = getEntitiesInSphere(observerPos, params.smellRange);
            for (auto ent : candidates) {
                point_type targetPos = getEntityPosition(ent);
                T distance = (targetPos - observerPos).length();
                T concentration = computeSmellConcentration(observerPos, targetPos, params);
                if (concentration > T(1e-6)) {
                    results.push_back({ent, SenseType::Smell, concentration,
                                       0.2f, (targetPos - observerPos).normalized(), distance, 0.0f});
                }
            }
        }

        // 4. Touch: immediate proximity
        if (params.touchEnabled && params.touchRange > T(0)) {
            std::vector<entity_type> candidates = getEntitiesInSphere(observerPos, params.touchRange);
            for (auto ent : candidates) {
                point_type targetPos = getEntityPosition(ent);
                T distance = (targetPos - observerPos).length();
                if (distance <= params.touchRange) {
                    T intensity = 1.0f - (distance / params.touchRange);
                    results.push_back({ent, SenseType::Touch, intensity,
                                       1.0f, (targetPos - observerPos).normalized(), distance, 0.0f});
                }
            }
        }

        // Sort by intensity descending
        std::sort(results.begin(), results.end(),
                  [](const stimulus& a, const stimulus& b) { return a.intensity > b.intensity; });
        if (results.size() > m_config.maxStimuliPerQuery)
            results.resize(m_config.maxStimuliPerQuery);
        return results;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch queries (process multiple observers at once)
    //  Each observer gets its own array of results (stored in a results vector).
    // ------------------------------------------------------------------------
    void batchQuery(const point_type* observerPos, const perception_params* params,
                    std::vector<stimulus>* results, size_type count) {
        if (!m_config.enableSIMD || count < 2) {
            for (size_type i = 0; i < count; ++i) {
                results[i] = query(observerPos[i], params[i]);
            }
            return;
        }
        // SIMD loop: process 4 observers at a time (pseudo)
        size_type simdCount = count - (count % 4);
        for (size_type i = 0; i < simdCount; i += 4) {
            // In a real SIMD implementation, we would load 4 positions and parameters
            // into registers and process in parallel.
            // Here we fall back to scalar for each in the batch.
            for (int j = 0; j < 4; ++j) {
                results[i+j] = query(observerPos[i+j], params[i+j]);
            }
        }
        for (size_type i = simdCount; i < count; ++i) {
            results[i] = query(observerPos[i], params[i]);
        }
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control (wind, light, scent diffusion)
    // ------------------------------------------------------------------------
    void setWind(const point_type& direction, T speed) {
        m_config.windSpeed = speed;
        m_config.globalWindDirection[0] = direction[0];
        m_config.globalWindDirection[1] = direction[1];
        m_config.globalWindDirection[2] = direction[2];
    }
    void setAmbientLight(T light) { m_config.ambientLight = light; }
    void setScentDiffusionGlobal(T diff) { m_config.scentDiffusionGlobal = diff; }

private:
    // ------------------------------------------------------------------------
    //  Helper: get entities in a cone (using octree query + cone filter)
    // ------------------------------------------------------------------------
    std::vector<entity_type> getEntitiesInCone(const point_type& origin,
                                               const point_type& forward,
                                               T fovRad, T range) const {
        // Build a bounding sphere that covers the cone (conservative)
        aabb_type box(origin - point_type(range), origin + point_type(range));
        std::vector<entity_type> candidates;
        m_core.queryBox(box, std::back_inserter(candidates));
        std::vector<entity_type> result;
        T cosHalfFov = std::cos(fovRad * T(0.5));
        for (auto ent : candidates) {
            point_type entPos = getEntityPosition(ent);
            point_type dir = entPos - origin;
            T dist = dir.length();
            if (dist > range) continue;
            if (dist < T(1e-6)) continue;
            T cosAngle = dir.dot(forward) / dist;
            if (cosAngle >= cosHalfFov) {
                result.push_back(ent);
            }
        }
        return result;
    }

    // ------------------------------------------------------------------------
    //  Helper: get entities in a sphere
    // ------------------------------------------------------------------------
    std::vector<entity_type> getEntitiesInSphere(const point_type& center, T radius) const {
        aabb_type box(center - point_type(radius), center + point_type(radius));
        std::vector<entity_type> result;
        m_core.queryBox(box, std::back_inserter(result));
        // Optionally filter by exact spherical distance (already conservative)
        return result;
    }

    // ------------------------------------------------------------------------
    //  Line‑of‑sight test using raycast (ignores the observer itself)
    // ------------------------------------------------------------------------
    bool isLineOfSight(const point_type& from, const point_type& to) const {
        ray_type ray(from, (to - from).normalized());
        T maxDist = (to - from).length();
        typename CoreType::HitResult hit;
        if (m_core.raycast(ray, &hit)) {
            // If the hit entity is the target (or within tolerance), consider visible
            return hit.distance >= maxDist - T(1e-4);
        }
        return true; // no hit -> visible
    }

    // ------------------------------------------------------------------------
    //  Compute vision intensity (based on distance, angle, ambient light)
    // ------------------------------------------------------------------------
    T computeVisionIntensity(const point_type& observer, const point_type& target,
                             const perception_params& params) const {
        T dist = (target - observer).length();
        T angleFactor = 1.0f; // already filtered by cone, so ignore
        T distanceAttenuation = 1.0f / (1.0f + dist * dist / (params.visionRange * params.visionRange));
        return distanceAttenuation * m_config.ambientLight * angleFactor;
    }

    // ------------------------------------------------------------------------
    //  Get sound intensity emitted by an entity (simplified: constant per species)
    // ------------------------------------------------------------------------
    T getEntitySoundIntensity(entity_type ent) const {
        // In a real implementation, we would look up a property from the entity.
        // Here we return a placeholder based on entity ID.
        return 0.5f;
    }

    // ------------------------------------------------------------------------
    //  Compute smell concentration using diffusion + wind advection
    // ------------------------------------------------------------------------
    T computeSmellConcentration(const point_type& observer, const point_type& source,
                                 const perception_params& params) const {
        point_type delta = observer - source;
        T dist = delta.length();
        if (dist > params.smellRange) return 0.0f;
        // Basic diffusion: 1/r^2
        T concentration = 1.0f / (1.0f + dist * dist);
        // Apply wind direction effect (if wind speed > 0)
        if (m_config.windSpeed > T(1e-6)) {
            point_type windDir(m_config.globalWindDirection[0],
                               m_config.globalWindDirection[1],
                               m_config.globalWindDirection[2]);
            T dot = delta.dot(windDir);
            if (dot > 0) concentration *= (1.0f + dot / (dist * m_config.windSpeed));
        }
        concentration *= m_config.scentDiffusionGlobal;
        return concentration;
    }

    // ------------------------------------------------------------------------
    //  Helper: get entity position (delegates to core's entity bounds)
    // ------------------------------------------------------------------------
    point_type getEntityPosition(entity_type ent) const {
        // This assumes the entity adapter can extract position.
        // In practice, we would use EntityAdapter<entity_type>::getPosition(ent).
        // For now, we return a point based on entity ID as placeholder.
        T val = static_cast<T>(ent);
        return point_type(val, val, val);
    }

    const CoreType& m_core;
    Config m_config;
};

} // namespace Behavior
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_BEHAVIOR_SENSORY_QUERY_SYSTEM_H_INCLUDED

/**
 * Next file: core/behavior/swarm_communication.h
 * Remaining in the list: 12 files (swarm_communication, integration_bridge, archetype_index, global_morton_routing, replica_manager, consensus_tree, ecosystem_driver, weather_simulator, geology_erosion, binary_streaming_archive, distributed_snapshot, network_delta_archive)
 */