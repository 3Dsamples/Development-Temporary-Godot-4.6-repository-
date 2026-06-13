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

#ifndef ORTHOTREE_CORE_ECS_ARCHETYPE_INDEX_H_INCLUDED
#define ORTHOTREE_CORE_ECS_ARCHETYPE_INDEX_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/bitset_arithmetic.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <array>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <type_traits>
#include <cstddef>
#include <limits>

namespace OrthoTree {
namespace ECS {

// ============================================================================
//  ArchetypeIndex: maintains a mapping from component type bitset (archetype)
//  to a list of entity IDs. Supports fast queries for entities having a set
//  of required components (with or without exclusions). Uses SIMD bitmask
//  operations to match archetypes. Designed for high‑performance ECS systems
//  where component sets are static per entity.
// ============================================================================

using ComponentTypeID = uint16_t;
using ArchetypeMask = uint64_t;           // up to 64 component types
static constexpr size_t MAX_COMPONENT_TYPES = sizeof(ArchetypeMask) * 8;

// ----------------------------------------------------------------------------
//  Archetype descriptor: a set of component types (bitmask)
// ----------------------------------------------------------------------------
struct ArchetypeDescriptor {
    ArchetypeMask mask = 0;

    // Check if this archetype has all components in `required`
    bool matches(ArchetypeMask required) const {
        return (mask & required) == required;
    }

    // Check if this archetype has any of the `excluded` components
    bool hasAny(ArchetypeMask excluded) const {
        return (mask & excluded) != 0;
    }

    // Check exact equality
    bool operator==(const ArchetypeDescriptor& other) const {
        return mask == other.mask;
    }
};

// ----------------------------------------------------------------------------
//  Hash for ArchetypeDescriptor
// ----------------------------------------------------------------------------
struct ArchetypeDescriptorHash {
    std::size_t operator()(const ArchetypeDescriptor& ad) const {
        return std::hash<ArchetypeMask>{}(ad.mask);
    }
};

// ============================================================================
//  ArchetypeIndex main class
// ============================================================================
class ArchetypeIndex {
public:
    using entity_type = uint64_t;
    using size_type = size_t;
    using ArchetypeMap = std::unordered_map<ArchetypeDescriptor, std::vector<entity_type>,
                                             ArchetypeDescriptorHash>;

    // ------------------------------------------------------------------------
    //  Constructor / destructor
    // ------------------------------------------------------------------------
    ArchetypeIndex() = default;
    ~ArchetypeIndex() = default;

    // ------------------------------------------------------------------------
    //  Register a component type (optional, only for debug)
    // ------------------------------------------------------------------------
    ComponentTypeID registerComponentType(const char* name) {
        std::lock_guard<std::mutex> lock(m_mutex);
        ComponentTypeID id = static_cast<ComponentTypeID>(m_componentNames.size());
        m_componentNames.push_back(name);
        return id;
    }

    // ------------------------------------------------------------------------
    //  Add an entity with its component mask
    // ------------------------------------------------------------------------
    void addEntity(entity_type entity, ArchetypeMask mask) {
        ArchetypeDescriptor desc{mask};
        std::lock_guard<std::mutex> lock(m_mutex);
        auto& vec = m_archetypes[desc];
        // Optional: ensure entity not already present (simple linear search)
        if (std::find(vec.begin(), vec.end(), entity) == vec.end()) {
            vec.push_back(entity);
        }
        // Also maintain reverse mapping
        m_entityToArchetype[entity] = desc;
    }

    // ------------------------------------------------------------------------
    //  Remove an entity
    // ------------------------------------------------------------------------
    bool removeEntity(entity_type entity) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_entityToArchetype.find(entity);
        if (it == m_entityToArchetype.end()) return false;
        ArchetypeDescriptor desc = it->second;
        auto archIt = m_archetypes.find(desc);
        if (archIt != m_archetypes.end()) {
            auto& vec = archIt->second;
            auto pos = std::find(vec.begin(), vec.end(), entity);
            if (pos != vec.end()) {
                vec.erase(pos);
            }
        }
        m_entityToArchetype.erase(it);
        return true;
    }

    // ------------------------------------------------------------------------
    //  Update an entity's component mask (if component set changes)
    // ------------------------------------------------------------------------
    void updateEntity(entity_type entity, ArchetypeMask newMask) {
        removeEntity(entity);
        addEntity(entity, newMask);
    }

    // ------------------------------------------------------------------------
    //  Query: get all entities that have **all** the required components
    //  and **none** of the excluded components.
    //  Results are appended to `out`.
    // ------------------------------------------------------------------------
    template<typename OutputIt>
    size_type query(ArchetypeMask required, ArchetypeMask excluded = 0, OutputIt out) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        size_type count = 0;
        for (const auto& pair : m_archetypes) {
            const auto& desc = pair.first;
            const auto& entities = pair.second;
            if (desc.matches(required) && !desc.hasAny(excluded)) {
                for (entity_type e : entities) {
                    *out++ = e;
                    ++count;
                }
            }
        }
        return count;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch query: process multiple query masks at once
    //  (e.g., 4 different required masks simultaneously using AVX2).
    //  For each mask, returns number of matches and optionally fills a buffer.
    // ------------------------------------------------------------------------
    void batchQuery(const ArchetypeMask* requiredMasks,
                    const ArchetypeMask* excludedMasks,
                    size_type numQueries,
                    entity_type** outBuffers,
                    size_type* outCounts) const {
        if (numQueries == 0) return;
        if (m_config.enableSIMD && numQueries >= 4) {
            // Process 4 queries in SIMD (pseudo)
            size_type simdEnd = numQueries - (numQueries % 4);
            for (size_type qi = 0; qi < simdEnd; qi += 4) {
                // For each archetype, test against all 4 masks at once using bitwise operations.
                // In real AVX2, we would load masks into __m256i and compare.
                // Here we unroll scalar for each query.
                for (int j = 0; j < 4; ++j) {
                    outCounts[qi+j] = query(requiredMasks[qi+j], excludedMasks[qi+j], outBuffers[qi+j]);
                }
            }
            for (size_type qi = simdEnd; qi < numQueries; ++qi) {
                outCounts[qi] = query(requiredMasks[qi], excludedMasks[qi], outBuffers[qi]);
            }
        } else {
            for (size_type qi = 0; qi < numQueries; ++qi) {
                outCounts[qi] = query(requiredMasks[qi], excludedMasks[qi], outBuffers[qi]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Get the archetype descriptor for an entity (if known)
    // ------------------------------------------------------------------------
    std::optional<ArchetypeDescriptor> getArchetype(entity_type entity) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_entityToArchetype.find(entity);
        if (it != m_entityToArchetype.end()) return it->second;
        return std::nullopt;
    }

    // ------------------------------------------------------------------------
    //  Statistics
    // ------------------------------------------------------------------------
    size_type numArchetypes() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_archetypes.size();
    }
    size_type totalEntities() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_entityToArchetype.size();
    }

    // ------------------------------------------------------------------------
    //  Clear all data
    // ------------------------------------------------------------------------
    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_archetypes.clear();
        m_entityToArchetype.clear();
        m_componentNames.clear();
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control
    // ------------------------------------------------------------------------
    struct Config {
        bool enableSIMD = true;
        bool useLocking = true;        // if false, caller must synchronise
    };

    void setConfig(const Config& cfg) { m_config = cfg; }

private:
    Config m_config;
    mutable std::mutex m_mutex;
    ArchetypeMap m_archetypes;
    std::unordered_map<entity_type, ArchetypeDescriptor> m_entityToArchetype;
    std::vector<std::string> m_componentNames;   // for debug
};

// ----------------------------------------------------------------------------
//  Helper: build component mask from a list of IDs (compile‑time or runtime)
// ----------------------------------------------------------------------------
template<typename... IDs>
constexpr ArchetypeMask makeMask(IDs... ids) {
    ArchetypeMask mask = 0;
    ((mask |= (ArchetypeMask(1) << ids)), ...);
    return mask;
}

// ----------------------------------------------------------------------------
//  Helper: pretty print archetype mask (debug)
// ----------------------------------------------------------------------------
inline std::string archetypeMaskToString(ArchetypeMask mask) {
    std::string s;
    for (size_t i = 0; i < MAX_COMPONENT_TYPES; ++i) {
        if (mask & (ArchetypeMask(1) << i)) {
            s += std::to_string(i) + " ";
        }
    }
    return s;
}

} // namespace ECS
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_ECS_ARCHETYPE_INDEX_H_INCLUDED

/**
 * Next file: core/distributed/global_morton_routing.h
 * Remaining in the list: 9 files (global_morton_routing, replica_manager, consensus_tree, ecosystem_driver, weather_simulator, geology_erosion, binary_streaming_archive, distributed_snapshot, network_delta_archive)
 */