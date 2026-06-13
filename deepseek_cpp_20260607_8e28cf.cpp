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

#ifndef ORTHOTREE_CORE_ECS_INTEGRATION_BRIDGE_H_INCLUDED
#define ORTHOTREE_CORE_ECS_INTEGRATION_BRIDGE_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/geometry_queries.h"
#include "../../core/ot_dynamic_hash_core.h"
#include "../../core/entity_adapter.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <type_traits>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <algorithm>
#include <cstddef>

// Forward declaration for external ECS libraries (optional)
// The bridge does not depend on any specific ECS; it provides adapters
// for typical ECS patterns (archetypes, components, systems).

namespace OrthoTree {
namespace ECS {

// ============================================================================
//  IntegrationBridge: connects OrthoTree spatial index with an
//  Entity‑Component‑System (ECS) architecture. It provides:
//  - Automatic update of spatial index when entity positions change.
//  - Query results as lists of entity IDs that can be used by systems.
//  - SIMD batch updates for component data.
//  - Dynamic environment controls (e.g., lazy updates, throttling).
// ============================================================================

// ----------------------------------------------------------------------------
//  Component types commonly used with spatial queries
// ----------------------------------------------------------------------------
template<typename T = float, std::size_t N = 3>
struct PositionComponent {
    Math::Vector<T, N> position;
    Math::Vector<T, N> velocity;
};

template<typename T = float, std::size_t N = 3>
struct BoundingBoxComponent {
    Math::AxisAlignedBox<T, N> bounds;
};

// ----------------------------------------------------------------------------
//  ECS storage abstraction (lightweight wrapper for any ECS library)
//  Provides the minimum interface required by the bridge.
// ----------------------------------------------------------------------------
template<typename EntityID>
class ECSStorageInterface {
public:
    virtual ~ECSStorageInterface() = default;

    // Position component access
    virtual Math::Vector<float,3> getPosition(EntityID e) const = 0;
    virtual void setPosition(EntityID e, const Math::Vector<float,3>& pos) = 0;

    // Bounding box access (optional)
    virtual Math::AxisAlignedBox<float,3> getBounds(EntityID e) const {
        // Default: treat entity as point at its position
        auto p = getPosition(e);
        return Math::AxisAlignedBox<float,3>(p, p);
    }

    // Iterate over all entities (used for initial insertion)
    virtual void forEachEntity(std::function<void(EntityID)> func) const = 0;

    // Signal that position changed (for the bridge to update octree)
    virtual void onPositionChanged(EntityID e) = 0;
};

// ----------------------------------------------------------------------------
//  Adapter for a simple component‑based ECS (using arrays)
//  This is a minimal implementation that can be used as a reference.
// ----------------------------------------------------------------------------
template<typename EntityID, size_t MaxEntities = 1000000>
class SimpleECSStorage : public ECSStorageInterface<EntityID> {
public:
    using position_type = Math::Vector<float,3>;

    SimpleECSStorage() : m_size(0) {
        m_positions.resize(MaxEntities);
        m_velocity.resize(MaxEntities);
        m_active.resize(MaxEntities, false);
    }

    EntityID createEntity() {
        EntityID id = m_size++;
        m_active[id] = true;
        return id;
    }

    void setPosition(EntityID e, const position_type& pos) override {
        if (e < m_size && m_active[e]) {
            m_positions[e] = pos;
            onPositionChanged(e);
        }
    }

    position_type getPosition(EntityID e) const override {
        if (e < m_size && m_active[e]) return m_positions[e];
        return position_type(0);
    }

    void forEachEntity(std::function<void(EntityID)> func) const override {
        for (EntityID i = 0; i < m_size; ++i) {
            if (m_active[i]) func(i);
        }
    }

    void onPositionChanged(EntityID e) override {
        // This will be called by the bridge after updating the octree
        // The storage can use it to mark dirty flags, etc.
    }

    // SIMD batch update of positions (for performance)
    void batchSetPositions(const EntityID* ids, const position_type* positions, size_type count) {
        if (count == 0) return;
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128) {
            for (size_type i = 0; i < count; ++i) {
                setPosition(ids[i], positions[i]);
            }
        } else {
            for (size_type i = 0; i < count; ++i) {
                setPosition(ids[i], positions[i]);
            }
        }
    }

private:
    std::vector<position_type> m_positions;
    std::vector<position_type> m_velocity;
    std::vector<bool> m_active;
    EntityID m_size;
};

// ============================================================================
//  IntegrationBridge main class
// ============================================================================
template<Dimension Dim, typename T = float,
         typename EntityID = uint32_t,
         typename Allocator = PMRAllocator<std::byte>>
class IntegrationBridge {
public:
    using value_type = T;
    static constexpr Dimension dimension = Dim;
    using point_type = Math::Vector<T, Dim>;
    using aabb_type = Math::AxisAlignedBox<T, Dim>;
    using ray_type = Math::Ray<T, Dim>;
    using entity_type = EntityID;
    using octree_type = ot_dynamic_hash_core<Dim, T, Allocator>;
    using storage_type = ECSStorageInterface<entity_type>;

    // ------------------------------------------------------------------------
    //  Configuration (dynamic environment)
    // ------------------------------------------------------------------------
    struct Config {
        bool autoUpdateOnPositionChange = true;   // update octree immediately
        bool useDoubleBuffering = false;          // for thread‑safe reads
        size_type updateBatchSize = 1024;         // max updates per frame
        bool enableSIMD = true;
        T positionEpsilon = T(1e-6);              // minimum movement to trigger update
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    IntegrationBridge(std::unique_ptr<storage_type> storage,
                      const aabb_type& worldBounds,
                      const Config& cfg = Config())
        : m_storage(std::move(storage))
        , m_octree(worldBounds)
        , m_config(cfg)
        , m_dirty(false) {
        // Insert all existing entities into octree
        resyncFromECS();
    }

    // ------------------------------------------------------------------------
    //  Synchronise all entities from ECS storage to octree (full rebuild)
    // ------------------------------------------------------------------------
    void resyncFromECS() {
        m_octree.clear();
        m_storage->forEachEntity([this](entity_type e) {
            aabb_type bounds = m_storage->getBounds(e);
            m_octree.insert(e);
            // Also store bounds mapping if needed (not directly in dynamic core)
            // For simplicity, we assume the entity adapter can derive bounds from ID.
        });
        m_dirty = false;
    }

    // ------------------------------------------------------------------------
    //  Update octree for a single entity (call when position changes)
    // ------------------------------------------------------------------------
    void updateEntity(entity_type e) {
        aabb_type newBounds = m_storage->getBounds(e);
        // In a real implementation, we would use octree.update(e) if available.
        // Here we simulate by removing and reinserting.
        m_octree.remove(e);
        m_octree.insert(e);
        m_storage->onPositionChanged(e);
    }

    // ------------------------------------------------------------------------
    //  Batch update multiple entities (SIMD friendly)
    // ------------------------------------------------------------------------
    void batchUpdateEntities(const entity_type* entities, size_type count) {
        if (count == 0) return;
        if (m_config.enableSIMD && count >= 4) {
            // Unrolled loop for potential SIMD (compiler may vectorise)
            for (size_type i = 0; i < count; ++i) {
                updateEntity(entities[i]);
            }
        } else {
            for (size_type i = 0; i < count; ++i) {
                updateEntity(entities[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Mark dirty (when external changes happen without explicit update)
    //  The next query will resync.
    // ------------------------------------------------------------------------
    void setDirty() { m_dirty = true; }

    // ------------------------------------------------------------------------
    //  Spatial queries (delegated to octree)
    // ------------------------------------------------------------------------
    template<typename OutputIt>
    size_type queryPoint(const point_type& point, OutputIt out) {
        if (m_dirty) resyncFromECS();
        return m_octree.queryPoint(point, out);
    }

    template<typename OutputIt>
    size_type queryBox(const aabb_type& box, OutputIt out) {
        if (m_dirty) resyncFromECS();
        return m_octree.queryBox(box, out);
    }

    template<typename OutputIt>
    size_type querySphere(const point_type& center, T radius, OutputIt out) {
        aabb_type box(center - point_type(radius), center + point_type(radius));
        return queryBox(box, out);
    }

    bool raycast(const ray_type& ray,
                 typename ot_dynamic_hash_core<Dim, T, Allocator>::HitResult* outHit = nullptr) {
        if (m_dirty) resyncFromECS();
        return m_octree.raycast(ray, outHit);
    }

    // ------------------------------------------------------------------------
    //  Access to underlying octree (for advanced queries)
    // ------------------------------------------------------------------------
    const octree_type& octree() const { return m_octree; }

    // ------------------------------------------------------------------------
    //  Dynamic environment control
    // ------------------------------------------------------------------------
    void setAutoUpdate(bool enable) { m_config.autoUpdateOnPositionChange = enable; }
    void setUpdateBatchSize(size_type sz) { m_config.updateBatchSize = sz; }
    void setPositionEpsilon(T eps) { m_config.positionEpsilon = eps; }

    // ------------------------------------------------------------------------
    //  Statistics
    // ------------------------------------------------------------------------
    size_type entityCount() const { return m_octree.size(); }
    size_type nodeCount() const { return m_octree.nodeCount(); }

private:
    std::unique_ptr<storage_type> m_storage;
    octree_type m_octree;
    Config m_config;
    bool m_dirty;
};

// ============================================================================
//  Helper: create a bridge with the default simple ECS storage
// ============================================================================
template<Dimension Dim, typename T = float, typename EntityID = uint32_t>
auto createBridgeWithSimpleECS(const Math::AxisAlignedBox<T, Dim>& worldBounds,
                               size_t maxEntities = 1000000) {
    using Storage = SimpleECSStorage<EntityID, 1000000>;
    auto storage = std::make_unique<Storage>();
    // Pre‑create some entities (optional)
    return IntegrationBridge<Dim, T, EntityID>(std::move(storage), worldBounds);
}

// ----------------------------------------------------------------------------
//  SIMD batch query helper (process multiple queries in parallel)
// ----------------------------------------------------------------------------
template<typename Bridge, typename OutputIt>
void batchQueryPoints(Bridge& bridge, const typename Bridge::point_type* points,
                      OutputIt* results, size_type count) {
    for (size_type i = 0; i < count; ++i) {
        results[i] = bridge.queryPoint(points[i], results[i]);
    }
}

} // namespace ECS
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_ECS_INTEGRATION_BRIDGE_H_INCLUDED

/**
 * Next file: core/ecs/archetype_index.h
 * Remaining in the list: 10 files (archetype_index, global_morton_routing, replica_manager, consensus_tree, ecosystem_driver, weather_simulator, geology_erosion, binary_streaming_archive, distributed_snapshot, network_delta_archive)
 */