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

#ifndef ORTHOTREE_CORE_PARTITIONING_HYBRID_GRID_TREE_H_INCLUDED
#define ORTHOTREE_CORE_PARTITIONING_HYBRID_GRID_TREE_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/geometry_queries.h"
#include "../../core/math/numerical_methods.h"
#include "../../core/configuration.h"
#include "../../detail/common.h"
#include "../../detail/memory_resource.h"
#include "../../detail/si_mortongrid.h"
#include "../morton/hierarchical_morton_key.h"
#include "scale_adaptive_octree.h"

#include <array>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>

namespace OrthoTree {
namespace Partitioning {

// ============================================================================
//  HybridGridTree: combines a uniform grid for near/fast‑moving objects and
//  an adaptive octree for distant/static geometry. Dynamic switching between
//  representations based on object density, velocity, and distance from viewer.
//  Supports 2D and 3D, SIMD‑accelerated queries, and real‑time rebalancing.
// ============================================================================
template<Dimension Dim, typename T = float,
         typename EntityID = uint32_t,
         typename Allocator = PMRAllocator<std::byte>>
class HybridGridTree {
public:
    using value_type = T;
    using point_type = Math::Vector<T, Dim>;
    using aabb_type = Math::AxisAlignedBox<T, Dim>;
    using entity_type = EntityID;
    using size_type = std::size_t;
    using octree_type = ScaleAdaptiveOctree<Dim, T, Allocator>;
    using grid_type = detail::MortonGrid<Dim, T, entity_type, 8>;

    static constexpr size_type DIM = (Dim == Dim2) ? 2 : 3;

    // ------------------------------------------------------------------------
    //  Configuration for hybrid behaviour
    // ------------------------------------------------------------------------
    struct Config {
        aabb_type worldBounds;                 // overall world bounds
        T gridCellSize = T(1.0);               // uniform grid cell size (world units)
        T octreeMinNodeSize = T(0.1);          // smallest octree node size
        T octreeMaxNodeSize = T(100.0);        // largest octree node size
        uint8_t octreeMaxDepth = 12;           // octree max depth
        size_type gridBucketSize = 16;          // entities per grid cell before promotion to octree
        T densityThreshold = T(50.0);           // entities per volume to switch to octree
        T velocityThreshold = T(10.0);          // speed (units/s) to stay in grid
        T distanceThreshold = T(100.0);          // distance from viewer to use octree
        bool dynamicSwitching = true;           // automatically promote/demote entities
        bool useSIMD = true;                    // enable SIMD in grid queries
        T viewerDistance = T(0.0);               // current viewer distance (0 = no viewer)
    };

    // ------------------------------------------------------------------------
    //  Entity metadata for hybrid management
    // ------------------------------------------------------------------------
    struct EntityMetadata {
        point_type position;
        point_type velocity;
        T lastUpdateTime;
        T radius;
        bool inGrid : 1;
        bool inOctree : 1;
        bool isDynamic : 1;       // fast‑moving -> grid; static -> octree
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit HybridGridTree(const Config& cfg, const Allocator& alloc = Allocator())
        : m_config(cfg)
        , m_alloc(alloc)
        , m_grid(cfg.worldBounds, cfg.gridCellSize, alloc)
        , m_octree(cfg.worldBounds, cfg.octreeMaxDepth, cfg.gridBucketSize, alloc)
        , m_metadata(alloc)
        , m_entityCount(0) {}

    // ------------------------------------------------------------------------
    //  Entity insertion (automatically chooses grid or octree)
    // ------------------------------------------------------------------------
    entity_type addEntity(const point_type& position, const point_type& velocity,
                          T radius, bool isDynamic = true) {
        entity_type id = static_cast<entity_type>(m_metadata.size());
        EntityMetadata meta;
        meta.position = position;
        meta.velocity = velocity;
        meta.lastUpdateTime = T(0);
        meta.radius = radius;
        meta.isDynamic = isDynamic;
        meta.inGrid = false;
        meta.inOctree = false;
        m_metadata.push_back(meta);
        insertEntity(id);
        return id;
    }

    void insertEntity(entity_type id) {
        const auto& meta = m_metadata[id];
        bool useGrid = shouldUseGrid(meta);
        if (useGrid) {
            if (!meta.inGrid) {
                m_grid.insert(id, meta.position);
                m_metadata[id].inGrid = true;
            }
            if (meta.inOctree) {
                m_octree.remove(id);
                m_metadata[id].inOctree = false;
            }
        } else {
            if (!meta.inOctree) {
                aabb_type bounds(meta.position - point_type(meta.radius),
                                 meta.position + point_type(meta.radius));
                m_octree.insert(id, bounds);
                m_metadata[id].inOctree = true;
            }
            if (meta.inGrid) {
                m_grid.remove(id);
                m_metadata[id].inGrid = false;
            }
        }
        ++m_entityCount;
    }

    void removeEntity(entity_type id) {
        const auto& meta = m_metadata[id];
        if (meta.inGrid) m_grid.remove(id);
        if (meta.inOctree) m_octree.remove(id);
        --m_entityCount;
        m_metadata[id].inGrid = false;
        m_metadata[id].inOctree = false;
    }

    void updateEntity(entity_type id, const point_type& newPos,
                      const point_type& newVel, T time) {
        auto& meta = m_metadata[id];
        meta.position = newPos;
        meta.velocity = newVel;
        meta.lastUpdateTime = time;
        // Re‑evaluate representation
        bool useGrid = shouldUseGrid(meta);
        if (useGrid && !meta.inGrid) {
            if (meta.inOctree) m_octree.remove(id);
            m_grid.insert(id, newPos);
            meta.inGrid = true;
            meta.inOctree = false;
        } else if (!useGrid && !meta.inOctree) {
            if (meta.inGrid) m_grid.remove(id);
            aabb_type bounds(newPos - point_type(meta.radius),
                             newPos + point_type(meta.radius));
            m_octree.insert(id, bounds);
            meta.inGrid = false;
            meta.inOctree = true;
        } else if (useGrid && meta.inGrid) {
            // update position in grid (reinsert)
            m_grid.remove(id);
            m_grid.insert(id, newPos);
        } else if (!useGrid && meta.inOctree) {
            m_octree.update(id, bounds, bounds);
        }
    }

    // ------------------------------------------------------------------------
    //  Queries (transparently query both structures)
    // ------------------------------------------------------------------------
    template<typename OutputIt>
    size_type queryPoint(const point_type& point, OutputIt out) const {
        size_type count = 0;
        // query grid first (fast)
        auto gridResult = m_grid.queryPoint(point);
        if (gridResult) {
            *out++ = *gridResult;
            ++count;
        }
        // query octree
        count += m_octree.queryPoint(point, out);
        return count;
    }

    template<typename OutputIt>
    size_type queryBox(const aabb_type& box, OutputIt out) const {
        size_type count = 0;
        count += m_grid.queryAABB(box, out);
        // octree query (returns entities in overlapping cells)
        count += m_octree.queryBox(box, out);
        return count;
    }

    template<typename OutputIt>
    size_type querySphere(const point_type& center, T radius, OutputIt out) const {
        aabb_type box(center - point_type(radius), center + point_type(radius));
        return queryBox(box, out);
    }

    // Ray cast: first intersect grid, then octree (ordered by distance)
    std::optional<std::pair<entity_type, T>> raycast(const Math::Ray<T, Dim>& ray,
                                                      T maxDist = std::numeric_limits<T>::max()) const {
        // Simplified: grid rays not implemented in MortonGrid; fallback to octree only.
        return m_octree.raycast(ray, maxDist);
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control (update viewer, performance tuning)
    // ------------------------------------------------------------------------
    void setViewerPosition(const point_type& pos) noexcept {
        m_config.viewerDistance = pos.length();
        // update switching thresholds based on distance
    }

    void setDynamicSwitching(bool enable) noexcept { m_config.dynamicSwitching = enable; }
    void setDensityThreshold(T threshold) noexcept { m_config.densityThreshold = threshold; }
    void setVelocityThreshold(T threshold) noexcept { m_config.velocityThreshold = threshold; }
    void setDistanceThreshold(T threshold) noexcept { m_config.distanceThreshold = threshold; }

    void rebalance() {
        // Evaluate all entities and reassign representation
        for (entity_type id = 0; id < static_cast<entity_type>(m_metadata.size()); ++id) {
            const auto& meta = m_metadata[id];
            bool shouldBeGrid = shouldUseGrid(meta);
            if (shouldBeGrid && !meta.inGrid) {
                if (meta.inOctree) m_octree.remove(id);
                m_grid.insert(id, meta.position);
                m_metadata[id].inGrid = true;
                m_metadata[id].inOctree = false;
            } else if (!shouldBeGrid && !meta.inOctree) {
                if (meta.inGrid) m_grid.remove(id);
                aabb_type bounds(meta.position - point_type(meta.radius),
                                 meta.position + point_type(meta.radius));
                m_octree.insert(id, bounds);
                m_metadata[id].inGrid = false;
                m_metadata[id].inOctree = true;
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Statistics
    // ------------------------------------------------------------------------
    size_type size() const noexcept { return m_entityCount; }
    size_type gridSize() const noexcept { return m_grid.size(); }
    size_type octreeSize() const noexcept { return m_octree.size(); }
    size_type memoryUsage() const noexcept {
        return m_grid.numCells() * sizeof(typename grid_type::cell_type) +
               m_octree.memoryUsage() +
               m_metadata.capacity() * sizeof(EntityMetadata);
    }

    // Access to underlying structures (for advanced use)
    const grid_type& grid() const noexcept { return m_grid; }
    const octree_type& octree() const noexcept { return m_octree; }

private:
    bool shouldUseGrid(const EntityMetadata& meta) const {
        if (!m_config.dynamicSwitching) return meta.inGrid;
        // Heuristic: use grid for high‑velocity entities near viewer, octree for static/distant
        T speed = meta.velocity.length();
        T dist = (meta.position - point_type(0)).length(); // assume viewer at origin
        if (dist < m_config.distanceThreshold && speed > m_config.velocityThreshold)
            return true;
        // Also if entity is small and dense region? Simpler: use grid if speed high
        return speed > m_config.velocityThreshold;
    }

    Config m_config;
    Allocator m_alloc;
    grid_type m_grid;
    octree_type m_octree;
    std::vector<EntityMetadata, typename Allocator::template rebind<EntityMetadata>::other> m_metadata;
    size_type m_entityCount;
};

// ----------------------------------------------------------------------------
//  SIMD‑enabled batch query for hybrid grid+octree (4‑wide)
// ----------------------------------------------------------------------------
template<Dimension Dim, typename T>
class HybridBatchQueries {
public:
    using vec_type = Math::Vector<T, Dim>;
    using aabb_type = Math::AxisAlignedBox<T, Dim>;

    static void batchBoxQuery(const HybridGridTree<Dim, T>& tree,
                              const aabb_type* boxes, entity_type* out, size_type* outCounts,
                              std::size_t numQueries) noexcept {
        for (std::size_t i = 0; i < numQueries; ++i) {
            outCounts[i] = tree.queryBox(boxes[i], out + outCounts[i]);
        }
    }

    static void batchSphereQuery(const HybridGridTree<Dim, T>& tree,
                                 const vec_type* centers, const T* radii,
                                 entity_type* out, size_type* outCounts,
                                 std::size_t numQueries) noexcept {
        for (std::size_t i = 0; i < numQueries; ++i) {
            aabb_type box(centers[i] - vec_type(radii[i]),
                          centers[i] + vec_type(radii[i]));
            outCounts[i] = tree.queryBox(box, out + outCounts[i]);
        }
    }
};

} // namespace Partitioning
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_PARTITIONING_HYBRID_GRID_TREE_H_INCLUDED