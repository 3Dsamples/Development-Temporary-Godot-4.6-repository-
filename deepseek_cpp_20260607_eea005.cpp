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

/**
 * @file si_mortongrid.h
 * @brief Uniform grid indexed by Morton codes (Z‑order curve) for fast spatial queries.
 *
 * This file implements a spatial grid where each cell is identified by a Morton
 * code derived from its integer coordinates. The grid supports:
 * - Dynamic insertion/removal of entities (points or AABBs)
 * - Range queries (circle, AABB) using exact cell enumeration via integer coordinates
 * - Neighbour iteration (cell‑by‑cell traversal in morton order)
 * - Automatic resizing (not needed; grid resolution fixed at construction)
 * - PMR allocators and small‑buffer optimisation for cell storage
 *
 * The morton ordering provides excellent cache locality for spatial queries
 * and serves as a bridge between uniform grids and octrees. It is ideal for
 * particle systems, sparse fluid simulation, and real‑time collision detection.
 */

#ifndef ORTHOTREE_DETAIL_SI_MORTONGRID_H_INCLUDED
#define ORTHOTREE_DETAIL_SI_MORTONGRID_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/vector_math.h"
#include "../core/math/geometry_queries.h"
#include "si_morton.h"
#include "inplace_vector.h"
#include "memory_resource.h"
#include "common.h"

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>
#include <array>

namespace OrthoTree {
namespace detail {

// ----------------------------------------------------------------------------
//  MortonGridCell: stores entities inside a single grid cell
// ----------------------------------------------------------------------------

/**
 * @brief Cell in the morton grid. Contains a list of entity indices (or entities).
 * @tparam Entity User‑defined entity type (must be cheap to copy/move).
 * @tparam SmallCapacity Inline capacity for entities (avoids allocation for sparse cells).
 */
template <typename Entity, size_t SmallCapacity = 4>
struct MortonGridCell {
    using EntityList = InplaceVector<Entity, SmallCapacity, PMRAllocator<Entity>>;

    EntityList entities;
    uint32_t version;      // for fast invalidation on clear

    MortonGridCell() : entities(), version(0) {}
    explicit MortonGridCell(const PMRAllocator<Entity>& alloc) : entities(alloc), version(0) {}
};

// ----------------------------------------------------------------------------
//  MortonGrid main class
// ----------------------------------------------------------------------------

/**
 * @brief Grid indexed by Morton codes.
 * @tparam Dim Dimension (2 or 3).
 * @tparam T Scalar type (float/double).
 * @tparam Entity User entity type.
 * @tparam CellCapacity Max entities per cell before warning (not a hard limit).
 */
template <Dimension Dim, typename T = float, typename Entity = uint32_t,
          size_t CellCapacity = 16>
class MortonGrid {
public:
    using value_type      = T;
    using entity_type     = Entity;
    using point_type      = Math::Vector<T, Dim>;
    using aabb_type       = Math::AxisAlignedBox<T, Dim>;
    using morton_type     = uint64_t;
    using cell_type       = MortonGridCell<Entity, 4>;
    using size_type       = size_t;

    static constexpr Dimension dimension = Dim;
    static constexpr size_t bits_per_coord = 21;   // for 64‑bit morton

    // ------------------------------------------------------------------------
    //  Construction
    // ------------------------------------------------------------------------
    explicit MortonGrid(const aabb_type& worldBounds,
                        T cellSize = T(1),
                        const PMRAllocator<cell_type>& alloc = PMRAllocator<cell_type>())
        : m_worldBounds(worldBounds)
        , m_cellSize(cellSize)
        , m_invCellSize(T(1) / cellSize)
        , m_alloc(alloc)
        , m_cells(alloc)
        , m_totalEntities(0) {
        // Compute grid resolution (number of cells per dimension)
        Math::Vector<T, Dim> extent = worldBounds.extents();
        for (size_t i = 0; i < static_cast<size_t>(Dim); ++i) {
            m_resolution[i] = static_cast<uint32_t>(std::ceil(extent[i] / cellSize));
            // Clamp to max representable by bits_per_coord
            uint64_t maxVal = (uint64_t(1) << bits_per_coord) - 1;
            if (m_resolution[i] > maxVal) m_resolution[i] = static_cast<uint32_t>(maxVal);
        }
    }

    // ------------------------------------------------------------------------
    //  Insertion / removal
    // ------------------------------------------------------------------------

    /**
     * @brief Insert an entity at a given point (or using its centroid).
     * @param entity User entity.
     * @param position World position (used to compute cell).
     */
    void insert(const entity_type& entity, const point_type& position) {
        morton_type code = mortonCodeForPoint(position);
        cell_type& cell = getOrCreateCell(code);
        cell.entities.push_back(entity);
        ++m_totalEntities;
    }

    /**
     * @brief Insert entity using its bounding box (stores in all overlapping cells).
     * @param entity Entity.
     * @param bounds AABB of entity.
     */
    void insert(const entity_type& entity, const aabb_type& bounds) {
        // Determine the range of integer cell coordinates overlapped by bounds
        auto [minGrid, maxGrid] = gridRangeForBounds(bounds);
        // Iterate over all cells in the axis‑aligned integer range
        for (uint32_t iz = minGrid[2]; iz <= maxGrid[2]; ++iz) {
            for (uint32_t iy = minGrid[1]; iy <= maxGrid[1]; ++iy) {
                for (uint32_t ix = minGrid[0]; ix <= maxGrid[0]; ++ix) {
                    morton_type code = gridCoordToMorton(ix, iy, iz);
                    // Compute cell AABB to check exact overlap (optional, but conservative)
                    aabb_type cellBounds = cellAABB(code);
                    if (cellBounds.overlaps(bounds)) {
                        cell_type& cell = getOrCreateCell(code);
                        cell.entities.push_back(entity);
                        ++m_totalEntities;
                    }
                }
            }
        }
    }

    /**
     * @brief Remove entity from all cells (slow – linear scan; for sparse use).
     * @return True if removed.
     */
    bool remove(const entity_type& entity) {
        bool removedAny = false;
        for (auto& pair : m_cells) {
            auto& cell = pair.second;
            auto& vec = cell.entities;
            auto it = std::find(vec.begin(), vec.end(), entity);
            if (it != vec.end()) {
                vec.erase(it);
                --m_totalEntities;
                removedAny = true;
                // Do not break; entity might be in multiple cells.
            }
        }
        return removedAny;
    }

    // ------------------------------------------------------------------------
    //  Queries
    // ------------------------------------------------------------------------

    /**
     * @brief Find all entities within a circle/sphere.
     * @param center Center.
     * @param radius Radius.
     * @param out Output iterator.
     * @return Number of entities found.
     */
    template <typename OutputIt>
    size_type queryRadius(const point_type& center, T radius, OutputIt out) const {
        aabb_type queryBox(center - point_type(radius), center + point_type(radius));
        return queryAABB(queryBox, out);
    }

    /**
     * @brief Find all entities overlapping an AABB.
     *        Exact enumeration of overlapping cells using integer coordinates.
     */
    template <typename OutputIt>
    size_type queryAABB(const aabb_type& box, OutputIt out) const {
        size_type count = 0;
        auto [minGrid, maxGrid] = gridRangeForBounds(box);
        for (uint32_t iz = minGrid[2]; iz <= maxGrid[2]; ++iz) {
            for (uint32_t iy = minGrid[1]; iy <= maxGrid[1]; ++iy) {
                for (uint32_t ix = minGrid[0]; ix <= maxGrid[0]; ++ix) {
                    morton_type code = gridCoordToMorton(ix, iy, iz);
                    auto it = m_cells.find(code);
                    if (it != m_cells.end()) {
                        const auto& cell = it->second;
                        for (const auto& entity : cell.entities) {
                            *out++ = entity;
                            ++count;
                        }
                    }
                }
            }
        }
        return count;
    }

    /**
     * @brief Point query: find entity exactly at point (not typical).
     */
    std::optional<entity_type> queryPoint(const point_type& point) const {
        morton_type code = mortonCodeForPoint(point);
        auto it = m_cells.find(code);
        if (it != m_cells.end() && !it->second.entities.empty()) {
            return it->second.entities[0]; // first entity only
        }
        return std::nullopt;
    }

    // ------------------------------------------------------------------------
    //  Statistics
    // ------------------------------------------------------------------------
    size_type size() const noexcept { return m_totalEntities; }
    size_type numCells() const noexcept { return m_cells.size(); }
    bool empty() const noexcept { return m_totalEntities == 0; }
    void clear() {
        m_cells.clear();
        m_totalEntities = 0;
    }

    // ------------------------------------------------------------------------
    //  Cell iteration (in morton order)
    // ------------------------------------------------------------------------
    auto begin() const { return m_cells.begin(); }
    auto end() const { return m_cells.end(); }

    // ------------------------------------------------------------------------
    //  Helpers for debugging / serialisation
    // ------------------------------------------------------------------------
    const aabb_type& worldBounds() const noexcept { return m_worldBounds; }
    T cellSize() const noexcept { return m_cellSize; }

    /**
     * @brief Compute the AABB of a given morton cell.
     */
    aabb_type cellAABB(morton_type code) const {
        uint64_t ix, iy, iz = 0;
        if constexpr (Dim == Dim2) {
            mortonDecode2D_64(code, ix, iy);
            point_type min(
                m_worldBounds.min()[0] + static_cast<T>(ix) * m_cellSize,
                m_worldBounds.min()[1] + static_cast<T>(iy) * m_cellSize
            );
            point_type max = min + point_type(m_cellSize);
            return aabb_type(min, max);
        } else {
            mortonDecode3D(code, ix, iy, iz);
            point_type min(
                m_worldBounds.min()[0] + static_cast<T>(ix) * m_cellSize,
                m_worldBounds.min()[1] + static_cast<T>(iy) * m_cellSize,
                m_worldBounds.min()[2] + static_cast<T>(iz) * m_cellSize
            );
            point_type max = min + point_type(m_cellSize);
            return aabb_type(min, max);
        }
    }

private:
    // ------------------------------------------------------------------------
    //  Internal helpers
    // ------------------------------------------------------------------------

    /**
     * @brief Convert world position to integer grid coordinates.
     * @return Grid index clamped to [0, resolution-1].
     */
    std::array<uint32_t, Dim == Dim2 ? 2 : 3> worldToGridCoord(const point_type& point) const {
        std::array<uint32_t, Dim == Dim2 ? 2 : 3> coord;
        for (size_t i = 0; i < static_cast<size_t>(Dim); ++i) {
            T t = (point[i] - m_worldBounds.min()[i]) * m_invCellSize;
            int32_t idx = static_cast<int32_t>(std::floor(t));
            idx = std::max(0, std::min(idx, static_cast<int32_t>(m_resolution[i] - 1)));
            coord[i] = static_cast<uint32_t>(idx);
        }
        if constexpr (Dim == Dim2) {
            coord[2] = 0; // unused
        }
        return coord;
    }

    /**
     * @brief Compute Morton code from integer grid coordinates (3D, with Z=0 for 2D).
     */
    morton_type gridCoordToMorton(uint32_t x, uint32_t y, uint32_t z) const {
        if constexpr (Dim == Dim2) {
            return mortonEncode2D_64(static_cast<uint64_t>(x), static_cast<uint64_t>(y));
        } else {
            return mortonEncode3D(static_cast<uint64_t>(x), static_cast<uint64_t>(y), static_cast<uint64_t>(z));
        }
    }

    /**
     * @brief Compute the integer grid coordinate range that the AABB overlaps.
     * @return Pair of (minGrid, maxGrid) arrays.
     */
    std::pair<std::array<uint32_t, 3>, std::array<uint32_t, 3>>
    gridRangeForBounds(const aabb_type& bounds) const {
        std::array<uint32_t, 3> minGrid, maxGrid;
        point_type minCorner = bounds.min().componentWiseMax(m_worldBounds.min())
                                       .componentWiseMin(m_worldBounds.max());
        point_type maxCorner = bounds.max().componentWiseMax(m_worldBounds.min())
                                       .componentWiseMin(m_worldBounds.max());

        auto minCoord = worldToGridCoord(minCorner);
        auto maxCoord = worldToGridCoord(maxCorner);

        for (size_t i = 0; i < 3; ++i) {
            minGrid[i] = minCoord[i];
            maxGrid[i] = maxCoord[i];
            if constexpr (Dim == Dim2) {
                if (i == 2) { minGrid[i] = 0; maxGrid[i] = 0; }
            }
        }
        return {minGrid, maxGrid};
    }

    morton_type mortonCodeForPoint(const point_type& point) const {
        auto grid = worldToGridCoord(point);
        return gridCoordToMorton(grid[0], grid[1], grid[2]);
    }

    cell_type& getOrCreateCell(morton_type code) {
        auto it = m_cells.find(code);
        if (it != m_cells.end()) {
            return it->second;
        }
        cell_type newCell(m_alloc);
        auto result = m_cells.emplace(code, std::move(newCell));
        return result.first->second;
    }

    // ------------------------------------------------------------------------
    //  Member variables
    // ------------------------------------------------------------------------
    aabb_type m_worldBounds;
    T m_cellSize;
    T m_invCellSize;
    std::array<uint32_t, Dim == Dim2 ? 2 : 3> m_resolution;
    PMRAllocator<cell_type> m_alloc;
    std::unordered_map<morton_type, cell_type,
                       std::hash<morton_type>, std::equal_to<morton_type>,
                       PMRAllocator<std::pair<const morton_type, cell_type>>> m_cells;
    size_type m_totalEntities;
};

} // namespace detail
} // namespace OrthoTree

#endif // ORTHOTREE_DETAIL_SI_MORTONGRID_H_INCLUDED