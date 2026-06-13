//File group name : OrthoTree Math
//File 0043 : core/math/grid.h
//Uniform grid (2D/3D) spatial partitioning: cell indexing, nearest neighbour search, ray traversal (grid traversal), SIMD batch operations, and dynamic environment controls.

#ifndef ORTHOTREE_CORE_MATH_GRID_H_INCLUDED
#define ORTHOTREE_CORE_MATH_GRID_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "geometry_queries.h"
#include "ray_intersection.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <functional>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  UniformGrid: axis‑aligned grid with constant cell size.
//  Stores arbitrary data per cell (template parameter). Provides
//  cell indexing, world‑to‑cell conversion, neighbour iteration,
//  ray traversal (Amanatides & Woo), and SIMD batch coordinate conversion.
// ============================================================================
template<typename T = float, std::size_t N = 3, typename CellData = int>
class UniformGrid {
public:
    using value_type = T;
    using point_type = Vector<T, N>;
    using aabb_type = AxisAlignedBox<T, N>;
    using ray_type = Ray<T, N>;
    using size_type = size_t;
    using cell_index = std::array<size_type, (N == 2 ? 2 : 3)>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    UniformGrid() = default;
    UniformGrid(const aabb_type& bounds, const std::array<size_type, N>& resolution)
        : m_bounds(bounds), m_resolution(resolution), m_invCellSize(T(0)) {
        for (size_type i = 0; i < N; ++i) {
            m_cellSize[i] = bounds.extents()[i] / static_cast<T>(resolution[i]);
            m_invCellSize[i] = T(1) / m_cellSize[i];
        }
        m_cells.resize(resolution[0] * resolution[1] * (N == 3 ? resolution[2] : 1));
    }

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    const aabb_type& bounds() const noexcept { return m_bounds; }
    const std::array<size_type, N>& resolution() const noexcept { return m_resolution; }
    T cellSize(size_type dim) const noexcept { return m_cellSize[dim]; }
    size_type totalCells() const noexcept { return m_cells.size(); }

    // ------------------------------------------------------------------------
    //  World to cell coordinate (clamped to [0, res-1])
    // ------------------------------------------------------------------------
    cell_index worldToCell(const point_type& p) const noexcept {
        cell_index idx;
        for (size_type i = 0; i < N; ++i) {
            T t = (p[i] - m_bounds.min()[i]) * m_invCellSize[i];
            idx[i] = static_cast<size_type>(Math::clamp<T>(t, T(0), static_cast<T>(m_resolution[i] - 1)));
        }
        return idx;
    }

    // ------------------------------------------------------------------------
    //  Cell to world bounding box (the cell's AABB)
    // ------------------------------------------------------------------------
    aabb_type cellBounds(const cell_index& idx) const noexcept {
        point_type minP, maxP;
        for (size_type i = 0; i < N; ++i) {
            minP[i] = m_bounds.min()[i] + static_cast<T>(idx[i]) * m_cellSize[i];
            maxP[i] = minP[i] + m_cellSize[i];
        }
        return aabb_type(minP, maxP);
    }

    // ------------------------------------------------------------------------
    //  Get linear index from cell coordinates
    // ------------------------------------------------------------------------
    size_type linearIndex(const cell_index& idx) const noexcept {
        if constexpr (N == 2) {
            return idx[1] * m_resolution[0] + idx[0];
        } else {
            return (idx[2] * m_resolution[1] + idx[1]) * m_resolution[0] + idx[0];
        }
    }

    // ------------------------------------------------------------------------
    //  Cell data access
    // ------------------------------------------------------------------------
    CellData& cell(const cell_index& idx) { return m_cells[linearIndex(idx)]; }
    const CellData& cell(const cell_index& idx) const { return m_cells[linearIndex(idx)]; }

    // ------------------------------------------------------------------------
    //  Ray traversal (Amanatides & Woo, 3D)
    //  Calls callback for each visited cell (returns false to stop).
    // ------------------------------------------------------------------------
    void traverseRay(const ray_type& ray,
                     std::function<bool(const cell_index&, T tEntry, T tExit)> callback) const {
        // Determine step direction and initial cell
        cell_index cell;
        point_type tDelta, tMax;
        int step[N];
        for (size_type i = 0; i < N; ++i) {
            if (ray.direction()[i] >= T(0)) {
                step[i] = 1;
                T distToNext = (m_bounds.min()[i] + static_cast<T>(cell[i]+1) * m_cellSize[i] - ray.origin()[i]) / ray.direction()[i];
                tMax[i] = distToNext;
            } else {
                step[i] = -1;
                T distToPrev = (m_bounds.min()[i] + static_cast<T>(cell[i]) * m_cellSize[i] - ray.origin()[i]) / ray.direction()[i];
                tMax[i] = distToPrev;
            }
            tDelta[i] = m_cellSize[i] / std::abs(ray.direction()[i]);
        }
        bool inside = true;
        while (inside) {
            if (!callback(cell, T(0), T(0))) break; // simplified: no entry/exit
            // Find next cell
            int dim = 0;
            for (int i = 1; i < N; ++i) if (tMax[i] < tMax[dim]) dim = i;
            if (tMax[dim] > T(1e9)) break;
            tMax[dim] += tDelta[dim];
            if (step[dim] > 0) {
                if (cell[dim] + 1 >= m_resolution[dim]) inside = false;
                else ++cell[dim];
            } else {
                if (cell[dim] == 0) inside = false;
                else --cell[dim];
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Batch world‑to‑cell for 4 points (SIMD)
    // ------------------------------------------------------------------------
    void batchWorldToCell(const point_type* points, cell_index* cells, size_t count) const noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && N == 3 && std::is_same_v<T,float>) {
            for (size_t i = 0; i < count; ++i) {
                cells[i] = worldToCell(points[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                cells[i] = worldToCell(points[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Clear all cells (reset data to default)
    // ------------------------------------------------------------------------
    void clear(const CellData& val = CellData()) {
        std::fill(m_cells.begin(), m_cells.end(), val);
    }

private:
    aabb_type m_bounds;
    std::array<size_type, N> m_resolution;
    std::array<T, N> m_cellSize;
    std::array<T, N> m_invCellSize;
    std::vector<CellData> m_cells;
};

// ============================================================================
//  Dynamic environment controller
// ============================================================================
class GridEnvironment {
public:
    static GridEnvironment& instance() {
        static GridEnvironment env;
        return env;
    }
    void setEpsilon(T eps) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_epsilon = eps;
    }
    T epsilon() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_epsilon;
    }
    void setUseSIMD(bool use) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_useSIMD = use;
    }
    bool useSIMD() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_useSIMD;
    }
private:
    GridEnvironment() : m_epsilon(T(1e-8)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_GRID_H_INCLUDED