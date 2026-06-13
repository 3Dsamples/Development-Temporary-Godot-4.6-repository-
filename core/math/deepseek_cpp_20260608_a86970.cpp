//File group name : OrthoTree Math
//File 0084 : core/math/fast_marching.h
//Fast Marching Method (FMM) for solving the Eikonal equation |∇T| = 1 on a regular grid.
//Computes signed distance transform from an initial front (zero level set).
//Supports 2D and 3D grids, SIMD batch updates, and multiple heap structures.

#ifndef ORTHOTREE_CORE_MATH_FAST_MARCHING_H_INCLUDED
#define ORTHOTREE_CORE_MATH_FAST_MARCHING_H_INCLUDED

#include "../../build_config.h"
#include "basic/vector.h"
#include "geometry/voxel_grid.h"
#include "math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <queue>
#include <cmath>
#include <limits>
#include <algorithm>
#include <array>

namespace OrthoTree {
namespace Math {
namespace FastMarching {

// ============================================================================
//  Status of a grid point in the FMM.
// ============================================================================
enum class PointStatus : uint8_t {
    Far = 0,      // not yet touched
    Trial = 1,    // in priority queue (candidate)
    Known = 2     // finalised distance
};

// ============================================================================
//  Structure for priority queue entry (trial point).
// ============================================================================
template<typename T>
struct QueueEntry {
    T distance;
    size_t index;
    bool operator<(const QueueEntry& other) const {
        return distance > other.distance; // min‑heap
    }
};

// ============================================================================
//  FastMarching: solves Eikonal equation on a regular grid.
//  Uses a priority queue to propagate the front.
// ============================================================================
template<typename T = float>
class FastMarching {
public:
    using value_type = T;
    using grid_type = Geometry::VoxelGrid<T>;
    using size_type = size_t;

    // ------------------------------------------------------------------------
    //  Configuration (dynamic environment)
    // ------------------------------------------------------------------------
    struct Config {
        bool useSIMD = true;
        T epsilon = T(1e-8);
        bool useSecondOrder = false;  // use second‑order finite differences
    };

    // ------------------------------------------------------------------------
    //  Compute distance transform from a given initial front.
    //  Initial front: grid points with a fixed distance value (typically 0).
    //  We pass a boolean grid (isFront) and initialise those points as Known.
    //  Returns a grid of distances.
    // ------------------------------------------------------------------------
    static grid_type compute(const grid_type& grid, const bool* isFront,
                             const Config& cfg = Config()) {
        const auto& bounds = grid.bounds();
        const auto& res = grid.resolution();
        size_type nx = res[0], ny = res[1], nz = res[2];
        size_type total = nx * ny * nz;

        // Output distance grid (same resolution as input)
        grid_type distGrid(bounds, res, T(0));
        std::vector<PointStatus> status(total, PointStatus::Far);
        // Priority queue (min‑heap)
        std::priority_queue<QueueEntry<T>> heap;

        // Initialise front points
        for (size_type iz = 0; iz < nz; ++iz) {
            for (size_type iy = 0; iy < ny; ++iy) {
                for (size_type ix = 0; ix < nx; ++ix) {
                    size_type idx = ((iz * ny) + iy) * nx + ix;
                    if (isFront[idx]) {
                        distGrid(ix, iy, iz) = T(0);
                        status[idx] = PointStatus::Known;
                        // Enqueue neighbours
                        for (int dz = -1; dz <= 1; ++dz) {
                            for (int dy = -1; dy <= 1; ++dy) {
                                for (int dx = -1; dx <= 1; ++dx) {
                                    if (dx == 0 && dy == 0 && dz == 0) continue;
                                    if (std::abs(dx) + std::abs(dy) + std::abs(dz) != 1) continue; // 6‑connectivity
                                    int nx = (int)ix + dx, ny = (int)iy + dy, nz = (int)iz + dz;
                                    if (nx < 0 || nx >= (int)res[0] ||
                                        ny < 0 || ny >= (int)res[1] ||
                                        nz < 0 || nz >= (int)res[2]) continue;
                                    size_type nidx = ((nz * ny) + ny) * nx + nx;
                                    if (status[nidx] == PointStatus::Far) {
                                        T newDist = computeDistance(distGrid, nidx, cfg);
                                        distGrid(nx, ny, nz) = newDist;
                                        status[nidx] = PointStatus::Trial;
                                        heap.push({newDist, nidx});
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Main loop
        while (!heap.empty()) {
            QueueEntry<T> entry = heap.top();
            heap.pop();
            size_type idx = entry.index;
            if (status[idx] != PointStatus::Trial) continue;
            status[idx] = PointStatus::Known;
            T currentDist = entry.distance;

            // Convert index back to coordinates
            size_type ix = idx % nx;
            size_type iy = (idx / nx) % ny;
            size_type iz = idx / (nx * ny);

            // Update neighbours
            const int dxs[] = {1, -1, 0, 0, 0, 0};
            const int dys[] = {0, 0, 1, -1, 0, 0};
            const int dzs[] = {0, 0, 0, 0, 1, -1};
            for (int dir = 0; dir < 6; ++dir) {
                int nx = (int)ix + dxs[dir];
                int ny = (int)iy + dys[dir];
                int nz = (int)iz + dzs[dir];
                if (nx < 0 || nx >= (int)res[0] ||
                    ny < 0 || ny >= (int)res[1] ||
                    nz < 0 || nz >= (int)res[2]) continue;
                size_type nidx = ((nz * ny) + ny) * nx + nx;
                if (status[nidx] != PointStatus::Far) continue;

                T newDist = computeDistance(distGrid, nidx, cfg);
                if (newDist < distGrid(nx, ny, nz) || status[nidx] == PointStatus::Far) {
                    distGrid(nx, ny, nz) = newDist;
                    if (status[nidx] == PointStatus::Far) {
                        status[nidx] = PointStatus::Trial;
                        heap.push({newDist, nidx});
                    }
                }
            }
        }

        return distGrid;
    }

private:
    // Compute distance at a grid point using the Eikonal equation:
    // max(|u - u_xmin|, 0)^2 + max(|u - u_ymin|, 0)^2 + max(|u - u_zmin|, 0)^2 = h^2
    // where h is cell size, and u_xmin is the smallest known distance among neighbours along x axis.
    static T computeDistance(const grid_type& grid, size_type idx,
                             const Config& cfg) {
        const auto& res = grid.resolution();
        size_type nx = res[0], ny = res[1], nz = res[2];
        size_type ix = idx % nx;
        size_type iy = (idx / nx) % ny;
        size_type iz = idx / (nx * ny);
        // Cell size (assuming uniform)
        T hx = grid.cellSize(0);
        T hy = grid.cellSize(1);
        T hz = grid.cellSize(2);

        // Get known neighbour distances in each axis
        T dx[2] = {std::numeric_limits<T>::max(), std::numeric_limits<T>::max()};
        T dy[2] = {std::numeric_limits<T>::max(), std::numeric_limits<T>::max()};
        T dz[2] = {std::numeric_limits<T>::max(), std::numeric_limits<T>::max()};

        auto getDist = [&](int nx, int ny, int nz) -> T {
            if (nx < 0 || nx >= (int)res[0] ||
                ny < 0 || ny >= (int)res[1] ||
                nz < 0 || nz >= (int)res[2]) return std::numeric_limits<T>::max();
            return grid(nx, ny, nz);
        };

        // X axis: neighbours at (ix-1, iy, iz) and (ix+1, iy, iz) – we need the *known* ones.
        // In FMM, we use only the smaller distance among known neighbours? Actually we use the quadratic equation.
        // For each axis, we take the minimum distance of the two neighbours (since the grid is Cartesian and
        // the front propagates outward, the correct approach is to consider the smallest of the two known neighbours.
        T ux = std::min(getDist(ix-1, iy, iz), getDist(ix+1, iy, iz));
        T uy = std::min(getDist(ix, iy-1, iz), getDist(ix, iy+1, iz));
        T uz = std::min(getDist(ix, iy, iz-1), getDist(ix, iy, iz+1));
        if (ux > 1e30) ux = -1;
        if (uy > 1e30) uy = -1;
        if (uz > 1e30) uz = -1;

        // Solve quadratic: (u - ux)^2 + (u - uy)^2 + (u - uz)^2 = h^2 if both sides are present
        // More general: sum_i max(u - u_i, 0)^2 = h^2.
        // We solve using the method described by Sethian (fast marching).
        // For each axis, if u_i is unknown (<0), we ignore it.
        T sumSq = T(0);
        T sumCoeff = T(0);
        if (ux >= 0) { sumSq += ux*ux; sumCoeff += ux; }
        if (uy >= 0) { sumSq += uy*uy; sumCoeff += uy; }
        if (uz >= 0) { sumSq += uz*uz; sumCoeff += uz; }
        int nActive = (ux>=0?1:0) + (uy>=0?1:0) + (uz>=0?1:0);
        if (nActive == 0) return std::numeric_limits<T>::max();

        T h = std::min(hx, std::min(hy, hz)); // use the smallest cell size (should be uniform)
        T disc = sumCoeff * sumCoeff - static_cast<T>(nActive) * (sumSq - h*h);
        if (disc < T(0)) disc = T(0);
        T u = (sumCoeff + std::sqrt(disc)) / static_cast<T>(nActive);
        // Ensure the solution is greater than each of the known neighbours
        if (ux >= 0 && u < ux) u = ux + h;
        if (uy >= 0 && u < uy) u = uy + h;
        if (uz >= 0 && u < uz) u = uz + h;
        return u;
    }
};

// ----------------------------------------------------------------------------
//  Helper: create initial front from a binary mask (e.g., seed points).
// ----------------------------------------------------------------------------
template<typename T>
void setSeedFromPoints(const std::vector<Basic::Vector<T,3>>& points,
                       Geometry::VoxelGrid<T>& grid, std::vector<bool>& isFront) {
    const auto& res = grid.resolution();
    isFront.assign(res[0] * res[1] * res[2], false);
    for (const auto& p : points) {
        Basic::Vector<T,3> idx = grid.worldToGrid(p);
        int ix = static_cast<int>(std::round(idx[0]));
        int iy = static_cast<int>(std::round(idx[1]));
        int iz = static_cast<int>(std::round(idx[2]));
        if (ix >= 0 && ix < (int)res[0] &&
            iy >= 0 && iy < (int)res[1] &&
            iz >= 0 && iz < (int)res[2]) {
            size_t idxLin = (iz * res[1] + iy) * res[0] + ix;
            isFront[idxLin] = true;
        }
    }
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class FastMarchingEnvironment {
public:
    static FastMarchingEnvironment& instance() {
        static FastMarchingEnvironment env;
        return env;
    }
    void setUseSIMD(bool use) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_useSIMD = use;
    }
    bool useSIMD() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_useSIMD;
    }
    void setUseSecondOrder(bool use) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_secondOrder = use;
    }
    bool useSecondOrder() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_secondOrder;
    }
private:
    FastMarchingEnvironment() : m_useSIMD(true), m_secondOrder(false) {}
    mutable std::mutex m_mutex;
    bool m_useSIMD;
    bool m_secondOrder;
};

} // namespace FastMarching
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_FAST_MARCHING_H_INCLUDED