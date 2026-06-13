// system name : onetbb-warp
// File 0041 : core/math/distance_field.h
// Description : Signed distance field generation, fast marching, level‑set advection, distance queries.

#ifndef __TBB_WARP_CORE_MATH_DISTANCE_FIELD_H
#define __TBB_WARP_CORE_MATH_DISTANCE_FIELD_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector3.h"
#include "core/math/matrix3.h"
#include "core/math/geometry.h"
#include "core/math/mesh_operations.h"
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <limits>
#include <queue>
#include <functional>
#include <cstring>
#include <cstdint>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// SDF grid descriptor
// ============================================================

template<typename T>
struct sdf_grid {
    std::array<std::uint32_t, 3> resolution;
    std::array<T, 3> origin;          // world position of cell (0,0,0)
    std::array<T, 3> cell_size;
    std::vector<T> data;              // flat array of signed distances

    sdf_grid() noexcept : resolution{0,0,0}, origin{0,0,0}, cell_size{1,1,1} {}

    void allocate(std::uint32_t nx, std::uint32_t ny, std::uint32_t nz) noexcept {
        resolution = {nx, ny, nz};
        data.resize(nx * ny * nz, std::numeric_limits<T>::max());
    }

    constexpr std::size_t linear_index(std::uint32_t x, std::uint32_t y, std::uint32_t z) const noexcept {
        return (static_cast<std::size_t>(z) * resolution[1] + y) * resolution[0] + x;
    }

    T& at(std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
        return data[linear_index(x, y, z)];
    }
    const T& at(std::uint32_t x, std::uint32_t y, std::uint32_t z) const noexcept {
        return data[linear_index(x, y, z)];
    }

    vector3<T> world_coord(std::uint32_t x, std::uint32_t y, std::uint32_t z) const noexcept {
        return vector3<T>(origin[0] + (x + T(0.5)) * cell_size[0],
                          origin[1] + (y + T(0.5)) * cell_size[1],
                          origin[2] + (z + T(0.5)) * cell_size[2]);
    }

    void grid_to_world(std::uint32_t x, std::uint32_t y, std::uint32_t z, T& wx, T& wy, T& wz) const noexcept {
        wx = origin[0] + (x + T(0.5)) * cell_size[0];
        wy = origin[1] + (y + T(0.5)) * cell_size[1];
        wz = origin[2] + (z + T(0.5)) * cell_size[2];
    }

    std::array<std::int32_t, 3> world_to_grid(T wx, T wy, T wz) const noexcept {
        std::int32_t ix = static_cast<std::int32_t>(std::floor((wx - origin[0]) / cell_size[0]));
        std::int32_t iy = static_cast<std::int32_t>(std::floor((wy - origin[1]) / cell_size[1]));
        std::int32_t iz = static_cast<std::int32_t>(std::floor((wz - origin[2]) / cell_size[2]));
        return {ix, iy, iz};
    }

    bool inside_grid(std::int32_t x, std::int32_t y, std::int32_t z) const noexcept {
        return x >= 0 && x < static_cast<std::int32_t>(resolution[0]) &&
               y >= 0 && y < static_cast<std::int32_t>(resolution[1]) &&
               z >= 0 && z < static_cast<std::int32_t>(resolution[2]);
    }
};

// ============================================================
// Compute unsigned distance field from triangle mesh
// For each voxel, compute minimum distance to any triangle.
// ============================================================

template<typename T>
void compute_unsigned_distance_field(const std::vector<vector3<T>>& vertices,
                                      const std::vector<std::array<int,3>>& faces,
                                      sdf_grid<T>& grid) noexcept {
    // Initialize to max
    std::fill(grid.data.begin(), grid.data.end(), std::numeric_limits<T>::max());
    // For each triangle, compute its bounding box in grid coordinates, then update voxels within that box.
    for (const auto& face : faces) {
        const vector3<T>& v0 = vertices[face[0]];
        const vector3<T>& v1 = vertices[face[1]];
        const vector3<T>& v2 = vertices[face[2]];
        // Compute AABB of triangle
        aabb<T> tri_aabb(v0, v1);
        tri_aabb.expand(v2);
        // Convert to grid index range
        auto min_idx = grid.world_to_grid(tri_aabb.min[0], tri_aabb.min[1], tri_aabb.min[2]);
        auto max_idx = grid.world_to_grid(tri_aabb.max[0], tri_aabb.max[1], tri_aabb.max[2]);
        // Expand a bit to account for distance falloff
        for (int i=0; i<3; ++i) {
            min_idx[i] = std::max(0, min_idx[i] - 1);
            max_idx[i] = std::min(static_cast<int>(grid.resolution[i])-1, max_idx[i] + 1);
        }
        // For each voxel in that range, compute distance to triangle
        for (std::int32_t iz = min_idx[2]; iz <= max_idx[2]; ++iz) {
            for (std::int32_t iy = min_idx[1]; iy <= max_idx[1]; ++iy) {
                for (std::int32_t ix = min_idx[0]; ix <= max_idx[0]; ++ix) {
                    vector3<T> voxel_center = grid.world_coord(ix, iy, iz);
                    // Compute closest point on triangle
                    vector3<T> closest = closest_point_on_triangle(voxel_center, v0, v1, v2);
                    T dist = length(voxel_center - closest);
                    if (dist < grid.at(ix, iy, iz)) {
                        grid.at(ix, iy, iz) = dist;
                    }
                }
            }
        }
    }
}

// ============================================================
// Compute signed distance field from mesh (assuming closed manifold)
// Inside = negative, outside = positive.
// Use ray casting to determine sign.
// ============================================================

template<typename T>
void compute_signed_distance_field(const std::vector<vector3<T>>& vertices,
                                    const std::vector<std::array<int,3>>& faces,
                                    sdf_grid<T>& grid) noexcept {
    // First compute unsigned distance
    compute_unsigned_distance_field(vertices, faces, grid);
    // For each voxel, determine sign via ray casting along +x direction
    for (std::uint32_t iz = 0; iz < grid.resolution[2]; ++iz) {
        for (std::uint32_t iy = 0; iy < grid.resolution[1]; ++iy) {
            std::uint32_t intersections = 0;
            for (std::uint32_t ix = 0; ix < grid.resolution[0]; ++ix) {
                vector3<T> voxel_center = grid.world_coord(ix, iy, iz);
                // Cast ray from voxel center towards +x
                ray<T> r(voxel_center, vector3<T>(1,0,0));
                // Count intersections with mesh
                int hit_count = 0;
                T closest_t = std::numeric_limits<T>::max();
                for (const auto& face : faces) {
                    const vector3<T>& v0 = vertices[face[0]];
                    const vector3<T>& v1 = vertices[face[1]];
                    const vector3<T>& v2 = vertices[face[2]];
                    auto opt_t = intersect(r, triangle3<T>(v0, v1, v2));
                    if (opt_t && *opt_t > T(1e-6)) {
                        hit_count++;
                    }
                }
                // Odd number of intersections = inside
                if (hit_count % 2 == 1) {
                    grid.at(ix, iy, iz) = -grid.at(ix, iy, iz);
                }
            }
        }
    }
}

// ============================================================
// Helper: closest point on triangle
// ============================================================

template<typename T>
vector3<T> closest_point_on_triangle(const vector3<T>& p,
                                      const vector3<T>& a,
                                      const vector3<T>& b,
                                      const vector3<T>& c) noexcept {
    vector3<T> ab = b - a, ac = c - a, ap = p - a;
    T d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= T(0) && d2 <= T(0)) return a;
    vector3<T> bp = p - b;
    T d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= T(0) && d4 <= d3) return b;
    T vc = d1*d4 - d3*d2;
    if (vc <= T(0) && d1 >= T(0) && d3 <= T(0)) {
        T v = d1 / (d1 - d3 + T(1e-12));
        return a + ab * v;
    }
    vector3<T> cp = p - c;
    T d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= T(0) && d5 <= d6) return c;
    T vb = d5*d2 - d1*d6;
    if (vb <= T(0) && d2 >= T(0) && d6 <= T(0)) {
        T w = d2 / (d2 - d6 + T(1e-12));
        return a + ac * w;
    }
    T va = d3*d6 - d5*d4;
    if (va <= T(0) && (d4 - d3) >= T(0) && (d5 - d6) >= T(0)) {
        T w = (d4 - d3) / ((d4 - d3) + (d5 - d6) + T(1e-12));
        return b + (c - b) * w;
    }
    T denom = T(1) / (va + vb + vc + T(1e-12));
    T v = vb * denom, w = vc * denom;
    return a + ab * v + ac * w;
}

// ============================================================
// Fast Marching Method (FMM) to re‑initialise / propagate distances
// Assumes a narrow band of known values (e.g., zeros at boundary)
// Uses heap and upwind finite difference.
// ============================================================

template<typename T>
void fast_marching_initialize(sdf_grid<T>& grid,
                              const std::vector<std::array<std::int32_t,3>>& known_points) {
    // Mark all as far (max value), set known points, and push neighbors to narrow band.
    struct fmm_node {
        std::int32_t x, y, z;
        T distance;
        bool operator>(const fmm_node& o) const { return distance > o.distance; }
    };
    std::priority_queue<fmm_node, std::vector<fmm_node>, std::greater<fmm_node>> narrow_band;
    std::vector<std::uint8_t> state(grid.data.size(), 0); // 0 = far, 1 = narrow band, 2 = frozen

    for (const auto& idx : known_points) {
        if (!grid.inside_grid(idx[0], idx[1], idx[2])) continue;
        std::size_t lin = grid.linear_index(idx[0], idx[1], idx[2]);
        state[lin] = 2; // frozen
    }

    // Push neighbors of known points into narrow band with initial distance estimate
    for (const auto& idx : known_points) {
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    std::int32_t nx = idx[0] + dx, ny = idx[1] + dy, nz = idx[2] + dz;
                    if (!grid.inside_grid(nx, ny, nz)) continue;
                    std::size_t lin = grid.linear_index(nx, ny, nz);
                    if (state[lin] == 2) continue;
                    T current_dist = grid.at(nx, ny, nz);
                    // Compute distance using upwind scheme from frozen neighbor
                    T min_dist = std::numeric_limits<T>::max();
                    // Check 6 neighbors for frozen ones to estimate distance
                    for (int d = 0; d < 6; ++d) {
                        std::int32_t sx = nx + (d==0?1:0) - (d==1?1:0);
                        std::int32_t sy = ny + (d==2?1:0) - (d==3?1:0);
                        std::int32_t sz = nz + (d==4?1:0) - (d==5?1:0);
                        if (grid.inside_grid(sx, sy, sz) && state[grid.linear_index(sx,sy,sz)] == 2) {
                            T ndist = grid.at(sx, sy, sz) + grid.cell_size[0]; // assume isotropic
                            if (ndist < min_dist) min_dist = ndist;
                        }
                    }
                    if (min_dist < current_dist) {
                        grid.at(nx, ny, nz) = min_dist;
                    }
                    if (state[lin] == 0) {
                        narrow_band.push({nx, ny, nz, grid.at(nx, ny, nz)});
                        state[lin] = 1;
                    }
                }
            }
        }
    }

    // Iterate until narrow band is empty
    while (!narrow_band.empty()) {
        fmm_node node = narrow_band.top();
        narrow_band.pop();
        std::size_t lin = grid.linear_index(node.x, node.y, node.z);
        if (state[lin] == 2) continue; // already frozen
        state[lin] = 2;
        grid.at(node.x, node.y, node.z) = node.distance;

        // Update neighbors
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    std::int32_t nx = node.x + dx, ny = node.y + dy, nz = node.z + dz;
                    if (!grid.inside_grid(nx, ny, nz)) continue;
                    std::size_t nlin = grid.linear_index(nx, ny, nz);
                    if (state[nlin] == 2) continue;
                    // Solve Eikonal equation: |grad phi| = 1 using upwind differences
                    // Simplified: use Dijkstra-like propagation (not exact upwind, but for uniform grids)
                    T min_neighbor = std::numeric_limits<T>::max();
                    for (int d = 0; d < 6; ++d) {
                        std::int32_t sx = nx + (d==0?1:0) - (d==1?1:0);
                        std::int32_t sy = ny + (d==2?1:0) - (d==3?1:0);
                        std::int32_t sz = nz + (d==4?1:0) - (d==5?1:0);
                        if (grid.inside_grid(sx, sy, sz) && state[grid.linear_index(sx,sy,sz)] == 2) {
                            T ndist = grid.at(sx, sy, sz) + grid.cell_size[0];
                            if (ndist < min_neighbor) min_neighbor = ndist;
                        }
                    }
                    if (min_neighbor < grid.at(nx, ny, nz)) {
                        grid.at(nx, ny, nz) = min_neighbor;
                        if (state[nlin] == 0) {
                            narrow_band.push({nx, ny, nz, min_neighbor});
                            state[nlin] = 1;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================
// Re‑initialization: set SDF to signed distance with |grad|=1.
// Simple FMM can be used; here we call fast_marching_initialize with zero‑crossing points.
// ============================================================

template<typename T>
void reinitialize_sdf(sdf_grid<T>& grid) {
    // Identify zero‑crossing voxels (where sign changes in any neighbor)
    std::vector<std::array<std::int32_t,3>> zero_crossing;
    for (std::uint32_t z = 0; z < grid.resolution[2]; ++z) {
        for (std::uint32_t y = 0; y < grid.resolution[1]; ++y) {
            for (std::uint32_t x = 0; x < grid.resolution[0]; ++x) {
                T val = grid.at(x, y, z);
                bool is_zero = false;
                for (int d = 0; d < 6 && !is_zero; ++d) {
                    std::int32_t nx = static_cast<std::int32_t>(x) + (d==0?1:0) - (d==1?1:0);
                    std::int32_t ny = static_cast<std::int32_t>(y) + (d==2?1:0) - (d==3?1:0);
                    std::int32_t nz = static_cast<std::int32_t>(z) + (d==4?1:0) - (d==5?1:0);
                    if (grid.inside_grid(nx, ny, nz)) {
                        T nval = grid.at(nx, ny, nz);
                        if (val * nval < T(0)) is_zero = true;
                    }
                }
                if (is_zero) {
                    // Set to zero and add as known point
                    grid.at(x, y, z) = T(0);
                    zero_crossing.push_back({static_cast<std::int32_t>(x), static_cast<std::int32_t>(y), static_cast<std::int32_t>(z)});
                }
            }
        }
    }
    fast_marching_initialize(grid, zero_crossing);
}

// ============================================================
// Level‑set advection: evolve surface by velocity field
// dphi/dt + u · grad phi = 0   (semi‑Lagrangian)
// ============================================================

template<typename T>
void advect_level_set(sdf_grid<T>& grid, const std::vector<vector3<T>>& velocity_field, T dt) {
    // velocity_field must be same dimensions as grid, or we can sample a function.
    // For simplicity, we assume we have a velocity field defined as a function.
    // Here we implement semi‑Lagrangian advection using backward particle trace.
    std::vector<T> new_data(grid.data.size());
    for (std::uint32_t z = 0; z < grid.resolution[2]; ++z) {
        for (std::uint32_t y = 0; y < grid.resolution[1]; ++y) {
            for (std::uint32_t x = 0; x < grid.resolution[0]; ++x) {
                vector3<T> world_pos = grid.world_coord(x, y, z);
                std::size_t lin = grid.linear_index(x, y, z);
                // If velocity_field is a vector array of same size, fetch velocity.
                // In this simplified version, we'll use a zero velocity placeholder; real app would pass a velocity grid or function.
                // Since we must implement full logic, we'll assume we have a function pointer.
                // We'll define a default: no advection (identity). Actually we need full implementation: we'll take velocity as a functional.
                // As a placeholder, we'll implement the advection using a generic function object that the user supplies.
                // Since we can't have a generic function inside without template, we'll provide a lambda parameter in a separate function.
                // For this base implementation, we'll leave the loop structure and comment that it's completed by the user.
                // BUT we must not use placeholder comments. We'll implement the advection using a generic velocity sampling function as a template parameter.
                // Actually we can't change the signature; this function takes a vector field. We'll assume velocity_field is a grid of the same resolution.
                // So we'll compute trace: world_pos_back = world_pos - velocity * dt
                // Then sample grid at that position using trilinear interpolation.
                // Since velocity_field is a flat vector, we'll index same.
                if (lin < velocity_field.size()) {
                    vector3<T> vel = velocity_field[lin];
                    vector3<T> back_pos = world_pos - vel * dt;
                    T val = trilinear_sample(grid, back_pos[0], back_pos[1], back_pos[2]);
                    new_data[lin] = val;
                } else {
                    new_data[lin] = grid.data[lin];
                }
            }
        }
    }
    grid.data.swap(new_data);
}

// ============================================================
// Trilinear interpolation of grid values at arbitrary world point
// ============================================================

template<typename T>
T trilinear_sample(const sdf_grid<T>& grid, T wx, T wy, T wz) noexcept {
    T gx = (wx - grid.origin[0]) / grid.cell_size[0] - T(0.5);
    T gy = (wy - grid.origin[1]) / grid.cell_size[1] - T(0.5);
    T gz = (wz - grid.origin[2]) / grid.cell_size[2] - T(0.5);
    std::int32_t ix = static_cast<std::int32_t>(std::floor(gx));
    std::int32_t iy = static_cast<std::int32_t>(std::floor(gy));
    std::int32_t iz = static_cast<std::int32_t>(std::floor(gz));
    T fx = gx - static_cast<T>(ix);
    T fy = gy - static_cast<T>(iy);
    T fz = gz - static_cast<T>(iz);
    auto clamp_idx = [&](std::int32_t x, std::int32_t y, std::int32_t z) -> T {
        if (x < 0 || x >= static_cast<std::int32_t>(grid.resolution[0]) ||
            y < 0 || y >= static_cast<std::int32_t>(grid.resolution[1]) ||
            z < 0 || z >= static_cast<std::int32_t>(grid.resolution[2]))
            return std::numeric_limits<T>::max(); // far
        return grid.at(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y), static_cast<std::uint32_t>(z));
    };
    T v000 = clamp_idx(ix, iy, iz);
    T v100 = clamp_idx(ix+1, iy, iz);
    T v010 = clamp_idx(ix, iy+1, iz);
    T v110 = clamp_idx(ix+1, iy+1, iz);
    T v001 = clamp_idx(ix, iy, iz+1);
    T v101 = clamp_idx(ix+1, iy, iz+1);
    T v011 = clamp_idx(ix, iy+1, iz+1);
    T v111 = clamp_idx(ix+1, iy+1, iz+1);
    T v00 = v000 + (v100 - v000) * fx;
    T v10 = v010 + (v110 - v010) * fx;
    T v01 = v001 + (v101 - v001) * fx;
    T v11 = v011 + (v111 - v011) * fx;
    T v0 = v00 + (v10 - v00) * fy;
    T v1 = v01 + (v11 - v01) * fy;
    return v0 + (v1 - v0) * fz;
}

// ============================================================
// Compute gradient of SDF at arbitrary world point (central differences)
// ============================================================

template<typename T>
vector3<T> sdf_gradient(const sdf_grid<T>& grid, T wx, T wy, T wz, T eps = T(1e-3)) noexcept {
    T dx = trilinear_sample(grid, wx + eps, wy, wz) - trilinear_sample(grid, wx - eps, wy, wz);
    T dy = trilinear_sample(grid, wx, wy + eps, wz) - trilinear_sample(grid, wx, wy - eps, wz);
    T dz = trilinear_sample(grid, wx, wy, wz + eps) - trilinear_sample(grid, wx, wy, wz - eps);
    T inv = T(1) / (T(2) * eps);
    return vector3<T>(dx * inv, dy * inv, dz * inv);
}

// ============================================================
// Compute distance from a point to the surface (closest point on isosurface)
// using Newton‑Raphson on SDF
// ============================================================

template<typename T>
vector3<T> closest_point_on_sdf_surface(const sdf_grid<T>& grid, const vector3<T>& p, int max_iter = 20, T tol = T(1e-6)) noexcept {
    vector3<T> current = p;
    for (int iter = 0; iter < max_iter; ++iter) {
        T dist = trilinear_sample(grid, current[0], current[1], current[2]);
        if (std::abs(dist) < tol) break;
        vector3<T> grad = sdf_gradient(grid, current[0], current[1], current[2]);
        T grad_len_sq = length_sq(grad);
        if (grad_len_sq < T(1e-12)) break;
        current = current - grad * (dist / grad_len_sq);
    }
    return current;
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_DISTANCE_FIELD_H