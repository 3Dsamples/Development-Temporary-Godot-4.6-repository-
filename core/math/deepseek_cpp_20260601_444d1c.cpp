//4/40
//File 0083 : core/math/surface_reconstruction.h
//Poisson surface reconstruction from oriented point clouds: uniform grid Laplacian assembly, divergence computation, conjugate gradient solver, and full marching cubes iso‑surface extraction with complete tables.
#ifndef CORE_MATH_SURFACE_RECONSTRUCTION_H
#define CORE_MATH_SURFACE_RECONSTRUCTION_H

#include "vector_math.h"
#include "geometry_primitives.h"
#include "math_constants.h"
#include "linear_algebra.h"            // Eigen types and sparse solvers
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <cstring>

namespace SimulationMath {
namespace surface_reconstruction {

// -----------------------------------------------------------------------------
// 1. Compute bounding box of a point cloud (min, max)
// -----------------------------------------------------------------------------
inline void compute_bbox(const std::vector<DirectX::XMVECTOR>& points,
                         DirectX::XMVECTOR& out_min, DirectX::XMVECTOR& out_max) noexcept {
    if (points.empty()) return;
    out_min = points[0]; out_max = points[0];
    for (const auto& p : points) {
        out_min = DirectX::XMVectorMin(out_min, p);
        out_max = DirectX::XMVectorMax(out_max, p);
    }
}

// -----------------------------------------------------------------------------
// 2. Poisson reconstruction on a uniform grid (fully implemented)
//    - points / normals : input
//    - voxel_size        : edge length of a grid cell
//    - out_indicator     : scalar field (size = nx*ny*nz)
//    - out_origin        : coordinates of grid corner (min corner)
//    - out_nx,out_ny,out_nz : grid dimensions
// -----------------------------------------------------------------------------
inline void uniform_poisson_reconstruction(
    const std::vector<DirectX::XMVECTOR>& points,
    const std::vector<DirectX::XMVECTOR>& normals,
    float voxel_size,
    std::vector<float>& out_indicator,
    DirectX::XMVECTOR& out_origin,
    int& out_nx, int& out_ny, int& out_nz) noexcept
{
    if (points.empty()) return;

    DirectX::XMVECTOR bb_min, bb_max;
    compute_bbox(points, bb_min, bb_max);
    // Expand bounding box slightly
    bb_min = DirectX::XMVectorSubtract(bb_min, DirectX::XMVectorReplicate(voxel_size * 0.5f));
    bb_max = DirectX::XMVectorAdd(bb_max, DirectX::XMVectorReplicate(voxel_size * 0.5f));
    out_origin = bb_min;

    float size_x = vector_math::get_x(bb_max) - vector_math::get_x(bb_min);
    float size_y = vector_math::get_y(bb_max) - vector_math::get_y(bb_min);
    float size_z = vector_math::get_z(bb_max) - vector_math::get_z(bb_min);
    out_nx = std::max(2, (int)(size_x / voxel_size) + 1);
    out_ny = std::max(2, (int)(size_y / voxel_size) + 1);
    out_nz = std::max(2, (int)(size_z / voxel_size) + 1);
    int nx = out_nx, ny = out_ny, nz = out_nz;
    size_t total_cells = (size_t)nx * ny * nz;

    // Mark active cells: those that contain at least one point or are adjacent to one.
    std::vector<bool> active(total_cells, false);
    for (size_t i = 0; i < points.size(); ++i) {
        float x = vector_math::get_x(DirectX::XMVectorSubtract(points[i], bb_min)) / voxel_size;
        float y = vector_math::get_y(DirectX::XMVectorSubtract(points[i], bb_min)) / voxel_size;
        float z = vector_math::get_z(DirectX::XMVectorSubtract(points[i], bb_min)) / voxel_size;
        int ix = std::clamp((int)x, 0, nx-1);
        int iy = std::clamp((int)y, 0, ny-1);
        int iz = std::clamp((int)z, 0, nz-1);
        active[(size_t)ix + nx*((size_t)iy + ny*(size_t)iz)] = true;
    }
    // Expand active mask to include 1‑ring neighborhood
    std::vector<bool> active2 = active;
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                size_t idx = (size_t)ix + nx*((size_t)iy + ny*(size_t)iz);
                if (active[idx]) {
                    for (int dz = -1; dz <= 1; ++dz)
                        for (int dy = -1; dy <= 1; ++dy)
                            for (int dx = -1; dx <= 1; ++dx) {
                                int nix = ix+dx, niy = iy+dy, niz = iz+dz;
                                if (nix>=0 && nix<nx && niy>=0 && niy<ny && niz>=0 && niz<nz)
                                    active2[(size_t)nix + nx*((size_t)niy + ny*(size_t)niz)] = true;
                            }
                }
            }
        }
    }
    active = active2;

    // Map active cells to contiguous indices
    std::vector<int> cell_to_var(total_cells, -1);
    int var_count = 0;
    for (size_t i = 0; i < total_cells; ++i) {
        if (active[i]) cell_to_var[i] = var_count++;
    }

    // Accumulate average normal per active cell
    std::vector<float> avgNx(var_count, 0.0f), avgNy(var_count, 0.0f), avgNz(var_count, 0.0f);
    std::vector<float> cell_weight(var_count, 0.0f);
    for (size_t i = 0; i < points.size(); ++i) {
        float x = vector_math::get_x(DirectX::XMVectorSubtract(points[i], bb_min)) / voxel_size;
        float y = vector_math::get_y(DirectX::XMVectorSubtract(points[i], bb_min)) / voxel_size;
        float z = vector_math::get_z(DirectX::XMVectorSubtract(points[i], bb_min)) / voxel_size;
        int ix = std::clamp((int)x, 0, nx-1);
        int iy = std::clamp((int)y, 0, ny-1);
        int iz = std::clamp((int)z, 0, nz-1);
        size_t cid = (size_t)ix + nx*((size_t)iy + ny*(size_t)iz);
        if (!active[cid]) continue;
        int vi = cell_to_var[cid];
        avgNx[vi] += vector_math::get_x(normals[i]);
        avgNy[vi] += vector_math::get_y(normals[i]);
        avgNz[vi] += vector_math::get_z(normals[i]);
        cell_weight[vi] += 1.0f;
    }
    for (int i = 0; i < var_count; ++i) {
        if (cell_weight[i] > 0.0f) {
            float inv = 1.0f / cell_weight[i];
            avgNx[i] *= inv;
            avgNy[i] *= inv;
            avgNz[i] *= inv;
        }
    }

    // Helper to get average normal component at a cell (returns 0 outside active)
    auto get_Nx = [&](int ix, int iy, int iz) -> float {
        if (ix<0||ix>=nx||iy<0||iy>=ny||iz<0||iz>=nz) return 0.0f;
        size_t cid = (size_t)ix + nx*((size_t)iy + ny*(size_t)iz);
        if (!active[cid]) return 0.0f;
        return avgNx[cell_to_var[cid]];
    };
    auto get_Ny = [&](int ix, int iy, int iz) -> float {
        if (ix<0||ix>=nx||iy<0||iy>=ny||iz<0||iz>=nz) return 0.0f;
        size_t cid = (size_t)ix + nx*((size_t)iy + ny*(size_t)iz);
        if (!active[cid]) return 0.0f;
        return avgNy[cell_to_var[cid]];
    };
    auto get_Nz = [&](int ix, int iy, int iz) -> float {
        if (ix<0||ix>=nx||iy<0||iy>=ny||iz<0||iz>=nz) return 0.0f;
        size_t cid = (size_t)ix + nx*((size_t)iy + ny*(size_t)iz);
        if (!active[cid]) return 0.0f;
        return avgNz[cell_to_var[cid]];
    };

    // Assemble Laplacian (7‑point stencil) and divergence vector b
    std::vector<Eigen::Triplet<float>> triplets;
    Eigen::VectorXf b(var_count);
    b.setZero();
    float inv_h2 = 1.0f / (voxel_size * voxel_size);
    float inv_2h = 0.5f / voxel_size;
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                size_t cid = (size_t)ix + nx*((size_t)iy + ny*(size_t)iz);
                if (!active[cid]) continue;
                int vi = cell_to_var[cid];
                // Laplacian stencil
                const int neigh[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
                for (int k = 0; k < 6; ++k) {
                    int nx_ = ix + neigh[k][0];
                    int ny_ = iy + neigh[k][1];
                    int nz_ = iz + neigh[k][2];
                    if (nx_>=0&&nx_<nx&&ny_>=0&&ny_<ny&&nz_>=0&&nz_<nz) {
                        size_t nid = (size_t)nx_ + nx*((size_t)ny_ + ny*(size_t)nz_);
                        if (active[nid]) {
                            int vj = cell_to_var[nid];
                            triplets.emplace_back(vi, vj, -inv_h2);
                        }
                    }
                }
                triplets.emplace_back(vi, vi, 6.0f * inv_h2);

                // Divergence: central differences of the averaged normal field
                float div = 0.0f;
                // dNx/dx
                float Nx_r = get_Nx(ix+1, iy, iz);
                float Nx_l = get_Nx(ix-1, iy, iz);
                if (ix == 0) div += (Nx_r - get_Nx(ix,iy,iz)) / voxel_size;
                else if (ix == nx-1) div += (get_Nx(ix,iy,iz) - Nx_l) / voxel_size;
                else div += (Nx_r - Nx_l) * inv_2h;
                // dNy/dy
                float Ny_r = get_Ny(ix, iy+1, iz);
                float Ny_l = get_Ny(ix, iy-1, iz);
                if (iy == 0) div += (Ny_r - get_Ny(ix,iy,iz)) / voxel_size;
                else if (iy == ny-1) div += (get_Ny(ix,iy,iz) - Ny_l) / voxel_size;
                else div += (Ny_r - Ny_l) * inv_2h;
                // dNz/dz
                float Nz_r = get_Nz(ix, iy, iz+1);
                float Nz_l = get_Nz(ix, iy, iz-1);
                if (iz == 0) div += (Nz_r - get_Nz(ix,iy,iz)) / voxel_size;
                else if (iz == nz-1) div += (get_Nz(ix,iy,iz) - Nz_l) / voxel_size;
                else div += (Nz_r - Nz_l) * inv_2h;

                b[vi] = div;
            }
        }
    }

    // Solve sparse linear system
    Eigen::SparseMatrix<float> A(var_count, var_count);
    A.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower|Eigen::Upper> cg;
    cg.compute(A);
    cg.setTolerance(1e-6f);
    cg.setMaxIterations(500);
    Eigen::VectorXf chi = cg.solve(b);

    // Map solution back to full grid
    out_indicator.assign(total_cells, 0.0f);
    for (size_t i = 0; i < total_cells; ++i) {
        if (active[i]) out_indicator[i] = chi[cell_to_var[i]];
    }
}

// -----------------------------------------------------------------------------
// 3. Marching Cubes lookup tables (full 256 entries)
// -----------------------------------------------------------------------------
namespace marching_cubes {

    // Edge table: which edges are intersected for each of the 256 cases (12 bits)
    inline constexpr int edge_table[256] = {
        0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
        0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
        0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
        0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
        0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
        0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
        0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
        0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
        0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
        0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
        0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
        0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
        0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
        0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
        0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
        0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
        0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
        0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
        0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
        0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
        0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
        0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
        0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
        0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
        0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
        0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
        0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
        0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
        0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
        0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
        0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
        0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
    };

    // Triangle table: for each case, list of edge indices to connect into triangles; terminated by -1.
    // Edge numbering: 0: (0,0,0)-(1,0,0), 1: (1,0,0)-(1,1,0), 2: (0,1,0)-(1,1,0), 3: (0,0,0)-(0,1,0),
    // 4: (0,0,1)-(1,0,1), 5: (1,0,1)-(1,1,1), 6: (0,1,1)-(1,1,1), 7: (0,0,1)-(0,1,1),
    // 8: (0,0,0)-(0,0,1), 9: (1,0,0)-(1,0,1), 10: (1,1,0)-(1,1,1), 11: (0,1,0)-(0,1,1).
    // The table entries are signed integers (the edge indices).
    // We'll store as int16_t to save space; but for simplicity, int.
    inline constexpr int tri_table[256][16] = {
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
        {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
        {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
        {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
        {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
        {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
        {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
        {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
        {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
        {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
        {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
        {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
        {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
        {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
        {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
        {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
        {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
        {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
        {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
        {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
        {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
        {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
        {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
        {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
        {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
        {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
        {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
        {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 2, 5, -1, -1, -1, -1},
        {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
        {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
        {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
        {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
        {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
        {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
        {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
        {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
        {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
        {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
        {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
        {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
        {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
        {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
        {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
        {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
        {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
        {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
        {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
        {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
        {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
        {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
        {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
        {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
        {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
        {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
        {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
        {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
        {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
        {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
        {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
        {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
        {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
        {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
        {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
        {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
        {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
        {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
        {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
        {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
        {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
        {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
        {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
        {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
        {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
        {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
        {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
        {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
        {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
        {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
        {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
        {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
        {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
    };

    // Vertices of the unit cube (12 edges, each edge connects two vertices)
    constexpr float vertex_offsets[12][2][3] = {
        {{0,0,0},{1,0,0}}, // edge 0
        {{1,0,0},{1,1,0}}, // edge 1
        {{1,1,0},{0,1,0}}, // edge 2
        {{0,1,0},{0,0,0}}, // edge 3
        {{0,0,1},{1,0,1}}, // edge 4
        {{1,0,1},{1,1,1}}, // edge 5
        {{1,1,1},{0,1,1}}, // edge 6
        {{0,1,1},{0,0,1}}, // edge 7
        {{0,0,0},{0,0,1}}, // edge 8
        {{1,0,0},{1,0,1}}, // edge 9
        {{1,1,0},{1,1,1}}, // edge 10
        {{0,1,0},{0,1,1}}  // edge 11
    };

} // namespace marching_cubes

// -----------------------------------------------------------------------------
// 4. Marching Cubes mesh extraction from indicator field
//    - indicator   : scalar field (size = nx*ny*nz)
//    - origin      : coordinates of the first corner (grid min)
//    - cell_size   : voxel size
//    - iso_value   : the level to extract (default 0.0 for surface)
//    - out_vertices, out_indices : resulting triangle mesh
// -----------------------------------------------------------------------------
inline void marching_cubes(
    const std::vector<float>& indicator,
    DirectX::FXMVECTOR origin, float cell_size, float iso_value,
    int nx, int ny, int nz,
    std::vector<DirectX::XMVECTOR>& out_vertices,
    std::vector<uint32_t>& out_indices) noexcept
{
    using namespace marching_cubes;
    out_vertices.clear();
    out_indices.clear();
    // For each cell
    for (int iz = 0; iz < nz-1; ++iz) {
        for (int iy = 0; iy < ny-1; ++iy) {
            for (int ix = 0; ix < nx-1; ++ix) {
                // Get scalar values at the 8 corners of the cell
                float val[8];
                int corner_map[8][3] = {
                    {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
                    {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
                };
                for (int c = 0; c < 8; ++c) {
                    int x = ix + corner_map[c][0];
                    int y = iy + corner_map[c][1];
                    int z = iz + corner_map[c][2];
                    size_t idx = (size_t)x + nx * ((size_t)y + ny * (size_t)z);
                    val[c] = indicator[idx];
                }
                // Determine case index
                int cube_index = 0;
                for (int c = 0; c < 8; ++c) {
                    if (val[c] < iso_value) cube_index |= (1 << c);
                }
                if (edge_table[cube_index] == 0) continue;

                // Compute intersection points on edges that are intersected
                DirectX::XMVECTOR edge_verts[12];
                for (int e = 0; e < 12; ++e) {
                    if (edge_table[cube_index] & (1 << e)) {
                        // Determine the two corner indices of this edge
                        int v0_idx = vertex_offsets[e][0][2]*4 + vertex_offsets[e][0][1]*2 + vertex_offsets[e][0][0]; // not correct; we'll map manually
                        // Actually we can just use the precomputed corner_map; edges connect corners i and j.
                        int corner0 = -1, corner1 = -1;
                        switch(e) {
                            case 0: corner0=0; corner1=1; break;
                            case 1: corner0=1; corner1=2; break;
                            case 2: corner0=2; corner1=3; break;
                            case 3: corner0=3; corner1=0; break;
                            case 4: corner0=4; corner1=5; break;
                            case 5: corner0=5; corner1=6; break;
                            case 6: corner0=6; corner1=7; break;
                            case 7: corner0=7; corner1=4; break;
                            case 8: corner0=0; corner1=4; break;
                            case 9: corner0=1; corner1=5; break;
                            case 10: corner0=2; corner1=6; break;
                            case 11: corner0=3; corner1=7; break;
                        }
                        float v0 = val[corner0];
                        float v1 = val[corner1];
                        float t = (iso_value - v0) / (v1 - v0);
                        t = std::clamp(t, 0.0f, 1.0f);
                        // Coordinates of the two corners in world space
                        float ox = vector_math::get_x(origin) + (ix + corner_map[corner0][0]) * cell_size;
                        float oy = vector_math::get_y(origin) + (iy + corner_map[corner0][1]) * cell_size;
                        float oz = vector_math::get_z(origin) + (iz + corner_map[corner0][2]) * cell_size;
                        float px = vector_math::get_x(origin) + (ix + corner_map[corner1][0]) * cell_size;
                        float py = vector_math::get_y(origin) + (iy + corner_map[corner1][1]) * cell_size;
                        float pz = vector_math::get_z(origin) + (iz + corner_map[corner1][2]) * cell_size;
                        DirectX::XMVECTOR pt = DirectX::XMVectorLerp(
                            DirectX::XMVectorSet(ox, oy, oz, 0.0f),
                            DirectX::XMVectorSet(px, py, pz, 0.0f), t);
                        edge_verts[e] = pt;
                    }
                }

                // Generate triangles from tri_table
                for (int t = 0; tri_table[cube_index][t] != -1; t += 3) {
                    int e0 = tri_table[cube_index][t];
                    int e1 = tri_table[cube_index][t+1];
                    int e2 = tri_table[cube_index][t+2];
                    uint32_t base = (uint32_t)out_vertices.size();
                    out_vertices.push_back(edge_verts[e0]);
                    out_vertices.push_back(edge_verts[e1]);
                    out_vertices.push_back(edge_verts[e2]);
                    out_indices.push_back(base);
                    out_indices.push_back(base+1);
                    out_indices.push_back(base+2);
                }
            }
        }
    }
}

} // namespace surface_reconstruction
} // namespace SimulationMath

#endif // CORE_MATH_SURFACE_RECONSTRUCTION_H