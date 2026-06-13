// system name : Octree Spatial Master
//File 0031 : core/math/fixed_fluid_dynamics.h
//Incompressible Navier‑Stokes MAC‑grid solvers: advection, diffusion, pressure projection, vorticity confinement, SIMD batch, perceptual colour diagnostics
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "core/math/fixed_tensor.h"
#include "core/math/fixed_sparse_solver.h"
#include "core/math/fixed_geometry.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <limits>

namespace fixed_math {

// ============================================================================
// MAC‑Grid data structure
// ============================================================================
struct MACGrid {
    int nx, ny, nz;
    fixed64_t cell_size;
    fvec3 origin;

    std::vector<fixed64_t> u;        // x‑faces: (nx+1)*ny*nz
    std::vector<fixed64_t> v;        // y‑faces: nx*(ny+1)*nz
    std::vector<fixed64_t> w;        // z‑faces: nx*ny*(nz+1)
    std::vector<fixed64_t> pressure; // cell‑centered: nx*ny*nz
    std::vector<fixed64_t> divergence;
    std::vector<fixed64_t> density;
    std::vector<fixed64_t> viscosity;

    MACGrid(int nx_, int ny_, int nz_, fixed64_t h_, const fvec3& org)
        : nx(nx_), ny(ny_), nz(nz_), cell_size(h_), origin(org) {
        u.resize((nx+1)*ny*nz, 0);
        v.resize(nx*(ny+1)*nz, 0);
        w.resize(nx*ny*(nz+1), 0);
        pressure.resize(nx*ny*nz, 0);
        divergence.resize(nx*ny*nz, 0);
        density.resize(nx*ny*nz, 1000LL << FRAC_BITS);
        viscosity.resize(nx*ny*nz, fixed_from_double(0.001));
    }

    // Cell‑centered index
    int cc_idx(int i, int j, int k) const noexcept {
        return (k*ny + j)*nx + i;
    }
    // Face indices
    int u_idx(int i, int j, int k) const noexcept {
        return (k*ny + j)*(nx+1) + i;
    }
    int v_idx(int i, int j, int k) const noexcept {
        return (k*(ny+1) + j)*nx + i;
    }
    int w_idx(int i, int j, int k) const noexcept {
        return (k*ny + j)*nx + i;
    }

    // Accessors
    fixed64_t& u_face(int i, int j, int k) noexcept { return u[u_idx(i,j,k)]; }
    fixed64_t& v_face(int i, int j, int k) noexcept { return v[v_idx(i,j,k)]; }
    fixed64_t& w_face(int i, int j, int k) noexcept { return w[w_idx(i,j,k)]; }
    fixed64_t& p_cell(int i, int j, int k) noexcept { return pressure[cc_idx(i,j,k)]; }
    fixed64_t& div_cell(int i, int j, int k) noexcept { return divergence[cc_idx(i,j,k)]; }
    fixed64_t& dens_cell(int i, int j, int k) noexcept { return density[cc_idx(i,j,k)]; }

    const fixed64_t& u_face(int i, int j, int k) const noexcept { return u[u_idx(i,j,k)]; }
    const fixed64_t& v_face(int i, int j, int k) const noexcept { return v[v_idx(i,j,k)]; }
    const fixed64_t& w_face(int i, int j, int k) const noexcept { return w[w_idx(i,j,k)]; }
    const fixed64_t& p_cell(int i, int j, int k) const noexcept { return pressure[cc_idx(i,j,k)]; }
};

// ---------------------------------------------------------------------------
// Trilinear interpolation of a scalar field defined on a uniform cell‑centered grid
// ---------------------------------------------------------------------------
inline fixed64_t trilinear_cell_centered(const std::vector<fixed64_t>& field,
                                         int nx, int ny, int nz,
                                         fixed64_t cell_size, const fvec3& origin,
                                         const fvec3& pos) noexcept {
    fvec3 local = fvec3_sub(pos, origin);
    fixed64_t fx = fixed_div(local.x, cell_size);
    fixed64_t fy = fixed_div(local.y, cell_size);
    fixed64_t fz = fixed_div(local.z, cell_size);
    int ix = (int)(fx >> FRAC_BITS);
    int iy = (int)(fy >> FRAC_BITS);
    int iz = (int)(fz >> FRAC_BITS);
    ix = std::clamp(ix, 0, nx-2);
    iy = std::clamp(iy, 0, ny-2);
    iz = std::clamp(iz, 0, nz-2);
    fixed64_t tx = fx - (fixed64_t(ix) << FRAC_BITS);
    fixed64_t ty = fy - (fixed64_t(iy) << FRAC_BITS);
    fixed64_t tz = fz - (fixed64_t(iz) << FRAC_BITS);
    auto get = [&](int i, int j, int k) -> fixed64_t {
        return field[(k*ny + j)*nx + i];
    };
    fixed64_t c000 = get(ix,iy,iz), c100 = get(ix+1,iy,iz);
    fixed64_t c010 = get(ix,iy+1,iz), c110 = get(ix+1,iy+1,iz);
    fixed64_t c001 = get(ix,iy,iz+1), c101 = get(ix+1,iy,iz+1);
    fixed64_t c011 = get(ix,iy+1,iz+1), c111 = get(ix+1,iy+1,iz+1);
    return fixed_mul(FIXED64_ONE-tz,
                fixed_mul(FIXED64_ONE-ty,
                    fixed_mul(FIXED64_ONE-tx, c000) + fixed_mul(tx, c100))
              + fixed_mul(ty,
                    fixed_mul(FIXED64_ONE-tx, c010) + fixed_mul(tx, c110)))
         + fixed_mul(tz,
                fixed_mul(FIXED64_ONE-ty,
                    fixed_mul(FIXED64_ONE-tx, c001) + fixed_mul(tx, c101))
              + fixed_mul(ty,
                    fixed_mul(FIXED64_ONE-tx, c011) + fixed_mul(tx, c111)));
}

// ---------------------------------------------------------------------------
// Trilinear interpolation of u‑face staggered values to arbitrary point
//   u stored at (i+0.5, j, k). We use bilinear in y,z for fixed x‑index.
// ---------------------------------------------------------------------------
inline fixed64_t interpolate_u_face(const MACGrid& grid, const fvec3& pos,
                                    const std::vector<fixed64_t>& u_data) noexcept {
    fvec3 local = fvec3_sub(pos, grid.origin);
    // x coordinate relative to faces
    fixed64_t fx = fixed_div(local.x, grid.cell_size) - FIXED64_HALF; // i index
    fixed64_t fy = fixed_div(local.y, grid.cell_size);
    fixed64_t fz = fixed_div(local.z, grid.cell_size);
    int ix = (int)(fx >> FRAC_BITS);
    int iy = (int)(fy >> FRAC_BITS);
    int iz = (int)(fz >> FRAC_BITS);
    ix = std::clamp(ix, 0, grid.nx-1); // valid face range 0..nx
    iy = std::clamp(iy, 0, grid.ny-1);
    iz = std::clamp(iz, 0, grid.nz-1);
    fixed64_t ty = fy - (fixed64_t(iy) << FRAC_BITS);
    fixed64_t tz = fz - (fixed64_t(iz) << FRAC_BITS);
    // Get u values at the four corners in y-z plane for this face i+1/2
    auto u_at = [&](int j, int k) -> fixed64_t {
        if (j<0 || j>=grid.ny || k<0 || k>=grid.nz) return 0;
        return u_data[grid.u_idx(ix, j, k)];
    };
    fixed64_t u00 = u_at(iy, iz);
    fixed64_t u10 = u_at(iy+1, iz);
    fixed64_t u01 = u_at(iy, iz+1);
    fixed64_t u11 = u_at(iy+1, iz+1);
    // bilinear in y,z
    fixed64_t u0 = fixed_mul(FIXED64_ONE-ty, u00) + fixed_mul(ty, u10);
    fixed64_t u1 = fixed_mul(FIXED64_ONE-ty, u01) + fixed_mul(ty, u11);
    return fixed_mul(FIXED64_ONE-tz, u0) + fixed_mul(tz, u1);
}

// Similarly for v and w
inline fixed64_t interpolate_v_face(const MACGrid& grid, const fvec3& pos,
                                    const std::vector<fixed64_t>& v_data) noexcept {
    fvec3 local = fvec3_sub(pos, grid.origin);
    fixed64_t fx = fixed_div(local.x, grid.cell_size);
    fixed64_t fy = fixed_div(local.y, grid.cell_size) - FIXED64_HALF;
    fixed64_t fz = fixed_div(local.z, grid.cell_size);
    int ix = (int)(fx >> FRAC_BITS);
    int iy = (int)(fy >> FRAC_BITS);
    int iz = (int)(fz >> FRAC_BITS);
    ix = std::clamp(ix, 0, grid.nx-1);
    iy = std::clamp(iy, 0, grid.ny-1); // v faces range 0..ny
    iz = std::clamp(iz, 0, grid.nz-1);
    fixed64_t tx = fx - (fixed64_t(ix) << FRAC_BITS);
    fixed64_t tz = fz - (fixed64_t(iz) << FRAC_BITS);
    auto v_at = [&](int i, int k) -> fixed64_t {
        if (i<0 || i>=grid.nx || k<0 || k>=grid.nz) return 0;
        return v_data[grid.v_idx(i, iy, k)];
    };
    fixed64_t v00 = v_at(ix, iz);
    fixed64_t v10 = v_at(ix+1, iz);
    fixed64_t v01 = v_at(ix, iz+1);
    fixed64_t v11 = v_at(ix+1, iz+1);
    fixed64_t v0 = fixed_mul(FIXED64_ONE-tx, v00) + fixed_mul(tx, v10);
    fixed64_t v1 = fixed_mul(FIXED64_ONE-tx, v01) + fixed_mul(tx, v11);
    return fixed_mul(FIXED64_ONE-tz, v0) + fixed_mul(tz, v1);
}

inline fixed64_t interpolate_w_face(const MACGrid& grid, const fvec3& pos,
                                    const std::vector<fixed64_t>& w_data) noexcept {
    fvec3 local = fvec3_sub(pos, grid.origin);
    fixed64_t fx = fixed_div(local.x, grid.cell_size);
    fixed64_t fy = fixed_div(local.y, grid.cell_size);
    fixed64_t fz = fixed_div(local.z, grid.cell_size) - FIXED64_HALF;
    int ix = (int)(fx >> FRAC_BITS);
    int iy = (int)(fy >> FRAC_BITS);
    int iz = (int)(fz >> FRAC_BITS);
    ix = std::clamp(ix, 0, grid.nx-1);
    iy = std::clamp(iy, 0, grid.ny-1);
    iz = std::clamp(iz, 0, grid.nz-1); // w faces range 0..nz
    fixed64_t tx = fx - (fixed64_t(ix) << FRAC_BITS);
    fixed64_t ty = fy - (fixed64_t(iy) << FRAC_BITS);
    auto w_at = [&](int i, int j) -> fixed64_t {
        if (i<0 || i>=grid.nx || j<0 || j>=grid.ny) return 0;
        return w_data[grid.w_idx(i, j, iz)];
    };
    fixed64_t w00 = w_at(ix, iy);
    fixed64_t w10 = w_at(ix+1, iy);
    fixed64_t w01 = w_at(ix, iy+1);
    fixed64_t w11 = w_at(ix+1, iy+1);
    fixed64_t w0 = fixed_mul(FIXED64_ONE-tx, w00) + fixed_mul(tx, w10);
    fixed64_t w1 = fixed_mul(FIXED64_ONE-tx, w01) + fixed_mul(tx, w11);
    return fixed_mul(FIXED64_ONE-ty, w0) + fixed_mul(ty, w1);
}

// ---------------------------------------------------------------------------
// Cell‑centered velocity interpolation (from staggered averages)
// ---------------------------------------------------------------------------
inline fvec3 mac_velocity_at(const MACGrid& grid, const fvec3& pos) noexcept {
    fvec3 local = fvec3_sub(pos, grid.origin);
    fixed64_t fx = fixed_div(local.x, grid.cell_size);
    fixed64_t fy = fixed_div(local.y, grid.cell_size);
    fixed64_t fz = fixed_div(local.z, grid.cell_size);
    int ix = (int)(fx >> FRAC_BITS);
    int iy = (int)(fy >> FRAC_BITS);
    int iz = (int)(fz >> FRAC_BITS);
    ix = std::clamp(ix, 0, grid.nx-1);
    iy = std::clamp(iy, 0, grid.ny-1);
    iz = std::clamp(iz, 0, grid.nz-1);
    fixed64_t tx = fx - (fixed64_t(ix) << FRAC_BITS);
    fixed64_t ty = fy - (fixed64_t(iy) << FRAC_BITS);
    fixed64_t tz = fz - (fixed64_t(iz) << FRAC_BITS);

    auto avg_u = [&](int i, int j, int k) {
        return fixed_mul(FIXED64_HALF, grid.u_face(i,j,k) + grid.u_face(i+1,j,k));
    };
    auto avg_v = [&](int i, int j, int k) {
        return fixed_mul(FIXED64_HALF, grid.v_face(i,j,k) + grid.v_face(i,j+1,k));
    };
    auto avg_w = [&](int i, int j, int k) {
        return fixed_mul(FIXED64_HALF, grid.w_face(i,j,k) + grid.w_face(i,j,k+1));
    };

    // Trilinear interpolation using the 8 corners of the cell
    auto get_corner = [&](int di, int dj, int dk, const auto& avg) -> fixed64_t {
        int ni = ix + di, nj = iy + dj, nk = iz + dk;
        if (ni<0 || ni>=grid.nx || nj<0 || nj>=grid.ny || nk<0 || nk>=grid.nz) return 0;
        return avg(ni, nj, nk);
    };

    fixed64_t u_val = trilinear_8(tx, ty, tz, [&](int di,int dj,int dk){ return get_corner(di,dj,dk, avg_u); });
    fixed64_t v_val = trilinear_8(tx, ty, tz, [&](int di,int dj,int dk){ return get_corner(di,dj,dk, avg_v); });
    fixed64_t w_val = trilinear_8(tx, ty, tz, [&](int di,int dj,int dk){ return get_corner(di,dj,dk, avg_w); });
    return {u_val, v_val, w_val};
}

// 8‑point trilinear combination
template<typename F>
inline fixed64_t trilinear_8(fixed64_t tx, fixed64_t ty, fixed64_t tz, F&& val) noexcept {
    fixed64_t c000 = val(0,0,0), c100 = val(1,0,0);
    fixed64_t c010 = val(0,1,0), c110 = val(1,1,0);
    fixed64_t c001 = val(0,0,1), c101 = val(1,0,1);
    fixed64_t c011 = val(0,1,1), c111 = val(1,1,1);
    return fixed_mul(FIXED64_ONE-tz,
                fixed_mul(FIXED64_ONE-ty,
                    fixed_mul(FIXED64_ONE-tx, c000) + fixed_mul(tx, c100))
              + fixed_mul(ty,
                    fixed_mul(FIXED64_ONE-tx, c010) + fixed_mul(tx, c110)))
         + fixed_mul(tz,
                fixed_mul(FIXED64_ONE-ty,
                    fixed_mul(FIXED64_ONE-tx, c001) + fixed_mul(tx, c101))
              + fixed_mul(ty,
                    fixed_mul(FIXED64_ONE-tx, c011) + fixed_mul(tx, c111)));
}

// ---------------------------------------------------------------------------
// Compute divergence of staggered velocity field
// ---------------------------------------------------------------------------
inline void compute_divergence(MACGrid& grid) noexcept {
    for (int k=0; k<grid.nz; ++k) {
        for (int j=0; j<grid.ny; ++j) {
            for (int i=0; i<grid.nx; ++i) {
                fixed64_t du = grid.u_face(i+1,j,k) - grid.u_face(i,j,k);
                fixed64_t dv = grid.v_face(i,j+1,k) - grid.v_face(i,j,k);
                fixed64_t dw = grid.w_face(i,j,k+1) - grid.w_face(i,j,k);
                grid.div_cell(i,j,k) = fixed_div(du + dv + dw, grid.cell_size);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Semi‑Lagrangian advection of velocity (backward trace + interpolation)
// ---------------------------------------------------------------------------
inline void advect_velocity_semilag(MACGrid& grid, fixed64_t dt,
                                    const std::vector<fixed64_t>& u_old,
                                    const std::vector<fixed64_t>& v_old,
                                    const std::vector<fixed64_t>& w_old) noexcept {
    // Save old velocities to temporary vectors for interpolation
    std::vector<fixed64_t> u_save = u_old;
    std::vector<fixed64_t> v_save = v_old;
    std::vector<fixed64_t> w_save = w_old;

    // Advect u faces
    for (int k=0; k<grid.nz; ++k) {
        for (int j=0; j<grid.ny; ++j) {
            for (int i=0; i<=grid.nx; ++i) {
                fvec3 face_pos;
                face_pos.x = grid.origin.x + fixed_mul(grid.cell_size, (fixed64_t(i) << FRAC_BITS) + FIXED64_HALF);
                face_pos.y = grid.origin.y + fixed_mul(grid.cell_size, fixed64_t(j) << FRAC_BITS);
                face_pos.z = grid.origin.z + fixed_mul(grid.cell_size, fixed64_t(k) << FRAC_BITS);
                // Velocity at face position: use current staggered values (old)
                fixed64_t u_vel = (i==0 || i==grid.nx) ? 0 : u_save[grid.u_idx(i, j, k)];
                fixed64_t v_vel = 0, w_vel = 0;
                if (j<grid.ny && k<grid.nz) {
                    v_vel = fixed_mul(FIXED64_HALF, v_save[grid.v_idx(i, j, k)] + v_save[grid.v_idx(i-1, j, k)]);
                    w_vel = fixed_mul(FIXED64_HALF, w_save[grid.w_idx(i, j, k)] + w_save[grid.w_idx(i-1, j, k)]);
                }
                fvec3 vel = {u_vel, v_vel, w_vel};
                fvec3 back_pos = fvec3_sub(face_pos, fvec3_scale(vel, dt));
                // Clamp to domain
                back_pos.x = std::max(grid.origin.x + grid.cell_size*FIXED64_HALF,
                                      std::min(back_pos.x, grid.origin.x + grid.cell_size*(grid.nx-0.5)));
                back_pos.y = std::max(grid.origin.y,
                                      std::min(back_pos.y, grid.origin.y + grid.cell_size*grid.ny));
                back_pos.z = std::max(grid.origin.z,
                                      std::min(back_pos.z, grid.origin.z + grid.cell_size*grid.nz));
                // Interpolate u from old u faces
                grid.u_face(i,j,k) = interpolate_u_face(grid, back_pos, u_save);
            }
        }
    }
    // Advect v faces
    for (int k=0; k<grid.nz; ++k) {
        for (int j=0; j<=grid.ny; ++j) {
            for (int i=0; i<grid.nx; ++i) {
                fvec3 face_pos;
                face_pos.x = grid.origin.x + fixed_mul(grid.cell_size, fixed64_t(i) << FRAC_BITS);
                face_pos.y = grid.origin.y + fixed_mul(grid.cell_size, (fixed64_t(j) << FRAC_BITS) + FIXED64_HALF);
                face_pos.z = grid.origin.z + fixed_mul(grid.cell_size, fixed64_t(k) << FRAC_BITS);
                fixed64_t u_vel = fixed_mul(FIXED64_HALF, u_save[grid.u_idx(i, j, k)] + u_save[grid.u_idx(i+1, j, k)]);
                fixed64_t v_vel = (j==0 || j==grid.ny) ? 0 : v_save[grid.v_idx(i, j, k)];
                fixed64_t w_vel = fixed_mul(FIXED64_HALF, w_save[grid.w_idx(i, j, k)] + w_save[grid.w_idx(i, j+1, k)]);
                fvec3 vel = {u_vel, v_vel, w_vel};
                fvec3 back_pos = fvec3_sub(face_pos, fvec3_scale(vel, dt));
                back_pos.y = std::max(grid.origin.y + grid.cell_size*FIXED64_HALF,
                                      std::min(back_pos.y, grid.origin.y + grid.cell_size*(grid.ny-0.5)));
                back_pos.x = std::max(grid.origin.x,
                                      std::min(back_pos.x, grid.origin.x + grid.cell_size*grid.nx));
                back_pos.z = std::max(grid.origin.z,
                                      std::min(back_pos.z, grid.origin.z + grid.cell_size*grid.nz));
                grid.v_face(i,j,k) = interpolate_v_face(grid, back_pos, v_save);
            }
        }
    }
    // Advect w faces
    for (int k=0; k<=grid.nz; ++k) {
        for (int j=0; j<grid.ny; ++j) {
            for (int i=0; i<grid.nx; ++i) {
                fvec3 face_pos;
                face_pos.x = grid.origin.x + fixed_mul(grid.cell_size, fixed64_t(i) << FRAC_BITS);
                face_pos.y = grid.origin.y + fixed_mul(grid.cell_size, fixed64_t(j) << FRAC_BITS);
                face_pos.z = grid.origin.z + fixed_mul(grid.cell_size, (fixed64_t(k) << FRAC_BITS) + FIXED64_HALF);
                fixed64_t u_vel = fixed_mul(FIXED64_HALF, u_save[grid.u_idx(i, j, k)] + u_save[grid.u_idx(i+1, j, k)]);
                fixed64_t v_vel = fixed_mul(FIXED64_HALF, v_save[grid.v_idx(i, j, k)] + v_save[grid.v_idx(i, j+1, k)]);
                fixed64_t w_vel = (k==0 || k==grid.nz) ? 0 : w_save[grid.w_idx(i, j, k)];
                fvec3 vel = {u_vel, v_vel, w_vel};
                fvec3 back_pos = fvec3_sub(face_pos, fvec3_scale(vel, dt));
                back_pos.z = std::max(grid.origin.z + grid.cell_size*FIXED64_HALF,
                                      std::min(back_pos.z, grid.origin.z + grid.cell_size*(grid.nz-0.5)));
                back_pos.x = std::max(grid.origin.x,
                                      std::min(back_pos.x, grid.origin.x + grid.cell_size*grid.nx));
                back_pos.y = std::max(grid.origin.y,
                                      std::min(back_pos.y, grid.origin.y + grid.cell_size*grid.ny));
                grid.w_face(i,j,k) = interpolate_w_face(grid, back_pos, w_save);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Build sparse matrix for pressure Poisson equation: ∇·(1/ρ ∇p)
// ---------------------------------------------------------------------------
inline void build_pressure_matrix(const MACGrid& grid, SparseMatrixCRS& A) noexcept {
    int n = grid.nx * grid.ny * grid.nz;
    A.resize(n, n, n * 7);
    std::vector<std::vector<std::pair<int, fixed64_t>>> row_entries(n);
    fixed64_t invh2 = fixed_rcp(fixed_mul(grid.cell_size, grid.cell_size));

    for (int k=0; k<grid.nz; ++k) {
        for (int j=0; j<grid.ny; ++j) {
            for (int i=0; i<grid.nx; ++i) {
                int row = grid.cc_idx(i,j,k);
                fixed64_t diag = 0;
                // x‑neighbors
                if (i < grid.nx-1) {
                    fixed64_t rho = fixed_mul(FIXED64_HALF, grid.dens_cell(i,j,k) + grid.dens_cell(i+1,j,k));
                    fixed64_t coeff = fixed_div(invh2, rho);
                    diag += coeff;
                    row_entries[row].emplace_back(grid.cc_idx(i+1,j,k), -coeff);
                }
                if (i > 0) {
                    fixed64_t rho = fixed_mul(FIXED64_HALF, grid.dens_cell(i,j,k) + grid.dens_cell(i-1,j,k));
                    fixed64_t coeff = fixed_div(invh2, rho);
                    diag += coeff;
                    row_entries[row].emplace_back(grid.cc_idx(i-1,j,k), -coeff);
                }
                // y‑neighbors
                if (j < grid.ny-1) {
                    fixed64_t rho = fixed_mul(FIXED64_HALF, grid.dens_cell(i,j,k) + grid.dens_cell(i,j+1,k));
                    fixed64_t coeff = fixed_div(invh2, rho);
                    diag += coeff;
                    row_entries[row].emplace_back(grid.cc_idx(i,j+1,k), -coeff);
                }
                if (j > 0) {
                    fixed64_t rho = fixed_mul(FIXED64_HALF, grid.dens_cell(i,j,k) + grid.dens_cell(i,j-1,k));
                    fixed64_t coeff = fixed_div(invh2, rho);
                    diag += coeff;
                    row_entries[row].emplace_back(grid.cc_idx(i,j-1,k), -coeff);
                }
                // z‑neighbors
                if (k < grid.nz-1) {
                    fixed64_t rho = fixed_mul(FIXED64_HALF, grid.dens_cell(i,j,k) + grid.dens_cell(i,j,k+1));
                    fixed64_t coeff = fixed_div(invh2, rho);
                    diag += coeff;
                    row_entries[row].emplace_back(grid.cc_idx(i,j,k+1), -coeff);
                }
                if (k > 0) {
                    fixed64_t rho = fixed_mul(FIXED64_HALF, grid.dens_cell(i,j,k) + grid.dens_cell(i,j,k-1));
                    fixed64_t coeff = fixed_div(invh2, rho);
                    diag += coeff;
                    row_entries[row].emplace_back(grid.cc_idx(i,j,k-1), -coeff);
                }
                row_entries[row].emplace_back(row, diag);
            }
        }
    }
    // Build CRS
    CRSBuilder builder(n, n);
    for (int i=0; i<n; ++i) {
        for (auto& p : row_entries[i]) {
            builder.add(i, p.first, p.second);
        }
    }
    builder.build(A);
}

// ---------------------------------------------------------------------------
// Pressure projection step
// ---------------------------------------------------------------------------
inline void pressure_projection(MACGrid& grid, fixed64_t dt) noexcept {
    compute_divergence(grid);
    int n = grid.nx * grid.ny * grid.nz;
    std::vector<fixed64_t> rhs(n);
    for (int i=0; i<n; ++i) rhs[i] = grid.divergence[i]; // = ∇·u

    SparseMatrixCRS A;
    build_pressure_matrix(grid, A);
    std::vector<fixed64_t> p_sol(n, 0);
    JacobiPreconditioner precond;
    precond.build(A);
    conjugate_gradient(A, rhs.data(), p_sol.data(), 1000, fixed_from_double(1e-6), precond);
    for (int i=0; i<n; ++i) grid.pressure[i] = p_sol[i];

    fixed64_t inv_h = fixed_rcp(grid.cell_size);
    for (int k=0; k<grid.nz; ++k) {
        for (int j=0; j<grid.ny; ++j) {
            for (int i=0; i<grid.nx; ++i) {
                int idx = grid.cc_idx(i,j,k);
                fixed64_t p0 = grid.pressure[idx];
                // u face update
                if (i < grid.nx-1) {
                    fixed64_t p1 = grid.pressure[grid.cc_idx(i+1,j,k)];
                    fixed64_t rho = fixed_mul(FIXED64_HALF, grid.dens_cell(i,j,k) + grid.dens_cell(i+1,j,k));
                    fixed64_t grad = (p1 - p0) * inv_h;
                    grid.u_face(i+1,j,k) -= fixed_div(dt, rho) * grad;
                }
                // v face update
                if (j < grid.ny-1) {
                    fixed64_t p1 = grid.pressure[grid.cc_idx(i,j+1,k)];
                    fixed64_t rho = fixed_mul(FIXED64_HALF, grid.dens_cell(i,j,k) + grid.dens_cell(i,j+1,k));
                    fixed64_t grad = (p1 - p0) * inv_h;
                    grid.v_face(i,j+1,k) -= fixed_div(dt, rho) * grad;
                }
                // w face update
                if (k < grid.nz-1) {
                    fixed64_t p1 = grid.pressure[grid.cc_idx(i,j,k+1)];
                    fixed64_t rho = fixed_mul(FIXED64_HALF, grid.dens_cell(i,j,k) + grid.dens_cell(i,j,k+1));
                    fixed64_t grad = (p1 - p0) * inv_h;
                    grid.w_face(i,j,k+1) -= fixed_div(dt, rho) * grad;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Vorticity confinement force
// ---------------------------------------------------------------------------
inline void vorticity_confinement(MACGrid& grid, fixed64_t dt, fixed64_t epsilon) noexcept {
    int n = grid.nx * grid.ny * grid.nz;
    std::vector<fvec3> vort(n);
    // Compute cell-centered vorticity
    for (int k=0; k<grid.nz; ++k) {
        for (int j=0; j<grid.ny; ++j) {
            for (int i=0; i<grid.nx; ++i) {
                fixed64_t dwdy = (grid.w_face(i,j+1,k) - grid.w_face(i,j,k)) / grid.cell_size;
                fixed64_t dvdz = (grid.v_face(i,j,k+1) - grid.v_face(i,j,k)) / grid.cell_size;
                fixed64_t dudz = (grid.u_face(i,j,k+1) - grid.u_face(i,j,k)) / grid.cell_size;
                fixed64_t dwdx = (grid.w_face(i+1,j,k) - grid.w_face(i,j,k)) / grid.cell_size;
                fixed64_t dvdx = (grid.v_face(i+1,j,k) - grid.v_face(i,j,k)) / grid.cell_size;
                fixed64_t dudy = (grid.u_face(i,j+1,k) - grid.u_face(i,j,k)) / grid.cell_size;
                vort[grid.cc_idx(i,j,k)] = {dwdy - dvdz, dudz - dwdx, dvdx - dudy};
            }
        }
    }

    // Apply confinement force
    for (int k=1; k<grid.nz-1; ++k) {
        for (int j=1; j<grid.ny-1; ++j) {
            for (int i=1; i<grid.nx-1; ++i) {
                int idx = grid.cc_idx(i,j,k);
                fvec3 omega = vort[idx];
                fixed64_t len = fvec3_length(omega);
                if (len == 0) continue;
                fvec3 N = fvec3_scale(omega, fixed_rcp(len));
                fixed64_t dabs_dx = (fvec3_length(vort[grid.cc_idx(i+1,j,k)]) - fvec3_length(vort[grid.cc_idx(i-1,j,k)])) / (2*grid.cell_size);
                fixed64_t dabs_dy = (fvec3_length(vort[grid.cc_idx(i,j+1,k)]) - fvec3_length(vort[grid.cc_idx(i,j-1,k)])) / (2*grid.cell_size);
                fixed64_t dabs_dz = (fvec3_length(vort[grid.cc_idx(i,j,k+1)]) - fvec3_length(vort[grid.cc_idx(i,j,k-1)])) / (2*grid.cell_size);
                fvec3 grad_abs = {dabs_dx, dabs_dy, dabs_dz};
                fvec3 force = fvec3_cross(N, grad_abs);
                force = fvec3_scale(force, epsilon * grid.cell_size);
                // Apply to u,v,w at faces (simple: average to cell center and redistribute)
                grid.u_face(i,j,k) += fixed_mul(dt, force.x);
                grid.v_face(i,j,k) += fixed_mul(dt, force.y);
                grid.w_face(i,j,k) += fixed_mul(dt, force.z);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Perceptual colour mapping of velocity magnitude
// ---------------------------------------------------------------------------
inline void velocity_to_colors(const MACGrid& grid, std::vector<fvec3>& colors) noexcept {
    int nc = grid.nx * grid.ny * grid.nz;
    colors.resize(nc);
    fixed64_t max_speed = 0;
    // First pass: max speed
    for (int idx=0; idx<nc; ++idx) {
        int i = idx % grid.nx;
        int j = (idx / grid.nx) % grid.ny;
        int k = idx / (grid.nx * grid.ny);
        fixed64_t u = fixed_mul(FIXED64_HALF, grid.u_face(i,j,k) + grid.u_face(i+1,j,k));
        fixed64_t v = fixed_mul(FIXED64_HALF, grid.v_face(i,j,k) + grid.v_face(i,j+1,k));
        fixed64_t w = fixed_mul(FIXED64_HALF, grid.w_face(i,j,k) + grid.w_face(i,j,k+1));
        fixed64_t sp = fixed_sqrt(u*u + v*v + w*w);
        if (sp > max_speed) max_speed = sp;
    }
    if (max_speed == 0) max_speed = FIXED64_ONE;
    for (int idx=0; idx<nc; ++idx) {
        int i = idx % grid.nx;
        int j = (idx / grid.nx) % grid.ny;
        int k = idx / (grid.nx * grid.ny);
        fixed64_t u = fixed_mul(FIXED64_HALF, grid.u_face(i,j,k) + grid.u_face(i+1,j,k));
        fixed64_t v = fixed_mul(FIXED64_HALF, grid.v_face(i,j,k) + grid.v_face(i,j+1,k));
        fixed64_t w = fixed_mul(FIXED64_HALF, grid.w_face(i,j,k) + grid.w_face(i,j,k+1));
        fixed64_t sp = fixed_sqrt(u*u + v*v + w*w);
        fixed64_t t = fixed_div(sp, max_speed);
        if (t > FIXED64_ONE) t = FIXED64_ONE;
        fvec3 linear = {t, 0, FIXED64_ONE - t};
        colors[idx] = perceptual_color::linear_srgb_to_oklab(linear);
    }
}

} // namespace fixed_math