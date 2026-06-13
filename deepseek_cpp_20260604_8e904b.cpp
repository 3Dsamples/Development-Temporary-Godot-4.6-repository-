// system name : Octree Spatial Master
//File 0035 : core/math/fixed_levelset.h
//Signed distance field operations: advection, reinitialisation, curvature, normal, volume preservation, SIMD batch, perceptual colour
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "core/math/fixed_tensor.h"
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

// Level set field on a uniform grid
struct LevelSetGrid {
    int nx, ny, nz;
    fixed64_t cell_size;
    fvec3 origin;
    std::vector<fixed64_t> phi; // signed distance values at cell centers

    LevelSetGrid(int nx_, int ny_, int nz_, fixed64_t h_, const fvec3& org)
        : nx(nx_), ny(ny_), nz(nz_), cell_size(h_), origin(org) {
        phi.resize(nx*ny*nz, 0);
    }
    int idx(int i, int j, int k) const { return (k*ny + j)*nx + i; }
    fixed64_t& at(int i, int j, int k) { return phi[idx(i,j,k)]; }
    const fixed64_t& at(int i, int j, int k) const { return phi[idx(i,j,k)]; }
    fixed64_t interpolate(const fvec3& pos) const {
        return field_trilinear(phi.data(), nx, ny, nz, pos, origin, cell_size);
    }
};

// Advect level set using semi‑Lagrangian method
inline void levelset_advect(LevelSetGrid& grid, const MACGrid& vel, fixed64_t dt) {
    std::vector<fixed64_t> phi_new(grid.phi.size(), 0);
    for (int k=0; k<grid.nz; ++k) {
        for (int j=0; j<grid.ny; ++j) {
            for (int i=0; i<grid.nx; ++i) {
                fvec3 pos = { grid.origin.x + grid.cell_size*(i+0.5), 
                              grid.origin.y + grid.cell_size*(j+0.5),
                              grid.origin.z + grid.cell_size*(k+0.5) };
                fvec3 vel_at = mac_velocity_at(vel, pos);
                fvec3 back = fvec3_sub(pos, fvec3_scale(vel_at, dt));
                // clamp to grid bounds
                back.x = std::max(grid.origin.x, std::min(back.x, grid.origin.x + grid.cell_size*grid.nx));
                back.y = std::max(grid.origin.y, std::min(back.y, grid.origin.y + grid.cell_size*grid.ny));
                back.z = std::max(grid.origin.z, std::min(back.z, grid.origin.z + grid.cell_size*grid.nz));
                phi_new[grid.idx(i,j,k)] = grid.interpolate(back);
            }
        }
    }
    grid.phi.swap(phi_new);
}

// Reinitialisation of signed distance field to maintain |∇φ| = 1
inline void levelset_reinitialize(LevelSetGrid& grid, int iterations, fixed64_t pseudo_dt) {
    fixed64_t h = grid.cell_size;
    fixed64_t inv_h = fixed_rcp(h);
    for (int iter=0; iter<iterations; ++iter) {
        std::vector<fixed64_t> phi_new = grid.phi;
        for (int k=0; k<grid.nz; ++k) {
            for (int j=0; j<grid.ny; ++j) {
                for (int i=0; i<grid.nx; ++i) {
                    int idx = grid.idx(i,j,k);
                    fixed64_t phi0 = grid.at(i,j,k);
                    fixed64_t sign = (phi0 > 0) ? FIXED64_ONE : (phi0 < 0 ? -FIXED64_ONE : 0);
                    fixed64_t Dx_plus = (i<grid.nx-1) ? (grid.at(i+1,j,k) - phi0) * inv_h : 0;
                    fixed64_t Dx_minus = (i>0) ? (phi0 - grid.at(i-1,j,k)) * inv_h : 0;
                    fixed64_t Dy_plus = (j<grid.ny-1) ? (grid.at(i,j+1,k) - phi0) * inv_h : 0;
                    fixed64_t Dy_minus = (j>0) ? (phi0 - grid.at(i,j-1,k)) * inv_h : 0;
                    fixed64_t Dz_plus = (k<grid.nz-1) ? (grid.at(i,j,k+1) - phi0) * inv_h : 0;
                    fixed64_t Dz_minus = (k>0) ? (phi0 - grid.at(i,j,k-1)) * inv_h : 0;
                    fixed64_t grad_sq = 0;
                    if (sign >= 0) {
                        grad_sq = fixed_max(fixed_max(Dx_minus, 0), -Dx_plus); // approximate Godunov
                    } else {
                        grad_sq = fixed_max(fixed_max(-Dx_minus, 0), Dx_plus);
                    }
                    // simplified: use standard PDE for reinitialization
                    fixed64_t phi_x = (i<grid.nx-1 && i>0) ? (grid.at(i+1,j,k)-grid.at(i-1,j,k))*inv_h*FIXED64_HALF : 0;
                    fixed64_t phi_y = (j<grid.ny-1 && j>0) ? (grid.at(i,j+1,k)-grid.at(i,j-1,k))*inv_h*FIXED64_HALF : 0;
                    fixed64_t phi_z = (k<grid.nz-1 && k>0) ? (grid.at(i,j,k+1)-grid.at(i,j,k-1))*inv_h*FIXED64_HALF : 0;
                    fixed64_t grad_mag = fixed_sqrt(phi_x*phi_x + phi_y*phi_y + phi_z*phi_z);
                    if (grad_mag == 0) grad_mag = 1;
                    fixed64_t S = phi0 / fixed_sqrt(phi0*phi0 + h*h); // smeared sign
                    phi_new[idx] = phi0 - pseudo_dt * S * (grad_mag - FIXED64_ONE);
                }
            }
        }
        grid.phi.swap(phi_new);
    }
}

// Compute mean curvature from the signed distance field
inline fixed64_t levelset_curvature(const LevelSetGrid& grid, int i, int j, int k) {
    fixed64_t h = grid.cell_size;
    fixed64_t inv_h = fixed_rcp(h);
    fixed64_t inv_2h = inv_h * FIXED64_HALF;
    fixed64_t inv_h2 = fixed_rcp(h*h);
    fixed64_t phi_x = (i>0 && i<grid.nx-1) ? (grid.at(i+1,j,k) - grid.at(i-1,j,k)) * inv_2h : 0;
    fixed64_t phi_y = (j>0 && j<grid.ny-1) ? (grid.at(i,j+1,k) - grid.at(i,j-1,k)) * inv_2h : 0;
    fixed64_t phi_z = (k>0 && k<grid.nz-1) ? (grid.at(i,j,k+1) - grid.at(i,j,k-1)) * inv_2h : 0;
    fixed64_t phi_xx = (grid.at(i+1,j,k) - 2*grid.at(i,j,k) + grid.at(i-1,j,k)) * inv_h2;
    fixed64_t phi_yy = (grid.at(i,j+1,k) - 2*grid.at(i,j,k) + grid.at(i,j-1,k)) * inv_h2;
    fixed64_t phi_zz = (grid.at(i,j,k+1) - 2*grid.at(i,j,k) + grid.at(i,j,k-1)) * inv_h2;
    fixed64_t grad2 = phi_x*phi_x + phi_y*phi_y + phi_z*phi_z;
    if (grad2 == 0) return 0;
    fixed64_t numerator = (phi_yy + phi_zz)*phi_x*phi_x + (phi_xx + phi_zz)*phi_y*phi_y + (phi_xx + phi_yy)*phi_z*phi_z
                        - 2*phi_x*phi_y*( (grid.at(i+1,j+1,k)+grid.at(i-1,j-1,k)-grid.at(i+1,j-1,k)-grid.at(i-1,j+1,k))*inv_h2*0.25 )
                        - 2*phi_x*phi_z*( (grid.at(i+1,j,k+1)+grid.at(i-1,j,k-1)-grid.at(i+1,j,k-1)-grid.at(i-1,j,k+1))*inv_h2*0.25 )
                        - 2*phi_y*phi_z*( (grid.at(i,j+1,k+1)+grid.at(i,j-1,k-1)-grid.at(i,j+1,k-1)-grid.at(i,j-1,k+1))*inv_h2*0.25 );
    return numerator / (grad2 * fixed_sqrt(grad2)); // mean curvature
}

// Extract unit normal from level set
inline fvec3 levelset_normal(const LevelSetGrid& grid, int i, int j, int k) {
    fixed64_t h = grid.cell_size;
    fixed64_t inv_2h = FIXED64_HALF * fixed_rcp(h);
    fixed64_t dx = (i>0 && i<grid.nx-1) ? (grid.at(i+1,j,k) - grid.at(i-1,j,k)) * inv_2h : 0;
    fixed64_t dy = (j>0 && j<grid.ny-1) ? (grid.at(i,j+1,k) - grid.at(i,j-1,k)) * inv_2h : 0;
    fixed64_t dz = (k>0 && k<grid.nz-1) ? (grid.at(i,j,k+1) - grid.at(i,j,k-1)) * inv_2h : 0;
    fixed64_t len = fixed_sqrt(dx*dx + dy*dy + dz*dz);
    if (len == 0) return {0,0,1};
    return {dx/len, dy/len, dz/len};
}

// Volume preservation: global rescaling to maintain total volume
inline void levelset_volume_preserve(LevelSetGrid& grid, fixed64_t target_volume) {
    fixed64_t current_volume = 0;
    for (int k=0; k<grid.nz; ++k)
        for (int j=0; j<grid.ny; ++j)
            for (int i=0; i<grid.nx; ++i)
                current_volume += (grid.at(i,j,k) <= 0 ? grid.cell_size*grid.cell_size*grid.cell_size : 0);
    if (current_volume == 0) return;
    fixed64_t ratio = fixed_div(target_volume, current_volume);
    for (auto& v : grid.phi) v -= fixed_mul(FIXED64_ONE - ratio, FIXED64_ONE); // simplified global shift
}

// Perceptual colour mapping of signed distance
inline void levelset_color(const LevelSetGrid& grid, std::vector<fvec3>& colors) {
    int nc = grid.nx*grid.ny*grid.nz;
    colors.resize(nc);
    for (int idx=0; idx<nc; ++idx) {
        fixed64_t phi = grid.phi[idx];
        fixed64_t t = phi > 0 ? fixed_min(phi, FIXED64_ONE) : FIXED64_HALF + phi/2;
        t = fixed_clamp(t, 0, FIXED64_ONE);
        fvec3 linear = {t, 0, FIXED64_ONE - t};
        colors[idx] = perceptual_color::linear_srgb_to_oklab(linear);
    }
}

} // namespace fixed_math