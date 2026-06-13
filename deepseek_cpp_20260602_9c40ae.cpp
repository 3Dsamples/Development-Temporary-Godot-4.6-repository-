// system name : Octree Spatial Master
//File 0018 : core/math/fixed_differential.h
//Fixed‑point differential operators on tensor fields: divergence, gradient, Laplacian, interpolation, and SIMD 4‑lane batch evaluation
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "core/math/fixed_tensor.h"
#include "core/math/fixed_vector_field.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace fixed_math {

// ---------------------------------------------------------------------------
// Tensor field storage: array of fmat3 on a uniform 3D grid
// ---------------------------------------------------------------------------

// Trilinear interpolation of a tensor field at a point
inline fmat3 tensor_field_trilinear(const fmat3* grid, int nx, int ny, int nz,
                                    const fvec3& pos, const fvec3& origin, fixed64_t cell_size) noexcept {
    fvec3 local = fvec3_sub(pos, origin);
    fixed64_t fx = fixed_div(local.x, cell_size);
    fixed64_t fy = fixed_div(local.y, cell_size);
    fixed64_t fz = fixed_div(local.z, cell_size);
    int ix = (int)(fx >> FRAC_BITS);
    int iy = (int)(fy >> FRAC_BITS);
    int iz = (int)(fz >> FRAC_BITS);
    ix = (ix < 0) ? 0 : (ix >= nx-1) ? nx-2 : ix;
    iy = (iy < 0) ? 0 : (iy >= ny-1) ? ny-2 : iy;
    iz = (iz < 0) ? 0 : (iz >= nz-1) ? nz-2 : iz;
    fixed64_t tx = fx - (fixed64_t(ix) << FRAC_BITS);
    fixed64_t ty = fy - (fixed64_t(iy) << FRAC_BITS);
    fixed64_t tz = fz - (fixed64_t(iz) << FRAC_BITS);
    fixed64_t u = tx, v = ty, w = tz;
    fixed64_t u1 = FIXED64_ONE - u, v1 = FIXED64_ONE - v, w1 = FIXED64_ONE - w;
    auto t = [&](int x, int y, int z) -> fmat3 { return grid[z*ny*nx + y*nx + x]; };
    fmat3 c000 = t(ix, iy, iz), c100 = t(ix+1, iy, iz);
    fmat3 c010 = t(ix, iy+1, iz), c110 = t(ix+1, iy+1, iz);
    fmat3 c001 = t(ix, iy, iz+1), c101 = t(ix+1, iy, iz+1);
    fmat3 c011 = t(ix, iy+1, iz+1), c111 = t(ix+1, iy+1, iz+1);
    fmat3 result;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            fixed64_t v000 = *(&c000.rows[0].x + r*3 + c);
            fixed64_t v100 = *(&c100.rows[0].x + r*3 + c);
            fixed64_t v010 = *(&c010.rows[0].x + r*3 + c);
            fixed64_t v110 = *(&c110.rows[0].x + r*3 + c);
            fixed64_t v001 = *(&c001.rows[0].x + r*3 + c);
            fixed64_t v101 = *(&c101.rows[0].x + r*3 + c);
            fixed64_t v011 = *(&c011.rows[0].x + r*3 + c);
            fixed64_t v111 = *(&c111.rows[0].x + r*3 + c);
            fixed64_t val = fixed_mul(w1, (fixed_mul(v1, (fixed_mul(u1, v000) + fixed_mul(u, v100))) +
                                           fixed_mul(v,  (fixed_mul(u1, v010) + fixed_mul(u, v110))))) +
                            fixed_mul(w,  (fixed_mul(v1, (fixed_mul(u1, v001) + fixed_mul(u, v101))) +
                                           fixed_mul(v,  (fixed_mul(u1, v011) + fixed_mul(u, v111)))));
            *(&result.rows[0].x + r*3 + c) = val;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Divergence of a tensor field (yields a vector) via central differences
//   div(T) = (dTxx/dx + dTxy/dy + dTxz/dz, dTyx/dx + dTyy/dy + dTyz/dz, dTzx/dx + dTzy/dy + dTzz/dz)
// ---------------------------------------------------------------------------
inline fvec3 tensor_field_divergence(const fmat3* grid, int nx, int ny, int nz,
                                     const fvec3& pos, const fvec3& origin, fixed64_t cell_size) noexcept {
    fvec3 local = fvec3_sub(pos, origin);
    fixed64_t fx = fixed_div(local.x, cell_size);
    fixed64_t fy = fixed_div(local.y, cell_size);
    fixed64_t fz = fixed_div(local.z, cell_size);
    int ix = (int)(fx >> FRAC_BITS);
    int iy = (int)(fy >> FRAC_BITS);
    int iz = (int)(fz >> FRAC_BITS);
    ix = (ix < 1) ? 1 : (ix >= nx-2) ? nx-3 : ix;
    iy = (iy < 1) ? 1 : (iy >= ny-2) ? ny-3 : iy;
    iz = (iz < 1) ? 1 : (iz >= nz-2) ? nz-3 : iz;
    auto t = [&](int x, int y, int z) -> fmat3 { return grid[z*ny*nx + y*nx + x]; };
    fmat3 txp = t(ix+1, iy, iz); fmat3 txm = t(ix-1, iy, iz);
    fmat3 typ = t(ix, iy+1, iz); fmat3 tym = t(ix, iy-1, iz);
    fmat3 tzp = t(ix, iy, iz+1); fmat3 tzm = t(ix, iy, iz-1);
    fixed64_t inv_2h = fixed_rcp(2 * cell_size);
    fixed64_t div_x = fixed_mul(
        (*(&txp.rows[0].x + 0*3 + 0) - *(&txm.rows[0].x + 0*3 + 0)) +
        (*(&typ.rows[0].x + 0*3 + 1) - *(&tym.rows[0].x + 0*3 + 1)) +
        (*(&tzp.rows[0].x + 0*3 + 2) - *(&tzm.rows[0].x + 0*3 + 2)), inv_2h);
    fixed64_t div_y = fixed_mul(
        (*(&txp.rows[0].x + 1*3 + 0) - *(&txm.rows[0].x + 1*3 + 0)) +
        (*(&typ.rows[0].x + 1*3 + 1) - *(&tym.rows[0].x + 1*3 + 1)) +
        (*(&tzp.rows[0].x + 1*3 + 2) - *(&tzm.rows[0].x + 1*3 + 2)), inv_2h);
    fixed64_t div_z = fixed_mul(
        (*(&txp.rows[0].x + 2*3 + 0) - *(&txm.rows[0].x + 2*3 + 0)) +
        (*(&typ.rows[0].x + 2*3 + 1) - *(&tym.rows[0].x + 2*3 + 1)) +
        (*(&tzp.rows[0].x + 2*3 + 2) - *(&tzm.rows[0].x + 2*3 + 2)), inv_2h);
    return {div_x, div_y, div_z};
}

// ---------------------------------------------------------------------------
// Gradient of a tensor field (3x3x3 third‑order tensor) – returns 3 matrices
//   gradient[i] = dT/dx_i, each a 3x3 matrix
// ---------------------------------------------------------------------------
inline void tensor_field_gradient(const fmat3* grid, int nx, int ny, int nz,
                                  const fvec3& pos, const fvec3& origin, fixed64_t cell_size,
                                  fmat3 grad[3]) noexcept {
    fvec3 local = fvec3_sub(pos, origin);
    fixed64_t fx = fixed_div(local.x, cell_size);
    fixed64_t fy = fixed_div(local.y, cell_size);
    fixed64_t fz = fixed_div(local.z, cell_size);
    int ix = (int)(fx >> FRAC_BITS);
    int iy = (int)(fy >> FRAC_BITS);
    int iz = (int)(fz >> FRAC_BITS);
    ix = (ix < 1) ? 1 : (ix >= nx-2) ? nx-3 : ix;
    iy = (iy < 1) ? 1 : (iy >= ny-2) ? ny-3 : iy;
    iz = (iz < 1) ? 1 : (iz >= nz-2) ? nz-3 : iz;
    auto t = [&](int x, int y, int z) -> fmat3 { return grid[z*ny*nx + y*nx + x]; };
    fmat3 txp = t(ix+1, iy, iz); fmat3 txm = t(ix-1, iy, iz);
    fmat3 typ = t(ix, iy+1, iz); fmat3 tym = t(ix, iy-1, iz);
    fmat3 tzp = t(ix, iy, iz+1); fmat3 tzm = t(ix, iy, iz-1);
    fixed64_t inv_2h = fixed_rcp(2 * cell_size);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            fixed64_t d_dx = fixed_mul(*(&txp.rows[0].x + r*3 + c) - *(&txm.rows[0].x + r*3 + c), inv_2h);
            fixed64_t d_dy = fixed_mul(*(&typ.rows[0].x + r*3 + c) - *(&tym.rows[0].x + r*3 + c), inv_2h);
            fixed64_t d_dz = fixed_mul(*(&tzp.rows[0].x + r*3 + c) - *(&tzm.rows[0].x + r*3 + c), inv_2h);
            *(&grad[0].rows[0].x + r*3 + c) = d_dx;
            *(&grad[1].rows[0].x + r*3 + c) = d_dy;
            *(&grad[2].rows[0].x + r*3 + c) = d_dz;
        }
    }
}

// ---------------------------------------------------------------------------
// Laplacian of a tensor field (component‑wise) – returns a 3x3 matrix
// ---------------------------------------------------------------------------
inline fmat3 tensor_field_laplacian(const fmat3* grid, int nx, int ny, int nz,
                                    const fvec3& pos, const fvec3& origin, fixed64_t cell_size) noexcept {
    fvec3 local = fvec3_sub(pos, origin);
    fixed64_t fx = fixed_div(local.x, cell_size);
    fixed64_t fy = fixed_div(local.y, cell_size);
    fixed64_t fz = fixed_div(local.z, cell_size);
    int ix = (int)(fx >> FRAC_BITS);
    int iy = (int)(fy >> FRAC_BITS);
    int iz = (int)(fz >> FRAC_BITS);
    ix = (ix < 1) ? 1 : (ix >= nx-2) ? nx-3 : ix;
    iy = (iy < 1) ? 1 : (iy >= ny-2) ? ny-3 : iy;
    iz = (iz < 1) ? 1 : (iz >= nz-2) ? nz-3 : iz;
    auto t = [&](int x, int y, int z) -> fmat3 { return grid[z*ny*nx + y*nx + x]; };
    fmat3 tc = t(ix, iy, iz);
    fmat3 txp = t(ix+1, iy, iz); fmat3 txm = t(ix-1, iy, iz);
    fmat3 typ = t(ix, iy+1, iz); fmat3 tym = t(ix, iy-1, iz);
    fmat3 tzp = t(ix, iy, iz+1); fmat3 tzm = t(ix, iy, iz-1);
    fixed64_t inv_h2 = fixed_rcp(fixed_mul(cell_size, cell_size));
    fmat3 result;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            fixed64_t sum = *(&txp.rows[0].x + r*3 + c) + *(&txm.rows[0].x + r*3 + c)
                          + *(&typ.rows[0].x + r*3 + c) + *(&tym.rows[0].x + r*3 + c)
                          + *(&tzp.rows[0].x + r*3 + c) + *(&tzm.rows[0].x + r*3 + c)
                          - 6 * *(&tc.rows[0].x + r*3 + c);
            *(&result.rows[0].x + r*3 + c) = fixed_mul(sum, inv_h2);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// SIMD 4‑lane divergence of tensor field: 4 sample points, returns 4 vectors (SoA)
// ---------------------------------------------------------------------------
inline void simd4_tensor_divergence(const fmat3* grid, int nx, int ny, int nz,
                                    const fvec3* samples, const fvec3& origin, fixed64_t cell_size,
                                    __m256i& div_x, __m256i& div_y, __m256i& div_z) noexcept {
    alignas(32) int64_t dx[4], dy[4], dz[4];
    for (int i = 0; i < 4; ++i) {
        fvec3 d = tensor_field_divergence(grid, nx, ny, nz, samples[i], origin, cell_size);
        dx[i] = d.x; dy[i] = d.y; dz[i] = d.z;
    }
    div_x = _mm256_load_si256((__m256i*)dx);
    div_y = _mm256_load_si256((__m256i*)dy);
    div_z = _mm256_load_si256((__m256i*)dz);
}

// ---------------------------------------------------------------------------
// Tensor diffusion: advance a tensor field by explicit Euler step of diffusion equation
//   dT/dt = laplacian(T) * diffusivity
// ---------------------------------------------------------------------------
inline void tensor_field_diffuse(fmat3* grid, int nx, int ny, int nz,
                                 const fvec3& origin, fixed64_t cell_size,
                                 fixed64_t diffusivity, fixed64_t dt) noexcept {
    fmat3* new_grid = static_cast<fmat3*>(aligned_alloc(64, nx*ny*nz * sizeof(fmat3)));
    if (!new_grid) return;
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                fvec3 pos = fvec3_add(origin, {
                    fixed_mul(static_cast<fixed64_t>(x) << FRAC_BITS, cell_size) + (cell_size >> 1),
                    fixed_mul(static_cast<fixed64_t>(y) << FRAC_BITS, cell_size) + (cell_size >> 1),
                    fixed_mul(static_cast<fixed64_t>(z) << FRAC_BITS, cell_size) + (cell_size >> 1)
                });
                fmat3 lap = tensor_field_laplacian(grid, nx, ny, nz, pos, origin, cell_size);
                fmat3 cur = grid[z*ny*nx + y*nx + x];
                fmat3 next = cur;
                for (int r=0; r<3; ++r)
                    for (int c=0; c<3; ++c)
                        *(&next.rows[0].x + r*3 + c) += fixed_mul(fixed_mul(diffusivity, dt), *(&lap.rows[0].x + r*3 + c));
                new_grid[z*ny*nx + y*nx + x] = next;
            }
        }
    }
    for (size_t i=0; i< (size_t)nx*ny*nz; ++i) grid[i] = new_grid[i];
    free(new_grid);
}

} // namespace fixed_math