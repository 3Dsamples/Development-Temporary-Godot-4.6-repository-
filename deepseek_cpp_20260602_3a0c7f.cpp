// system name : Octree Spatial Master
//File 0016 : core/math/fixed_vector_field.h
//Fixed‑point vector field operations: divergence, curl, Laplacian, gradient tensor, strain/stress tensors, SIMD 4‑lane on uniform grids
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
#include <cmath>

namespace fixed_math {

// ---------------------------------------------------------------------------
// Vector field stored on a uniform 3D grid (array of fvec3)
// ---------------------------------------------------------------------------

// Evaluate vector field at a point via trilinear interpolation
inline fvec3 vector_field_trilinear(const fvec3* grid, int nx, int ny, int nz,
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
    auto vtx = [&](int x, int y, int z) -> fvec3 { return grid[z*ny*nx + y*nx + x]; };
    // Trilinear blend
    fvec3 c000 = vtx(ix, iy, iz), c100 = vtx(ix+1, iy, iz);
    fvec3 c010 = vtx(ix, iy+1, iz), c110 = vtx(ix+1, iy+1, iz);
    fvec3 c001 = vtx(ix, iy, iz+1), c101 = vtx(ix+1, iy, iz+1);
    fvec3 c011 = vtx(ix, iy+1, iz+1), c111 = vtx(ix+1, iy+1, iz+1);
    fvec3 sum;
    sum = fvec3_add(fvec3_scale(c000, fixed_mul(u1, fixed_mul(v1, w1))),
                    fvec3_scale(c100, fixed_mul(u,  fixed_mul(v1, w1))));
    sum = fvec3_add(sum, fvec3_scale(c010, fixed_mul(u1, fixed_mul(v, w1))));
    sum = fvec3_add(sum, fvec3_scale(c110, fixed_mul(u,  fixed_mul(v, w1))));
    sum = fvec3_add(sum, fvec3_scale(c001, fixed_mul(u1, fixed_mul(v1, w))));
    sum = fvec3_add(sum, fvec3_scale(c101, fixed_mul(u,  fixed_mul(v1, w))));
    sum = fvec3_add(sum, fvec3_scale(c011, fixed_mul(u1, fixed_mul(v, w))));
    sum = fvec3_add(sum, fvec3_scale(c111, fixed_mul(u,  fixed_mul(v, w))));
    return sum;
}

// ---------------------------------------------------------------------------
// Divergence of a vector field via central differences
// ---------------------------------------------------------------------------
inline fixed64_t vector_field_divergence(const fvec3* grid, int nx, int ny, int nz,
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
    auto v = [&](int x, int y, int z) -> fvec3 { return grid[z*ny*nx + y*nx + x]; };
    fvec3 vxp = v(ix+1, iy, iz);
    fvec3 vxm = v(ix-1, iy, iz);
    fvec3 vyp = v(ix, iy+1, iz);
    fvec3 vym = v(ix, iy-1, iz);
    fvec3 vzp = v(ix, iy, iz+1);
    fvec3 vzm = v(ix, iy, iz-1);
    fixed64_t dx = fixed_div(vxp.x - vxm.x, 2 * cell_size);
    fixed64_t dy = fixed_div(vyp.y - vym.y, 2 * cell_size);
    fixed64_t dz = fixed_div(vzp.z - vzm.z, 2 * cell_size);
    return dx + dy + dz;
}

// ---------------------------------------------------------------------------
// Curl of a vector field
// ---------------------------------------------------------------------------
inline fvec3 vector_field_curl(const fvec3* grid, int nx, int ny, int nz,
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
    auto v = [&](int x, int y, int z) -> fvec3 { return grid[z*ny*nx + y*nx + x]; };
    fvec3 vz_p = v(ix, iy, iz+1);
    fvec3 vz_m = v(ix, iy, iz-1);
    fvec3 vy_p = v(ix, iy+1, iz);
    fvec3 vy_m = v(ix, iy-1, iz);
    fvec3 vx_p = v(ix+1, iy, iz);
    fvec3 vx_m = v(ix-1, iy, iz);
    fixed64_t curl_x = fixed_div(vz_p.y - vz_m.y - (vy_p.z - vy_m.z), 2 * cell_size);
    fixed64_t curl_y = fixed_div(vx_p.z - vx_m.z - (vz_p.x - vz_m.x), 2 * cell_size);
    fixed64_t curl_z = fixed_div(vy_p.x - vy_m.x - (vx_p.y - vx_m.y), 2 * cell_size);
    return {curl_x, curl_y, curl_z};
}

// ---------------------------------------------------------------------------
// Gradient tensor (3x3 Jacobian) of a vector field
// ---------------------------------------------------------------------------
inline fmat3 vector_field_gradient_tensor(const fvec3* grid, int nx, int ny, int nz,
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
    auto v = [&](int x, int y, int z) -> fvec3 { return grid[z*ny*nx + y*nx + x]; };
    fvec3 vxp = v(ix+1, iy, iz); fvec3 vxm = v(ix-1, iy, iz);
    fvec3 vyp = v(ix, iy+1, iz); fvec3 vym = v(ix, iy-1, iz);
    fvec3 vzp = v(ix, iy, iz+1); fvec3 vzm = v(ix, iy, iz-1);
    fixed64_t dvx_dx = fixed_div(vxp.x - vxm.x, 2 * cell_size);
    fixed64_t dvx_dy = fixed_div(vyp.x - vym.x, 2 * cell_size);
    fixed64_t dvx_dz = fixed_div(vzp.x - vzm.x, 2 * cell_size);
    fixed64_t dvy_dx = fixed_div(vxp.y - vxm.y, 2 * cell_size);
    fixed64_t dvy_dy = fixed_div(vyp.y - vym.y, 2 * cell_size);
    fixed64_t dvy_dz = fixed_div(vzp.y - vzm.y, 2 * cell_size);
    fixed64_t dvz_dx = fixed_div(vxp.z - vxm.z, 2 * cell_size);
    fixed64_t dvz_dy = fixed_div(vyp.z - vym.z, 2 * cell_size);
    fixed64_t dvz_dz = fixed_div(vzp.z - vzm.z, 2 * cell_size);
    fmat3 J;
    J.rows[0] = {dvx_dx, dvx_dy, dvx_dz};
    J.rows[1] = {dvy_dx, dvy_dy, dvy_dz};
    J.rows[2] = {dvz_dx, dvz_dy, dvz_dz};
    return J;
}

// ---------------------------------------------------------------------------
// Strain tensor (linearised) = 0.5 * (J + J^T)
// ---------------------------------------------------------------------------
inline fmat3 vector_field_strain_tensor(const fvec3* grid, int nx, int ny, int nz,
                                       const fvec3& pos, const fvec3& origin, fixed64_t cell_size) noexcept {
    fmat3 J = vector_field_gradient_tensor(grid, nx, ny, nz, pos, origin, cell_size);
    fmat3 JT = fmat3_transpose(J);
    // E = 0.5 * (J + JT)
    fmat3 E;
    for (int r=0; r<3; ++r)
        for (int c=0; c<3; ++c)
            *(&E.rows[0].x + r*3 + c) = fixed_mul(FIXED64_HALF, *(&J.rows[0].x + r*3 + c) + *(&JT.rows[0].x + r*3 + c));
    return E;
}

// ---------------------------------------------------------------------------
// Stress tensor (linear isotropic elasticity: sigma = lambda * tr(E)*I + 2*mu*E)
// ---------------------------------------------------------------------------
inline fmat3 vector_field_stress_tensor(const fvec3* grid, int nx, int ny, int nz,
                                        const fvec3& pos, const fvec3& origin, fixed64_t cell_size,
                                        fixed64_t lambda, fixed64_t mu) noexcept {
    fmat3 E = vector_field_strain_tensor(grid, nx, ny, nz, pos, origin, cell_size);
    fixed64_t traceE = fmat3_trace(E);
    fmat3 I = fmat3_identity();
    fmat3 sigma;
    for (int r=0; r<3; ++r)
        for (int c=0; c<3; ++c) {
            fixed64_t term1 = fixed_mul(lambda, fixed_mul(traceE, *(&I.rows[0].x + r*3 + c)));
            fixed64_t term2 = fixed_mul(2 * mu, *(&E.rows[0].x + r*3 + c));
            *(&sigma.rows[0].x + r*3 + c) = term1 + term2;
        }
    return sigma;
}

// ---------------------------------------------------------------------------
// Vector Laplacian (component‑wise)
// ---------------------------------------------------------------------------
inline fvec3 vector_field_laplacian(const fvec3* grid, int nx, int ny, int nz,
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
    auto v = [&](int x, int y, int z) -> fvec3 { return grid[z*ny*nx + y*nx + x]; };
    fvec3 vc = v(ix, iy, iz);
    fvec3 vxp = v(ix+1, iy, iz); fvec3 vxm = v(ix-1, iy, iz);
    fvec3 vyp = v(ix, iy+1, iz); fvec3 vym = v(ix, iy-1, iz);
    fvec3 vzp = v(ix, iy, iz+1); fvec3 vzm = v(ix, iy, iz-1);
    fixed64_t inv_h2 = fixed_rcp(fixed_mul(cell_size, cell_size));
    fvec3 lap;
    lap.x = fixed_mul((vxp.x + vxm.x + vyp.x + vym.x + vzp.x + vzm.x - 6 * vc.x), inv_h2);
    lap.y = fixed_mul((vxp.y + vxm.y + vyp.y + vym.y + vzp.y + vzm.y - 6 * vc.y), inv_h2);
    lap.z = fixed_mul((vxp.z + vxm.z + vyp.z + vym.z + vzp.z + vzm.z - 6 * vc.z), inv_h2);
    return lap;
}

// ---------------------------------------------------------------------------
// SIMD 4‑lane divergence for 4 sample points (using scalar extraction)
// ---------------------------------------------------------------------------
inline __m256i simd4_divergence(const fvec3* grid, int nx, int ny, int nz,
                                const fvec3* samples, const fvec3& origin, fixed64_t cell_size) noexcept {
    alignas(32) int64_t res[4];
    for (int i=0; i<4; ++i) {
        res[i] = vector_field_divergence(grid, nx, ny, nz, samples[i], origin, cell_size);
    }
    return _mm256_load_si256((__m256i*)res);
}

// ---------------------------------------------------------------------------
// Tensor: vorticity tensor = 0.5 * (J - J^T) (antisymmetric part)
// ---------------------------------------------------------------------------
inline fmat3 vector_field_vorticity_tensor(const fvec3* grid, int nx, int ny, int nz,
                                          const fvec3& pos, const fvec3& origin, fixed64_t cell_size) noexcept {
    fmat3 J = vector_field_gradient_tensor(grid, nx, ny, nz, pos, origin, cell_size);
    fmat3 JT = fmat3_transpose(J);
    fmat3 V;
    for (int r=0; r<3; ++r)
        for (int c=0; c<3; ++c)
            *(&V.rows[0].x + r*3 + c) = fixed_mul(FIXED64_HALF, *(&J.rows[0].x + r*3 + c) - *(&JT.rows[0].x + r*3 + c));
    return V;
}

// ---------------------------------------------------------------------------
// Deformation gradient F = I + J (linearised)
// ---------------------------------------------------------------------------
inline fmat3 deformation_gradient(const fvec3* grid, int nx, int ny, int nz,
                                 const fvec3& pos, const fvec3& origin, fixed64_t cell_size) noexcept {
    fmat3 J = vector_field_gradient_tensor(grid, nx, ny, nz, pos, origin, cell_size);
    fmat3 I = fmat3_identity();
    for (int r=0; r<3; ++r)
        for (int c=0; c<3; ++c)
            *(&J.rows[0].x + r*3 + c) = *(&I.rows[0].x + r*3 + c) + *(&J.rows[0].x + r*3 + c);
    return J;
}

} // namespace fixed_math