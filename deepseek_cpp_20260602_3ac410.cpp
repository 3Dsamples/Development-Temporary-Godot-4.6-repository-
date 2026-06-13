// system name : Octree Spatial Master
//File 0015 : core/math/fixed_optimization.h
//Fixed‑point optimization routines: gradient descent, Newton's method, line search, scalar field interpolation, gradient/Hessian of fields, and tensor‑based Newton steps
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "core/math/fixed_tensor.h"
#include "core/math/fixed_interpolation.h"
#include "core/math/fixed_noise.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <functional>

namespace fixed_math {

// ---------------------------------------------------------------------------
// Scalar field evaluation at a point using trilinear interpolation on a uniform 3D grid
// ---------------------------------------------------------------------------
inline fixed64_t field_trilinear(const fixed64_t* grid, int nx, int ny, int nz,
                                 const fvec3& pos, const fvec3& origin, fixed64_t cell_size) noexcept {
    // Convert position to fractional grid coordinates
    fvec3 local = fvec3_sub(pos, origin);
    fixed64_t fx = fixed_div(local.x, cell_size);
    fixed64_t fy = fixed_div(local.y, cell_size);
    fixed64_t fz = fixed_div(local.z, cell_size);
    int ix = (int)(fx >> FRAC_BITS);
    int iy = (int)(fy >> FRAC_BITS);
    int iz = (int)(fz >> FRAC_BITS);
    // Clamp to grid bounds
    ix = (ix < 0) ? 0 : (ix >= nx-1) ? nx-2 : ix;
    iy = (iy < 0) ? 0 : (iy >= ny-1) ? ny-2 : iy;
    iz = (iz < 0) ? 0 : (iz >= nz-1) ? nz-2 : iz;
    fixed64_t tx = fx - (fixed64_t(ix) << FRAC_BITS);
    fixed64_t ty = fy - (fixed64_t(iy) << FRAC_BITS);
    fixed64_t tz = fz - (fixed64_t(iz) << FRAC_BITS);
    // Trilinear weights
    fixed64_t u = tx, v = ty, w = tz;
    fixed64_t u1 = FIXED64_ONE - u, v1 = FIXED64_ONE - v, w1 = FIXED64_ONE - w;
    // Access grid values
    auto val = [&](int x, int y, int z) -> fixed64_t { return grid[z*ny*nx + y*nx + x]; };
    fixed64_t c000 = val(ix,   iy,   iz);
    fixed64_t c100 = val(ix+1, iy,   iz);
    fixed64_t c010 = val(ix,   iy+1, iz);
    fixed64_t c110 = val(ix+1, iy+1, iz);
    fixed64_t c001 = val(ix,   iy,   iz+1);
    fixed64_t c101 = val(ix+1, iy,   iz+1);
    fixed64_t c011 = val(ix,   iy+1, iz+1);
    fixed64_t c111 = val(ix+1, iy+1, iz+1);
    return fixed_mul(w1, (fixed_mul(v1, (fixed_mul(u1, c000) + fixed_mul(u, c100))) +
                          fixed_mul(v,  (fixed_mul(u1, c010) + fixed_mul(u, c110))))) +
           fixed_mul(w,  (fixed_mul(v1, (fixed_mul(u1, c001) + fixed_mul(u, c101))) +
                          fixed_mul(v,  (fixed_mul(u1, c011) + fixed_mul(u, c111)))));
}

// ---------------------------------------------------------------------------
// Gradient of a scalar field via central finite differences on the grid
// ---------------------------------------------------------------------------
inline fvec3 field_gradient(const fixed64_t* grid, int nx, int ny, int nz,
                            const fvec3& pos, const fvec3& origin, fixed64_t cell_size) noexcept {
    fvec3 local = fvec3_sub(pos, origin);
    fixed64_t fx = fixed_div(local.x, cell_size);
    fixed64_t fy = fixed_div(local.y, cell_size);
    fixed64_t fz = fixed_div(local.z, cell_size);
    int ix = (int)(fx >> FRAC_BITS);
    int iy = (int)(fy >> FRAC_BITS);
    int iz = (int)(fz >> FRAC_BITS);
    // Clamp to allow central difference within interior
    ix = (ix < 1) ? 1 : (ix >= nx-2) ? nx-3 : ix;
    iy = (iy < 1) ? 1 : (iy >= ny-2) ? ny-3 : iy;
    iz = (iz < 1) ? 1 : (iz >= nz-2) ? nz-3 : iz;
    auto v = [&](int x, int y, int z) -> fixed64_t { return grid[z*ny*nx + y*nx + x]; };
    fixed64_t dx = fixed_div(v(ix+1, iy, iz) - v(ix-1, iy, iz), 2 * cell_size);
    fixed64_t dy = fixed_div(v(ix, iy+1, iz) - v(ix, iy-1, iz), 2 * cell_size);
    fixed64_t dz = fixed_div(v(ix, iy, iz+1) - v(ix, iy, iz-1), 2 * cell_size);
    return {dx, dy, dz};
}

// ---------------------------------------------------------------------------
// Hessian of a scalar field (3x3 matrix) via finite differences on the grid
// ---------------------------------------------------------------------------
inline fmat3 field_hessian(const fixed64_t* grid, int nx, int ny, int nz,
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
    auto v = [&](int x, int y, int z) -> fixed64_t { return grid[z*ny*nx + y*nx + x]; };
    fixed64_t h2 = fixed_mul(cell_size, cell_size);
    fixed64_t dxx = fixed_div(v(ix+1, iy, iz) - 2*v(ix, iy, iz) + v(ix-1, iy, iz), h2);
    fixed64_t dyy = fixed_div(v(ix, iy+1, iz) - 2*v(ix, iy, iz) + v(ix, iy-1, iz), h2);
    fixed64_t dzz = fixed_div(v(ix, iy, iz+1) - 2*v(ix, iy, iz) + v(ix, iy, iz-1), h2);
    fixed64_t dxy = fixed_div(v(ix+1, iy+1, iz) - v(ix-1, iy+1, iz) - v(ix+1, iy-1, iz) + v(ix-1, iy-1, iz), 4 * h2);
    fixed64_t dxz = fixed_div(v(ix+1, iy, iz+1) - v(ix-1, iy, iz+1) - v(ix+1, iy, iz-1) + v(ix-1, iy, iz-1), 4 * h2);
    fixed64_t dyz = fixed_div(v(ix, iy+1, iz+1) - v(ix, iy-1, iz+1) - v(ix, iy+1, iz-1) + v(ix, iy-1, iz-1), 4 * h2);
    fmat3 H;
    H.rows[0] = {dxx, dxy, dxz};
    H.rows[1] = {dxy, dyy, dyz};
    H.rows[2] = {dxz, dyz, dzz};
    return H;
}

// ---------------------------------------------------------------------------
// Gradient descent solver for a scalar function f: R^3 -> R
//   x0 is initial guess, alpha is step size, max_iter iterations, ftol convergence.
// ---------------------------------------------------------------------------
inline fvec3 gradient_descent(const std::function<fixed64_t(const fvec3&)>& f,
                              const std::function<fvec3(const fvec3&)>& grad,
                              fvec3 x0, fixed64_t alpha, int max_iter, fixed64_t ftol) noexcept {
    fvec3 x = x0;
    fixed64_t fx = f(x);
    for (int i = 0; i < max_iter; ++i) {
        fvec3 g = grad(x);
        fixed64_t len_g = fvec3_length(g);
        if (len_g < ftol) break;
        fvec3 x_new = fvec3_sub(x, fvec3_scale(g, alpha));
        fixed64_t fx_new = f(x_new);
        if (fx_new < fx) {
            x = x_new;
            fx = fx_new;
        } else {
            alpha = fixed_mul(alpha, FIXED64_HALF); // reduce step
        }
    }
    return x;
}

// ---------------------------------------------------------------------------
// Newton's method for scalar function f: R^3 -> R using gradient and Hessian
//   solve H * dx = -g for dx, then x = x + dx
// ---------------------------------------------------------------------------
inline fvec3 newton_raphson(const std::function<fvec3(const fvec3&)>& grad,
                            const std::function<fmat3(const fvec3&)>& hess,
                            fvec3 x0, int max_iter, fixed64_t tol) noexcept {
    fvec3 x = x0;
    for (int i = 0; i < max_iter; ++i) {
        fvec3 g = grad(x);
        if (fvec3_length(g) < tol) break;
        fmat3 H = hess(x);
        // Solve H * dx = -g using Cholesky if positive definite, else LU
        fmat3 L, U;
        int perm[3];
        fvec3 rhs = fvec3_neg(g);
        if (fmat3_cholesky(H, L)) {
            // Forward substitution L * y = rhs, then L^T * dx = y
            fvec3 y;
            for (int r=0; r<3; ++r) {
                fixed64_t sum = 0;
                for (int c=0; c<r; ++c) sum += fixed_mul(*(&L.rows[0].x + r*3 + c), *(&y.x + c));
                *(&y.x + r) = fixed_div(*(&rhs.x + r) - sum, *(&L.rows[0].x + r*3 + r));
            }
            fvec3 dx;
            for (int r=2; r>=0; --r) {
                fixed64_t sum = 0;
                for (int c=r+1; c<3; ++c) sum += fixed_mul(*(&L.rows[0].x + c*3 + r), *(&dx.x + c)); // L^T(r,c) = L(c,r)
                *(&dx.x + r) = fixed_div(*(&y.x + r) - sum, *(&L.rows[0].x + r*3 + r));
            }
            x = fvec3_add(x, dx);
        } else if (fmat3_lu(H, L, U, perm)) {
            if (fmat3_solve_lu(L, U, perm, rhs)) {
                x = fvec3_add(x, rhs); // rhs overwritten with dx
            }
        } else break;
    }
    return x;
}

// ---------------------------------------------------------------------------
// Backtracking line search for a step x + t*direction
//   returns t that satisfies Armijo condition
// ---------------------------------------------------------------------------
inline fixed64_t line_search_backtrack(const std::function<fixed64_t(const fvec3&)>& f,
                                       const std::function<fvec3(const fvec3&)>& grad,
                                       const fvec3& x, const fvec3& direction,
                                       fixed64_t alpha_init = FIXED64_ONE,
                                       fixed64_t beta = FIXED64_HALF,
                                       fixed64_t c = FIXED64_ONE / 10000, // ~0.0001
                                       int max_iter = 16) noexcept {
    fixed64_t t = alpha_init;
    fixed64_t fx = f(x);
    fixed64_t gdir = fvec3_dot(grad(x), direction);
    fvec3 x_new = fvec3_add(x, fvec3_scale(direction, t));
    for (int i = 0; i < max_iter; ++i) {
        if (f(x_new) <= fx + fixed_mul(c, fixed_mul(t, gdir))) break;
        t = fixed_mul(t, beta);
        x_new = fvec3_add(x, fvec3_scale(direction, t));
    }
    return t;
}

} // namespace fixed_math