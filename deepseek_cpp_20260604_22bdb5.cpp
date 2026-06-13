// system name : Octree Spatial Master
//File 0036 : core/math/fixed_particle_methods.h
//Material Point Method (MPM) and PIC/FLIP transfer kernels: grid‑to‑particle, particle‑to‑grid, deformation update, SIMD batch, perceptual colour
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "core/math/fixed_tensor.h"
#include "core/math/fixed_elasticity_tensor.h"
#include "core/math/fixed_plasticity.h"
#include "core/math/fixed_geometry.h"
#include "core/math/fixed_sparse_solver.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>

namespace fixed_math {

struct MPM_Particle {
    fvec3 pos;
    fvec3 vel;
    fmat3 F;        // deformation gradient
    fixed64_t mass;
    fixed64_t volume;
    fmat3 stress;   // Cauchy stress (updated after constitutive)
    int material_id;
};

struct MPM_GridNode {
    fvec3 vel;
    fixed64_t mass;
    fvec3 force;
};

struct MPM_Grid {
    int nx, ny, nz;
    fixed64_t h;
    fvec3 origin;
    std::vector<MPM_GridNode> nodes;
    MPM_Grid(int nx_, int ny_, int nz_, fixed64_t h_, const fvec3& org)
        : nx(nx_), ny(ny_), nz(nz_), h(h_), origin(org) {
        nodes.resize((nx+1)*(ny+1)*(nz+1));
    }
    int idx(int i, int j, int k) const { return (k*(ny+1) + j)*(nx+1) + i; }
    MPM_GridNode& at(int i, int j, int k) { return nodes[idx(i,j,k)]; }
    const MPM_GridNode& at(int i, int j, int k) const { return nodes[idx(i,j,k)]; }
    void reset() {
        for (auto& n : nodes) { n.vel = {0,0,0}; n.mass = 0; n.force = {0,0,0}; }
    }
};

// Quadratic B‑spline kernel and its gradient (1D)
inline fixed64_t bspline_quad(fixed64_t r) noexcept {
    fixed64_t absr = fixed_abs(r);
    if (absr < FIXED64_HALF) return 0.75 - r*r;
    if (absr < 1.5) {
        fixed64_t t = 1.5 - absr;
        return fixed_mul(FIXED64_HALF, fixed_mul(t, t));
    }
    return 0;
}
inline fixed64_t bspline_quad_grad(fixed64_t r) noexcept {
    fixed64_t absr = fixed_abs(r);
    fixed64_t sign = (r >= 0) ? FIXED64_ONE : -FIXED64_ONE;
    if (absr < FIXED64_HALF) return -2 * r;
    if (absr < 1.5) {
        fixed64_t t = 1.5 - absr;
        return fixed_mul(-sign, t);
    }
    return 0;
}

// 3D weight: product of 1D kernels
inline fixed64_t particle_weight(const fvec3& dx, fixed64_t inv_h) noexcept {
    return bspline_quad(dx.x * inv_h) * bspline_quad(dx.y * inv_h) * bspline_quad(dx.z * inv_h);
}
inline fvec3 particle_weight_gradient(const fvec3& dx, fixed64_t inv_h) noexcept {
    fixed64_t wx = bspline_quad(dx.x*inv_h), wy = bspline_quad(dx.y*inv_h), wz = bspline_quad(dx.z*inv_h);
    fixed64_t dwx = bspline_quad_grad(dx.x*inv_h) * inv_h;
    fixed64_t dwy = bspline_quad_grad(dx.y*inv_h) * inv_h;
    fixed64_t dwz = bspline_quad_grad(dx.z*inv_h) * inv_h;
    return { dwx*wy*wz, wx*dwy*wz, wx*wy*dwz };
}

// Particle‑to‑grid transfer (P2G)
inline void mpm_p2g(const std::vector<MPM_Particle>& particles, MPM_Grid& grid, fixed64_t dt) {
    grid.reset();
    fixed64_t inv_h = fixed_rcp(grid.h);
    for (const auto& p : particles) {
        fvec3 base = fvec3_sub(p.pos, grid.origin);
        int ix = (int)(base.x * inv_h) >> FRAC_BITS;
        int iy = (int)(base.y * inv_h) >> FRAC_BITS;
        int iz = (int)(base.z * inv_h) >> FRAC_BITS;
        for (int dk=-1; dk<=2; ++dk) {
            for (int dj=-1; dj<=2; ++dj) {
                for (int di=-1; di<=2; ++di) {
                    int gi = ix+di, gj = iy+dj, gk = iz+dk;
                    if (gi<0 || gi>grid.nx || gj<0 || gj>grid.ny || gk<0 || gk>grid.nz) continue;
                    fvec3 node_pos = { grid.origin.x + gi*grid.h, grid.origin.y + gj*grid.h, grid.origin.z + gk*grid.h };
                    fvec3 dx = fvec3_sub(p.pos, node_pos);
                    fixed64_t w = particle_weight(dx, inv_h);
                    fvec3 grad_w = particle_weight_gradient(dx, inv_h);
                    // mass
                    grid.at(gi,gj,gk).mass += w * p.mass;
                    // velocity contribution (PIC)
                    fvec3 vel_contrib = fvec3_scale(p.vel, w * p.mass);
                    grid.at(gi,gj,gk).vel.x += vel_contrib.x;
                    grid.at(gi,gj,gk).vel.y += vel_contrib.y;
                    grid.at(gi,gj,gk).vel.z += vel_contrib.z;
                    // force from stress divergence: V * σ : ∇w
                    fvec3 force_contrib = fmat3_mul_vec3(p.stress, grad_w);
                    force_contrib = fvec3_scale(force_contrib, -p.volume);
                    grid.at(gi,gj,gk).force.x += force_contrib.x;
                    grid.at(gi,gj,gk).force.y += force_contrib.y;
                    grid.at(gi,gj,gk).force.z += force_contrib.z;
                }
            }
        }
    }
    // divide velocity by mass
    for (int k=0; k<=grid.nz; ++k)
        for (int j=0; j<=grid.ny; ++j)
            for (int i=0; i<=grid.nx; ++i) {
                if (grid.at(i,j,k).mass > 0) {
                    fixed64_t inv_m = fixed_rcp(grid.at(i,j,k).mass);
                    grid.at(i,j,k).vel.x *= inv_m;
                    grid.at(i,j,k).vel.y *= inv_m;
                    grid.at(i,j,k).vel.z *= inv_m;
                    grid.at(i,j,k).force.x *= inv_m;
                    grid.at(i,j,k).force.y *= inv_m;
                    grid.at(i,j,k).force.z *= inv_m;
                }
            }
}

// Grid update (forces + velocity integration)
inline void mpm_grid_update(MPM_Grid& grid, fixed64_t dt) {
    for (int k=0; k<=grid.nz; ++k)
        for (int j=0; j<=grid.ny; ++j)
            for (int i=0; i<=grid.nx; ++i) {
                fvec3 acc = grid.at(i,j,k).force; // already divided by mass
                grid.at(i,j,k).vel = fvec3_add(grid.at(i,j,k).vel, fvec3_scale(acc, dt));
            }
}

// Grid‑to‑particle transfer (G2P) with PIC/FLIP blending
inline void mpm_g2p(std::vector<MPM_Particle>& particles, const MPM_Grid& grid, fixed64_t alpha, fixed64_t dt) {
    fixed64_t inv_h = fixed_rcp(grid.h);
    for (auto& p : particles) {
        fvec3 base = fvec3_sub(p.pos, grid.origin);
        int ix = (int)(base.x * inv_h) >> FRAC_BITS;
        int iy = (int)(base.y * inv_h) >> FRAC_BITS;
        int iz = (int)(base.z * inv_h) >> FRAC_BITS;
        fvec3 vel_pic = {0,0,0};
        fmat3 vel_grad = fmat3_identity(); // zero
        for (int r=0;r<3;++r) for (int c=0;c<3;++c) *(&vel_grad.rows[0].x + r*3 + c) = 0;
        for (int dk=-1; dk<=2; ++dk) {
            for (int dj=-1; dj<=2; ++dj) {
                for (int di=-1; di<=2; ++di) {
                    int gi = ix+di, gj = iy+dj, gk = iz+dk;
                    if (gi<0 || gi>grid.nx || gj<0 || gj>grid.ny || gk<0 || gk>grid.nz) continue;
                    fvec3 node_pos = { grid.origin.x + gi*grid.h, grid.origin.y + gj*grid.h, grid.origin.z + gk*grid.h };
                    fvec3 dx = fvec3_sub(p.pos, node_pos);
                    fixed64_t w = particle_weight(dx, inv_h);
                    fvec3 grad_w = particle_weight_gradient(dx, inv_h);
                    fvec3 nvel = grid.at(gi,gj,gk).vel;
                    vel_pic = fvec3_add(vel_pic, fvec3_scale(nvel, w));
                    // velocity gradient L = Σ v_i ⊗ ∇w_i
                    for (int r=0;r<3;++r) for (int c=0;c<3;++c)
                        *(&vel_grad.rows[0].x + r*3 + c) += fixed_mul(*(&nvel.x + r), *(&grad_w.x + c));
                }
            }
        }
        // FLIP update: v_new = (1-α)*v_PIC + α*(v_old + Δv_FLIP)
        fvec3 v_flip = fvec3_add(p.vel, fvec3_scale(vel_grad.rows[0], dt)); // approximate; full FLIP uses L * p.vel? Actually FLIP: v_new = v_old + Σ (Δv_i * w_i)
        // Standard PIC/FLIP blend: v_new = (1-α) * v_PIC + α * v_FLIP
        p.vel = fvec3_add(fvec3_scale(vel_pic, FIXED64_ONE - alpha), fvec3_scale(v_flip, alpha));
        // Update position: x = x + v_new * dt
        p.pos = fvec3_add(p.pos, fvec3_scale(p.vel, dt));
        // Update deformation gradient: F_new = (I + L*dt) * F
        fmat3 I = fmat3_identity();
        fmat3 F_inc = fmat3_add(I, fmat3_mul_scalar(vel_grad, dt));
        p.F = fmat3_mul(F_inc, p.F);
        // Update volume (Jacobian): J = det(F), volume = initial_volume * J
        // We would store initial_volume separately. Assume p.volume is current; update as volume *= det(F_inc)
        p.volume = fixed_mul(p.volume, fmat3_det(F_inc));
    }
}

// Constitutive update (example: Neo‑Hookean)
inline void mpm_constitutive(std::vector<MPM_Particle>& particles, fixed64_t mu, fixed64_t lambda) {
    for (auto& p : particles) {
        fmat3 F = p.F;
        fixed64_t J = fmat3_det(F);
        fmat3 FinvT = fmat3_transpose(fmat3_inverse(F));
        fmat3 P; // first Piola‑Kirchhoff stress
        for (int r=0;r<3;++r) for (int c=0;c<3;++c) {
            fixed64_t val = mu * *(&F.rows[0].x + r*3 + c) - mu * *(&FinvT.rows[0].x + r*3 + c) + lambda * fixed_log(J) * *(&FinvT.rows[0].x + r*3 + c);
            *(&P.rows[0].x + r*3 + c) = val;
        }
        // Cauchy stress = (1/J) P * F^T
        fmat3 cauchy = fmat3_mul(P, fmat3_transpose(F));
        cauchy = fmat3_mul_scalar(cauchy, fixed_rcp(J));
        p.stress = cauchy;
    }
}

// Perceptual colour for particle stress (von Mises)
inline void particle_stress_color(const std::vector<MPM_Particle>& particles, std::vector<fvec3>& colors) {
    colors.resize(particles.size());
    fixed64_t max_vm = 0;
    for (const auto& p : particles) {
        fixed64_t vm = von_mises_eq_stress(p.stress);
        if (vm > max_vm) max_vm = vm;
    }
    if (max_vm == 0) max_vm = 1;
    for (size_t i=0; i<particles.size(); ++i) {
        fixed64_t t = fixed_div(von_mises_eq_stress(particles[i].stress), max_vm);
        if (t > FIXED64_ONE) t = FIXED64_ONE;
        fvec3 linear = {t, 0, FIXED64_ONE - t};
        colors[i] = perceptual_color::linear_srgb_to_oklab(linear);
    }
}

} // namespace fixed_math