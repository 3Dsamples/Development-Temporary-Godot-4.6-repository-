// system name : Octree Spatial Master
//File 0025 : core/math/fixed_inertia_tensor.h
//Rigid body inertia tensor operations: compute from points/solids, Steiner's theorem, principal axes, angular momentum, kinetic energy, tetrahedron inertia, SIMD batch
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "core/math/fixed_tensor.h"
#include "core/math/fixed_spectral_decomposition.h"
#include "core/math/fixed_geometry.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace fixed_math {

// ------------------------------------------------------------
// Local helper: negate a vector
// ------------------------------------------------------------
inline fvec3 fvec3_neg(const fvec3& v) noexcept { return {-v.x, -v.y, -v.z}; }

// ------------------------------------------------------------
// Local helper: scalar multiply matrix
// ------------------------------------------------------------
inline fmat3 fmat3_mul_scalar(const fmat3& m, fixed64_t s) noexcept {
    fmat3 r;
    for (int i=0;i<3;++i) for (int j=0;j<3;++j)
        *(&r.rows[0].x + i*3 + j) = fixed_mul(*(&m.rows[0].x + i*3 + j), s);
    return r;
}

// ------------------------------------------------------------
// Local helper: matrix add
// ------------------------------------------------------------
inline fmat3 fmat3_add(const fmat3& a, const fmat3& b) noexcept {
    fmat3 r;
    for (int i=0;i<3;++i) for (int j=0;j<3;++j)
        *(&r.rows[0].x + i*3 + j) = *(&a.rows[0].x + i*3 + j) + *(&b.rows[0].x + i*3 + j);
    return r;
}

// ---------------------------------------------------------------------------
// Inertia tensor of a point mass m at position p relative to origin
// ---------------------------------------------------------------------------
inline fmat3 point_inertia_tensor(fixed64_t mass, const fvec3& p) noexcept {
    fixed64_t x2 = fixed_mul(p.x, p.x);
    fixed64_t y2 = fixed_mul(p.y, p.y);
    fixed64_t z2 = fixed_mul(p.z, p.z);
    fixed64_t xy = fixed_mul(p.x, p.y);
    fixed64_t xz = fixed_mul(p.x, p.z);
    fixed64_t yz = fixed_mul(p.y, p.z);
    fmat3 I;
    I.rows[0] = { mass * (y2 + z2), -mass * xy,        -mass * xz };
    I.rows[1] = { -mass * xy,        mass * (x2 + z2),  -mass * yz };
    I.rows[2] = { -mass * xz,       -mass * yz,          mass * (x2 + y2) };
    return I;
}

// ---------------------------------------------------------------------------
// Inertia tensor of a set of point masses (positions and masses)
// ---------------------------------------------------------------------------
inline fmat3 points_inertia_tensor(const fvec3* positions, const fixed64_t* masses, size_t count) noexcept {
    fmat3 I_total = fmat3_identity();
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) *(&I_total.rows[0].x + r*3 + c) = 0;
    for (size_t i = 0; i < count; ++i) {
        fmat3 Ii = point_inertia_tensor(masses[i], positions[i]);
        I_total = fmat3_add(I_total, Ii);
    }
    return I_total;
}

// ---------------------------------------------------------------------------
// Inertia tensor of a solid AABB with uniform density (returns I for unit density)
// ---------------------------------------------------------------------------
inline fmat3 solid_aabb_inertia_tensor(const AABB& box) noexcept {
    fvec3 e = box.extent();
    fixed64_t volume = fixed_mul(fixed_mul(e.x, e.y), e.z);
    fixed64_t dx2 = fixed_mul(e.x, e.x);
    fixed64_t dy2 = fixed_mul(e.y, e.y);
    fixed64_t dz2 = fixed_mul(e.z, e.z);
    fmat3 I;
    // I_ij = (ρ * dx * dy * dz / 12) * (δ_ij * (dx^2+dy^2+dz^2) - e_i e_j) for i==j? Actually the standard formula:
    // I_xx = (1/12) * M * (dy^2 + dz^2), etc. M = ρ*dx*dy*dz. For unit density, M = volume.
    I.rows[0] = { fixed_mul(volume, (dy2+dz2)/12), 0, 0 };
    I.rows[1] = { 0, fixed_mul(volume, (dx2+dz2)/12), 0 };
    I.rows[2] = { 0, 0, fixed_mul(volume, (dx2+dy2)/12) };
    return I;
}

// ---------------------------------------------------------------------------
// Steiner's theorem (parallel axis): translate inertia tensor I_cm to new origin
// ---------------------------------------------------------------------------
inline fmat3 translate_inertia(const fmat3& I_cm, fixed64_t mass, const fvec3& displacement) noexcept {
    fixed64_t r2 = fvec3_length_sq(displacement);
    fmat3 outer = tensor_product(displacement, displacement);
    fmat3 identity = fmat3_identity();
    fmat3 delta;
    for (int i=0;i<3;++i) {
        for (int j=0;j<3;++j) {
            fixed64_t term = (i==j) ? r2 : 0;
            term -= *(&outer.rows[0].x + i*3 + j);
            *(&delta.rows[0].x + i*3 + j) = fixed_mul(mass, term);
        }
    }
    return fmat3_add(I_cm, delta);
}

// ---------------------------------------------------------------------------
// Principal axes and moments via spectral decomposition (sorted descending)
// ---------------------------------------------------------------------------
inline void principal_inertia(const fmat3& I, fmat3& R, fixed64_t principal[3]) noexcept {
    fmat3 V;
    symmetric_eigen_decomposition(I, V, principal);
    sort_eigen_descending(V, principal);
    R = V;
}

// ---------------------------------------------------------------------------
// Rotate inertia tensor: I' = R * I * R^T
// ---------------------------------------------------------------------------
inline fmat3 rotate_inertia(const fmat3& I, const fmat3& R) noexcept {
    fmat3 RT = fmat3_transpose(R);
    return fmat3_mul(fmat3_mul(R, I), RT);
}

// ---------------------------------------------------------------------------
// Angular momentum: L = I * ω
// ---------------------------------------------------------------------------
inline fvec3 angular_momentum(const fmat3& I, const fvec3& omega) noexcept {
    return fmat3_mul_vec3(I, omega);
}

// ---------------------------------------------------------------------------
// Rotational kinetic energy: T = 0.5 * ω^T * I * ω
// ---------------------------------------------------------------------------
inline fixed64_t rotational_kinetic_energy(const fmat3& I, const fvec3& omega) noexcept {
    fvec3 L = angular_momentum(I, omega);
    return fixed_mul(FIXED64_HALF, fvec3_dot(omega, L));
}

// ---------------------------------------------------------------------------
// Inertia tensor of a solid tetrahedron (uniform unit density) about its centroid
//   Uses closed-form expression for second moments.
// ---------------------------------------------------------------------------
inline fmat3 tetrahedron_inertia_tensor(const fvec3& A, const fvec3& B, const fvec3& C, const fvec3& D) noexcept {
    // Compute the signed 6*volume
    fvec3 d1 = fvec3_sub(B, A);
    fvec3 d2 = fvec3_sub(C, A);
    fvec3 d3 = fvec3_sub(D, A);
    fixed64_t det = d1.x*(d2.y*d3.z - d2.z*d3.y) - d1.y*(d2.x*d3.z - d2.z*d3.x) + d1.z*(d2.x*d3.y - d2.y*d3.x);
    if (det < 0) det = -det;
    fixed64_t volume = det / 6; // absolute volume

    // The inertia tensor about the origin for unit density is:
    // I_origin = trace(M)*I_3 - M, where M = ∫ (x⊗x) dV over the tetrahedron.
    // M can be computed exactly as:
    // M = (V/20) * ( ∑_i a_i a_i^T + ∑_{i<j} (a_i a_j^T + a_j a_i^T) )
    // where a_i are the vertex position vectors.
    auto outer = [](const fvec3& u, const fvec3& v) -> fmat3 {
        fmat3 m;
        for (int r=0;r<3;++r) for (int c=0;c<3;++c)
            *(&m.rows[0].x + r*3 + c) = fixed_mul(*(&u.x + r), *(&v.x + c));
        return m;
    };
    auto add_outer_pair = [&](const fvec3& u, const fvec3& v, fmat3& acc) {
        fmat3 uv = outer(u, v);
        fmat3 vu = outer(v, u);
        acc = fmat3_add(acc, fmat3_add(uv, vu));
    };

    fmat3 M;
    // zero M
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) *(&M.rows[0].x + r*3 + c) = 0;

    // First sum: a_i a_i^T for i=0..3
    fmat3 temp = outer(A, A); M = fmat3_add(M, temp);
    temp = outer(B, B); M = fmat3_add(M, temp);
    temp = outer(C, C); M = fmat3_add(M, temp);
    temp = outer(D, D); M = fmat3_add(M, temp);

    // Second sum: pairs
    add_outer_pair(A, B, M);
    add_outer_pair(A, C, M);
    add_outer_pair(A, D, M);
    add_outer_pair(B, C, M);
    add_outer_pair(B, D, M);
    add_outer_pair(C, D, M);

    // M *= volume / 20
    fixed64_t factor = volume / 20;
    M = fmat3_mul_scalar(M, factor);

    // Compute trace of M
    fixed64_t tr = *(&M.rows[0].x + 0*3+0) + *(&M.rows[0].x + 1*3+1) + *(&M.rows[0].x + 2*3+2);

    // I_origin = tr*I - M
    fmat3 I_origin;
    for (int r=0;r<3;++r) {
        for (int c=0;c<3;++c) {
            fixed64_t val = (r==c) ? tr : 0;
            val -= *(&M.rows[0].x + r*3 + c);
            *(&I_origin.rows[0].x + r*3 + c) = val;
        }
    }

    // Translate to centroid: centroid = (A+B+C+D)/4
    fvec3 centroid = fvec3_scale(fvec3_add(fvec3_add(A, B), fvec3_add(C, D)), FIXED64_ONE/4);
    fmat3 I_centroid = translate_inertia(I_origin, volume, fvec3_neg(centroid));
    return I_centroid;
}

// ---------------------------------------------------------------------------
// Combine multiple bodies' inertia tensors at a common origin (using parallel axis)
// ---------------------------------------------------------------------------
inline fmat3 combine_inertia(const fmat3* I_local, const fvec3* offsets, const fixed64_t* masses, size_t count) noexcept {
    fmat3 I_total;
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) *(&I_total.rows[0].x + r*3 + c) = 0;
    for (size_t i=0; i<count; ++i) {
        fmat3 Ii = translate_inertia(I_local[i], masses[i], offsets[i]);
        I_total = fmat3_add(I_total, Ii);
    }
    return I_total;
}

// ---------------------------------------------------------------------------
// Angular acceleration from Euler's equation: I * dω/dt = τ - ω × (I ω)
// ---------------------------------------------------------------------------
inline fvec3 angular_acceleration(const fmat3& I, const fvec3& omega, const fvec3& torque) noexcept {
    fvec3 L = angular_momentum(I, omega);
    fvec3 gyro = fvec3_cross(omega, L);
    fvec3 rhs = fvec3_sub(torque, gyro);
    // Solve linear system I * dw = rhs
    fmat3 L_mat, U;
    int perm[3];
    if (fmat3_lu(I, L_mat, U, perm)) {
        fmat3_solve_lu(L_mat, U, perm, rhs); // rhs overwritten with solution
        return rhs;
    }
    // Fallback: use inverse
    return fmat3_mul_vec3(fmat3_inverse(I), rhs);
}

// ---------------------------------------------------------------------------
// SIMD batch operations
// ---------------------------------------------------------------------------
inline void simd4_angular_momentum(const fmat3 I[4], const fvec3 omega[4], fvec3 L[4]) noexcept {
    for (int i=0;i<4;++i) L[i] = angular_momentum(I[i], omega[i]);
}
inline __m256i simd4_rotational_kinetic_energy(const fmat3 I[4], const fvec3 omega[4]) noexcept {
    alignas(32) fixed64_t res[4];
    for (int i=0;i<4;++i) res[i] = rotational_kinetic_energy(I[i], omega[i]);
    return _mm256_load_si256((__m256i*)res);
}

} // namespace fixed_math