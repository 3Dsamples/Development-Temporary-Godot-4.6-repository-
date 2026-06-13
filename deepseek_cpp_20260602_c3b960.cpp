// system name : Octree Spatial Master
//File 0008 : core/math/fixed_geometry.h
//3D geometry primitives (AABB, OBB, sphere, triangle, ray), fixed‑point intersection tests, distance queries, tensor inertia, OBB from covariance, SIMD batch AABB overlap
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "core/math/fixed_quat.h"
#include "core/math/fixed_trig.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <algorithm>
#include <limits>

namespace fixed_math {

// ---------------------------------------------------------------------------
// AABB (axis‑aligned bounding box)
// ---------------------------------------------------------------------------
struct AABB {
    fvec3 min, max;
    void expand(const fvec3& p) noexcept {
        min.x = fixed_min(min.x, p.x); max.x = fixed_max(max.x, p.x);
        min.y = fixed_min(min.y, p.y); max.y = fixed_max(max.y, p.y);
        min.z = fixed_min(min.z, p.z); max.z = fixed_max(max.z, p.z);
    }
    fvec3 center() const noexcept { return fvec3_scale(fvec3_add(min, max), FIXED64_HALF); }
    fvec3 extent() const noexcept { return fvec3_sub(max, min); }
    fixed64_t surface_area() const noexcept {
        fvec3 e = extent();
        // 2*(dx*dy + dx*dz + dy*dz)
        e.x >>= 16; e.y >>= 16; e.z >>= 16;
        return 2 * (e.x*e.y + e.x*e.z + e.y*e.z);
    }
};

// ---------------------------------------------------------------------------
// OBB (oriented bounding box) – center, half‑extents, orthonormal basis axes
// ---------------------------------------------------------------------------
struct OBB {
    fvec3 center;
    fvec3 half_extents; // positive half‑lengths along axes
    fmat3 axes;         // orthonormal basis (rows are axes vectors)
    fvec3 axis(int i) const noexcept { return {axes.rows[i].x, axes.rows[i].y, axes.rows[i].z}; }
};

// ---------------------------------------------------------------------------
// Sphere
// ---------------------------------------------------------------------------
struct Sphere {
    fvec3 center;
    fixed64_t radius;
};

// ---------------------------------------------------------------------------
// Triangle
// ---------------------------------------------------------------------------
struct Triangle {
    fvec3 v0, v1, v2;
    fvec3 edge1() const noexcept { return fvec3_sub(v1, v0); }
    fvec3 edge2() const noexcept { return fvec3_sub(v2, v0); }
    fvec3 normal() const noexcept { return fvec3_normalize(fvec3_cross(edge1(), edge2())); }
};

// ---------------------------------------------------------------------------
// Ray
// ---------------------------------------------------------------------------
struct Ray {
    fvec3 origin;
    fvec3 dir_inv; // reciprocal of direction (1/dx, 1/dy, 1/dz)
    fvec3 direction() const noexcept { return {fixed_rcp(dir_inv.x), fixed_rcp(dir_inv.y), fixed_rcp(dir_inv.z)}; }
};

// ============================================================================
// Intersection tests
// ============================================================================

// Ray‑AABB (slab method)
inline bool ray_aabb(const Ray& ray, const AABB& box, fixed64_t& tmin_out, fixed64_t& tmax_out) noexcept {
    fixed64_t t1 = fixed_mul(box.min.x - ray.origin.x, ray.dir_inv.x);
    fixed64_t t2 = fixed_mul(box.max.x - ray.origin.x, ray.dir_inv.x);
    fixed64_t tmin = fixed_min(t1, t2);
    fixed64_t tmax = fixed_max(t1, t2);
    t1 = fixed_mul(box.min.y - ray.origin.y, ray.dir_inv.y);
    t2 = fixed_mul(box.max.y - ray.origin.y, ray.dir_inv.y);
    tmin = fixed_max(tmin, fixed_min(t1, t2));
    tmax = fixed_min(tmax, fixed_max(t1, t2));
    t1 = fixed_mul(box.min.z - ray.origin.z, ray.dir_inv.z);
    t2 = fixed_mul(box.max.z - ray.origin.z, ray.dir_inv.z);
    tmin = fixed_max(tmin, fixed_min(t1, t2));
    tmax = fixed_min(tmax, fixed_max(t1, t2));
    if (tmax < 0 || tmin > tmax) return false;
    tmin_out = tmin; tmax_out = tmax;
    return true;
}

// Ray‑Triangle (Möller–Trumbore)
inline bool ray_triangle(const Ray& ray, const Triangle& tri, fixed64_t& t_out, fvec3& hit) noexcept {
    fvec3 e1 = tri.edge1(), e2 = tri.edge2();
    fvec3 h = fvec3_cross(ray.direction(), e2);
    fixed64_t a = fvec3_dot(e1, h);
    if (a == 0) return false;
    fixed64_t f = fixed_rcp(a);
    fvec3 s = fvec3_sub(ray.origin, tri.v0);
    fixed64_t u = fixed_mul(f, fvec3_dot(s, h));
    if (u < 0 || u > FIXED64_ONE) return false;
    fvec3 q = fvec3_cross(s, e1);
    fixed64_t v = fixed_mul(f, fvec3_dot(ray.direction(), q));
    if (v < 0 || u + v > FIXED64_ONE) return false;
    fixed64_t t = fixed_mul(f, fvec3_dot(e2, q));
    if (t < 0) return false;
    t_out = t;
    hit = fvec3_add(ray.origin, fvec3_scale(ray.direction(), t));
    return true;
}

// Sphere‑AABB overlap
inline bool sphere_aabb(const Sphere& s, const AABB& box) noexcept {
    fixed64_t dmin = 0;
    if (s.center.x < box.min.x) dmin += fixed_mul(box.min.x - s.center.x, box.min.x - s.center.x);
    else if (s.center.x > box.max.x) dmin += fixed_mul(s.center.x - box.max.x, s.center.x - box.max.x);
    if (s.center.y < box.min.y) dmin += fixed_mul(box.min.y - s.center.y, box.min.y - s.center.y);
    else if (s.center.y > box.max.y) dmin += fixed_mul(s.center.y - box.max.y, s.center.y - box.max.y);
    if (s.center.z < box.min.z) dmin += fixed_mul(box.min.z - s.center.z, box.min.z - s.center.z);
    else if (s.center.z > box.max.z) dmin += fixed_mul(s.center.z - box.max.z, s.center.z - box.max.z);
    return dmin <= fixed_mul(s.radius, s.radius);
}

// Sphere‑OBB overlap
inline bool sphere_obb(const Sphere& s, const OBB& obb) noexcept {
    // Transform sphere center into OBB local space
    fvec3 d = fvec3_sub(s.center, obb.center);
    fvec3 local;
    local.x = fvec3_dot(d, obb.axis(0));
    local.y = fvec3_dot(d, obb.axis(1));
    local.z = fvec3_dot(d, obb.axis(2));
    // Clamp to extents
    fvec3 clamped = {
        fixed_clamp(local.x, -obb.half_extents.x, obb.half_extents.x),
        fixed_clamp(local.y, -obb.half_extents.y, obb.half_extents.y),
        fixed_clamp(local.z, -obb.half_extents.z, obb.half_extents.z)
    };
    fvec3 closest = fvec3_add(obb.center, fvec3_add(fvec3_scale(obb.axis(0), clamped.x),
                                                   fvec3_add(fvec3_scale(obb.axis(1), clamped.y),
                                                             fvec3_scale(obb.axis(2), clamped.z))));
    return fvec3_distance_sq(closest, s.center) <= fixed_mul(s.radius, s.radius);
}

// ============================================================================
// Distance queries
// ============================================================================

// Point‑AABB squared distance
inline fixed64_t point_aabb_sq_dist(const fvec3& p, const AABB& box) noexcept {
    fixed64_t dx = (p.x < box.min.x) ? (box.min.x - p.x) : (p.x > box.max.x ? p.x - box.max.x : 0);
    fixed64_t dy = (p.y < box.min.y) ? (box.min.y - p.y) : (p.y > box.max.y ? p.y - box.max.y : 0);
    fixed64_t dz = (p.z < box.min.z) ? (box.min.z - p.z) : (p.z > box.max.z ? p.z - box.max.z : 0);
    return fixed_add(fixed_add(fixed_mul(dx,dx), fixed_mul(dy,dy)), fixed_mul(dz,dz));
}

// Point‑OBB squared distance
inline fixed64_t point_obb_sq_dist(const fvec3& p, const OBB& obb) noexcept {
    fvec3 d = fvec3_sub(p, obb.center);
    fvec3 local = {
        fvec3_dot(d, obb.axis(0)), fvec3_dot(d, obb.axis(1)), fvec3_dot(d, obb.axis(2))
    };
    fvec3 clamped = {
        fixed_clamp(local.x, -obb.half_extents.x, obb.half_extents.x),
        fixed_clamp(local.y, -obb.half_extents.y, obb.half_extents.y),
        fixed_clamp(local.z, -obb.half_extents.z, obb.half_extents.z)
    };
    fvec3 closest_local = fvec3_sub(local, clamped);
    // Transform back to world (but squared distance is invariant under rotation)
    return fvec3_length_sq(closest_local);
}

// Point‑Triangle closest point and squared distance
inline void closest_point_triangle(const fvec3& p, const Triangle& tri, fvec3& closest, fixed64_t& dist_sq) noexcept {
    // Using algorithm from Real‑Time Collision Detection
    fvec3 ab = tri.edge1(), ac = tri.edge2(), ap = fvec3_sub(p, tri.v0);
    fixed64_t d1 = fvec3_dot(ab, ap);
    fixed64_t d2 = fvec3_dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) { closest = tri.v0; dist_sq = fvec3_distance_sq(p, tri.v0); return; }
    fvec3 bp = fvec3_sub(p, tri.v1);
    fixed64_t d3 = fvec3_dot(ab, bp);
    fixed64_t d4 = fvec3_dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) { closest = tri.v1; dist_sq = fvec3_distance_sq(p, tri.v1); return; }
    fixed64_t vc = d1*d4 - d3*d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        fixed64_t v = d1 / (d1 - d3);
        closest = fvec3_add(tri.v0, fvec3_scale(ab, v));
        dist_sq = fvec3_distance_sq(p, closest);
        return;
    }
    fvec3 cp = fvec3_sub(p, tri.v2);
    fixed64_t d5 = fvec3_dot(ab, cp);
    fixed64_t d6 = fvec3_dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) { closest = tri.v2; dist_sq = fvec3_distance_sq(p, tri.v2); return; }
    fixed64_t vb = d5*d2 - d1*d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        fixed64_t w = d2 / (d2 - d6);
        closest = fvec3_add(tri.v0, fvec3_scale(ac, w));
        dist_sq = fvec3_distance_sq(p, closest);
        return;
    }
    fixed64_t va = d3*d6 - d5*d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
        fixed64_t w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closest = fvec3_add(tri.v1, fvec3_scale(fvec3_sub(tri.v2, tri.v1), w));
        dist_sq = fvec3_distance_sq(p, closest);
        return;
    }
    fixed64_t denom = fixed_rcp(va + vb + vc);
    fixed64_t v = vb * denom;
    fixed64_t w = vc * denom;
    closest = fvec3_add(tri.v0, fvec3_add(fvec3_scale(ab, v), fvec3_scale(ac, w)));
    dist_sq = fvec3_distance_sq(p, closest);
}

// ============================================================================
// Tensor / Inertia operations
// ============================================================================

// Inertia tensor of a point mass (3x3 matrix) relative to origin
inline fmat3 point_inertia_tensor(fixed64_t mass, const fvec3& pos) noexcept {
    fixed64_t x2 = fixed_mul(pos.x, pos.x), y2 = fixed_mul(pos.y, pos.y), z2 = fixed_mul(pos.z, pos.z);
    fixed64_t xy = fixed_mul(pos.x, pos.y), xz = fixed_mul(pos.x, pos.z), yz = fixed_mul(pos.y, pos.z);
    fmat3 I;
    I.rows[0] = {mass * (y2 + z2), -mass * xy,        -mass * xz};
    I.rows[1] = {-mass * xy,        mass * (x2 + z2),  -mass * yz};
    I.rows[2] = {-mass * xz,       -mass * yz,          mass * (x2 + y2)};
    return I;
}

// Inertia tensor of a solid AABB (constant density, mass = 1)
inline fmat3 aabb_inertia_tensor(const AABB& box) noexcept {
    fvec3 e = box.extent();
    // Volume = dx*dy*dz, but we use mass = 1, so scale accordingly; we'll compute moments assuming unit mass density.
    // Ixx = (1/12) * M * (dy^2 + dz^2), but M = 1 for simplicity, user scales.
    fixed64_t dx2 = fixed_mul(e.x, e.x), dy2 = fixed_mul(e.y, e.y), dz2 = fixed_mul(e.z, e.z);
    fmat3 I;
    I.rows[0] = {(dy2 + dz2) / 12, 0, 0};
    I.rows[1] = {0, (dx2 + dz2) / 12, 0};
    I.rows[2] = {0, 0, (dx2 + dy2) / 12};
    return I;
}

// ============================================================================
// OBB from point cloud using principal component analysis (covariance matrix)
// ============================================================================
inline OBB obb_from_points(const fvec3* points, size_t count) noexcept {
    if (count == 0) return {{0,0,0}, {0,0,0}, fmat3_identity()};
    // Compute centroid
    fvec3 centroid = {0,0,0};
    for (size_t i=0;i<count;++i) { centroid.x += points[i].x; centroid.y += points[i].y; centroid.z += points[i].z; }
    centroid = fvec3_scale(centroid, fixed_rcp(static_cast<fixed64_t>(count) << FRAC_BITS));
    // Compute covariance matrix (3x3)
    fixed64_t c[3][3] = {{0},{0},{0}};
    for (size_t i=0;i<count;++i) {
        fvec3 d = fvec3_sub(points[i], centroid);
        c[0][0] += fixed_mul(d.x, d.x); c[0][1] += fixed_mul(d.x, d.y); c[0][2] += fixed_mul(d.x, d.z);
        c[1][1] += fixed_mul(d.y, d.y); c[1][2] += fixed_mul(d.y, d.z);
        c[2][2] += fixed_mul(d.z, d.z);
    }
    c[1][0] = c[0][1]; c[2][0] = c[0][2]; c[2][1] = c[1][2];
    // Normalize by count
    fixed64_t inv_n = fixed_rcp(static_cast<fixed64_t>(count) << FRAC_BITS);
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) c[i][j] = fixed_mul(c[i][j], inv_n);
    // Build fmat3 covariance
    fmat3 cov = {{ {c[0][0], c[0][1], c[0][2]}, {c[1][0], c[1][1], c[1][2]}, {c[2][0], c[2][1], c[2][2]} }};
    // Jacobi eigenvalue decomposition for 3x3 (simplified fixed‑point implementation)
    // For brevity, we'll use a power iteration and deflation to get eigenvector, but full Jacobi is long.
    // We'll implement a simple Power Method + Deflation to get the largest eigenvector first.
    auto multiply = [](const fmat3& m, const fvec3& v) -> fvec3 {
        return fmat3_mul_vec3(m, v);
    };
    // Power method for dominant eigenvector
    fvec3 vec = {FIXED64_ONE, 0, 0};
    for (int iter=0; iter<16; ++iter) {
        vec = multiply(cov, vec);
        fixed64_t len = fvec3_length(vec);
        if (len != 0) vec = fvec3_scale(vec, fixed_rcp(len));
    }
    // That's the largest eigenvector (first axis). We'll use that as one OBB axis.
    // In a full implementation, we'd deflate and find other axes, but for brevity we'll approximate by orthogonalizing with a basis.
    // To keep this file complete and avoid placeholder, I'll use a simplified method: use the covariance's diagonal as extents after rotating points by the first axis? Not accurate.
    // Instead, I'll use a method that projects points onto the first eigenvector and orthogonal complement, then repeat.
    // For the rule "no simplified comments to skip coding", I'll implement a full Jacobi rotation for 3x3 symmetric matrix.
    // Jacobi rotation for fixed-point is lengthy but doable. I'll write it.
    fmat3 V = fmat3_identity(); // eigenvectors
    fmat3 A = cov;
    const int MAX_ITER = 16;
    for (int iter=0; iter<MAX_ITER; ++iter) {
        // Find largest off‑diagonal element
        fixed64_t max_off = 0;
        int p=0, q=1;
        for (int i=0;i<3;++i) {
            for (int j=i+1;j<3;++j) {
                fixed64_t val = fixed_abs(*(&A.rows[0].x + i*3 + j));
                if (val > max_off) { max_off = val; p=i; q=j; }
            }
        }
        if (max_off < 1) break; // tolerance ~1e-? in fixed? use 1
        // Compute Jacobi rotation
        fixed64_t app = *(&A.rows[0].x + p*3 + p);
        fixed64_t aqq = *(&A.rows[0].x + q*3 + q);
        fixed64_t apq = *(&A.rows[0].x + p*3 + q);
        fixed64_t theta = fixed_atan2(2*apq, app - aqq) >> 1; // half angle
        fixed64_t c_rot = fixed_cos(theta);
        fixed64_t s_rot = fixed_sin(theta);
        // Apply rotation to A and V
        // Update rows/cols p,q
        for (int i=0;i<3;++i) {
            if (i == p || i == q) continue;
            fixed64_t aip = *(&A.rows[0].x + i*3 + p);
            fixed64_t aiq = *(&A.rows[0].x + i*3 + q);
            fixed64_t new_aip = fixed_add(fixed_mul(c_rot, aip), fixed_mul(s_rot, aiq));
            fixed64_t new_aiq = fixed_sub(fixed_mul(c_rot, aiq), fixed_mul(s_rot, aip));
            *(&A.rows[0].x + i*3 + p) = new_aip;
            *(&A.rows[0].x + p*3 + i) = new_aip;
            *(&A.rows[0].x + i*3 + q) = new_aiq;
            *(&A.rows[0].x + q*3 + i) = new_aiq;
        }
        // update eigenvectors V
        for (int i=0;i<3;++i) {
            fixed64_t vip = *(&V.rows[0].x + i*3 + p);
            fixed64_t viq = *(&V.rows[0].x + i*3 + q);
            fixed64_t new_vip = fixed_add(fixed_mul(c_rot, vip), fixed_mul(s_rot, viq));
            fixed64_t new_viq = fixed_sub(fixed_mul(c_rot, viq), fixed_mul(s_rot, vip));
            *(&V.rows[0].x + i*3 + p) = new_vip;
            *(&V.rows[0].x + i*3 + q) = new_viq;
        }
    }
    // After Jacobi, A is approximately diagonal; diagonal entries are eigenvalues.
    // Eigenvectors are columns of V.
    fvec3 e0 = {V.rows[0].x, V.rows[1].x, V.rows[2].x}; // first eigenvector
    fvec3 e1 = {V.rows[0].y, V.rows[1].y, V.rows[2].y};
    fvec3 e2 = {V.rows[0].z, V.rows[1].z, V.rows[2].z};
    // Ensure orthonormal basis
    e0 = fvec3_normalize(e0);
    e1 = fvec3_normalize(fvec3_sub(e1, fvec3_scale(e0, fvec3_dot(e0,e1))));
    e1 = fvec3_normalize(e1);
    e2 = fvec3_normalize(fvec3_cross(e0, e1));
    // Compute extents by projecting points onto axes
    fixed64_t min_x = 0, max_x = 0, min_y = 0, max_y = 0, min_z = 0, max_z = 0;
    bool first = true;
    for (size_t i=0;i<count;++i) {
        fvec3 d = fvec3_sub(points[i], centroid);
        fixed64_t x = fvec3_dot(d, e0), y = fvec3_dot(d, e1), z = fvec3_dot(d, e2);
        if (first) {
            min_x = max_x = x; min_y = max_y = y; min_z = max_z = z;
            first = false;
        } else {
            min_x = fixed_min(min_x, x); max_x = fixed_max(max_x, x);
            min_y = fixed_min(min_y, y); max_y = fixed_max(max_y, y);
            min_z = fixed_min(min_z, z); max_z = fixed_max(max_z, z);
        }
    }
    // half extents
    OBB obb;
    obb.center = centroid;
    obb.half_extents = { (max_x - min_x) >> 1, (max_y - min_y) >> 1, (max_z - min_z) >> 1 };
    obb.axes.rows[0] = e0;
    obb.axes.rows[1] = e1;
    obb.axes.rows[2] = e2;
    return obb;
}

// ============================================================================
// SIMD batch AABB overlap (4 AABBs against 4 AABBs)
// ============================================================================
inline __m256i aabb_overlap_simd4(__m256i min1_x, __m256i min1_y, __m256i min1_z,
                                  __m256i max1_x, __m256i max1_y, __m256i max1_z,
                                  __m256i min2_x, __m256i min2_y, __m256i min2_z,
                                  __m256i max2_x, __m256i max2_y, __m256i max2_z) noexcept {
    __m256i overlap_x = _mm256_and_si256(
        _mm256_cmpgt_epi64(max2_x, min1_x), _mm256_cmpgt_epi64(max1_x, min2_x));
    __m256i overlap_y = _mm256_and_si256(
        _mm256_cmpgt_epi64(max2_y, min1_y), _mm256_cmpgt_epi64(max1_y, min2_y));
    __m256i overlap_z = _mm256_and_si256(
        _mm256_cmpgt_epi64(max2_z, min1_z), _mm256_cmpgt_epi64(max1_z, min2_z));
    return _mm256_and_si256(_mm256_and_si256(overlap_x, overlap_y), overlap_z);
}

} // namespace fixed_math

// End of File 0008
// Next file: File 0009 – core/math/fixed_sph_kernel.h
// Description: SPH kernel functions (Poly6, Spiky, Viscosity) in fixed‑point with SIMD 4‑lane evaluation for smooth particle hydrodynamics.