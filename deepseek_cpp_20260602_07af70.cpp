// system name : Octree Spatial Master
//File 0005 : core/math/fixed_quat.h
//Fixed‑point quaternion operations: mul, conjugate, rotate, slerp, to/from rotation matrix, and SIMD 4‑lane support
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <algorithm>

namespace fixed_math {

// Quaternion type (assumed defined in core/types.h; repeated for self‑containment)
struct fquat { fixed64_t w, x, y, z; };

// ---------------------------------------------------------------------------
// Constructors / constants
// ---------------------------------------------------------------------------
inline fquat fquat_identity() noexcept { return {FIXED64_ONE, 0, 0, 0}; }
inline fquat fquat_from_axis_angle(const fvec3& axis, fixed64_t angle) noexcept {
    fixed64_t half = angle >> 1;
    fixed64_t s, c;
    fixed_sincos(half, s, c);
    return {c, fixed_mul(axis.x, s), fixed_mul(axis.y, s), fixed_mul(axis.z, s)};
}

// ---------------------------------------------------------------------------
// Conjugate / inverse
// ---------------------------------------------------------------------------
inline fquat fquat_conjugate(const fquat& q) noexcept {
    return {q.w, -q.x, -q.y, -q.z};
}
inline fixed64_t fquat_norm_sq(const fquat& q) noexcept {
    return fixed_add(fixed_add(fixed_mul(q.w, q.w), fixed_mul(q.x, q.x)),
                     fixed_add(fixed_mul(q.y, q.y), fixed_mul(q.z, q.z)));
}
inline fquat fquat_inverse(const fquat& q) noexcept {
    fixed64_t n2 = fquat_norm_sq(q);
    if (n2 == 0) return q;
    fixed64_t inv = fixed_rcp(n2);
    fquat conj = fquat_conjugate(q);
    return {fixed_mul(conj.w, inv), fixed_mul(conj.x, inv),
            fixed_mul(conj.y, inv), fixed_mul(conj.z, inv)};
}

// ---------------------------------------------------------------------------
// Normalize
// ---------------------------------------------------------------------------
inline fquat fquat_normalize(const fquat& q) noexcept {
    fixed64_t len = fixed_sqrt(fquat_norm_sq(q));
    if (len == 0) return fquat_identity();
    fixed64_t inv = fixed_rcp(len);
    return {fixed_mul(q.w, inv), fixed_mul(q.x, inv),
            fixed_mul(q.y, inv), fixed_mul(q.z, inv)};
}

// ---------------------------------------------------------------------------
// Multiplication (q1 * q2)
// ---------------------------------------------------------------------------
inline fquat fquat_mul(const fquat& q1, const fquat& q2) noexcept {
    return {
        fixed_sub(fixed_sub(fixed_sub(fixed_mul(q1.w, q2.w), fixed_mul(q1.x, q2.x)),
                            fixed_mul(q1.y, q2.y)), fixed_mul(q1.z, q2.z)),
        fixed_add(fixed_add(fixed_add(fixed_mul(q1.w, q2.x), fixed_mul(q1.x, q2.w)),
                            fixed_mul(q1.y, q2.z)), fixed_mul(q1.z, q2.y)),
        fixed_add(fixed_add(fixed_sub(fixed_mul(q1.w, q2.y), fixed_mul(q1.x, q2.z)),
                            fixed_mul(q1.y, q2.w)), fixed_mul(q1.z, q2.x)),
        fixed_add(fixed_add(fixed_sub(fixed_mul(q1.w, q2.z), fixed_mul(q1.x, q2.y)),
                            fixed_mul(q1.y, q2.x)), fixed_mul(q1.z, q2.w))
    };
}

// ---------------------------------------------------------------------------
// Rotate vector by unit quaternion
// ---------------------------------------------------------------------------
inline fvec3 fquat_rotate(const fquat& q, const fvec3& v) noexcept {
    fquat p = {0, v.x, v.y, v.z};
    fquat conj = fquat_conjugate(q);
    fquat rotated = fquat_mul(fquat_mul(q, p), conj);
    return {rotated.x, rotated.y, rotated.z};
}

// ---------------------------------------------------------------------------
// Spherical linear interpolation (slerp)
// ---------------------------------------------------------------------------
inline fquat fquat_slerp(const fquat& a, const fquat& b, fixed64_t t) noexcept {
    fixed64_t dot = fixed_add(fixed_add(fixed_mul(a.w, b.w), fixed_mul(a.x, b.x)),
                              fixed_add(fixed_mul(a.y, b.y), fixed_mul(a.z, b.z)));
    fquat b_flip = b;
    if (dot < 0) { dot = -dot; b_flip = {-b.w, -b.x, -b.y, -b.z}; }
    const fixed64_t THRESHOLD = FIXED64_ONE - (1LL << 12); // ~0.9998
    if (dot > THRESHOLD) {
        fquat lerp = {
            a.w + fixed_mul(b_flip.w - a.w, t),
            a.x + fixed_mul(b_flip.x - a.x, t),
            a.y + fixed_mul(b_flip.y - a.y, t),
            a.z + fixed_mul(b_flip.z - a.z, t)
        };
        return fquat_normalize(lerp);
    }
    fixed64_t theta = fixed_acos(dot);
    fixed64_t sin_theta = fixed_sin(theta);
    fixed64_t w1 = fixed_div(fixed_sin(fixed_mul(theta, FIXED64_ONE - t)), sin_theta);
    fixed64_t w2 = fixed_div(fixed_sin(fixed_mul(theta, t)), sin_theta);
    return {
        fixed_add(fixed_mul(w1, a.w), fixed_mul(w2, b_flip.w)),
        fixed_add(fixed_mul(w1, a.x), fixed_mul(w2, b_flip.x)),
        fixed_add(fixed_mul(w1, a.y), fixed_mul(w2, b_flip.y)),
        fixed_add(fixed_mul(w1, a.z), fixed_mul(w2, b_flip.z))
    };
}

// ---------------------------------------------------------------------------
// Conversion to 3x3 rotation matrix (row‑major)
// ---------------------------------------------------------------------------
inline fmat3 fquat_to_mat3(const fquat& q) noexcept {
    fixed64_t xx = fixed_mul(q.x, q.x), yy = fixed_mul(q.y, q.y), zz = fixed_mul(q.z, q.z);
    fixed64_t xy = fixed_mul(q.x, q.y), xz = fixed_mul(q.x, q.z), yz = fixed_mul(q.y, q.z);
    fixed64_t wx = fixed_mul(q.w, q.x), wy = fixed_mul(q.w, q.y), wz = fixed_mul(q.w, q.z);
    fmat3 m;
    m.rows[0] = {FIXED64_ONE - 2*(yy + zz), 2*(xy - wz),              2*(xz + wy)};
    m.rows[1] = {2*(xy + wz),              FIXED64_ONE - 2*(xx + zz),  2*(yz - wx)};
    m.rows[2] = {2*(xz - wy),              2*(yz + wx),                FIXED64_ONE - 2*(xx + yy)};
    return m;
}

// ---------------------------------------------------------------------------
// 4x4 rotation matrix (extension of 3x3)
// ---------------------------------------------------------------------------
inline fmat4 fquat_to_mat4(const fquat& q) noexcept {
    fmat3 m3 = fquat_to_mat3(q);
    fmat4 m;
    m.rows[0] = {m3.rows[0].x, m3.rows[0].y, m3.rows[0].z, 0};
    m.rows[1] = {m3.rows[1].x, m3.rows[1].y, m3.rows[1].z, 0};
    m.rows[2] = {m3.rows[2].x, m3.rows[2].y, m3.rows[2].z, 0};
    m.rows[3] = {0, 0, 0, FIXED64_ONE};
    return m;
}

// ---------------------------------------------------------------------------
// SIMD 4‑lane quaternion multiplication (scalar extraction)
// ---------------------------------------------------------------------------
inline void fquat_mul_simd4(__m256i w1, __m256i x1, __m256i y1, __m256i z1,
                             __m256i w2, __m256i x2, __m256i y2, __m256i z2,
                             __m256i& wo, __m256i& xo, __m256i& yo, __m256i& zo) noexcept {
    alignas(32) int64_t a[4][4], b[4][4];
    _mm256_store_si256((__m256i*)a[0], w1); _mm256_store_si256((__m256i*)a[1], x1);
    _mm256_store_si256((__m256i*)a[2], y1); _mm256_store_si256((__m256i*)a[3], z1);
    _mm256_store_si256((__m256i*)b[0], w2); _mm256_store_si256((__m256i*)b[1], x2);
    _mm256_store_si256((__m256i*)b[2], y2); _mm256_store_si256((__m256i*)b[3], z2);
    alignas(32) int64_t out[4][4];
    for (int i = 0; i < 4; ++i) {
        fquat q1 = {a[0][i], a[1][i], a[2][i], a[3][i]};
        fquat q2 = {b[0][i], b[1][i], b[2][i], b[3][i]};
        fquat r = fquat_mul(q1, q2);
        out[0][i] = r.w; out[1][i] = r.x; out[2][i] = r.y; out[3][i] = r.z;
    }
    wo = _mm256_load_si256((__m256i*)out[0]);
    xo = _mm256_load_si256((__m256i*)out[1]);
    yo = _mm256_load_si256((__m256i*)out[2]);
    zo = _mm256_load_si256((__m256i*)out[3]);
}

// SIMD 4‑lane rotate vector (each lane individual)
inline void fquat_rotate_simd4(__m256i w, __m256i x, __m256i y, __m256i z,
                                __m256i vx, __m256i vy, __m256i vz,
                                __m256i& rx, __m256i& ry, __m256i& rz) noexcept {
    alignas(32) int64_t qw[4], qx[4], qy[4], qz[4];
    _mm256_store_si256((__m256i*)qw, w); _mm256_store_si256((__m256i*)qx, x);
    _mm256_store_si256((__m256i*)qy, y); _mm256_store_si256((__m256i*)qz, z);
    alignas(32) int64_t vx_a[4], vy_a[4], vz_a[4];
    _mm256_store_si256((__m256i*)vx_a, vx); _mm256_store_si256((__m256i*)vy_a, vy); _mm256_store_si256((__m256i*)vz_a, vz);
    alignas(32) int64_t rvx[4], rvy[4], rvz[4];
    for (int i = 0; i < 4; ++i) {
        fquat q = {qw[i], qx[i], qy[i], qz[i]};
        fvec3 vv = {vx_a[i], vy_a[i], vz_a[i]};
        fvec3 res = fquat_rotate(q, vv);
        rvx[i] = res.x; rvy[i] = res.y; rvz[i] = res.z;
    }
    rx = _mm256_load_si256((__m256i*)rvx);
    ry = _mm256_load_si256((__m256i*)rvy);
    rz = _mm256_load_si256((__m256i*)rvz);
}

// SIMD 4‑lane slerp
inline void fquat_slerp_simd4(__m256i aw, __m256i ax, __m256i ay, __m256i az,
                               __m256i bw, __m256i bx, __m256i by, __m256i bz,
                               __m256i t,
                               __m256i& rw, __m256i& rx, __m256i& ry, __m256i& rz) noexcept {
    alignas(32) int64_t qaw[4], qax[4], qay[4], qaz[4];
    _mm256_store_si256((__m256i*)qaw, aw); _mm256_store_si256((__m256i*)qax, ax);
    _mm256_store_si256((__m256i*)qay, ay); _mm256_store_si256((__m256i*)qaz, az);
    alignas(32) int64_t qbw[4], qbx[4], qby[4], qbz[4];
    _mm256_store_si256((__m256i*)qbw, bw); _mm256_store_si256((__m256i*)qbx, bx);
    _mm256_store_si256((__m256i*)qby, by); _mm256_store_si256((__m256i*)qbz, bz);
    alignas(32) int64_t tv[4];
    _mm256_store_si256((__m256i*)tv, t);
    alignas(32) int64_t rwv[4], rxv[4], ryv[4], rzv[4];
    for (int i = 0; i < 4; ++i) {
        fquat qa = {qaw[i], qax[i], qay[i], qaz[i]};
        fquat qb = {qbw[i], qbx[i], qby[i], qbz[i]};
        fquat r = fquat_slerp(qa, qb, tv[i]);
        rwv[i] = r.w; rxv[i] = r.x; ryv[i] = r.y; rzv[i] = r.z;
    }
    rw = _mm256_load_si256((__m256i*)rwv);
    rx = _mm256_load_si256((__m256i*)rxv);
    ry = _mm256_load_si256((__m256i*)ryv);
    rz = _mm256_load_si256((__m256i*)rzv);
}

} // namespace fixed_math

// End of File 0005
// Next file: File 0006 – core/math/fixed_mat.h
// Description: Fixed‑point 3x3 and 4x4 matrix operations (mul, transpose, inverse, determinant, transform point/vector) with SIMD 4‑lane support.