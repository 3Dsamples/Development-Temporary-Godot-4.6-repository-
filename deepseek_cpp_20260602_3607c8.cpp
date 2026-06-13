// system name : Octree Spatial Master
//File 0004 : core/math/fixed_vec3.h
//Fixed‑point 3D vector operations (add, sub, dot, cross, length, normalize, lerp, reflect, refract) with SIMD 4‑lane SoA/AoS conversion and tensor outer product
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_trig.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <array>
#include <algorithm>

namespace fixed_math {

// ---------------------------------------------------------------------------
// 3D vector type (already defined elsewhere; we repeat here for completeness)
// ---------------------------------------------------------------------------
struct fvec3 { fixed64_t x, y, z; };

// ---------------------------------------------------------------------------
// Constructors / constants
// ---------------------------------------------------------------------------
inline fvec3 fvec3_zero() noexcept { return {0, 0, 0}; }
inline fvec3 fvec3_one()  noexcept { return {FIXED64_ONE, FIXED64_ONE, FIXED64_ONE}; }

// ---------------------------------------------------------------------------
// Scalar component operations
// ---------------------------------------------------------------------------
inline fvec3 fvec3_add(const fvec3& a, const fvec3& b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
inline fvec3 fvec3_sub(const fvec3& a, const fvec3& b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline fvec3 fvec3_mul(const fvec3& a, const fvec3& b) noexcept {
    return {fixed_mul(a.x, b.x), fixed_mul(a.y, b.y), fixed_mul(a.z, b.z)};
}
inline fvec3 fvec3_div(const fvec3& a, const fvec3& b) noexcept {
    return {fixed_div(a.x, b.x), fixed_div(a.y, b.y), fixed_div(a.z, b.z)};
}
inline fvec3 fvec3_scale(const fvec3& v, fixed64_t s) noexcept {
    return {fixed_mul(v.x, s), fixed_mul(v.y, s), fixed_mul(v.z, s)};
}
inline fvec3 fvec3_neg(const fvec3& v) noexcept { return {-v.x, -v.y, -v.z}; }

// ---------------------------------------------------------------------------
// Dot product
// ---------------------------------------------------------------------------
inline fixed64_t fvec3_dot(const fvec3& a, const fvec3& b) noexcept {
    return fixed_add(fixed_add(fixed_mul(a.x, b.x), fixed_mul(a.y, b.y)), fixed_mul(a.z, b.z));
}

// ---------------------------------------------------------------------------
// Cross product
// ---------------------------------------------------------------------------
inline fvec3 fvec3_cross(const fvec3& a, const fvec3& b) noexcept {
    return {
        fixed_sub(fixed_mul(a.y, b.z), fixed_mul(a.z, b.y)),
        fixed_sub(fixed_mul(a.z, b.x), fixed_mul(a.x, b.z)),
        fixed_sub(fixed_mul(a.x, b.y), fixed_mul(a.y, b.x))
    };
}

// ---------------------------------------------------------------------------
// Length squared
// ---------------------------------------------------------------------------
inline fixed64_t fvec3_length_sq(const fvec3& v) noexcept {
    return fvec3_dot(v, v);
}

// ---------------------------------------------------------------------------
// Length
// ---------------------------------------------------------------------------
inline fixed64_t fvec3_length(const fvec3& v) noexcept {
    return fixed_sqrt(fvec3_length_sq(v));
}

// ---------------------------------------------------------------------------
// Normalize (returns zero vector if length zero)
// ---------------------------------------------------------------------------
inline fvec3 fvec3_normalize(const fvec3& v) noexcept {
    fixed64_t len = fvec3_length(v);
    if (len == 0) return {0,0,0};
    return fvec3_scale(v, fixed_rcp(len));
}

// ---------------------------------------------------------------------------
// Distance / distance squared
// ---------------------------------------------------------------------------
inline fixed64_t fvec3_distance_sq(const fvec3& a, const fvec3& b) noexcept {
    fixed64_t dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return fixed_add(fixed_add(fixed_mul(dx, dx), fixed_mul(dy, dy)), fixed_mul(dz, dz));
}
inline fixed64_t fvec3_distance(const fvec3& a, const fvec3& b) noexcept {
    return fixed_sqrt(fvec3_distance_sq(a, b));
}

// ---------------------------------------------------------------------------
// Linear interpolation (lerp): a + t*(b - a)
// ---------------------------------------------------------------------------
inline fvec3 fvec3_lerp(const fvec3& a, const fvec3& b, fixed64_t t) noexcept {
    return fvec3_add(a, fvec3_scale(fvec3_sub(b, a), t));
}

// ---------------------------------------------------------------------------
// Reflection: v reflect about normal n (assumed unit)
// ---------------------------------------------------------------------------
inline fvec3 fvec3_reflect(const fvec3& v, const fvec3& n) noexcept {
    fixed64_t d = fvec3_dot(v, n);
    return fvec3_sub(v, fvec3_scale(n, 2 * d));
}

// ---------------------------------------------------------------------------
// Refraction (Snell's law): v incident, n normal, eta ratio of IORs
// ---------------------------------------------------------------------------
inline fvec3 fvec3_refract(const fvec3& v, const fvec3& n, fixed64_t eta) noexcept {
    fixed64_t d = fvec3_dot(v, n);
    fixed64_t k = FIXED64_ONE - fixed_mul(eta, eta) * (FIXED64_ONE - fixed_mul(d, d));
    if (k < 0) return {0,0,0}; // total internal reflection
    return fvec3_add(fvec3_scale(v, eta), fvec3_scale(n, fixed_mul(eta, d) - fixed_sqrt(k)));
}

// ---------------------------------------------------------------------------
// Outer product (tensor): 3x3 matrix from two vectors (v ⊗ w)
// ---------------------------------------------------------------------------
struct fmat3 { fvec3 rows[3]; };
inline fmat3 fvec3_outer(const fvec3& a, const fvec3& b) noexcept {
    fmat3 m;
    m.rows[0] = fvec3_scale(b, a.x);
    m.rows[1] = fvec3_scale(b, a.y);
    m.rows[2] = fvec3_scale(b, a.z);
    return m;
}

// ---------------------------------------------------------------------------
// Matrix‑vector multiply
// ---------------------------------------------------------------------------
inline fvec3 fmat3_mul_vec3(const fmat3& m, const fvec3& v) noexcept {
    return {
        fvec3_dot(m.rows[0], v),
        fvec3_dot(m.rows[1], v),
        fvec3_dot(m.rows[2], v)
    };
}

// ============================================================================
// SIMD 4‑lane vector operations (SoA layout: 4 vectors stored as 3x __m256i)
// ============================================================================

// Load 4 fvec3 from AoS array into SoA registers
inline void fvec3_aos_to_soa4(const fvec3 src[4], __m256i& x, __m256i& y, __m256i& z) noexcept {
    x = _mm256_set_epi64x(src[3].x, src[2].x, src[1].x, src[0].x);
    y = _mm256_set_epi64x(src[3].y, src[2].y, src[1].y, src[0].y);
    z = _mm256_set_epi64x(src[3].z, src[2].z, src[1].z, src[0].z);
}

// Store 4 fvec3 from SoA registers to AoS array
inline void fvec3_soa4_to_aos(__m256i x, __m256i y, __m256i z, fvec3 dst[4]) noexcept {
    alignas(32) int64_t xv[4], yv[4], zv[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(xv), x);
    _mm256_store_si256(reinterpret_cast<__m256i*>(yv), y);
    _mm256_store_si256(reinterpret_cast<__m256i*>(zv), z);
    for (int i = 0; i < 4; ++i) dst[i] = {xv[i], yv[i], zv[i]};
}

// SIMD 4‑lane add
inline void fvec3_add_simd4(__m256i ax, __m256i ay, __m256i az,
                             __m256i bx, __m256i by, __m256i bz,
                             __m256i& rx, __m256i& ry, __m256i& rz) noexcept {
    rx = simd4_add_epi64(ax, bx);
    ry = simd4_add_epi64(ay, by);
    rz = simd4_add_epi64(az, bz);
}

// SIMD 4‑lane sub
inline void fvec3_sub_simd4(__m256i ax, __m256i ay, __m256i az,
                             __m256i bx, __m256i by, __m256i bz,
                             __m256i& rx, __m256i& ry, __m256i& rz) noexcept {
    rx = simd4_sub_epi64(ax, bx);
    ry = simd4_sub_epi64(ay, by);
    rz = simd4_sub_epi64(az, bz);
}

// SIMD 4‑lane scale
inline void fvec3_scale_simd4(__m256i x, __m256i y, __m256i z, __m256i s,
                               __m256i& rx, __m256i& ry, __m256i& rz) noexcept {
    rx = simd4_mul_epi64(x, s);
    ry = simd4_mul_epi64(y, s);
    rz = simd4_mul_epi64(z, s);
}

// SIMD 4‑lane dot product (returns 4 values in an __m256i)
inline __m256i fvec3_dot_simd4(__m256i ax, __m256i ay, __m256i az,
                                __m256i bx, __m256i by, __m256i bz) noexcept {
    __m256i prod_x = simd4_mul_epi64(ax, bx);
    __m256i prod_y = simd4_mul_epi64(ay, by);
    __m256i prod_z = simd4_mul_epi64(az, bz);
    return simd4_add_epi64(simd4_add_epi64(prod_x, prod_y), prod_z);
}

// SIMD 4‑lane cross product
inline void fvec3_cross_simd4(__m256i ax, __m256i ay, __m256i az,
                               __m256i bx, __m256i by, __m256i bz,
                               __m256i& cx, __m256i& cy, __m256i& cz) noexcept {
    cx = simd4_sub_epi64(simd4_mul_epi64(ay, bz), simd4_mul_epi64(az, by));
    cy = simd4_sub_epi64(simd4_mul_epi64(az, bx), simd4_mul_epi64(ax, bz));
    cz = simd4_sub_epi64(simd4_mul_epi64(ax, by), simd4_mul_epi64(ay, bx));
}

// SIMD 4‑lane length squared
inline __m256i fvec3_length_sq_simd4(__m256i x, __m256i y, __m256i z) noexcept {
    return fvec3_dot_simd4(x, y, z, x, y, z);
}

// SIMD 4‑lane length (scalar sqrt per lane)
inline __m256i fvec3_length_simd4(__m256i x, __m256i y, __m256i z) noexcept {
    __m256i len_sq = fvec3_length_sq_simd4(x, y, z);
    return simd4_sqrt_epi64(len_sq);
}

// SIMD 4‑lane normalize (with zero check)
inline void fvec3_normalize_simd4(__m256i x, __m256i y, __m256i z,
                                   __m256i& nx, __m256i& ny, __m256i& nz) noexcept {
    __m256i len_sq = fvec3_length_sq_simd4(x, y, z);
    __m256i len = simd4_sqrt_epi64(len_sq);
    __m256i zero_mask = _mm256_cmpeq_epi64(len, _mm256_setzero_si256());
    __m256i inv_len = simd4_rcp_epi64(len);
    // blend: if len==0, keep zero
    __m256i inv_len_fixed = _mm256_andnot_si256(zero_mask, inv_len);
    nx = simd4_mul_epi64(x, inv_len_fixed);
    ny = simd4_mul_epi64(y, inv_len_fixed);
    nz = simd4_mul_epi64(z, inv_len_fixed);
}

} // namespace fixed_math

// End of File 0004
// Next file: File 0005 – core/math/fixed_quat.h
// Description: Fixed‑point quaternion operations: mul, conjugate, rotate vector, slerp, to/from rotation matrix, with SIMD 4‑lane support.