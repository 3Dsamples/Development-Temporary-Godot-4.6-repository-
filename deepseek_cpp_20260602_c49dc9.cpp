// system name : Octree Spatial Master
//File 0006 : core/math/fixed_mat.h
//Fixed‑point 3x3/4x4 matrix operations (mul, inverse, determinant, transform), tensor outer product, SIMD batch transform, 4‑lane point transforms
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_quat.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <algorithm>

namespace fixed_math {

// fvec4 type (used by fmat4)
struct fvec4 { fixed64_t x, y, z, w; };

// 4x4 matrix type
struct fmat4 { fvec4 rows[4]; };

// ---------------------------------------------------------------------------
// Identity matrices
// ---------------------------------------------------------------------------
inline fmat3 fmat3_identity() noexcept {
    fmat3 m;
    m.rows[0] = {FIXED64_ONE, 0, 0};
    m.rows[1] = {0, FIXED64_ONE, 0};
    m.rows[2] = {0, 0, FIXED64_ONE};
    return m;
}
inline fmat4 fmat4_identity() noexcept {
    fmat4 m;
    m.rows[0] = {FIXED64_ONE, 0, 0, 0};
    m.rows[1] = {0, FIXED64_ONE, 0, 0};
    m.rows[2] = {0, 0, FIXED64_ONE, 0};
    m.rows[3] = {0, 0, 0, FIXED64_ONE};
    return m;
}

// ---------------------------------------------------------------------------
// 3x3 matrix multiplication
// ---------------------------------------------------------------------------
inline fmat3 fmat3_mul(const fmat3& a, const fmat3& b) noexcept {
    fmat3 res;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            fixed64_t sum = 0;
            for (int k = 0; k < 3; ++k) {
                fixed64_t ak = *(&a.rows[0].x + r*3 + k);
                fixed64_t bk = *(&b.rows[0].x + k*3 + c);
                sum += fixed_mul(ak, bk);
            }
            *(&res.rows[0].x + r*3 + c) = sum;
        }
    }
    return res;
}

// ---------------------------------------------------------------------------
// 4x4 matrix multiplication
// ---------------------------------------------------------------------------
inline fmat4 fmat4_mul(const fmat4& a, const fmat4& b) noexcept {
    fmat4 res;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            fixed64_t sum = 0;
            for (int k = 0; k < 4; ++k) {
                sum += fixed_mul(*(&a.rows[0].x + r*4 + k),
                                 *(&b.rows[0].x + k*4 + c));
            }
            *(&res.rows[0].x + r*4 + c) = sum;
        }
    }
    return res;
}

// ---------------------------------------------------------------------------
// Transpose 3x3
// ---------------------------------------------------------------------------
inline fmat3 fmat3_transpose(const fmat3& m) noexcept {
    fmat3 t;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            *(&t.rows[0].x + r*3 + c) = *(&m.rows[0].x + c*3 + r);
    return t;
}

// ---------------------------------------------------------------------------
// Transpose 4x4
// ---------------------------------------------------------------------------
inline fmat4 fmat4_transpose(const fmat4& m) noexcept {
    fmat4 t;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            *(&t.rows[0].x + r*4 + c) = *(&m.rows[0].x + c*4 + r);
    return t;
}

// ---------------------------------------------------------------------------
// Determinant of 3x3 matrix
// ---------------------------------------------------------------------------
inline fixed64_t fmat3_det(const fmat3& m) noexcept {
    const fvec3& r0 = m.rows[0];
    const fvec3& r1 = m.rows[1];
    const fvec3& r2 = m.rows[2];
    return fixed_mul(r0.x, fixed_sub(fixed_mul(r1.y, r2.z), fixed_mul(r1.z, r2.y)))
         - fixed_mul(r0.y, fixed_sub(fixed_mul(r1.x, r2.z), fixed_mul(r1.z, r2.x)))
         + fixed_mul(r0.z, fixed_sub(fixed_mul(r1.x, r2.y), fixed_mul(r1.y, r2.x)));
}

// ---------------------------------------------------------------------------
// Determinant of 4x4 matrix via Laplace expansion along first row
// ---------------------------------------------------------------------------
inline fixed64_t fmat4_det(const fmat4& m) noexcept {
    // Minor matrices (3x3)
    auto minor3 = [&](int row, int col) noexcept -> fixed64_t {
        int rr = 0;
        fmat3 mat;
        for (int r = 0; r < 4; ++r) {
            if (r == row) continue;
            int cc = 0;
            for (int c = 0; c < 4; ++c) {
                if (c == col) continue;
                *(&mat.rows[0].x + rr*3 + cc) = *(&m.rows[0].x + r*4 + c);
                ++cc;
            }
            ++rr;
        }
        return fmat3_det(mat);
    };
    fixed64_t d0 = fixed_mul(m.rows[0].x, minor3(0, 0));
    fixed64_t d1 = fixed_mul(m.rows[0].y, minor3(0, 1));
    fixed64_t d2 = fixed_mul(m.rows[0].z, minor3(0, 2));
    fixed64_t d3 = fixed_mul(m.rows[0].w, minor3(0, 3));
    return d0 - d1 + d2 - d3;
}

// ---------------------------------------------------------------------------
// Inverse of 3x3 matrix (using adjugate)
// ---------------------------------------------------------------------------
inline fmat3 fmat3_inverse(const fmat3& m) noexcept {
    fixed64_t det = fmat3_det(m);
    if (det == 0) return fmat3_identity();
    fixed64_t inv_det = fixed_rcp(det);
    fmat3 adj;
    adj.rows[0].x = fixed_sub(fixed_mul(m.rows[1].y, m.rows[2].z), fixed_mul(m.rows[1].z, m.rows[2].y));
    adj.rows[0].y = fixed_sub(fixed_mul(m.rows[0].z, m.rows[2].y), fixed_mul(m.rows[0].y, m.rows[2].z));
    adj.rows[0].z = fixed_sub(fixed_mul(m.rows[0].y, m.rows[1].z), fixed_mul(m.rows[0].z, m.rows[1].y));
    adj.rows[1].x = fixed_sub(fixed_mul(m.rows[1].z, m.rows[2].x), fixed_mul(m.rows[1].x, m.rows[2].z));
    adj.rows[1].y = fixed_sub(fixed_mul(m.rows[0].x, m.rows[2].z), fixed_mul(m.rows[0].z, m.rows[2].x));
    adj.rows[1].z = fixed_sub(fixed_mul(m.rows[0].z, m.rows[1].x), fixed_mul(m.rows[0].x, m.rows[1].z));
    adj.rows[2].x = fixed_sub(fixed_mul(m.rows[1].x, m.rows[2].y), fixed_mul(m.rows[1].y, m.rows[2].x));
    adj.rows[2].y = fixed_sub(fixed_mul(m.rows[0].y, m.rows[2].x), fixed_mul(m.rows[0].x, m.rows[2].y));
    adj.rows[2].z = fixed_sub(fixed_mul(m.rows[0].x, m.rows[1].y), fixed_mul(m.rows[0].y, m.rows[1].x));
    // Transpose to get cofactor matrix, then multiply by 1/det
    fmat3 inv;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            *(&inv.rows[0].x + r*3 + c) = fixed_mul(*(&adj.rows[0].x + c*3 + r), inv_det);
    return inv;
}

// ---------------------------------------------------------------------------
// Inverse of 4x4 matrix (cofactor + adjugate)
// ---------------------------------------------------------------------------
inline fmat4 fmat4_inverse(const fmat4& m) noexcept {
    fixed64_t det = fmat4_det(m);
    if (det == 0) return fmat4_identity();
    fixed64_t inv_det = fixed_rcp(det);
    fmat4 adj;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            // Compute cofactor sign
            fixed64_t sign = ((r + c) & 1) ? -FIXED64_ONE : FIXED64_ONE;
            // Build 3x3 minor
            fmat3 minor;
            int rr = 0;
            for (int row = 0; row < 4; ++row) {
                if (row == r) continue;
                int cc = 0;
                for (int col = 0; col < 4; ++col) {
                    if (col == c) continue;
                    *(&minor.rows[0].x + rr*3 + cc) = *(&m.rows[0].x + row*4 + col);
                    ++cc;
                }
                ++rr;
            }
            fixed64_t cofactor = fixed_mul(sign, fmat3_det(minor));
            *(&adj.rows[0].x + c*4 + r) = fixed_mul(cofactor, inv_det); // transposed placement
        }
    }
    return adj;
}

// ---------------------------------------------------------------------------
// Transform point by 4x4 matrix (perspective divide)
// ---------------------------------------------------------------------------
inline fvec3 fmat4_transform_point(const fmat4& m, const fvec3& p) noexcept {
    fvec4 v = {p.x, p.y, p.z, FIXED64_ONE};
    fvec4 res;
    res.x = fvec3_dot({m.rows[0].x, m.rows[0].y, m.rows[0].z}, p) + m.rows[0].w;
    res.y = fvec3_dot({m.rows[1].x, m.rows[1].y, m.rows[1].z}, p) + m.rows[1].w;
    res.z = fvec3_dot({m.rows[2].x, m.rows[2].y, m.rows[2].z}, p) + m.rows[2].w;
    res.w = fvec3_dot({m.rows[3].x, m.rows[3].y, m.rows[3].z}, p) + m.rows[3].w;
    if (res.w == 0) return {res.x, res.y, res.z};
    return {fixed_div(res.x, res.w), fixed_div(res.y, res.w), fixed_div(res.z, res.w)};
}

// ---------------------------------------------------------------------------
// Transform vector by 4x4 matrix (w=0)
// ---------------------------------------------------------------------------
inline fvec3 fmat4_transform_vector(const fmat4& m, const fvec3& v) noexcept {
    return {
        fvec3_dot({m.rows[0].x, m.rows[0].y, m.rows[0].z}, v),
        fvec3_dot({m.rows[1].x, m.rows[1].y, m.rows[1].z}, v),
        fvec3_dot({m.rows[2].x, m.rows[2].y, m.rows[2].z}, v)
    };
}

// ---------------------------------------------------------------------------
// Build translation matrix
// ---------------------------------------------------------------------------
inline fmat4 fmat4_translation(const fvec3& t) noexcept {
    fmat4 m = fmat4_identity();
    m.rows[0].w = t.x;
    m.rows[1].w = t.y;
    m.rows[2].w = t.z;
    return m;
}

// ---------------------------------------------------------------------------
// Build scale matrix
// ---------------------------------------------------------------------------
inline fmat4 fmat4_scale(const fvec3& s) noexcept {
    fmat4 m = fmat4_identity();
    m.rows[0].x = s.x;
    m.rows[1].y = s.y;
    m.rows[2].z = s.z;
    return m;
}

// ---------------------------------------------------------------------------
// Build rotation matrix from quaternion
// ---------------------------------------------------------------------------
inline fmat4 fmat4_rotation(const fquat& q) noexcept {
    fmat3 rot3 = fquat_to_mat3(q);
    fmat4 m;
    m.rows[0] = {rot3.rows[0].x, rot3.rows[0].y, rot3.rows[0].z, 0};
    m.rows[1] = {rot3.rows[1].x, rot3.rows[1].y, rot3.rows[1].z, 0};
    m.rows[2] = {rot3.rows[2].x, rot3.rows[2].y, rot3.rows[2].z, 0};
    m.rows[3] = {0, 0, 0, FIXED64_ONE};
    return m;
}

// ---------------------------------------------------------------------------
// Outer product of two vec4 -> 4x4 matrix
// ---------------------------------------------------------------------------
inline fmat4 fmat4_outer(const fvec4& a, const fvec4& b) noexcept {
    fmat4 m;
    m.rows[0] = {fixed_mul(a.x, b.x), fixed_mul(a.x, b.y), fixed_mul(a.x, b.z), fixed_mul(a.x, b.w)};
    m.rows[1] = {fixed_mul(a.y, b.x), fixed_mul(a.y, b.y), fixed_mul(a.y, b.z), fixed_mul(a.y, b.w)};
    m.rows[2] = {fixed_mul(a.z, b.x), fixed_mul(a.z, b.y), fixed_mul(a.z, b.z), fixed_mul(a.z, b.w)};
    m.rows[3] = {fixed_mul(a.w, b.x), fixed_mul(a.w, b.y), fixed_mul(a.w, b.z), fixed_mul(a.w, b.w)};
    return m;
}

// ---------------------------------------------------------------------------
// Outer product of two vec3 -> 3x3 matrix (already in fixed_vec3.h; alias)
// ---------------------------------------------------------------------------
inline fmat3 fmat3_outer(const fvec3& a, const fvec3& b) noexcept {
    return fvec3_outer(a, b);
}

// ---------------------------------------------------------------------------
// SIMD batch transform: apply a single 4x4 matrix to 4 points (SoA)
// ---------------------------------------------------------------------------
inline void fmat4_transform_points_simd4(const fmat4& m,
                                         __m256i px, __m256i py, __m256i pz,
                                         __m256i& rx, __m256i& ry, __m256i& rz) noexcept {
    // Load matrix coefficients broadcast across lanes
    __m256i m00 = _mm256_set1_epi64x(m.rows[0].x);
    __m256i m01 = _mm256_set1_epi64x(m.rows[0].y);
    __m256i m02 = _mm256_set1_epi64x(m.rows[0].z);
    __m256i m03 = _mm256_set1_epi64x(m.rows[0].w);
    __m256i m10 = _mm256_set1_epi64x(m.rows[1].x);
    __m256i m11 = _mm256_set1_epi64x(m.rows[1].y);
    __m256i m12 = _mm256_set1_epi64x(m.rows[1].z);
    __m256i m13 = _mm256_set1_epi64x(m.rows[1].w);
    __m256i m20 = _mm256_set1_epi64x(m.rows[2].x);
    __m256i m21 = _mm256_set1_epi64x(m.rows[2].y);
    __m256i m22 = _mm256_set1_epi64x(m.rows[2].z);
    __m256i m23 = _mm256_set1_epi64x(m.rows[2].w);
    __m256i m30 = _mm256_set1_epi64x(m.rows[3].x);
    __m256i m31 = _mm256_set1_epi64x(m.rows[3].y);
    __m256i m32 = _mm256_set1_epi64x(m.rows[3].z);
    __m256i m33 = _mm256_set1_epi64x(m.rows[3].w);

    // Compute each output component
    __m256i x = simd4_add_epi64(
                    simd4_add_epi64(simd4_mul_epi64(m00, px), simd4_mul_epi64(m01, py)),
                    simd4_add_epi64(simd4_mul_epi64(m02, pz), m03));
    __m256i y = simd4_add_epi64(
                    simd4_add_epi64(simd4_mul_epi64(m10, px), simd4_mul_epi64(m11, py)),
                    simd4_add_epi64(simd4_mul_epi64(m12, pz), m13));
    __m256i z = simd4_add_epi64(
                    simd4_add_epi64(simd4_mul_epi64(m20, px), simd4_mul_epi64(m21, py)),
                    simd4_add_epi64(simd4_mul_epi64(m22, pz), m23));
    __m256i w = simd4_add_epi64(
                    simd4_add_epi64(simd4_mul_epi64(m30, px), simd4_mul_epi64(m31, py)),
                    simd4_add_epi64(simd4_mul_epi64(m32, pz), m33));

    // Perform perspective division (if w != 0)
    __m256i zero = _mm256_setzero_si256();
    __m256i w_eq_zero = _mm256_cmpeq_epi64(w, zero);
    // For lanes where w==0, we leave x,y,z as is; otherwise, divide by w
    __m256i inv_w = simd4_rcp_epi64(w); // approximate reciprocal
    __m256i rx_div = simd4_mul_epi64(x, inv_w);
    __m256i ry_div = simd4_mul_epi64(y, inv_w);
    __m256i rz_div = simd4_mul_epi64(z, inv_w);
    // Blend: if w==0 keep original, else keep divided
    rx = _mm256_blendv_epi8(rx_div, x, w_eq_zero);
    ry = _mm256_blendv_epi8(ry_div, y, w_eq_zero);
    rz = _mm256_blendv_epi8(rz_div, z, w_eq_zero);
}

} // namespace fixed_math

// End of File 0006
// Next file: File 0007 – core/math/fixed_rand.h
// Description: Fixed‑point pseudo‑random number generators (XorShift, SplitMix, Lehmer) and sampling distributions (uniform, normal, sphere, disc) with SIMD batch generation.