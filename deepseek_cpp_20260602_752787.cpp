// system name : Octree Spatial Master
//File 0012 : core/math/fixed_simd_util.h
//SIMD utility functions for fixed‑point 64‑bit integer operations: load/store, blend, gather/scatter, shuffle, tensor operations (4x4 matmul, outer product, batch transforms)
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_quat.h"
#include "core/math/fixed_mat.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cstring>

namespace fixed_math {

// ---------------------------------------------------------------------------
// 256‑bit aligned load/store
// ---------------------------------------------------------------------------
inline __m256i load_i64_4(const fixed64_t* p) noexcept {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
}
inline void store_i64_4(fixed64_t* p, __m256i v) noexcept {
    _mm256_store_si256(reinterpret_cast<__m256i*>(p), v);
}

// ---------------------------------------------------------------------------
// Broadcast scalar to all lanes
// ---------------------------------------------------------------------------
inline __m256i set1_i64(fixed64_t x) noexcept {
    return _mm256_set1_epi64x(x);
}

// ---------------------------------------------------------------------------
// Blend two 4‑lane vectors based on mask (0 = keep a, 1 = take b)
// ---------------------------------------------------------------------------
inline __m256i blend_i64(__m256i a, __m256i b, __m256i mask) noexcept {
    return _mm256_blendv_epi8(a, b, mask);
}

// ---------------------------------------------------------------------------
// Extract lane value
// ---------------------------------------------------------------------------
inline fixed64_t extract_i64(__m256i v, int lane) noexcept {
    alignas(32) int64_t tmp[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), v);
    return tmp[lane & 3];
}

// ---------------------------------------------------------------------------
// Gather 4 fixed64_t from memory using indices (scale = 8)
// ---------------------------------------------------------------------------
inline __m256i gather_i64(const fixed64_t* base, __m256i indices, int scale) noexcept {
    return _mm256_i64gather_epi64(reinterpret_cast<const long long*>(base), indices, scale);
}

// ---------------------------------------------------------------------------
// Scatter 4 fixed64_t to memory using indices
// ---------------------------------------------------------------------------
inline void scatter_i64(fixed64_t* base, __m256i indices, __m256i values, int scale) noexcept {
    _mm256_i64scatter_epi64(reinterpret_cast<long long*>(base), indices, values, scale);
}

// ---------------------------------------------------------------------------
// Shuffle lanes (control mask immediate)
// ---------------------------------------------------------------------------
inline __m256i shuffle_i64(__m256i a, __m256i b, int imm) noexcept {
    return _mm256_permute2f128_si256(a, b, imm);
}

// ---------------------------------------------------------------------------
// Mask from comparison (a == b)
// ---------------------------------------------------------------------------
inline __m256i cmpeq_i64(__m256i a, __m256i b) noexcept {
    return _mm256_cmpeq_epi64(a, b);
}

// ---------------------------------------------------------------------------
// Mask from comparison (a > b) – signed
// ---------------------------------------------------------------------------
inline __m256i cmpgt_i64(__m256i a, __m256i b) noexcept {
    return _mm256_cmpgt_epi64(a, b);
}

// ---------------------------------------------------------------------------
// Horizontal minimum of 4 lanes (scalar reduction)
// ---------------------------------------------------------------------------
inline fixed64_t hmin_i64(__m256i v) noexcept {
    alignas(32) int64_t tmp[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), v);
    return std::min({tmp[0], tmp[1], tmp[2], tmp[3]});
}

// ---------------------------------------------------------------------------
// Horizontal maximum of 4 lanes
// ---------------------------------------------------------------------------
inline fixed64_t hmax_i64(__m256i v) noexcept {
    alignas(32) int64_t tmp[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), v);
    return std::max({tmp[0], tmp[1], tmp[2], tmp[3]});
}

// ---------------------------------------------------------------------------
// Convert 4 fixed64_t lanes to 4 double lanes
// ---------------------------------------------------------------------------
inline __m256d to_double4(__m256i v) noexcept {
    __m256d d = _mm256_cvtepi64_pd(v);
    return _mm256_mul_pd(d, _mm256_set1_pd(1.0 / double(FIXED64_ONE)));
}

// ---------------------------------------------------------------------------
// Convert 4 double lanes to 4 fixed64_t lanes
// ---------------------------------------------------------------------------
inline __m256i from_double4(__m256d d) noexcept {
    d = _mm256_mul_pd(d, _mm256_set1_pd(double(FIXED64_ONE)));
    return _mm256_cvtpd_epi64(d);
}

// ============================================================================
// Tensor / Matrix SIMD operations
// ============================================================================

// 4x4 matrix multiply using SIMD: C = A * B  (row‑major, each row of A broadcast)
inline void fmat4_mul_simd4(const fmat4& A, const fmat4& B, fmat4& C) noexcept {
    // Load rows of B as 4‑lane vectors of columns (i.e., each column of B is a __m256i of its 4 row entries)
    // For row‑major C[r][c] = A[r][0]*B[0][c] + A[r][1]*B[1][c] + A[r][2]*B[2][c] + A[r][3]*B[3][c]
    // We'll compute each result column vector by accumulating
    for (int r = 0; r < 4; ++r) {
        __m256i sum = _mm256_setzero_si256();
        for (int k = 0; k < 4; ++k) {
            __m256i a_rk = set1_i64(*(&A.rows[0].x + r*4 + k));
            // column k of B: need B[0][k], B[1][k], B[2][k], B[3][k]
            // We can construct from rows of B: the k-th component of each row
            alignas(32) fixed64_t col_k[4] = {
                *(&B.rows[0].x + 0*4 + k),
                *(&B.rows[1].x + 1*4 + k),
                *(&B.rows[2].x + 2*4 + k),
                *(&B.rows[3].x + 3*4 + k)
            };
            __m256i b_col = load_i64_4(col_k);
            sum = simd4_add_epi64(sum, simd4_mul_epi64(a_rk, b_col));
        }
        // Store result row
        alignas(32) fixed64_t crow[4];
        store_i64_4(crow, sum);
        for (int c = 0; c < 4; ++c) *(&C.rows[0].x + r*4 + c) = crow[c];
    }
}

// Outer product of two fvec4 vectors using SIMD (4x4 matrix)
inline fmat4 outer_product_simd4(const fvec4& a, const fvec4& b) noexcept {
    fmat4 m;
    // a as four lanes
    __m256i ax = set1_i64(a.x), ay = set1_i64(a.y), az = set1_i64(a.z), aw = set1_i64(a.w);
    // b as a 4‑lane vector
    __m256i bv = _mm256_set_epi64x(b.w, b.z, b.y, b.x);
    // Row0 = a.x * b
    __m256i r0 = simd4_mul_epi64(ax, bv);
    // Row1 = a.y * b
    __m256i r1 = simd4_mul_epi64(ay, bv);
    // Row2 = a.z * b
    __m256i r2 = simd4_mul_epi64(az, bv);
    // Row3 = a.w * b
    __m256i r3 = simd4_mul_epi64(aw, bv);
    // Store rows
    alignas(32) fixed64_t row[4];
    store_i64_4(row, r0);
    m.rows[0] = {row[0], row[1], row[2], row[3]};
    store_i64_4(row, r1);
    m.rows[1] = {row[0], row[1], row[2], row[3]};
    store_i64_4(row, r2);
    m.rows[2] = {row[0], row[1], row[2], row[3]};
    store_i64_4(row, r3);
    m.rows[3] = {row[0], row[1], row[2], row[3]};
    return m;
}

// 4‑lane batch transform of 4 vectors by one 4x4 matrix
inline void fmat4_transform_points_simd4(const fmat4& m, const fvec3 p[4], fvec3 out[4]) noexcept {
    // Load points as SoA
    __m256i px = _mm256_set_epi64x(p[3].x, p[2].x, p[1].x, p[0].x);
    __m256i py = _mm256_set_epi64x(p[3].y, p[2].y, p[1].y, p[0].y);
    __m256i pz = _mm256_set_epi64x(p[3].z, p[2].z, p[1].z, p[0].z);
    // Broadcast matrix elements
    __m256i m00 = set1_i64(m.rows[0].x), m01 = set1_i64(m.rows[0].y), m02 = set1_i64(m.rows[0].z), m03 = set1_i64(m.rows[0].w);
    __m256i m10 = set1_i64(m.rows[1].x), m11 = set1_i64(m.rows[1].y), m12 = set1_i64(m.rows[1].z), m13 = set1_i64(m.rows[1].w);
    __m256i m20 = set1_i64(m.rows[2].x), m21 = set1_i64(m.rows[2].y), m22 = set1_i64(m.rows[2].z), m23 = set1_i64(m.rows[2].w);
    __m256i m30 = set1_i64(m.rows[3].x), m31 = set1_i64(m.rows[3].y), m32 = set1_i64(m.rows[3].z), m33 = set1_i64(m.rows[3].w);
    // Perform dot products
    __m256i rx = simd4_add_epi64(simd4_add_epi64(simd4_mul_epi64(m00, px), simd4_mul_epi64(m01, py)),
                                  simd4_add_epi64(simd4_mul_epi64(m02, pz), m03));
    __m256i ry = simd4_add_epi64(simd4_add_epi64(simd4_mul_epi64(m10, px), simd4_mul_epi64(m11, py)),
                                  simd4_add_epi64(simd4_mul_epi64(m12, pz), m13));
    __m256i rz = simd4_add_epi64(simd4_add_epi64(simd4_mul_epi64(m20, px), simd4_mul_epi64(m21, py)),
                                  simd4_add_epi64(simd4_mul_epi64(m22, pz), m23));
    __m256i rw = simd4_add_epi64(simd4_add_epi64(simd4_mul_epi64(m30, px), simd4_mul_epi64(m31, py)),
                                  simd4_add_epi64(simd4_mul_epi64(m32, pz), m33));
    // Perspective divide (using reciprocal)
    __m256i inv_w = simd4_rcp_epi64(rw);
    __m256i zero = _mm256_setzero_si256();
    __m256i w_eq_zero = cmpeq_i64(rw, zero);
    rx = _mm256_blendv_epi8(simd4_mul_epi64(rx, inv_w), rx, w_eq_zero);
    ry = _mm256_blendv_epi8(simd4_mul_epi64(ry, inv_w), ry, w_eq_zero);
    rz = _mm256_blendv_epi8(simd4_mul_epi64(rz, inv_w), rz, w_eq_zero);
    // Store results
    alignas(32) fixed64_t outx[4], outy[4], outz[4];
    store_i64_4(outx, rx); store_i64_4(outy, ry); store_i64_4(outz, rz);
    for (int i=0;i<4;++i) out[i] = {outx[i], outy[i], outz[i]};
}

} // namespace fixed_math