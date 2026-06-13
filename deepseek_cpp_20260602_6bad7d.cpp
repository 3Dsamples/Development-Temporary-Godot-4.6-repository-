// system name : Octree Spatial Master
//File 0011 : core/math/fixed_interpolation.h
//Fixed‑point interpolation functions (linear, cubic Hermite, Catmull‑Rom, Bézier) for scalars and vectors with SIMD 4‑lane support
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

// ---------------------------------------------------------------------------
// Linear interpolation (scalar): a + t*(b - a)
// ---------------------------------------------------------------------------
inline fixed64_t lerp(fixed64_t a, fixed64_t b, fixed64_t t) noexcept {
    return fixed_add(a, fixed_mul(fixed_sub(b, a), t));
}

// ---------------------------------------------------------------------------
// Vector linear interpolation
// ---------------------------------------------------------------------------
inline fvec3 lerp_vec3(const fvec3& a, const fvec3& b, fixed64_t t) noexcept {
    return fvec3_lerp(a, b, t);
}

// ---------------------------------------------------------------------------
// Cubic Hermite interpolation (scalar)
//   p0, p1 – points; m0, m1 – tangents; t in [0,1]
// ---------------------------------------------------------------------------
inline fixed64_t hermite(fixed64_t p0, fixed64_t m0, fixed64_t p1, fixed64_t m1, fixed64_t t) noexcept {
    fixed64_t t2 = fixed_mul(t, t);
    fixed64_t t3 = fixed_mul(t2, t);
    fixed64_t h00 = 2*t3 - 3*t2 + FIXED64_ONE;  // (2t^3 - 3t^2 + 1)
    fixed64_t h10 = t3 - 2*t2 + t;               // (t^3 - 2t^2 + t)
    fixed64_t h01 = -2*t3 + 3*t2;                // (-2t^3 + 3t^2)
    fixed64_t h11 = t3 - t2;                      // (t^3 - t^2)
    return fixed_add(fixed_add(fixed_mul(h00, p0), fixed_mul(h10, m0)),
                     fixed_add(fixed_mul(h01, p1), fixed_mul(h11, m1)));
}

// ---------------------------------------------------------------------------
// Cubic Hermite for vectors
// ---------------------------------------------------------------------------
inline fvec3 hermite_vec3(const fvec3& p0, const fvec3& m0, const fvec3& p1, const fvec3& m1, fixed64_t t) noexcept {
    return {
        hermite(p0.x, m0.x, p1.x, m1.x, t),
        hermite(p0.y, m0.y, p1.y, m1.y, t),
        hermite(p0.z, m0.z, p1.z, m1.z, t)
    };
}

// ---------------------------------------------------------------------------
// Catmull‑Rom spline (scalar) – interpolates between p1 and p2 using surrounding points
//   Assumes uniform parameterisation; returns point at t in [0,1] between p1 and p2.
// ---------------------------------------------------------------------------
inline fixed64_t catmull_rom(fixed64_t p0, fixed64_t p1, fixed64_t p2, fixed64_t p3, fixed64_t t) noexcept {
    fixed64_t t2 = fixed_mul(t, t);
    fixed64_t t3 = fixed_mul(t2, t);
    // 0.5 * (2*p1 + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t^2 + (-p0+3*p1-3*p2+p3)*t^3)
    return (FIXED64_HALF) * (
        2*p1 + (-p0+p2)*t + (2*p0 - 5*p1 + 4*p2 - p3)*t2 + (-p0 + 3*p1 - 3*p2 + p3)*t3
    );
}

// ---------------------------------------------------------------------------
// Catmull‑Rom for vectors
// ---------------------------------------------------------------------------
inline fvec3 catmull_rom_vec3(const fvec3& p0, const fvec3& p1, const fvec3& p2, const fvec3& p3, fixed64_t t) noexcept {
    return {
        catmull_rom(p0.x, p1.x, p2.x, p3.x, t),
        catmull_rom(p0.y, p1.y, p2.y, p3.y, t),
        catmull_rom(p0.z, p1.z, p2.z, p3.z, t)
    };
}

// ---------------------------------------------------------------------------
// Quadratic Bézier (scalar) – three control points P0, P1, P2
// ---------------------------------------------------------------------------
inline fixed64_t bezier_quad(fixed64_t p0, fixed64_t p1, fixed64_t p2, fixed64_t t) noexcept {
    fixed64_t u = FIXED64_ONE - t;
    // B(t) = u^2*P0 + 2*u*t*P1 + t^2*P2
    return fixed_add(fixed_add(fixed_mul(fixed_mul(u, u), p0),
                               fixed_mul(2 * FIXED64_ONE, fixed_mul(fixed_mul(u, t), p1))),
                     fixed_mul(fixed_mul(t, t), p2));
}

// ---------------------------------------------------------------------------
// Cubic Bézier (scalar) – four control points
// ---------------------------------------------------------------------------
inline fixed64_t bezier_cubic(fixed64_t p0, fixed64_t p1, fixed64_t p2, fixed64_t p3, fixed64_t t) noexcept {
    fixed64_t u = FIXED64_ONE - t;
    fixed64_t u2 = fixed_mul(u, u);
    fixed64_t t2 = fixed_mul(t, t);
    fixed64_t u3 = fixed_mul(u2, u);
    fixed64_t t3 = fixed_mul(t2, t);
    // B(t) = u^3*P0 + 3*u^2*t*P1 + 3*u*t^2*P2 + t^3*P3
    return fixed_add(fixed_add(fixed_mul(u3, p0),
                               fixed_mul(3 * FIXED64_ONE, fixed_mul(fixed_mul(u2, t), p1))),
                     fixed_add(fixed_mul(3 * FIXED64_ONE, fixed_mul(fixed_mul(u, t2), p2)),
                               fixed_mul(t3, p3)));
}

// ---------------------------------------------------------------------------
// Cubic Bézier for vectors
// ---------------------------------------------------------------------------
inline fvec3 bezier_cubic_vec3(const fvec3& p0, const fvec3& p1, const fvec3& p2, const fvec3& p3, fixed64_t t) noexcept {
    return {
        bezier_cubic(p0.x, p1.x, p2.x, p3.x, t),
        bezier_cubic(p0.y, p1.y, p2.y, p3.y, t),
        bezier_cubic(p0.z, p1.z, p2.z, p3.z, t)
    };
}

// ============================================================================
// SIMD 4‑lane interpolation functions
// ============================================================================

// 4‑lane linear interpolation (scalar fixed64_t lanes)
inline __m256i simd4_lerp(__m256i a, __m256i b, __m256i t) noexcept {
    __m256i diff = simd4_sub_epi64(b, a);
    __m256i term = simd4_mul_epi64(diff, t);
    return simd4_add_epi64(a, term);
}

// 4‑lane cubic Hermite
inline __m256i simd4_hermite(__m256i p0, __m256i m0, __m256i p1, __m256i m1, __m256i t) noexcept {
    __m256i t2 = simd4_mul_epi64(t, t);
    __m256i t3 = simd4_mul_epi64(t2, t);
    __m256i one = _mm256_set1_epi64x(FIXED64_ONE);
    __m256i h00 = simd4_add_epi64(simd4_sub_epi64(simd4_mul_epi64(_mm256_set1_epi64x(2), t3),
                                                   simd4_mul_epi64(_mm256_set1_epi64x(3), t2)), one);
    __m256i h10 = simd4_add_epi64(simd4_sub_epi64(t3, simd4_mul_epi64(_mm256_set1_epi64x(2), t2)), t);
    __m256i h01 = simd4_add_epi64(simd4_mul_epi64(_mm256_set1_epi64x(-2), t3),
                                  simd4_mul_epi64(_mm256_set1_epi64x(3), t2));
    __m256i h11 = simd4_sub_epi64(t3, t2);
    // Combine
    __m256i r0 = simd4_mul_epi64(h00, p0);
    __m256i r1 = simd4_mul_epi64(h10, m0);
    __m256i r2 = simd4_mul_epi64(h01, p1);
    __m256i r3 = simd4_mul_epi64(h11, m1);
    return simd4_add_epi64(simd4_add_epi64(r0, r1), simd4_add_epi64(r2, r3));
}

// 4‑lane Catmull‑Rom
inline __m256i simd4_catmull_rom(__m256i p0, __m256i p1, __m256i p2, __m256i p3, __m256i t) noexcept {
    __m256i t2 = simd4_mul_epi64(t, t);
    __m256i t3 = simd4_mul_epi64(t2, t);
    __m256i half = _mm256_set1_epi64x(FIXED64_HALF);
    __m256i term1 = simd4_add_epi64(simd4_mul_epi64(_mm256_set1_epi64x(2), p1),
                                    simd4_mul_epi64(simd4_sub_epi64(p2, p0), t));
    __m256i term2 = simd4_mul_epi64(simd4_add_epi64(simd4_sub_epi64(simd4_mul_epi64(_mm256_set1_epi64x(2), p0),
                                                                      simd4_mul_epi64(_mm256_set1_epi64x(5), p1)),
                                                     simd4_add_epi64(simd4_mul_epi64(_mm256_set1_epi64x(4), p2), simd4_neg_epi64(p3))), t2);
    __m256i term3 = simd4_mul_epi64(simd4_add_epi64(simd4_neg_epi64(p0),
                                                     simd4_add_epi64(simd4_mul_epi64(_mm256_set1_epi64x(3), p1),
                                                                     simd4_add_epi64(simd4_mul_epi64(_mm256_set1_epi64x(-3), p2), p3))), t3);
    return simd4_mul_epi64(half, simd4_add_epi64(simd4_add_epi64(term1, term2), term3));
}
// Helper for negation (sub from zero)
inline __m256i simd4_neg_epi64(__m256i x) noexcept { return simd4_sub_epi64(_mm256_setzero_si256(), x); }

// 4‑lane cubic Bézier
inline __m256i simd4_bezier_cubic(__m256i p0, __m256i p1, __m256i p2, __m256i p3, __m256i t) noexcept {
    __m256i u = simd4_sub_epi64(_mm256_set1_epi64x(FIXED64_ONE), t);
    __m256i u2 = simd4_mul_epi64(u, u);
    __m256i t2 = simd4_mul_epi64(t, t);
    __m256i u3 = simd4_mul_epi64(u2, u);
    __m256i t3 = simd4_mul_epi64(t2, t);
    __m256i three = _mm256_set1_epi64x(3 * FIXED64_ONE);
    __m256i term0 = simd4_mul_epi64(u3, p0);
    __m256i term1 = simd4_mul_epi64(three, simd4_mul_epi64(simd4_mul_epi64(u2, t), p1));
    __m256i term2 = simd4_mul_epi64(three, simd4_mul_epi64(simd4_mul_epi64(u, t2), p2));
    __m256i term3 = simd4_mul_epi64(t3, p3);
    return simd4_add_epi64(simd4_add_epi64(term0, term1), simd4_add_epi64(term2, term3));
}

} // namespace fixed_math

// End of File 0011
// Next file: File 0012 – core/math/fixed_simd_util.h
// Description: SIMD utility functions for fixed‑point 64‑bit integer operations: load/store, blend, gather/scatter, shuffle, and conversion helpers.