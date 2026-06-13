// system name : Octree Spatial Master
//File 0007 : core/math/fixed_rand.h
//Fixed‑point pseudo‑random number generators (XorShift128+, SplitMix64) and sampling distributions (uniform, normal, sphere, disc) with SIMD batch generation
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_trig.h"
#include "core/math/fixed_exp_log.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cmath>

namespace fixed_math {

// ============================================================================
// SplitMix64 – seed expansion and state mixing
// ============================================================================
inline uint64_t splitmix64(uint64_t& state) noexcept {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// ============================================================================
// XorShift128+ – fast 64‑bit PRNG
// ============================================================================
struct XorShift128Plus {
    uint64_t s[2];

    explicit XorShift128Plus(uint64_t seed = 123456789) noexcept {
        s[0] = splitmix64(seed);
        s[1] = splitmix64(seed);
    }

    uint64_t next_u64() noexcept {
        uint64_t s1 = s[0];
        const uint64_t s0 = s[1];
        s[0] = s0;
        s1 ^= s1 << 23;
        s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
        return s[1] + s0;
    }

    // Generate a fixed64_t in [0, 1) – top 32 bits scaled to Q32.32
    fixed64_t next_fixed() noexcept {
        return static_cast<fixed64_t>((next_u64() >> 32) & 0xFFFFFFFF) << (FRAC_BITS - 32);
    }

    // Generate a fixed64_t in [lo, hi]
    fixed64_t next_fixed_range(fixed64_t lo, fixed64_t hi) noexcept {
        fixed64_t range = hi - lo;
        return lo + fixed_mul(range, next_fixed());
    }

    // 2D random point in unit disc (rejection sampling)
    fvec3 next_on_disc(fixed64_t radius = FIXED64_ONE) noexcept {
        fixed64_t a, b, len2;
        do {
            a = next_fixed_range(-FIXED64_ONE, FIXED64_ONE);
            b = next_fixed_range(-FIXED64_ONE, FIXED64_ONE);
            len2 = fixed_mul(a, a) + fixed_mul(b, b);
        } while (len2 > FIXED64_ONE);
        return {a, b, 0};
    }

    // 3D random point on unit sphere (trigonometric method)
    fvec3 next_on_sphere() noexcept {
        fixed64_t theta = next_fixed_range(0, 2 * FIXED64_PI);
        fixed64_t phi = fixed_acos(next_fixed_range(-FIXED64_ONE, FIXED64_ONE));
        fixed64_t sin_phi = fixed_sin(phi);
        return {
            fixed_mul(sin_phi, fixed_cos(theta)),
            fixed_mul(sin_phi, fixed_sin(theta)),
            fixed_cos(phi)
        };
    }

    // Standard normal using Box‑Muller (two uniforms -> two normals)
    void next_gaussian_pair(fixed64_t& n0, fixed64_t& n1) noexcept {
        fixed64_t u1 = next_fixed();
        fixed64_t u2 = next_fixed();
        if (u1 == 0) u1 = 1;
        // Box‑Muller: n = sqrt(-2 ln u1) * sin(2 pi u2)
        fixed64_t r = fixed_sqrt(-2 * fixed_log(u1));
        fixed64_t theta = 2 * FIXED64_PI * u2;
        n0 = fixed_mul(r, fixed_cos(theta));
        n1 = fixed_mul(r, fixed_sin(theta));
    }

    fixed64_t next_gaussian() noexcept {
        fixed64_t a, b;
        next_gaussian_pair(a, b);
        return a;
    }
};

// ============================================================================
// SIMD 4‑lane PRNG – four independent generators in parallel
// ============================================================================
struct alignas(64) XorShift128Plusx4 {
    alignas(32) uint64_t s[2][4];

    void seed(uint64_t base) noexcept {
        uint64_t st = base;
        for (int i = 0; i < 4; ++i) {
            uint64_t t = splitmix64(st);
            s[0][i] = t;
            s[1][i] = splitmix64(st);
        }
    }

    // Generate 4 fixed64_t values in [0,1)
    __m256i next_fixed4() noexcept {
        alignas(32) int64_t out[4];
        for (int i = 0; i < 4; ++i) {
            uint64_t s1 = s[0][i];
            uint64_t s0 = s[1][i];
            s[0][i] = s0;
            s1 ^= s1 << 23;
            s[1][i] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
            out[i] = static_cast<int64_t>(((s[1][i] + s0) >> 32) & 0xFFFFFFFF) << (FRAC_BITS - 32);
        }
        return _mm256_load_si256((__m256i*)out);
    }

    // Generate 4 float random values (0..1) as __m256
    __m256 next_float4() noexcept {
        alignas(32) float fout[4];
        __m256i ifx = next_fixed4();
        __m256d d = _mm256_cvtepi64_pd(ifx);
        d = _mm256_mul_pd(d, _mm256_set1_pd(1.0 / double(FIXED64_ONE)));
        _mm256_store_pd((double*)fout, d); // misuse but convert to float after
        // Proper conversion: we'll extract to float array
        alignas(32) int64_t fx[4];
        _mm256_store_si256((__m256i*)fx, ifx);
        for (int i=0;i<4;++i) fout[i] = float_from_fixed(fx[i]);
        return _mm256_load_ps(fout);
    }

    // 4 random points on sphere
    void next_on_sphere4(fvec3* out) noexcept {
        __m256i theta = next_fixed4(); // scale later
        // theta = theta * 2PI / FIXED64_ONE
        alignas(32) int64_t th[4], ph[4];
        __m256i phi = next_fixed4();
        _mm256_store_si256((__m256i*)th, theta);
        _mm256_store_si256((__m256i*)ph, phi);
        for (int i=0;i<4;++i) {
            fixed64_t t = fixed_mul(th[i], 2 * FIXED64_PI);
            fixed64_t p = fixed_acos(fixed_mul(ph[i], 2) - FIXED64_ONE);
            fixed64_t sp = fixed_sin(p);
            out[i] = {
                fixed_mul(sp, fixed_cos(t)),
                fixed_mul(sp, fixed_sin(t)),
                fixed_cos(p)
            };
        }
    }
};

// ============================================================================
// Sampling distributions (standalone)
// ============================================================================
inline fvec3 sample_hemisphere_cosine(const fvec3& normal, XorShift128Plus& rng) noexcept {
    fvec3 on_disc = rng.next_on_disc();
    fixed64_t z = fixed_sqrt(FIXED64_ONE - fixed_mul(on_disc.x, on_disc.x) - fixed_mul(on_disc.y, on_disc.y));
    // Build local frame: tangent and bitangent
    fvec3 up = (normal.z != 0) ? fvec3{1,0,0} : fvec3{0,0,1};
    fvec3 tangent = fvec3_normalize(fvec3_cross(up, normal));
    fvec3 bitangent = fvec3_cross(normal, tangent);
    return fvec3_normalize(fvec3_add(fvec3_add(fvec3_scale(tangent, on_disc.x),
                                               fvec3_scale(bitangent, on_disc.y)),
                                     fvec3_scale(normal, z)));
}

} // namespace fixed_math

// End of File 0007
// Next file: File 0008 – core/math/fixed_geometry.h
// Description: 3D geometry primitives (AABB, OBB, sphere, triangle, ray) with fixed‑point intersection tests, distance queries, and SIMD batch overlap.