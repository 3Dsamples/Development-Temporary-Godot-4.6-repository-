// system name : Octree Spatial Master
//File 0014 : core/math/fixed_noise.h
//Fixed‑point Perlin noise 3D, FBM, and SIMD 4‑lane evaluation for procedural generation and simulation
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_interpolation.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <array>

namespace fixed_math {

// ---------------------------------------------------------------------------
// Permutation table (256 values)
// ---------------------------------------------------------------------------
alignas(64) static const uint8_t PERM[512] = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
    // duplicate for wrapping
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

// ---------------------------------------------------------------------------
// Gradient vectors for 3D (12 directions, normalized to approximate unit length)
// ---------------------------------------------------------------------------
alignas(64) static const fvec3 GRAD3[12] = {
    {0x100000000LL,0x100000000LL,0}, {-0x100000000LL,0x100000000LL,0},
    {0x100000000LL,-0x100000000LL,0}, {-0x100000000LL,-0x100000000LL,0},
    {0x100000000LL,0,0x100000000LL}, {-0x100000000LL,0,0x100000000LL},
    {0x100000000LL,0,-0x100000000LL}, {-0x100000000LL,0,-0x100000000LL},
    {0,0x100000000LL,0x100000000LL}, {0,-0x100000000LL,0x100000000LL},
    {0,0x100000000LL,-0x100000000LL}, {0,-0x100000000LL,-0x100000000LL}
};

// ---------------------------------------------------------------------------
// Fade curve (quintic): 6t^5 - 15t^4 + 10t^3
// ---------------------------------------------------------------------------
inline fixed64_t fade_quintic(fixed64_t t) noexcept {
    fixed64_t t2 = fixed_mul(t, t);
    fixed64_t t3 = fixed_mul(t2, t);
    fixed64_t t4 = fixed_mul(t3, t);
    fixed64_t t5 = fixed_mul(t4, t);
    return 6*t5 - 15*t4 + 10*t3;
}

// ---------------------------------------------------------------------------
// 3D Perlin noise (scalar) – returns value in [-ONE, ONE]
// ---------------------------------------------------------------------------
inline fixed64_t perlin3d(fvec3 p) noexcept {
    int64_t xi = p.x >> FRAC_BITS;
    int64_t yi = p.y >> FRAC_BITS;
    int64_t zi = p.z >> FRAC_BITS;
    fixed64_t xf = p.x & (FIXED64_ONE - 1);
    fixed64_t yf = p.y & (FIXED64_ONE - 1);
    fixed64_t zf = p.z & (FIXED64_ONE - 1);
    fixed64_t u = fade_quintic(xf);
    fixed64_t v = fade_quintic(yf);
    fixed64_t w = fade_quintic(zf);

    auto hash = [](int64_t x, int64_t y, int64_t z) -> int {
        return PERM[(PERM[(PERM[x & 255] + (y & 255)) & 255] + (z & 255)) & 255];
    };

    int aaa = hash(xi,   yi,   zi);
    int aba = hash(xi,   yi+1, zi);
    int aab = hash(xi,   yi,   zi+1);
    int abb = hash(xi,   yi+1, zi+1);
    int baa = hash(xi+1, yi,   zi);
    int bba = hash(xi+1, yi+1, zi);
    int bab = hash(xi+1, yi,   zi+1);
    int bbb = hash(xi+1, yi+1, zi+1);

    const fvec3& g000 = GRAD3[aaa % 12];
    const fvec3& g100 = GRAD3[baa % 12];
    const fvec3& g010 = GRAD3[aba % 12];
    const fvec3& g110 = GRAD3[bba % 12];
    const fvec3& g001 = GRAD3[aab % 12];
    const fvec3& g101 = GRAD3[bab % 12];
    const fvec3& g011 = GRAD3[abb % 12];
    const fvec3& g111 = GRAD3[bbb % 12];

    fvec3 d000 = {xf,      yf,      zf};
    fvec3 d100 = {xf - FIXED64_ONE, yf,      zf};
    fvec3 d010 = {xf,      yf - FIXED64_ONE, zf};
    fvec3 d110 = {xf - FIXED64_ONE, yf - FIXED64_ONE, zf};
    fvec3 d001 = {xf,      yf,      zf - FIXED64_ONE};
    fvec3 d101 = {xf - FIXED64_ONE, yf,      zf - FIXED64_ONE};
    fvec3 d011 = {xf,      yf - FIXED64_ONE, zf - FIXED64_ONE};
    fvec3 d111 = {xf - FIXED64_ONE, yf - FIXED64_ONE, zf - FIXED64_ONE};

    auto dot = [](const fvec3& a, const fvec3& b) -> fixed64_t {
        return fixed_add(fixed_add(fixed_mul(a.x, b.x), fixed_mul(a.y, b.y)), fixed_mul(a.z, b.z));
    };

    fixed64_t n000 = dot(g000, d000);
    fixed64_t n100 = dot(g100, d100);
    fixed64_t n010 = dot(g010, d010);
    fixed64_t n110 = dot(g110, d110);
    fixed64_t n001 = dot(g001, d001);
    fixed64_t n101 = dot(g101, d101);
    fixed64_t n011 = dot(g011, d011);
    fixed64_t n111 = dot(g111, d111);

    // Interpolate along X
    fixed64_t nx00 = lerp(n000, n100, u);
    fixed64_t nx10 = lerp(n010, n110, u);
    fixed64_t nx01 = lerp(n001, n101, u);
    fixed64_t nx11 = lerp(n011, n111, u);
    // Interpolate along Y
    fixed64_t nxy0 = lerp(nx00, nx10, v);
    fixed64_t nxy1 = lerp(nx01, nx11, v);
    // Interpolate along Z
    fixed64_t nxyz = lerp(nxy0, nxy1, w);
    return nxyz;
}

// ---------------------------------------------------------------------------
// Fractal Brownian Motion (FBM) – sum of octaves of Perlin noise
// ---------------------------------------------------------------------------
inline fixed64_t fbm3d(fvec3 p, int octaves, fixed64_t lacunarity, fixed64_t gain) noexcept {
    fixed64_t value = 0;
    fixed64_t amplitude = FIXED64_ONE;
    fixed64_t frequency = FIXED64_ONE;
    fixed64_t maxVal = 0;
    for (int i = 0; i < octaves; ++i) {
        fvec3 q = { fixed_mul(p.x, frequency), fixed_mul(p.y, frequency), fixed_mul(p.z, frequency) };
        value += fixed_mul(perlin3d(q), amplitude);
        maxVal += amplitude;
        frequency = fixed_mul(frequency, lacunarity);
        amplitude = fixed_mul(amplitude, gain);
    }
    return fixed_div(value, maxVal);
}

// ---------------------------------------------------------------------------
// SIMD 4‑lane 3D Perlin noise (scalar extraction per lane)
// ---------------------------------------------------------------------------
inline __m256i perlin3d_simd4(__m256i px, __m256i py, __m256i pz) noexcept {
    alignas(32) int64_t xv[4], yv[4], zv[4];
    _mm256_store_si256((__m256i*)xv, px);
    _mm256_store_si256((__m256i*)yv, py);
    _mm256_store_si256((__m256i*)zv, pz);
    alignas(32) int64_t res[4];
    for (int i = 0; i < 4; ++i) {
        fvec3 p = {xv[i], yv[i], zv[i]};
        res[i] = perlin3d(p);
    }
    return _mm256_load_si256((__m256i*)res);
}

// ---------------------------------------------------------------------------
// Tensor‑like gradient field: return the pseudorandom gradient vector at a given integer lattice point
// ---------------------------------------------------------------------------
inline fvec3 noise_gradient(int ix, int iy, int iz) noexcept {
    int h = PERM[(PERM[(PERM[ix & 255] + (iy & 255)) & 255] + (iz & 255)) & 255];
    return GRAD3[h % 12];
}

} // namespace fixed_math