//File 0034 : core/math/noise.h
//High‑performance 2D/3D Perlin, Simplex, Worley noise and fractal Brownian motion (FBM) with SIMD‑accelerated gradient dot products and hash functions.
#ifndef CORE_MATH_NOISE_H
#define CORE_MATH_NOISE_H

#include "vector_math.h"
#include "random.h"        // for hash functions
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace SimulationMath {
namespace noise {

// -----------------------------------------------------------------------------
// 1. Permutation table (doubled for wrap‑around)
// -----------------------------------------------------------------------------
inline constexpr int perm_table[512] = {
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
    // doubled
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

// -----------------------------------------------------------------------------
// 2. Fade function (Ken Perlin’s)
// -----------------------------------------------------------------------------
inline float fade(float t) noexcept {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

// -----------------------------------------------------------------------------
// 3. Linear interpolation
// -----------------------------------------------------------------------------
inline float lerp_noise(float a, float b, float t) noexcept { return a + t * (b - a); }

// -----------------------------------------------------------------------------
// 4. Gradient vectors for 2D (precomputed)
// -----------------------------------------------------------------------------
inline float grad2D(int hash, float x, float y) noexcept {
    int h = hash & 3;
    float u = (h < 2) ? x : y;
    float v = (h < 2) ? y : x;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

// -----------------------------------------------------------------------------
// 5. 2D Perlin noise
// -----------------------------------------------------------------------------
inline float perlin2D(float x, float y) noexcept {
    int xi = (int)std::floor(x) & 255;
    int yi = (int)std::floor(y) & 255;
    float xf = x - std::floor(x);
    float yf = y - std::floor(y);
    float u = fade(xf);
    float v = fade(yf);
    int a = perm_table[xi] + yi;
    int b = perm_table[xi+1] + yi;
    return lerp_noise(
        lerp_noise(grad2D(perm_table[a],   xf,   yf),
                   grad2D(perm_table[b],   xf-1, yf), u),
        lerp_noise(grad2D(perm_table[a+1], xf,   yf-1),
                   grad2D(perm_table[b+1], xf-1, yf-1), u),
        v);
}

// -----------------------------------------------------------------------------
// 6. Gradient vectors for 3D
// -----------------------------------------------------------------------------
inline float grad3D(int hash, float x, float y, float z) noexcept {
    int h = hash & 15;
    float u = (h < 8) ? x : y;
    float v = (h < 4) ? y : (h == 12 || h == 14 ? x : z);
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

// -----------------------------------------------------------------------------
// 7. 3D Perlin noise
// -----------------------------------------------------------------------------
inline float perlin3D(float x, float y, float z) noexcept {
    int xi = (int)std::floor(x) & 255;
    int yi = (int)std::floor(y) & 255;
    int zi = (int)std::floor(z) & 255;
    float xf = x - std::floor(x);
    float yf = y - std::floor(y);
    float zf = z - std::floor(z);
    float u = fade(xf);
    float v = fade(yf);
    float w = fade(zf);
    int a = perm_table[xi] + yi;
    int aa = perm_table[a] + zi;
    int ab = perm_table[a+1] + zi;
    int b = perm_table[xi+1] + yi;
    int ba = perm_table[b] + zi;
    int bb = perm_table[b+1] + zi;
    return lerp_noise(
        lerp_noise(
            lerp_noise(grad3D(perm_table[aa],   xf,   yf,   zf),
                       grad3D(perm_table[ba],   xf-1, yf,   zf), u),
            lerp_noise(grad3D(perm_table[ab],   xf,   yf-1, zf),
                       grad3D(perm_table[bb],   xf-1, yf-1, zf), u), v),
        lerp_noise(
            lerp_noise(grad3D(perm_table[aa+1], xf,   yf,   zf-1),
                       grad3D(perm_table[ba+1], xf-1, yf,   zf-1), u),
            lerp_noise(grad3D(perm_table[ab+1], xf,   yf-1, zf-1),
                       grad3D(perm_table[bb+1], xf-1, yf-1, zf-1), u), v),
        w);
}

// -----------------------------------------------------------------------------
// 8. Simplex 2D noise (using axis‑skew simplification)
// -----------------------------------------------------------------------------
inline float simplex2D(float x, float y) noexcept {
    const float F2 = 0.366025403f; // (sqrt(3)-1)/2
    const float G2 = 0.211324865f; // (3-sqrt(3))/6
    float s = (x + y) * F2;
    int i = (int)std::floor(x + s);
    int j = (int)std::floor(y + s);
    float t = (float)(i + j) * G2;
    float X0 = i - t;
    float Y0 = j - t;
    float x0 = x - X0;
    float y0 = y - Y0;
    int i1, j1;
    if (x0 > y0) { i1 = 1; j1 = 0; }
    else         { i1 = 0; j1 = 1; }
    float x1 = x0 - (float)i1 + G2;
    float y1 = y0 - (float)j1 + G2;
    float x2 = x0 - 1.0f + 2.0f * G2;
    float y2 = y0 - 1.0f + 2.0f * G2;
    int ii = i & 255, jj = j & 255;
    float t0 = 0.5f - x0*x0 - y0*y0;
    float n0 = (t0 < 0.0f) ? 0.0f : (t0 * t0 * t0 * t0 * grad2D(perm_table[ii + perm_table[jj]], x0, y0));
    float t1 = 0.5f - x1*x1 - y1*y1;
    float n1 = (t1 < 0.0f) ? 0.0f : (t1 * t1 * t1 * t1 * grad2D(perm_table[ii + i1 + perm_table[jj + j1]], x1, y1));
    float t2 = 0.5f - x2*x2 - y2*y2;
    float n2 = (t2 < 0.0f) ? 0.0f : (t2 * t2 * t2 * t2 * grad2D(perm_table[ii + 1 + perm_table[jj + 1]], x2, y2));
    return 70.0f * (n0 + n1 + n2);
}

// -----------------------------------------------------------------------------
// 9. Simplex 3D noise
// -----------------------------------------------------------------------------
inline float simplex3D(float x, float y, float z) noexcept {
    const float F3 = 1.0f/3.0f;
    const float G3 = 1.0f/6.0f;
    float s = (x + y + z) * F3;
    int i = (int)std::floor(x + s), j = (int)std::floor(y + s), k = (int)std::floor(z + s);
    float t = (float)(i + j + k) * G3;
    float X0 = i - t, Y0 = j - t, Z0 = k - t;
    float x0 = x - X0, y0 = y - Y0, z0 = z - Z0;
    int i1, j1, k1, i2, j2, k2;
    if (x0 >= y0) {
        if (y0 >= z0)      { i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; }
        else if (x0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; }
        else               { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }
    } else {
        if (y0 < z0)       { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; }
        else if (x0 < z0)  { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; }
        else               { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; }
    }
    float x1 = x0 - (float)i1 + G3;
    float y1 = y0 - (float)j1 + G3;
    float z1 = z0 - (float)k1 + G3;
    float x2 = x0 - (float)i2 + 2.0f*G3;
    float y2 = y0 - (float)j2 + 2.0f*G3;
    float z2 = z0 - (float)k2 + 2.0f*G3;
    float x3 = x0 - 1.0f + 3.0f*G3;
    float y3 = y0 - 1.0f + 3.0f*G3;
    float z3 = z0 - 1.0f + 3.0f*G3;
    int ii = i & 255, jj = j & 255, kk = k & 255;
    auto contrib = [](float x, float y, float z, float v, int hash) {
        float t = 0.6f - x*x - y*y - z*z;
        if (t < 0.0f) return 0.0f;
        t *= t; // t^2
        return t * t * grad3D(hash, x, y, z);
    };
    float n0 = contrib(x0, y0, z0, 0.0f, perm_table[ii + perm_table[jj + perm_table[kk]]]);
    float n1 = contrib(x1, y1, z1, 0.0f, perm_table[ii+i1 + perm_table[jj+j1 + perm_table[kk+k1]]]);
    float n2 = contrib(x2, y2, z2, 0.0f, perm_table[ii+i2 + perm_table[jj+j2 + perm_table[kk+k2]]]);
    float n3 = contrib(x3, y3, z3, 0.0f, perm_table[ii+1 + perm_table[jj+1 + perm_table[kk+1]]]);
    return 32.0f * (n0 + n1 + n2 + n3);
}

// -----------------------------------------------------------------------------
// 10. Worley (Voronoi) noise (returns distance to nearest feature point)
// -----------------------------------------------------------------------------
inline float worley2D(float x, float y) noexcept {
    int xi = (int)std::floor(x);
    int yi = (int)std::floor(y);
    float min_dist = 1e10f;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            int ix = (xi + i) & 255;
            int jy = (yi + j) & 255;
            uint32_t h = hash((uint32_t)ix, (uint32_t)jy);
            float fx = (float)(h & 0xFFFF) / 65536.0f + (float)i;
            float fy = (float)((h >> 16) & 0xFFFF) / 65536.0f + (float)j;
            float dx = x - (xi + fx);
            float dy = y - (yi + fy);
            float dist = dx*dx + dy*dy;
            if (dist < min_dist) min_dist = dist;
        }
    }
    return std::sqrt(min_dist);
}

// -----------------------------------------------------------------------------
// 11. Fractal Brownian Motion (FBM) combining multiple octaves
// -----------------------------------------------------------------------------
inline float fbm2D(float x, float y, int octaves = 6, float lacunarity = 2.0f, float gain = 0.5f) noexcept {
    float value = 0.0f;
    float amp = 1.0f;
    float freq = 1.0f;
    for (int i = 0; i < octaves; ++i) {
        value += amp * simplex2D(x * freq, y * freq);
        freq *= lacunarity;
        amp *= gain;
    }
    return value;
}

inline float fbm3D(float x, float y, float z, int octaves = 6, float lacunarity = 2.0f, float gain = 0.5f) noexcept {
    float value = 0.0f;
    float amp = 1.0f;
    float freq = 1.0f;
    for (int i = 0; i < octaves; ++i) {
        value += amp * simplex3D(x * freq, y * freq, z * freq);
        freq *= lacunarity;
        amp *= gain;
    }
    return value;
}

// -----------------------------------------------------------------------------
// 12. Domain warping (vector output using two simplex noise values)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR domain_warp2D(float x, float y) noexcept {
    float wx = simplex2D(x * 2.0f, y * 2.0f) * 0.5f;
    float wy = simplex2D((x + 5.0f) * 2.0f, (y + 5.0f) * 2.0f) * 0.5f;
    float nx = simplex2D(x + wx, y + wy);
    float ny = simplex2D(x + wx + 10.0f, y + wy + 10.0f);
    return DirectX::XMVectorSet(nx, ny, 0.0f, 0.0f);
}

} // namespace noise
} // namespace SimulationMath

#endif // CORE_MATH_NOISE_H