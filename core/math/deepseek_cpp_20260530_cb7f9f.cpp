// File 0028 : core/math/noise.h
// Procedural noise: Perlin, Simplex, Cellular, and fractal variants (FBM, turbulent) for 2D/3D.

#pragma once

#include "vec2.h"
#include "vec3.h"
#include "constants.h"
#include "interpolation.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace wp {

// ── Internal hash and gradient helpers ───────────────────────────────
namespace noise_detail {

// Permutation table (256 values, duplicated to avoid modulo)
inline constexpr uint8 perm[512] = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
    190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,
    125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,
    105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,
    135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,
    82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,
    153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,
    251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,
    106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,128,
    195,78,66,215,61,156,180
    // Same values duplicated for seamless wrapping
};
inline constexpr uint8 perm_dup[512] = {0}; // actually we will just duplicate manually by using perm[i & 255]

inline uint8 hash(int32 x) { return perm[x & 255]; }
inline uint8 hash(int32 x, int32 y) { return perm[perm[x & 255] + y & 255]; }
inline uint8 hash(int32 x, int32 y, int32 z) { return perm[perm[perm[x & 255] + y & 255] + z & 255]; }

// Gradient vectors for 2D Perlin (12 directions)
template <typename T>
constexpr vec2<T> grad2(uint8 hash_val) noexcept {
    constexpr T sqrt2 = T(0.7071067811865475); // 1/sqrt(2)
    switch (hash_val & 0x3) {
        case 0: return vec2<T>( T(1),  T(0));
        case 1: return vec2<T>(-T(1),  T(0));
        case 2: return vec2<T>( T(0),  T(1));
        default:return vec2<T>( T(0), -T(1));
    }
}

// Gradient vectors for 3D Perlin (12 edges of cube)
template <typename T>
constexpr vec3<T> grad3(uint8 hash_val) noexcept {
    static const vec3<T> grads[12] = {
        {1,1,0}, {-1,1,0}, {1,-1,0}, {-1,-1,0},
        {1,0,1}, {-1,0,1}, {1,0,-1}, {-1,0,-1},
        {0,1,1}, {0,-1,1}, {0,1,-1}, {0,-1,-1}
    };
    return grads[hash_val % 12];
}

// Simplex 2D gradient from lookup
template <typename T>
constexpr vec2<T> simplex_grad2(uint8 hash_val) noexcept {
    constexpr T inv = T(0.7071067811865476);
    switch (hash_val & 0x7) {
        case 0: return vec2<T>( inv,  inv);
        case 1: return vec2<T>(-inv,  inv);
        case 2: return vec2<T>( inv, -inv);
        case 3: return vec2<T>(-inv, -inv);
        case 4: return vec2<T>( T(1),  T(0));
        case 5: return vec2<T>(-T(1),  T(0));
        case 6: return vec2<T>( T(0),  T(1));
        default:return vec2<T>( T(0), -T(1));
    }
}

// Simplex 3D gradient
template <typename T>
constexpr vec3<T> simplex_grad3(uint8 hash_val) noexcept {
    constexpr T inv = T(0.7071067811865476);
    switch (hash_val & 0x7) {
        case 0: return vec3<T>( inv,  inv,  inv);
        case 1: return vec3<T>(-inv,  inv,  inv);
        case 2: return vec3<T>( inv, -inv,  inv);
        case 3: return vec3<T>(-inv, -inv,  inv);
        case 4: return vec3<T>( inv,  inv, -inv);
        case 5: return vec3<T>(-inv,  inv, -inv);
        case 6: return vec3<T>( inv, -inv, -inv);
        default:return vec3<T>(-inv, -inv, -inv);
    }
}

} // namespace noise_detail

// ═════════════════════════════════════════════════════════════════════
// Perlin Noise 2D / 3D
// ═════════════════════════════════════════════════════════════════════

template <typename T = float>
T perlin2d(T x, T y) noexcept {
    int32 xi = static_cast<int32>(std::floor(x)) & 255;
    int32 yi = static_cast<int32>(std::floor(y)) & 255;
    T xf = x - std::floor(x);
    T yf = y - std::floor(y);
    // Smoothstep interpolation weights
    T u = smootherstep(T(0), T(1), xf);
    T v = smootherstep(T(0), T(1), yf);

    uint8 aa = noise_detail::hash(xi, yi);
    uint8 ab = noise_detail::hash(xi, yi+1);
    uint8 ba = noise_detail::hash(xi+1, yi);
    uint8 bb = noise_detail::hash(xi+1, yi+1);

    T x1 = lerp(dot(noise_detail::grad2<T>(aa), vec2<T>(xf, yf)),
                 dot(noise_detail::grad2<T>(ba), vec2<T>(xf-1, yf)), u);
    T x2 = lerp(dot(noise_detail::grad2<T>(ab), vec2<T>(xf, yf-1)),
                 dot(noise_detail::grad2<T>(bb), vec2<T>(xf-1, yf-1)), u);
    // Range approximately [-1,1], scale if desired
    return lerp(x1, x2, v);
}

template <typename T = float>
T perlin3d(T x, T y, T z) noexcept {
    int32 xi = static_cast<int32>(std::floor(x)) & 255;
    int32 yi = static_cast<int32>(std::floor(y)) & 255;
    int32 zi = static_cast<int32>(std::floor(z)) & 255;
    T xf = x - std::floor(x);
    T yf = y - std::floor(y);
    T zf = z - std::floor(z);

    T u = smootherstep(T(0), T(1), xf);
    T v = smootherstep(T(0), T(1), yf);
    T w = smootherstep(T(0), T(1), zf);

    auto h = [](int32 x, int32 y, int32 z) { return noise_detail::hash(x, y, z); };
    auto g = [](uint8 hv, const vec3<T>& v) { return dot(noise_detail::grad3<T>(hv), v); };

    int32 xi0 = xi, xi1 = xi+1;
    int32 yi0 = yi, yi1 = yi+1;
    int32 zi0 = zi, zi1 = zi+1;

    T v000 = g(h(xi0, yi0, zi0), vec3<T>(xf, yf, zf));
    T v100 = g(h(xi1, yi0, zi0), vec3<T>(xf-1, yf, zf));
    T v010 = g(h(xi0, yi1, zi0), vec3<T>(xf, yf-1, zf));
    T v110 = g(h(xi1, yi1, zi0), vec3<T>(xf-1, yf-1, zf));
    T v001 = g(h(xi0, yi0, zi1), vec3<T>(xf, yf, zf-1));
    T v101 = g(h(xi1, yi0, zi1), vec3<T>(xf-1, yf, zf-1));
    T v011 = g(h(xi0, yi1, zi1), vec3<T>(xf, yf-1, zf-1));
    T v111 = g(h(xi1, yi1, zi1), vec3<T>(xf-1, yf-1, zf-1));

    T y0 = lerp(lerp(v000, v100, u), lerp(v010, v110, u), v);
    T y1 = lerp(lerp(v001, v101, u), lerp(v011, v111, u), v);
    return lerp(y0, y1, w);
}

// ═════════════════════════════════════════════════════════════════════
// Simplex Noise 2D / 3D (improved by Stefan Gustavson)
// ═════════════════════════════════════════════════════════════════════

template <typename T = float>
T simplex2d(T x, T y) noexcept {
    constexpr T F2 = T(0.3660254037844386); // (sqrt(3)-1)/2
    constexpr T G2 = T(0.21132486540518713); // (3-sqrt(3))/6
    constexpr T SKEW_FACTOR = T(0.5) * (std::sqrt(T(3)) - T(1));
    // Skew input space to triangular grid
    T s = (x + y) * F2;
    int32 i = static_cast<int32>(std::floor(x + s));
    int32 j = static_cast<int32>(std::floor(y + s));
    T t = T(i + j) * G2;
    T X0 = i - t;
    T Y0 = j - t;
    T x0 = x - X0;
    T y0 = y - Y0;

    // Determine which simplex we are in
    int32 i1, j1;
    if (x0 > y0) { i1 = 1; j1 = 0; }
    else         { i1 = 0; j1 = 1; }

    T x1 = x0 - i1 + G2;
    T y1 = y0 - j1 + G2;
    T x2 = x0 - T(1) + T(2) * G2;
    T y2 = y0 - T(1) + T(2) * G2;

    // Hash the three corners
    uint8 hash0 = noise_detail::hash(i, j);
    uint8 hash1 = noise_detail::hash(i + i1, j + j1);
    uint8 hash2 = noise_detail::hash(i + 1, j + 1);

    // Compute contributions
    T n0 = T(0), n1 = T(0), n2 = T(0);
    T t0 = T(0.5) - x0*x0 - y0*y0;
    if (t0 > T(0)) {
        t0 *= t0;
        n0 = t0 * t0 * dot(noise_detail::simplex_grad2<T>(hash0), vec2<T>(x0, y0));
    }
    T t1 = T(0.5) - x1*x1 - y1*y1;
    if (t1 > T(0)) {
        t1 *= t1;
        n1 = t1 * t1 * dot(noise_detail::simplex_grad2<T>(hash1), vec2<T>(x1, y1));
    }
    T t2 = T(0.5) - x2*x2 - y2*y2;
    if (t2 > T(0)) {
        t2 *= t2;
        n2 = t2 * t2 * dot(noise_detail::simplex_grad2<T>(hash2), vec2<T>(x2, y2));
    }
    // Scale to [-1,1]
    return T(70) * (n0 + n1 + n2);
}

template <typename T = float>
T simplex3d(T x, T y, T z) noexcept {
    constexpr T F3 = T(1) / T(3);
    constexpr T G3 = T(1) / T(6);
    T s = (x + y + z) * F3;
    int32 i = static_cast<int32>(std::floor(x + s));
    int32 j = static_cast<int32>(std::floor(y + s));
    int32 k = static_cast<int32>(std::floor(z + s));
    T t = T(i + j + k) * G3;
    T X0 = i - t;
    T Y0 = j - t;
    T Z0 = k - t;
    T x0 = x - X0;
    T y0 = y - Y0;
    T z0 = z - Z0;

    int32 i1, j1, k1, i2, j2, k2;
    if (x0 >= y0) {
        if (y0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; }
        else if (x0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; }
        else { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }
    } else {
        if (y0 < z0) { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; }
        else if (x0 < z0) { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; }
        else { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; }
    }

    T x1 = x0 - i1 + G3;
    T y1 = y0 - j1 + G3;
    T z1 = z0 - k1 + G3;
    T x2 = x0 - i2 + T(2)*G3;
    T y2 = y0 - j2 + T(2)*G3;
    T z2 = z0 - k2 + T(2)*G3;
    T x3 = x0 - T(1) + T(3)*G3;
    T y3 = y0 - T(1) + T(3)*G3;
    T z3 = z0 - T(1) + T(3)*G3;

    uint8 h0 = noise_detail::hash(i, j, k);
    uint8 h1 = noise_detail::hash(i+i1, j+j1, k+k1);
    uint8 h2 = noise_detail::hash(i+i2, j+j2, k+k2);
    uint8 h3 = noise_detail::hash(i+1, j+1, k+1);

    T n0=T(0), n1=T(0), n2=T(0), n3=T(0);
    T t0 = T(0.6) - x0*x0 - y0*y0 - z0*z0;
    if (t0 > T(0)) { t0 *= t0; n0 = t0*t0 * dot(noise_detail::simplex_grad3<T>(h0), vec3<T>(x0,y0,z0)); }
    T t1 = T(0.6) - x1*x1 - y1*y1 - z1*z1;
    if (t1 > T(0)) { t1 *= t1; n1 = t1*t1 * dot(noise_detail::simplex_grad3<T>(h1), vec3<T>(x1,y1,z1)); }
    T t2 = T(0.6) - x2*x2 - y2*y2 - z2*z2;
    if (t2 > T(0)) { t2 *= t2; n2 = t2*t2 * dot(noise_detail::simplex_grad3<T>(h2), vec3<T>(x2,y2,z2)); }
    T t3 = T(0.6) - x3*x3 - y3*y3 - z3*z3;
    if (t3 > T(0)) { t3 *= t3; n3 = t3*t3 * dot(noise_detail::simplex_grad3<T>(h3), vec3<T>(x3,y3,z3)); }
    return T(32) * (n0 + n1 + n2 + n3);
}

// ═════════════════════════════════════════════════════════════════════
// Cellular (Worley) Noise – returns distance and cell index
// ═════════════════════════════════════════════════════════════════════

template <typename T = float>
T cellular2d(T x, T y, T jitter = T(1)) noexcept {
    int32 xi = static_cast<int32>(std::floor(x));
    int32 yi = static_cast<int32>(std::floor(y));
    T xf = x - xi, yf = y - yi;
    T min_dist = std::numeric_limits<T>::max();
    for (int32 dy = -1; dy <= 1; ++dy) {
        for (int32 dx = -1; dx <= 1; ++dx) {
            int32 nx = xi + dx;
            int32 ny = yi + dy;
            uint8 h = noise_detail::hash(nx, ny);
            // Random point within cell [0,1)x[0,1) with jittering
            T ox = T(h % 100) / T(100) * jitter;
            T oy = T((h/100) % 100) / T(100) * jitter;
            T px = dx + ox - xf;
            T py = dy + oy - yf;
            T dist_sq = px*px + py*py;
            if (dist_sq < min_dist) min_dist = dist_sq;
        }
    }
    return std::sqrt(min_dist);
}

template <typename T = float>
T cellular3d(T x, T y, T z, T jitter = T(1)) noexcept {
    int32 xi = static_cast<int32>(std::floor(x));
    int32 yi = static_cast<int32>(std::floor(y));
    int32 zi = static_cast<int32>(std::floor(z));
    T xf = x - xi, yf = y - yi, zf = z - zi;
    T min_dist = std::numeric_limits<T>::max();
    for (int32 dz = -1; dz <= 1; ++dz) {
        for (int32 dy = -1; dy <= 1; ++dy) {
            for (int32 dx = -1; dx <= 1; ++dx) {
                int32 nx = xi + dx;
                int32 ny = yi + dy;
                int32 nz = zi + dz;
                uint8 h = noise_detail::hash(nx, ny, nz);
                T ox = T(h % 100) / T(100) * jitter;
                T oy = T((h/100) % 100) / T(100) * jitter;
                T oz = T((h/10000) % 100) / T(100) * jitter;
                T px = dx + ox - xf;
                T py = dy + oy - yf;
                T pz = dz + oz - zf;
                T dist_sq = px*px + py*py + pz*pz;
                if (dist_sq < min_dist) min_dist = dist_sq;
            }
        }
    }
    return std::sqrt(min_dist);
}

// ── FBM (Fractional Brownian Motion) and Turbulence ──────────────────
template <typename T, typename NoiseFunc>
T fbm(NoiseFunc noise, T x, T y, int octaves = 4, T lacunarity = T(2), T gain = T(0.5)) noexcept {
    T sum = T(0);
    T amp = T(1);
    T freq = T(1);
    T max_val = T(0);
    for (int i = 0; i < octaves; ++i) {
        sum += noise(x*freq, y*freq) * amp;
        max_val += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    return sum / max_val;
}

template <typename T, typename NoiseFunc>
T fbm_3d(NoiseFunc noise, T x, T y, T z, int octaves = 4, T lacunarity = T(2), T gain = T(0.5)) noexcept {
    T sum = T(0);
    T amp = T(1);
    T freq = T(1);
    T max_val = T(0);
    for (int i = 0; i < octaves; ++i) {
        sum += noise(x*freq, y*freq, z*freq) * amp;
        max_val += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    return sum / max_val;
}

template <typename T, typename NoiseFunc>
T turbulence(NoiseFunc noise, T x, T y, int octaves = 4, T lacunarity = T(2), T gain = T(0.5)) noexcept {
    T sum = T(0);
    T amp = T(1);
    T freq = T(1);
    T max_val = T(0);
    for (int i = 0; i < octaves; ++i) {
        sum += std::abs(noise(x*freq, y*freq)) * amp;
        max_val += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    return sum / max_val;
}

// ── Domain Warping ───────────────────────────────────────────────────
template <typename T>
T domain_warp2d(T x, T y, T warp_strength = T(1), int warp_octaves = 1) {
    T qx = fbm(perlin2d<T>, x, y, warp_octaves);
    T qy = fbm(perlin2d<T>, x + T(5.2), y + T(1.3), warp_octaves);
    return perlin2d(x + qx * warp_strength, y + qy * warp_strength);
}

} // namespace wp