// system name : onetbb-warp
// File 0013 : core/math/noise.h
// Description : Coherent noise functions (Perlin, Simplex, Worley, Curl, fBm) for procedural generation.

#ifndef __TBB_WARP_CORE_MATH_NOISE_H
#define __TBB_WARP_CORE_MATH_NOISE_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/vector4.h"
#include "core/math/random.h"
#include <cmath>
#include <cstdint>
#include <array>
#include <algorithm>
#include <numeric>

namespace tbb {
namespace core {
namespace math {
namespace noise {

// ============================================================
// Permutation table (doubled for wrap‑around)
// ============================================================

static constexpr std::array<std::uint8_t, 512> build_permutation(std::uint64_t seed = 0) {
    std::array<std::uint8_t, 512> p{};
    std::array<std::uint8_t, 256> source{};
    for (int i = 0; i < 256; ++i) source[i] = static_cast<std::uint8_t>(i);
    xorshift64 rng(seed);
    for (int i = 255; i >= 0; --i) {
        int j = static_cast<int>(rng.next() % (i + 1));
        std::swap(source[i], source[j]);
    }
    for (int i = 0; i < 256; ++i) p[i] = p[i + 256] = source[i];
    return p;
}

inline const std::array<std::uint8_t, 512>& default_permutation() {
    static const auto perm = build_permutation(0x42);
    return perm;
}

// ============================================================
// Gradient vectors for Perlin noise
// ============================================================

static constexpr std::array<std::array<float, 2>, 8> grad2 = {{
    { 1.f,  1.f}, {-1.f,  1.f}, { 1.f, -1.f}, {-1.f, -1.f},
    { 1.f,  0.f}, {-1.f,  0.f}, { 0.f,  1.f}, { 0.f, -1.f}
}};

static constexpr std::array<std::array<float, 3>, 12> grad3 = {{
    { 1.f, 1.f, 0.f}, {-1.f, 1.f, 0.f}, { 1.f,-1.f, 0.f}, {-1.f,-1.f, 0.f},
    { 1.f, 0.f, 1.f}, {-1.f, 0.f, 1.f}, { 1.f, 0.f,-1.f}, {-1.f, 0.f,-1.f},
    { 0.f, 1.f, 1.f}, { 0.f,-1.f, 1.f}, { 0.f, 1.f,-1.f}, { 0.f,-1.f,-1.f}
}};

static constexpr std::array<std::array<float, 4>, 32> grad4 = {{
    { 0.f, 1.f, 1.f, 1.f}, { 0.f, 1.f, 1.f,-1.f}, { 0.f, 1.f,-1.f, 1.f}, { 0.f, 1.f,-1.f,-1.f},
    { 0.f,-1.f, 1.f, 1.f}, { 0.f,-1.f, 1.f,-1.f}, { 0.f,-1.f,-1.f, 1.f}, { 0.f,-1.f,-1.f,-1.f},
    { 1.f, 0.f, 1.f, 1.f}, { 1.f, 0.f, 1.f,-1.f}, { 1.f, 0.f,-1.f, 1.f}, { 1.f, 0.f,-1.f,-1.f},
    {-1.f, 0.f, 1.f, 1.f}, {-1.f, 0.f, 1.f,-1.f}, {-1.f, 0.f,-1.f, 1.f}, {-1.f, 0.f,-1.f,-1.f},
    { 1.f, 1.f, 0.f, 1.f}, { 1.f, 1.f, 0.f,-1.f}, { 1.f,-1.f, 0.f, 1.f}, { 1.f,-1.f, 0.f,-1.f},
    {-1.f, 1.f, 0.f, 1.f}, {-1.f, 1.f, 0.f,-1.f}, {-1.f,-1.f, 0.f, 1.f}, {-1.f,-1.f, 0.f,-1.f},
    { 1.f, 1.f, 1.f, 0.f}, { 1.f, 1.f,-1.f, 0.f}, { 1.f,-1.f, 1.f, 0.f}, { 1.f,-1.f,-1.f, 0.f},
    {-1.f, 1.f, 1.f, 0.f}, {-1.f, 1.f,-1.f, 0.f}, {-1.f,-1.f, 1.f, 0.f}, {-1.f,-1.f,-1.f, 0.f}
}};

// ============================================================
// Perlin noise 1D
// ============================================================

inline float perlin_1d(float x, const std::array<std::uint8_t, 512>& perm = default_permutation()) {
    int xi = static_cast<int>(std::floor(x)) & 255;
    float xf = x - std::floor(x);
    float u = smootherstep(0.0f, 1.0f, xf);
    int a = perm[xi];
    int b = perm[xi + 1];
    float ga = (a & 1) ? xf : -xf;
    float gb = (b & 1) ? (xf - 1.0f) : -(xf - 1.0f);
    return lerp(ga, gb, u);
}

// ============================================================
// Perlin noise 2D
// ============================================================

inline float perlin_2d(float x, float y, const std::array<std::uint8_t, 512>& perm = default_permutation()) {
    int xi = static_cast<int>(std::floor(x)) & 255;
    int yi = static_cast<int>(std::floor(y)) & 255;
    float xf = x - std::floor(x);
    float yf = y - std::floor(y);
    float u = smootherstep(0.0f, 1.0f, xf);
    float v = smootherstep(0.0f, 1.0f, yf);
    int aa = perm[perm[xi] + yi];
    int ab = perm[perm[xi] + yi + 1];
    int ba = perm[perm[xi + 1] + yi];
    int bb = perm[perm[xi + 1] + yi + 1];
    const auto& g = grad2;
    float x1 = lerp(dot(vector2<float>(g[aa % 8][0], g[aa % 8][1]), vector2<float>(xf, yf)),
                    dot(vector2<float>(g[ba % 8][0], g[ba % 8][1]), vector2<float>(xf - 1.0f, yf)), u);
    float x2 = lerp(dot(vector2<float>(g[ab % 8][0], g[ab % 8][1]), vector2<float>(xf, yf - 1.0f)),
                    dot(vector2<float>(g[bb % 8][0], g[bb % 8][1]), vector2<float>(xf - 1.0f, yf - 1.0f)), u);
    return lerp(x1, x2, v);
}

// ============================================================
// Perlin noise 3D
// ============================================================

inline float perlin_3d(float x, float y, float z, const std::array<std::uint8_t, 512>& perm = default_permutation()) {
    int xi = static_cast<int>(std::floor(x)) & 255;
    int yi = static_cast<int>(std::floor(y)) & 255;
    int zi = static_cast<int>(std::floor(z)) & 255;
    float xf = x - std::floor(x);
    float yf = y - std::floor(y);
    float zf = z - std::floor(z);
    float u = smootherstep(0.0f, 1.0f, xf);
    float v = smootherstep(0.0f, 1.0f, yf);
    float w = smootherstep(0.0f, 1.0f, zf);
    int aaa = perm[perm[perm[xi] + yi] + zi];
    int baa = perm[perm[perm[xi + 1] + yi] + zi];
    int aba = perm[perm[perm[xi] + yi + 1] + zi];
    int bba = perm[perm[perm[xi + 1] + yi + 1] + zi];
    int aab = perm[perm[perm[xi] + yi] + zi + 1];
    int bab = perm[perm[perm[xi + 1] + yi] + zi + 1];
    int abb = perm[perm[perm[xi] + yi + 1] + zi + 1];
    int bbb = perm[perm[perm[xi + 1] + yi + 1] + zi + 1];
    const auto& g = grad3;
    auto grad = [&](int h, float dx, float dy, float dz) { return g[h%12][0]*dx + g[h%12][1]*dy + g[h%12][2]*dz; };
    float x1 = lerp(lerp(grad(aaa, xf, yf, zf), grad(baa, xf-1.0f, yf, zf), u),
                    lerp(grad(aba, xf, yf-1.0f, zf), grad(bba, xf-1.0f, yf-1.0f, zf), u), v);
    float x2 = lerp(lerp(grad(aab, xf, yf, zf-1.0f), grad(bab, xf-1.0f, yf, zf-1.0f), u),
                    lerp(grad(abb, xf, yf-1.0f, zf-1.0f), grad(bbb, xf-1.0f, yf-1.0f, zf-1.0f), u), v);
    return lerp(x1, x2, w);
}

// ============================================================
// Simplex noise 2D (with pre‑computed skew factors)
// ============================================================

inline float simplex_2d(float x, float y) {
    const float F2 = 0.3660254037844386f;   // (sqrt(3)-1)/2
    const float G2 = 0.21132486540518713f;   // (3-sqrt(3))/6
    float s = (x + y) * F2;
    int i = static_cast<int>(std::floor(x + s));
    int j = static_cast<int>(std::floor(y + s));
    float t = (i + j) * G2;
    float X0 = i - t, Y0 = j - t;
    float x0 = x - X0, y0 = y - Y0;
    int i1, j1;
    if (x0 > y0) { i1 = 1; j1 = 0; }
    else         { i1 = 0; j1 = 1; }
    float x1 = x0 - i1 + G2, y1 = y0 - j1 + G2;
    float x2 = x0 - 1.0f + 2.0f * G2, y2 = y0 - 1.0f + 2.0f * G2;
    int ii = i & 255, jj = j & 255;
    const auto& perm = default_permutation();
    auto grad2f = [&](int h, float dx, float dy) {
        int hh = h & 7;
        return grad2[hh][0]*dx + grad2[hh][1]*dy;
    };
    float n0 = 0.0f, n1 = 0.0f, n2 = 0.0f;
    float t0 = 0.5f - x0*x0 - y0*y0;
    if (t0 > 0) { t0 *= t0; n0 = t0 * t0 * grad2f(perm[perm[ii] + jj], x0, y0); }
    float t1 = 0.5f - x1*x1 - y1*y1;
    if (t1 > 0) { t1 *= t1; n1 = t1 * t1 * grad2f(perm[perm[ii + i1] + jj + j1], x1, y1); }
    float t2 = 0.5f - x2*x2 - y2*y2;
    if (t2 > 0) { t2 *= t2; n2 = t2 * t2 * grad2f(perm[perm[ii + 1] + jj + 1], x2, y2); }
    return 70.0f * (n0 + n1 + n2);
}

// ============================================================
// Simplex noise 3D
// ============================================================

inline float simplex_3d(float x, float y, float z) {
    const float F3 = 1.0f / 3.0f;
    const float G3 = 1.0f / 6.0f;
    float s = (x + y + z) * F3;
    int i = static_cast<int>(std::floor(x + s));
    int j = static_cast<int>(std::floor(y + s));
    int k = static_cast<int>(std::floor(z + s));
    float t = (i + j + k) * G3;
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
    float x1 = x0 - i1 + G3, y1 = y0 - j1 + G3, z1 = z0 - k1 + G3;
    float x2 = x0 - i2 + 2.0f*G3, y2 = y0 - j2 + 2.0f*G3, z2 = z0 - k2 + 2.0f*G3;
    float x3 = x0 - 1.0f + 3.0f*G3, y3 = y0 - 1.0f + 3.0f*G3, z3 = z0 - 1.0f + 3.0f*G3;
    int ii = i & 255, jj = j & 255, kk = k & 255;
    const auto& perm = default_permutation();
    auto grad3f = [&](int h, float dx, float dy, float dz) {
        int hh = h & 15;
        return grad3[hh%12][0]*dx + grad3[hh%12][1]*dy + grad3[hh%12][2]*dz;
    };
    float n0=0,n1=0,n2=0,n3=0;
    float t0 = 0.6f - x0*x0 - y0*y0 - z0*z0;
    if (t0>0) { t0*=t0; n0 = t0*t0*grad3f(perm[perm[perm[ii]+jj]+kk], x0,y0,z0); }
    float t1 = 0.6f - x1*x1 - y1*y1 - z1*z1;
    if (t1>0) { t1*=t1; n1 = t1*t1*grad3f(perm[perm[perm[ii+i1]+jj+j1]+kk+k1], x1,y1,z1); }
    float t2 = 0.6f - x2*x2 - y2*y2 - z2*z2;
    if (t2>0) { t2*=t2; n2 = t2*t2*grad3f(perm[perm[perm[ii+i2]+jj+j2]+kk+k2], x2,y2,z2); }
    float t3 = 0.6f - x3*x3 - y3*y3 - z3*z3;
    if (t3>0) { t3*=t3; n3 = t3*t3*grad3f(perm[perm[perm[ii+1]+jj+1]+kk+1], x3,y3,z3); }
    return 32.0f * (n0 + n1 + n2 + n3);
}

// ============================================================
// Worley (Voronoi) noise – distance to nearest feature point
// ============================================================

inline std::pair<float, float> worley_2d(float x, float y, const std::array<std::uint8_t, 512>& perm = default_permutation()) {
    int xi = static_cast<int>(std::floor(x));
    int yi = static_cast<int>(std::floor(y));
    float fx = x - xi, fy = y - yi;
    float min_dist1 = std::numeric_limits<float>::max();
    float min_dist2 = std::numeric_limits<float>::max();
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = xi + dx, ny = yi + dy;
            std::uint32_t h = perm[perm[nx & 255] + (ny & 255)];
            float px = dx + (h & 0xFF) / 256.0f;
            float py = dy + ((h >> 8) & 0xFF) / 256.0f;
            float d = (fx - px)*(fx - px) + (fy - py)*(fy - py);
            if (d < min_dist1) { min_dist2 = min_dist1; min_dist1 = d; }
            else if (d < min_dist2) { min_dist2 = d; }
        }
    }
    return {std::sqrt(min_dist1), std::sqrt(min_dist2)};
}

inline float worley_2d_f1(float x, float y) { return worley_2d(x, y).first; }
inline float worley_2d_f2(float x, float y) { return worley_2d(x, y).second; }

inline std::pair<float, float> worley_3d(float x, float y, float z, const std::array<std::uint8_t, 512>& perm = default_permutation()) {
    int xi = static_cast<int>(std::floor(x)), yi = static_cast<int>(std::floor(y)), zi = static_cast<int>(std::floor(z));
    float fx = x - xi, fy = y - yi, fz = z - zi;
    float min_dist1 = std::numeric_limits<float>::max();
    float min_dist2 = std::numeric_limits<float>::max();
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = xi + dx, ny = yi + dy, nz = zi + dz;
                std::uint32_t h = perm[perm[perm[nx & 255] + (ny & 255)] + (nz & 255)];
                float px = dx + (h & 0xFF) / 256.0f;
                float py = dy + ((h >> 8) & 0xFF) / 256.0f;
                float pz = dz + ((h >> 16) & 0xFF) / 256.0f;
                float d = (fx - px)*(fx - px) + (fy - py)*(fy - py) + (fz - pz)*(fz - pz);
                if (d < min_dist1) { min_dist2 = min_dist1; min_dist1 = d; }
                else if (d < min_dist2) { min_dist2 = d; }
            }
    return {std::sqrt(min_dist1), std::sqrt(min_dist2)};
}

// ============================================================
// Curl noise (divergence‑free vector field) – 2D / 3D
// ============================================================

inline vector2<float> curl_noise_2d(float x, float y, float step = 0.1f) {
    float dx = (perlin_2d(x, y + step) - perlin_2d(x, y - step)) / (2.0f * step);
    float dy = -(perlin_2d(x + step, y) - perlin_2d(x - step, y)) / (2.0f * step);
    return vector2<float>(dx, dy);
}

inline vector3<float> curl_noise_3d(float x, float y, float z, float step = 0.1f) {
    float nx = perlin_3d(x, y + step, z) - perlin_3d(x, y - step, z);
    float ny = perlin_3d(x, y, z + step) - perlin_3d(x, y, z - step);
    float nz = perlin_3d(x + step, y, z) - perlin_3d(x - step, y, z);
    float nx2 = perlin_3d(x + step, y, z) - perlin_3d(x - step, y, z);
    float ny2 = perlin_3d(x, y + step, z) - perlin_3d(x, y - step, z);
    float nz2 = perlin_3d(x, y, z + step) - perlin_3d(x, y, z - step);
    float cx = (ny - nz2) / (2.0f * step);
    float cy = (nz - nx2) / (2.0f * step);
    float cz = (nx - ny2) / (2.0f * step);
    return vector3<float>(cx, cy, cz);
}

// ============================================================
// Fractional Brownian motion (fBm) – octaves of noise
// ============================================================

inline float fbm_perlin_2d(float x, float y, int octaves = 6, float lacunarity = 2.0f, float gain = 0.5f) {
    float value = 0.0f, amplitude = 1.0f, frequency = 1.0f, max_value = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        value += amplitude * perlin_2d(x * frequency, y * frequency);
        max_value += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    return value / max_value;
}

inline float fbm_simplex_2d(float x, float y, int octaves = 6, float lacunarity = 2.0f, float gain = 0.5f) {
    float value = 0.0f, amplitude = 1.0f, frequency = 1.0f, max_value = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        value += amplitude * simplex_2d(x * frequency, y * frequency);
        max_value += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    return value / max_value;
}

inline float fbm_perlin_3d(float x, float y, float z, int octaves = 6, float lacunarity = 2.0f, float gain = 0.5f) {
    float value = 0.0f, amplitude = 1.0f, frequency = 1.0f, max_value = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        value += amplitude * perlin_3d(x * frequency, y * frequency, z * frequency);
        max_value += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    return value / max_value;
}

// ============================================================
// Turbulence (absolute value of noise)
// ============================================================

inline float turbulence_2d(float x, float y, int octaves = 6, float lacunarity = 2.0f, float gain = 0.5f) {
    float value = 0.0f, amplitude = 1.0f, frequency = 1.0f, max_value = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        value += amplitude * std::abs(perlin_2d(x * frequency, y * frequency) * 2.0f - 1.0f);
        max_value += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    return value / max_value;
}

// ============================================================
// Domain warping – feed noise coordinates through noise
// ============================================================

inline float domain_warp_2d(float x, float y, float warp_strength = 1.0f, int octaves = 4) {
    float qx = fbm_perlin_2d(x, y, octaves, 2.0f, 0.5f);
    float qy = fbm_perlin_2d(x + 5.2f, y + 1.3f, octaves, 2.0f, 0.5f);
    float rx = fbm_perlin_2d(x + 4.0f * qx * warp_strength + 1.7f, y + 4.0f * qy * warp_strength + 9.2f, octaves, 2.0f, 0.5f);
    float ry = fbm_perlin_2d(x + 4.0f * qx * warp_strength + 8.3f, y + 4.0f * qy * warp_strength + 2.8f, octaves, 2.0f, 0.5f);
    return fbm_perlin_2d(x + 4.0f * rx * warp_strength, y + 4.0f * ry * warp_strength, octaves, 2.0f, 0.5f);
}

// ============================================================
// Periodic noise variants
// ============================================================

inline float perlin_2d_periodic(float x, float y, float px, float py) {
    float sx = std::sin(x * TAU_F / px) * (px / TAU_F);
    float sy = std::sin(y * TAU_F / py) * (py / TAU_F);
    float cx = std::cos(x * TAU_F / px) * (px / TAU_F);
    float cy = std::cos(y * TAU_F / py) * (py / TAU_F);
    return perlin_2d(sx + cx, sy + cy);
}

} // namespace noise
} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_NOISE_H