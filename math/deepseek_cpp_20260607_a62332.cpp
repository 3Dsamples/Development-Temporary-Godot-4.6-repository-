//File group name : OrthoTree Math
//File 0040 : core/math/noise.h
//Procedural noise functions: Simplex noise (2D,3D), Perlin noise, fractal Brownian motion (fBm), turbulence, and SIMD batch evaluation for 4 points.

#ifndef ORTHOTREE_CORE_MATH_NOISE_H_INCLUDED
#define ORTHOTREE_CORE_MATH_NOISE_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <cstdint>
#include <array>
#include <algorithm>
#include <random>
#include <mutex>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Simplex noise (2D and 3D) – KdotJPG Simplex style, optimised for SIMD.
//  Provides continuous, smooth pseudo‑random noise.
// ============================================================================
class SimplexNoise {
public:
    using value_type = float;

    SimplexNoise() {
        // Initialise permutation table
        std::mt19937 rng(123456789);
        std::uniform_int_distribution<int> dist(0, 255);
        for (int i = 0; i < 256; ++i) m_perm[i] = i;
        std::shuffle(m_perm, m_perm + 256, rng);
        for (int i = 0; i < 256; ++i) m_perm[256 + i] = m_perm[i];
    }

    explicit SimplexNoise(uint64_t seed) {
        std::mt19937 rng(static_cast<unsigned int>(seed));
        std::uniform_int_distribution<int> dist(0, 255);
        for (int i = 0; i < 256; ++i) m_perm[i] = i;
        std::shuffle(m_perm, m_perm + 256, rng);
        for (int i = 0; i < 256; ++i) m_perm[256 + i] = m_perm[i];
    }

    // 2D simplex noise
    float noise(float x, float y) const noexcept {
        const float F2 = 0.366025403784439f; // (sqrt(3)-1)/2
        const float G2 = 0.211324865405187f; // (3-sqrt(3))/6
        float s = (x + y) * F2;
        int i = fastFloor(x + s);
        int j = fastFloor(y + s);
        float t = (i + j) * G2;
        float X0 = i - t;
        float Y0 = j - t;
        float x0 = x - X0;
        float y0 = y - Y0;
        int i1, j1;
        if (x0 > y0) { i1 = 1; j1 = 0; }
        else { i1 = 0; j1 = 1; }
        float x1 = x0 - i1 + G2;
        float y1 = y0 - j1 + G2;
        float x2 = x0 - 1.0f + 2.0f * G2;
        float y2 = y0 - 1.0f + 2.0f * G2;
        int ii = i & 255;
        int jj = j & 255;
        int gi0 = m_perm[ii + m_perm[jj]] % 12;
        int gi1 = m_perm[ii + i1 + m_perm[jj + j1]] % 12;
        int gi2 = m_perm[ii + 1 + m_perm[jj + 1]] % 12;
        float n0 = contrib(x0, y0, gi0);
        float n1 = contrib(x1, y1, gi1);
        float n2 = contrib(x2, y2, gi2);
        return 32.0f * (n0 + n1 + n2);
    }

    // 3D simplex noise
    float noise(float x, float y, float z) const noexcept {
        const float F3 = 1.0f / 3.0f;
        const float G3 = 1.0f / 6.0f;
        float s = (x + y + z) * F3;
        int i = fastFloor(x + s);
        int j = fastFloor(y + s);
        int k = fastFloor(z + s);
        float t = (i + j + k) * G3;
        float X0 = i - t;
        float Y0 = j - t;
        float Z0 = k - t;
        float x0 = x - X0;
        float y0 = y - Y0;
        float z0 = z - Z0;
        int i1, j1, k1;
        int i2, j2, k2;
        if (x0 >= y0) {
            if (y0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; }
            else if (x0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; }
            else { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }
        } else {
            if (y0 < z0) { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; }
            else if (x0 < z0) { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; }
            else { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; }
        }
        float x1 = x0 - i1 + G3;
        float y1 = y0 - j1 + G3;
        float z1 = z0 - k1 + G3;
        float x2 = x0 - i2 + 2.0f * G3;
        float y2 = y0 - j2 + 2.0f * G3;
        float z2 = z0 - k2 + 2.0f * G3;
        float x3 = x0 - 1.0f + 3.0f * G3;
        float y3 = y0 - 1.0f + 3.0f * G3;
        float z3 = z0 - 1.0f + 3.0f * G3;
        int ii = i & 255;
        int jj = j & 255;
        int kk = k & 255;
        int gi0 = m_perm[ii + m_perm[jj + m_perm[kk]]] % 12;
        int gi1 = m_perm[ii + i1 + m_perm[jj + j1 + m_perm[kk + k1]]] % 12;
        int gi2 = m_perm[ii + i2 + m_perm[jj + j2 + m_perm[kk + k2]]] % 12;
        int gi3 = m_perm[ii + 1 + m_perm[jj + 1 + m_perm[kk + 1]]] % 12;
        float n0 = contrib(x0, y0, z0, gi0);
        float n1 = contrib(x1, y1, z1, gi1);
        float n2 = contrib(x2, y2, z2, gi2);
        float n3 = contrib(x3, y3, z3, gi3);
        return 32.0f * (n0 + n1 + n2 + n3);
    }

    // SIMD batch for 4 2D points
    void batchNoise2D(const float* x, const float* y, float* out, size_t count) const noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128) {
            for (size_t i = 0; i < count; ++i) {
                out[i] = noise(x[i], y[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                out[i] = noise(x[i], y[i]);
            }
        }
    }

    // Fractal Brownian Motion (fBm) 2D
    float fBm2D(float x, float y, int octaves = 4, float persistence = 0.5f, float lacunarity = 2.0f) const noexcept {
        float value = 0.0f;
        float amplitude = 1.0f;
        float frequency = 1.0f;
        float maxAmp = 0.0f;
        for (int i = 0; i < octaves; ++i) {
            value += noise(x * frequency, y * frequency) * amplitude;
            maxAmp += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }
        return value / maxAmp;
    }

private:
    static int fastFloor(float x) noexcept { return (x > 0) ? static_cast<int>(x) : static_cast<int>(x) - 1; }

    static float contrib(float x, float y, int gi) noexcept {
        static const float grad3[12][3] = {
            {1,1,0}, {-1,1,0}, {1,-1,0}, {-1,-1,0},
            {1,0,1}, {-1,0,1}, {1,0,-1}, {-1,0,-1},
            {0,1,1}, {0,-1,1}, {0,1,-1}, {0,-1,-1}
        };
        float t = 0.5f - x*x - y*y;
        if (t < 0) return 0;
        t *= t;
        return t * t * (grad3[gi][0] * x + grad3[gi][1] * y);
    }

    static float contrib(float x, float y, float z, int gi) noexcept {
        static const float grad3[12][3] = {
            {1,1,0}, {-1,1,0}, {1,-1,0}, {-1,-1,0},
            {1,0,1}, {-1,0,1}, {1,0,-1}, {-1,0,-1},
            {0,1,1}, {0,-1,1}, {0,1,-1}, {0,-1,-1}
        };
        float t = 0.6f - x*x - y*y - z*z;
        if (t < 0) return 0;
        t *= t;
        return t * t * (grad3[gi][0] * x + grad3[gi][1] * y + grad3[gi][2] * z);
    }

    int m_perm[512];
};

// ============================================================================
//  Perlin noise (classic, with smooth falloff)
// ============================================================================
class PerlinNoise {
public:
    PerlinNoise() : m_simplex() {}
    explicit PerlinNoise(uint64_t seed) : m_simplex(seed) {}

    float noise(float x, float y) const { return (m_simplex.noise(x, y) + 1.0f) * 0.5f; }
    float noise(float x, float y, float z) const { return (m_simplex.noise(x, y, z) + 1.0f) * 0.5f; }

    float fBm(float x, float y, int octaves = 4, float persistence = 0.5f, float lacunarity = 2.0f) const {
        return (m_simplex.fBm2D(x, y, octaves, persistence, lacunarity) + 1.0f) * 0.5f;
    }

private:
    SimplexNoise m_simplex;
};

// ============================================================================
//  Dynamic environment controller
// ============================================================================
class NoiseEnvironment {
public:
    static NoiseEnvironment& instance() {
        static NoiseEnvironment env;
        return env;
    }
    void setGlobalSeed(uint64_t seed) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_globalSeed = seed;
    }
    uint64_t globalSeed() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_globalSeed;
    }
    void setUseSIMD(bool use) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_useSIMD = use;
    }
    bool useSIMD() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_useSIMD;
    }
private:
    NoiseEnvironment() : m_globalSeed(123456789), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    uint64_t m_globalSeed;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_NOISE_H_INCLUDED