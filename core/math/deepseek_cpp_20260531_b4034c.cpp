//File 0032 : core/math/random.h
//High‑performance pseudo‑random generators (PCG32/PCG64) with uniform, normal (Box‑Muller), exponential, and spatial (sphere/hemisphere/disk) distributions, plus 1D/2D/3D hash functions for procedural noise.
#ifndef CORE_MATH_RANDOM_H
#define CORE_MATH_RANDOM_H

#include "vector_math.h"
#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>
#include <array>
#include <utility>

namespace SimulationMath {
namespace random {

// -----------------------------------------------------------------------------
// 1. PCG32 generator (minimal state, 32‑bit output)
// -----------------------------------------------------------------------------
class PCG32 {
    uint64_t state_;
    static constexpr uint64_t multiplier = 6364136223846793005ULL;
    static constexpr uint64_t increment  = 1442695040888963407ULL;

public:
    explicit PCG32(uint64_t seed = 0x4F6CDD1F) noexcept : state_(0) {
        seed(seed);
    }

    void seed(uint64_t s) noexcept {
        state_ = 0;
        next();
        state_ += s;
        next();
    }

    uint32_t next_u32() noexcept {
        uint64_t old = state_;
        state_ = old * multiplier + increment;
        uint32_t xorshifted = static_cast<uint32_t>(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = old >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
    }

    // Uniform float in [0, 1)
    float next_float() noexcept {
        return static_cast<float>(next_u32()) * 0x1p-32f;
    }

    // Uniform float in [a, b)
    float next_float_range(float a, float b) noexcept {
        return a + next_float() * (b - a);
    }

    // Uniform int in [0, n-1]
    uint32_t next_int(uint32_t n) noexcept {
        uint32_t threshold = (~n + 1u) % n;
        for (;;) {
            uint32_t r = next_u32();
            if (r >= threshold)
                return r % n;
        }
    }
};

// -----------------------------------------------------------------------------
// 2. PCG64 generator (64‑bit output, higher precision)
// -----------------------------------------------------------------------------
class PCG64 {
    __uint128_t state_;
    static constexpr __uint128_t multiplier = (static_cast<__uint128_t>(6364136223846793005ULL) << 64)
                                             | static_cast<__uint128_t>(1442695040888963407ULL);
    static constexpr __uint128_t increment  = (static_cast<__uint128_t>(1442695040888963407ULL) << 64)
                                             | static_cast<__uint128_t>(1442695040888963407ULL);
public:
    explicit PCG64(uint64_t seed = 0xCAFEF00DD15EA5E5ULL) noexcept : state_(0) {
        seed(seed);
    }

    void seed(uint64_t s) noexcept {
        state_ = 0;
        next_u64();
        state_ += s;
        next_u64();
    }

    uint64_t next_u64() noexcept {
        __uint128_t old = state_;
        state_ = old * multiplier + increment;
        uint64_t xorshifted = static_cast<uint64_t>((old >> 64u) ^ (old >> 32u)) >> 32u;
        return xorshifted;
    }

    double next_double() noexcept {
        return static_cast<double>(next_u64()) * 0x1p-64;
    }

    float next_float() noexcept {
        return static_cast<float>(next_u64() >> 40) * 0x1p-24f;
    }
};

// -----------------------------------------------------------------------------
// 3. Uniform distribution helpers (using a generator)
// -----------------------------------------------------------------------------
template <typename Gen>
inline float uniform_float(Gen& gen) noexcept { return gen.next_float(); }

template <typename Gen>
inline double uniform_double(Gen& gen) noexcept { return gen.next_double(); }

// -----------------------------------------------------------------------------
// 4. Normal distribution (Box‑Muller)
// -----------------------------------------------------------------------------
template <typename Gen>
inline float gaussian(Gen& gen, float mean = 0.0f, float sigma = 1.0f) noexcept {
    // Generate two uniforms and apply Box‑Muller transform
    float u1, u2;
    do {
        u1 = gen.next_float();
    } while (u1 < 1e-12f);
    u2 = gen.next_float();
    float mag = sigma * std::sqrt(-2.0f * std::log(u1));
    float z0 = mag * std::cos(2.0f * 3.14159265358979f * u2);
    // Discard z1 = mag * sin(2*pi*u2) to save computation per call.
    return mean + z0;
}

// Pair of independent normals (returns both from same two uniforms)
template <typename Gen>
inline std::pair<float, float> gaussian_pair(Gen& gen, float mean = 0.0f, float sigma = 1.0f) noexcept {
    float u1, u2;
    do { u1 = gen.next_float(); } while (u1 < 1e-12f);
    u2 = gen.next_float();
    float mag = sigma * std::sqrt(-2.0f * std::log(u1));
    float z0 = mag * std::cos(2.0f * 3.14159265358979f * u2);
    float z1 = mag * std::sin(2.0f * 3.14159265358979f * u2);
    return {mean + z0, mean + z1};
}

// -----------------------------------------------------------------------------
// 5. Exponential distribution
// -----------------------------------------------------------------------------
template <typename Gen>
inline float exponential(Gen& gen, float lambda = 1.0f) noexcept {
    float u;
    do { u = gen.next_float(); } while (u < 1e-12f);
    return -std::log(u) / lambda;
}

// -----------------------------------------------------------------------------
// 6. Uniform sampling on unit sphere (3D)
// -----------------------------------------------------------------------------
template <typename Gen>
inline DirectX::XMVECTOR uniform_sphere(Gen& gen) noexcept {
    float u1 = gen.next_float();
    float u2 = gen.next_float();
    float z = 2.0f * u1 - 1.0f;
    float t = 2.0f * 3.14159265358979f * u2;
    float r = std::sqrt(std::max(1.0f - z * z, 0.0f));
    return DirectX::XMVectorSet(r * std::cos(t), r * std::sin(t), z, 0.0f);
}

// -----------------------------------------------------------------------------
// 7. Uniform sampling on hemisphere (cosine‑weighted)
// -----------------------------------------------------------------------------
template <typename Gen>
inline DirectX::XMVECTOR cosine_hemisphere(Gen& gen) noexcept {
    float u1 = gen.next_float();
    float u2 = gen.next_float();
    float r = std::sqrt(u1);
    float theta = 2.0f * 3.14159265358979f * u2;
    float x = r * std::cos(theta);
    float y = r * std::sin(theta);
    float z = std::sqrt(std::max(1.0f - u1, 0.0f));
    return DirectX::XMVectorSet(x, y, z, 0.0f);
}

// -----------------------------------------------------------------------------
// 8. Uniform sampling on disk (2D)
// -----------------------------------------------------------------------------
template <typename Gen>
inline std::pair<float, float> uniform_disk(Gen& gen) noexcept {
    float u1 = gen.next_float();
    float u2 = gen.next_float();
    float r = std::sqrt(u1);
    float theta = 2.0f * 3.14159265358979f * u2;
    return {r * std::cos(theta), r * std::sin(theta)};
}

// -----------------------------------------------------------------------------
// 9. Hash functions (good for seeding / procedural noise)
// -----------------------------------------------------------------------------
inline uint32_t hash(uint32_t x) noexcept {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

inline uint32_t hash(uint32_t x, uint32_t y) noexcept {
    return hash(x + hash(y) * 0x9e3779b9);
}

inline uint32_t hash(uint32_t x, uint32_t y, uint32_t z) noexcept {
    return hash(x + hash(y + hash(z) * 0x9e3779b9));
}

// 1D value noise (float in [0,1] using integer coordinate)
inline float value_noise(int32_t ix) noexcept {
    return static_cast<float>(hash(static_cast<uint32_t>(ix))) * 0x1p-32f;
}

inline float value_noise(int32_t ix, int32_t iy) noexcept {
    return static_cast<float>(hash(static_cast<uint32_t>(ix), static_cast<uint32_t>(iy))) * 0x1p-32f;
}

inline float value_noise(int32_t ix, int32_t iy, int32_t iz) noexcept {
    return static_cast<float>(hash(static_cast<uint32_t>(ix), static_cast<uint32_t>(iy), static_cast<uint32_t>(iz))) * 0x1p-32f;
}

// -----------------------------------------------------------------------------
// 10. SIMD batch random (4 floats at once) using scalar PCG and packing
// -----------------------------------------------------------------------------
template <typename Gen>
inline DirectX::XMVECTOR uniform_float4(Gen& gen) noexcept {
    float f[4];
    for (int i = 0; i < 4; ++i) f[i] = gen.next_float();
    return DirectX::XMVectorLoadFloat4(reinterpret_cast<const DirectX::XMFLOAT4*>(f));
}

} // namespace random
} // namespace SimulationMath

#endif // CORE_MATH_RANDOM_H