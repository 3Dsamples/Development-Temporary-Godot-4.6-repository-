// system name : onetbb-warp
// File 0012 : core/math/random.h
// Description : High‑performance random number generators and distribution functions for simulation.

#ifndef __TBB_WARP_CORE_MATH_RANDOM_H
#define __TBB_WARP_CORE_MATH_RANDOM_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/vector4.h"
#include <cstdint>
#include <cmath>
#include <random>
#include <limits>
#include <array>
#include <algorithm>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Xorshift64‑state
// ============================================================

struct xorshift64 {
    std::uint64_t state;

    constexpr xorshift64(std::uint64_t seed = 0x123456789ABCDEF0ULL) noexcept : state(seed) {}

    inline std::uint64_t next() noexcept {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }

    inline std::uint32_t next32() noexcept {
        return static_cast<std::uint32_t>(next() & 0xFFFFFFFFULL);
    }

    inline void jump() noexcept {
        std::uint64_t jump_const[2] = { 0x8a5cd789635d2dffULL, 0x121fd2155c472f96ULL };
        std::uint64_t s0 = 0, s1 = 0;
        for (int i = 0; i < 2; ++i) {
            for (int b = 0; b < 64; ++b) {
                if (jump_const[i] & (1ULL << b)) {
                    s0 ^= state;
                    s1 ^= state >> 32;
                }
                next();
            }
        }
        state = s0 | (s1 << 32);
    }
};

// ============================================================
// Xorshift128+ (for SIMD‑friendly 64‑bit generation)
// ============================================================

struct xorshift128plus {
    alignas(16) std::array<std::uint64_t, 2> state;

    constexpr xorshift128plus(std::uint64_t seed = 0xABCDEF0123456789ULL) noexcept
        : state{seed, seed * 0x9E3779B97F4A7C15ULL} {}

    inline std::uint64_t next() noexcept {
        std::uint64_t s1 = state[0];
        std::uint64_t s0 = state[1];
        state[0] = s0;
        s1 ^= s1 << 23;
        state[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
        return state[1] + s0;
    }
};

// ============================================================
// PCG32 (Permuted Congruential Generator, fast)
// ============================================================

struct pcg32 {
    std::uint64_t state;
    std::uint64_t inc;

    constexpr pcg32(std::uint64_t seed = 0, std::uint64_t seq = 1) noexcept
        : state(0), inc((seq << 1) | 1) {
        next();
        state += seed;
        next();
    }

    inline std::uint32_t next() noexcept {
        std::uint64_t oldstate = state;
        state = oldstate * 6364136223846793005ULL + inc;
        std::uint32_t xorshifted = static_cast<std::uint32_t>(((oldstate >> 18) ^ oldstate) >> 27);
        std::uint32_t rot = static_cast<std::uint32_t>(oldstate >> 59);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }
};

// ============================================================
// Mersenne Twister MT19937 (reference)
// ============================================================

struct mt19937 {
    static constexpr std::size_t N = 624;
    static constexpr std::size_t M = 397;
    std::array<std::uint32_t, N> mt;
    std::size_t mti = N + 1;

    explicit mt19937(std::uint32_t seed = 5489) noexcept {
        mt[0] = seed;
        for (mti = 1; mti < N; ++mti)
            mt[mti] = 1812433253 * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + static_cast<std::uint32_t>(mti);
    }

    std::uint32_t next() noexcept {
        if (mti >= N) {
            for (std::size_t kk = 0; kk < N - M; ++kk) {
                std::uint32_t y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7FFFFFFF);
                mt[kk] = mt[kk + M] ^ (y >> 1) ^ ((y & 1) ? 0x9908B0DF : 0);
            }
            for (std::size_t kk = N - M; kk < N - 1; ++kk) {
                std::uint32_t y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7FFFFFFF);
                mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ ((y & 1) ? 0x9908B0DF : 0);
            }
            std::uint32_t y = (mt[N - 1] & 0x80000000) | (mt[0] & 0x7FFFFFFF);
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ ((y & 1) ? 0x9908B0DF : 0);
            mti = 0;
        }
        std::uint32_t y = mt[mti++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9D2C5680;
        y ^= (y << 15) & 0xEFC60000;
        y ^= (y >> 18);
        return y;
    }
};

// ============================================================
// Uniform distributions
// ============================================================

template<typename RNG>
float uniform_float(RNG& rng) noexcept {
    return static_cast<float>(rng.next32() * 2.3283064365386963e-10f); // 1 / (2^32-1)
}

template<typename RNG>
double uniform_double(RNG& rng) noexcept {
    return static_cast<double>(rng.next() * 5.421010862427522e-20); // 1 / (2^64-1)
}

template<typename RNG>
float uniform_range(RNG& rng, float a, float b) noexcept {
    return a + uniform_float(rng) * (b - a);
}

template<typename RNG>
double uniform_range_double(RNG& rng, double a, double b) noexcept {
    return a + uniform_double(rng) * (b - a);
}

template<typename RNG>
std::uint32_t uniform_uint(RNG& rng, std::uint32_t min_val, std::uint32_t max_val) noexcept {
    return min_val + (rng.next32() % (max_val - min_val + 1));
}

template<typename RNG>
int uniform_int(RNG& rng, int min_val, int max_val) noexcept {
    return min_val + static_cast<int>(rng.next32() % (max_val - min_val + 1));
}

// ============================================================
// Normal (Gaussian) distribution (Box‑Muller)
// ============================================================

template<typename RNG>
std::pair<float, float> box_muller(RNG& rng) noexcept {
    float u1 = uniform_float(rng);
    float u2 = uniform_float(rng);
    float r = std::sqrt(-2.0f * std::log(std::max(u1, 1e-12f)));
    float theta = TAU_F * u2;
    return {r * std::cos(theta), r * std::sin(theta)};
}

template<typename RNG>
float normal_float(RNG& rng, float mean = 0.0f, float stddev = 1.0f) noexcept {
    float z0, z1;
    std::tie(z0, z1) = box_muller(rng);
    return mean + z0 * stddev;
}

template<typename RNG>
double normal_double(RNG& rng, double mean = 0.0, double stddev = 1.0) noexcept {
    float u1 = uniform_float(rng);
    float u2 = uniform_float(rng);
    double r = std::sqrt(-2.0 * std::log(std::max((double)u1, 1e-12)));
    double theta = TAU_D * u2;
    return mean + r * std::cos(theta) * stddev;
}

// ============================================================
// Log‑normal distribution
// ============================================================

template<typename RNG>
float log_normal(RNG& rng, float mean = 0.0f, float sigma = 1.0f) noexcept {
    float z = normal_float(rng, 0.0f, 1.0f);
    return std::exp(mean + sigma * z);
}

// ============================================================
// Exponential distribution (inverse CDF)
// ============================================================

template<typename RNG>
float exponential(RNG& rng, float lambda = 1.0f) noexcept {
    float u = uniform_float(rng);
    return -std::log(std::max(u, 1e-12f)) / lambda;
}

// ============================================================
// Poisson distribution
// ============================================================

template<typename RNG>
int poisson(RNG& rng, float lambda) noexcept {
    if (lambda < 30.0f) {
        float L = std::exp(-lambda);
        int k = 0;
        float p = 1.0f;
        do {
            ++k;
            p *= uniform_float(rng);
        } while (p > L);
        return k - 1;
    }
    float c = 0.767f - 3.36f / lambda;
    float beta = PI_F / std::sqrt(3.0f * lambda);
    float alpha = beta * lambda;
    float k = std::log(c) - lambda - std::log(beta);
    while (true) {
        float u = uniform_float(rng);
        float x = (alpha - std::log((1.0f - u) / u)) / beta;
        int n = static_cast<int>(std::floor(x + 0.5f));
        if (n < 0) continue;
        float v = uniform_float(rng);
        float y = alpha - beta * x;
        float lhs = y + std::log(v / ((1.0f + std::exp(y)) * (1.0f + std::exp(y))));
        float rhs = k + n * std::log(lambda) - lgamma(static_cast<float>(n + 1));
        if (lhs <= rhs) return n;
    }
}

// ============================================================
// Gamma distribution (Marsaglia‑Tsang)
// ============================================================

template<typename RNG>
float gamma(RNG& rng, float shape, float scale = 1.0f) noexcept {
    if (shape < 1.0f) {
        float u = uniform_float(rng);
        return gamma(rng, shape + 1.0f, scale) * std::pow(u, 1.0f / shape);
    }
    float d = shape - 1.0f / 3.0f;
    float c = 1.0f / std::sqrt(9.0f * d);
    while (true) {
        float x, v;
        do {
            x = normal_float(rng, 0.0f, 1.0f);
            v = 1.0f + c * x;
        } while (v <= 0.0f);
        v = v * v * v;
        float u = uniform_float(rng);
        if (u < 1.0f - 0.0331f * (x*x) * (x*x)) return d * v * scale;
        if (std::log(u) < 0.5f * x * x + d * (1.0f - v + std::log(v))) return d * v * scale;
    }
}

// ============================================================
// Beta distribution
// ============================================================

template<typename RNG>
float beta(RNG& rng, float a, float b) noexcept {
    float x = gamma(rng, a, 1.0f);
    float y = gamma(rng, b, 1.0f);
    return x / (x + y);
}

// ============================================================
// Binomial distribution
// ============================================================

template<typename RNG>
int binomial(RNG& rng, int n, float p) noexcept {
    if (p < 0.5f) return n - binomial(rng, n, 1.0f - p);
    float lambda = n * p;
    if (n > 100 && lambda < 10.0f) return std::min(n, poisson(rng, lambda));
    int k = 0;
    for (int i = 0; i < n; ++i) if (uniform_float(rng) < p) ++k;
    return k;
}

// ============================================================
// Random direction in 2D / 3D / sphere / hemisphere / cone
// ============================================================

template<typename RNG>
vector2<float> random_unit_circle(RNG& rng) noexcept {
    float angle = uniform_range(rng, 0.0f, TAU_F);
    return {std::cos(angle), std::sin(angle)};
}

template<typename RNG>
vector3<float> random_unit_sphere(RNG& rng) noexcept {
    float z = uniform_range(rng, -1.0f, 1.0f);
    float r = std::sqrt(1.0f - z * z);
    float angle = uniform_range(rng, 0.0f, TAU_F);
    return {r * std::cos(angle), r * std::sin(angle), z};
}

template<typename RNG>
vector3<float> random_unit_hemisphere(RNG& rng, const vector3<float>& normal) noexcept {
    vector3<float> dir = random_unit_sphere(rng);
    if (dot(dir, normal) < 0.0f) dir = -dir;
    return dir;
}

template<typename RNG>
vector3<float> random_cone(RNG& rng, const vector3<float>& axis, float half_angle) noexcept {
    float cos_theta = uniform_range(rng, std::cos(half_angle), 1.0f);
    float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);
    float phi = uniform_range(rng, 0.0f, TAU_F);
    vector3<float> local(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta);
    vector3<float> up{0.0f, 0.0f, 1.0f};
    if (std::abs(dot(axis, up)) > 0.999f) up = {1.0f, 0.0f, 0.0f};
    vector3<float> t1 = normalize(cross(up, axis));
    vector3<float> t2 = cross(axis, t1);
    return t1 * local.x + t2 * local.y + axis * local.z;
}

// ============================================================
// Shuffle and sample
// ============================================================

template<typename RNG, typename Iterator>
void shuffle(RNG& rng, Iterator first, Iterator last) noexcept {
    auto n = std::distance(first, last);
    for (decltype(n) i = n - 1; i > 0; --i) {
        auto j = uniform_uint(rng, 0, static_cast<std::uint32_t>(i));
        std::swap(*(first + i), *(first + j));
    }
}

template<typename RNG, typename Iterator>
Iterator sample(RNG& rng, Iterator first, Iterator last, std::size_t k, Iterator out) noexcept {
    std::size_t n = std::distance(first, last);
    if (k > n) k = n;
    for (std::size_t i = 0; i < k; ++i, ++out) {
        auto j = uniform_uint(rng, 0, static_cast<std::uint32_t>(n - i - 1));
        *out = first[j];
        std::swap(first[j], first[n - i - 1]);
    }
    return out;
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_RANDOM_H