// File 0019 : core/math/random.h
// Pseudo‑random number generators (SplitMix64, Xoshiro256**, PCG32) and distributions (uniform, normal, sphere, disc, aabb).

#pragma once

#include "constants.h"
#include "vec2.h"
#include "vec3.h"
#include "aabb.h"
#include "sphere.h"
#include <cstdint>
#include <cmath>
#include <limits>
#include <type_traits>

namespace wp {

// ── SplitMix64 (fast, good for seeding) ───────────────────────────
class SplitMix64 {
public:
    using result_type = uint64;
    static constexpr result_type default_seed = 0x9E3779B97F4A7C15ULL;

    constexpr explicit SplitMix64(result_type s = default_seed) noexcept : m_state(s) {}

    result_type operator()() noexcept {
        result_type z = (m_state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    constexpr result_type min() const noexcept { return std::numeric_limits<result_type>::min(); }
    constexpr result_type max() const noexcept { return std::numeric_limits<result_type>::max(); }

private:
    result_type m_state;
};

// ── Xoshiro256** (fast, high quality) ────────────────────────────
class Xoshiro256ss {
public:
    using result_type = uint64;

    constexpr explicit Xoshiro256ss(result_type seed = 0) noexcept {
        SplitMix64 sm(seed);
        for (int i = 0; i < 4; ++i) s[i] = sm();
    }

    Xoshiro256ss(result_type s0, result_type s1, result_type s2, result_type s3) noexcept
        : s{s0, s1, s2, s3} {}

    result_type operator()() noexcept {
        const result_type result = rotl(s[1] * 5, 7) * 9;
        const result_type t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    constexpr result_type min() const noexcept { return 0; }
    constexpr result_type max() const noexcept { return ~result_type(0); }

    void jump() noexcept {
        static constexpr uint64 JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
                                           0xa9582618e03fc9aa, 0x39abdc4529b1661c };
        uint64 s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int i = 0; i < 4; ++i) {
            for (int b = 0; b < 64; b++) {
                if (JUMP[i] & (uint64(1) << b)) {
                    s0 ^= s[0]; s1 ^= s[1]; s2 ^= s[2]; s3 ^= s[3];
                }
                operator()();
            }
        }
        s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
    }

private:
    uint64 s[4];

    static constexpr uint64 rotl(uint64 x, int k) noexcept { return (x << k) | (x >> (64 - k)); }
};

// ── PCG32 (permuted congruential generator, 32‑bit output) ─────
class PCG32 {
public:
    using result_type = uint32;

    constexpr explicit PCG32(uint64 seed = 0x853c49e6748fea9bULL, uint64 stream_id = 0xda3e39cb94b95bdbULL) noexcept
        : state(0), inc((stream_id << 1) | 1) { operator()(); state += seed; operator()(); }

    result_type operator()() noexcept {
        uint64 old_state = state;
        state = old_state * 6364136223846793005ULL + inc;
        uint32 xorshifted = uint32(((old_state >> 18u) ^ old_state) >> 27u);
        int rot = old_state >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    constexpr result_type min() const noexcept { return 0; }
    constexpr result_type max() const noexcept { return ~result_type(0); }

private:
    uint64 state, inc;
};

// ── Distribution helpers ──────────────────────────────────────────
template <typename RNG, typename T = float>
T uniform_float(RNG& rng) noexcept {
    if constexpr (sizeof(typename RNG::result_type) >= 4) {
        // 32‑bit random to float in [0,1)
        constexpr T scale = T(1) / (T(1ULL << 32));
        return T(rng()) * scale;
    } else {
        // fallback
        return T(rng()) / T(rng.max());
    }
}

template <typename RNG, typename T = float>
T uniform_float(RNG& rng, T lo, T hi) noexcept {
    return lo + uniform_float<RNG,T>(rng) * (hi - lo);
}

template <typename RNG>
int32 uniform_int(RNG& rng, int32 lo, int32 hi) noexcept {
    return lo + int32(rng() % uint32(hi - lo + 1));
}

// Generate a random point inside a unit sphere (rejection method)
template <typename RNG, typename T = float>
vec3<T> random_in_unit_sphere(RNG& rng) noexcept {
    while (true) {
        vec3<T> p(uniform_float<RNG,T>(rng, T(-1), T(1)),
                  uniform_float<RNG,T>(rng, T(-1), T(1)),
                  uniform_float<RNG,T>(rng, T(-1), T(1)));
        if (length_sq(p) <= T(1)) return p;
    }
}

// Random point on unit sphere surface (Marsaglia's method)
template <typename RNG, typename T = float>
vec3<T> random_on_unit_sphere(RNG& rng) noexcept {
    while (true) {
        T x = uniform_float<RNG,T>(rng, T(-1), T(1));
        T y = uniform_float<RNG,T>(rng, T(-1), T(1));
        T s = x * x + y * y;
        if (s < T(1)) {
            T f = std::sqrt(T(1) - s);
            return vec3<T>(T(2) * x * f, T(2) * y * f, T(1) - T(2) * s);
        }
    }
}

// Random point inside a disc (2D)
template <typename RNG, typename T = float>
vec2<T> random_in_unit_disc(RNG& rng) noexcept {
    while (true) {
        vec2<T> p(uniform_float<RNG,T>(rng, T(-1), T(1)),
                  uniform_float<RNG,T>(rng, T(-1), T(1)));
        if (length_sq(p) <= T(1)) return p;
    }
}

// Random inside AABB
template <typename RNG, typename T = float>
vec3<T> random_in_aabb(RNG& rng, const aabb<T>& box) noexcept {
    return vec3<T>(uniform_float<RNG,T>(rng, box.min.x, box.max.x),
                   uniform_float<RNG,T>(rng, box.min.y, box.max.y),
                   uniform_float<RNG,T>(rng, box.min.z, box.max.z));
}

// Random inside sphere
template <typename RNG, typename T = float>
vec3<T> random_in_sphere(RNG& rng, const sphere<T>& s) noexcept {
    return s.center + random_in_unit_sphere<RNG,T>(rng) * s.radius;
}

// Box‑Muller normal distribution (mean 0, stddev 1)
template <typename RNG, typename T = float>
T normal_float(RNG& rng) noexcept {
    T u1, u2;
    do { u1 = uniform_float<RNG,T>(rng); } while (u1 <= T(0));
    u2 = uniform_float<RNG,T>(rng);
    return std::sqrt(T(-2) * std::log(u1)) * std::cos(two_pi<T> * u2);
}

template <typename RNG, typename T = float>
T normal_float(RNG& rng, T mean, T stddev) noexcept {
    return mean + normal_float<RNG,T>(rng) * stddev;
}

} // namespace wp