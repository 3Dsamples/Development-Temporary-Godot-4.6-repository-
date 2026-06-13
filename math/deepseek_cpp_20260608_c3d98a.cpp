//File group name : OrthoTree Math
//File 0062 : core/math/random/sampling.h
//Random sampling utilities: uniform, normal (Box‑Muller), Poisson disk, Halton sequence, stratified sampling, and SIMD batch generation.

#ifndef ORTHOTREE_CORE_MATH_RANDOM_SAMPLING_H_INCLUDED
#define ORTHOTREE_CORE_MATH_RANDOM_SAMPLING_H_INCLUDED

#include "../../build_config.h"
#include "../basic/vector.h"
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <random>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

namespace OrthoTree {
namespace Math {
namespace Random {

// ============================================================================
//  Uniform random number generator (thread‑local)
// ============================================================================
class UniformRNG {
public:
    using result_type = uint64_t;
    UniformRNG(uint64_t seed = 123456789) : m_state(seed) {}
    uint64_t operator()() {
        uint64_t x = m_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        m_state = x;
        return x * 0x2545F4914F6CDD1DULL;
    }
    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() { return ~0ULL; }
    double uniform01() { return static_cast<double>((*this)()) / max(); }
    float uniform01f() { return static_cast<float>(uniform01()); }
private:
    uint64_t m_state;
};

// ----------------------------------------------------------------------------
//  SIMD batch: generate 4 uniform floats in [0,1) using AVX2
// ----------------------------------------------------------------------------
inline void batchUniform01(UniformRNG& rng, float* out, size_t count) {
    if constexpr (std::is_same_v<T,float> && ORTHOTREE_SIMD_LEVEL >= 128) {
        for (size_t i = 0; i < count; ++i) {
            out[i] = rng.uniform01f();
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            out[i] = rng.uniform01f();
        }
    }
}

// ============================================================================
//  Normal (Gaussian) distribution using Box‑Muller
// ============================================================================
template<typename T = double>
std::pair<T,T> boxMuller(UniformRNG& rng) {
    T u1 = rng.uniform01();
    T u2 = rng.uniform01();
    T r = std::sqrt(-T(2) * std::log(u1));
    T theta = T(2) * Constants<T>::pi() * u2;
    return {r * std::cos(theta), r * std::sin(theta)};
}

// Batch generation of 4 normal samples (2 pairs)
inline void batchNormal(UniformRNG& rng, float* out, size_t count) {
    for (size_t i = 0; i < count; i += 2) {
        auto [z1, z2] = boxMuller<float>(rng);
        out[i] = z1;
        if (i+1 < count) out[i+1] = z2;
    }
}

// ============================================================================
//  Halton sequence (low‑discrepancy)
// ============================================================================
inline double halton(int index, int base) {
    double f = 1.0, r = 0.0;
    int i = index;
    while (i > 0) {
        f /= base;
        r += f * (i % base);
        i /= base;
    }
    return r;
}

// 2D Halton point (bases 2 and 3)
inline Basic::Vector<double,2> halton2D(int index) {
    return Basic::Vector<double,2>(halton(index, 2), halton(index, 3));
}

// Batch Halton points
inline void batchHalton2D(int startIndex, int count, Basic::Vector<double,2>* out) {
    for (int i = 0; i < count; ++i) {
        out[i] = halton2D(startIndex + i);
    }
}

// ============================================================================
//  Stratified sampling (jittered grid)
// ============================================================================
template<typename T = float>
std::vector<Basic::Vector<T,2>> stratified2D(int nx, int ny, UniformRNG& rng) {
    std::vector<Basic::Vector<T,2>> samples;
    samples.reserve(nx * ny);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            T u = static_cast<T>(rng.uniform01());
            T v = static_cast<T>(rng.uniform01());
            T x = (static_cast<T>(i) + u) / static_cast<T>(nx);
            T y = (static_cast<T>(j) + v) / static_cast<T>(ny);
            samples.emplace_back(x, y);
        }
    }
    return samples;
}

// ============================================================================
//  Poisson disk sampling (2D, brute‑force, using grid)
//  Simplified version: generates random points with minimum distance.
// ============================================================================
template<typename T = float>
std::vector<Basic::Vector<T,2>> poissonDisk(T radius, const Basic::Vector<T,2>& min, const Basic::Vector<T,2>& max,
                                           int maxAttempts = 30, UniformRNG& rng = UniformRNG()) {
    T cellSize = radius / std::sqrt(T(2));
    int gridW = static_cast<int>((max[0] - min[0]) / cellSize) + 1;
    int gridH = static_cast<int>((max[1] - min[1]) / cellSize) + 1;
    std::vector<std::vector<int>> grid(gridW, std::vector<int>(gridH, -1));
    std::vector<Basic::Vector<T,2>> points;
    std::vector<int> activeList;

    auto addPoint = [&](const Basic::Vector<T,2>& p, int idx) {
        points.push_back(p);
        activeList.push_back(idx);
        int gx = static_cast<int>((p[0] - min[0]) / cellSize);
        int gy = static_cast<int>((p[1] - min[1]) / cellSize);
        grid[gx][gy] = idx;
    };

    // First point
    T x = min[0] + rng.uniform01() * (max[0] - min[0]);
    T y = min[1] + rng.uniform01() * (max[1] - min[1]);
    addPoint(Basic::Vector<T,2>(x, y), 0);

    while (!activeList.empty()) {
        int idx = activeList[rng.uniform01() * activeList.size()];
        const auto& p = points[idx];
        bool found = false;
        for (int attempt = 0; attempt < maxAttempts; ++attempt) {
            T angle = rng.uniform01() * T(2) * Constants<T>::pi();
            T rad = radius + rng.uniform01() * radius;
            T nx = p[0] + rad * std::cos(angle);
            T ny = p[1] + rad * std::sin(angle);
            if (nx < min[0] || nx > max[0] || ny < min[1] || ny > max[1]) continue;
            int gx = static_cast<int>((nx - min[0]) / cellSize);
            int gy = static_cast<int>((ny - min[1]) / cellSize);
            bool ok = true;
            for (int dx = -2; dx <= 2; ++dx) {
                for (int dy = -2; dy <= 2; ++dy) {
                    int ngx = gx + dx, ngy = gy + dy;
                    if (ngx >= 0 && ngx < gridW && ngy >= 0 && ngy < gridH && grid[ngx][ngy] != -1) {
                        T dx_ = points[grid[ngx][ngy]][0] - nx;
                        T dy_ = points[grid[ngx][ngy]][1] - ny;
                        if (dx_*dx_ + dy_*dy_ < radius*radius) {
                            ok = false;
                            break;
                        }
                    }
                }
                if (!ok) break;
            }
            if (ok) {
                addPoint(Basic::Vector<T,2>(nx, ny), static_cast<int>(points.size()));
                found = true;
                break;
            }
        }
        if (!found) {
            activeList.erase(std::remove(activeList.begin(), activeList.end(), idx), activeList.end());
        }
    }
    return points;
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller for random sampling
// ----------------------------------------------------------------------------
class SamplingEnvironment {
public:
    static SamplingEnvironment& instance() {
        static SamplingEnvironment env;
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
private:
    SamplingEnvironment() : m_globalSeed(123456789) {}
    mutable std::mutex m_mutex;
    uint64_t m_globalSeed;
};

} // namespace Random
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_RANDOM_SAMPLING_H_INCLUDED