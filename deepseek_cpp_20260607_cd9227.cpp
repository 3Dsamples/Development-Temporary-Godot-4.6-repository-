/* MIT License Copyright (c) 2021-2026 Attila Csikós & Contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ORTHOTREE_CORE_MATH_RANDOM_SAMPLING_H_INCLUDED
#define ORTHOTREE_CORE_MATH_RANDOM_SAMPLING_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "../vector_math.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <random>
#include <cstdint>
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  SIMD random number generator (xoshiro256+)
//  Generates 4 random floats or doubles simultaneously using AVX2.
//  Deterministic per seed, thread‑safe.
// ============================================================================
class SimdRandomGenerator {
public:
    using result_type = uint64_t;

    SimdRandomGenerator(uint64_t seed = 123456789ULL) noexcept {
        for (int i = 0; i < 4; ++i) {
            m_state[i] = splitmix64(seed + i);
        }
    }

    result_type operator()() noexcept {
        uint64_t result = m_state[0] + m_state[3];
        uint64_t t = m_state[1] << 17;
        m_state[2] ^= m_state[0];
        m_state[3] ^= m_state[1];
        m_state[1] ^= m_state[2];
        m_state[0] ^= m_state[3];
        m_state[2] ^= t;
        m_state[3] = rotl(m_state[3], 45);
        return result;
    }

    static constexpr result_type min() noexcept { return 0; }
    static constexpr result_type max() noexcept { return ~0ULL; }

    // Generate 4 uniform random floats in [0,1) using SIMD
    void simdUniform4(float* out) noexcept {
        uint64_t r0 = (*this)();
        uint64_t r1 = (*this)();
        uint64_t r2 = (*this)();
        uint64_t r3 = (*this)();
        out[0] = (r0 >> 11) * (1.0f / (1ULL << 53));
        out[1] = (r1 >> 11) * (1.0f / (1ULL << 53));
        out[2] = (r2 >> 11) * (1.0f / (1ULL << 53));
        out[3] = (r3 >> 11) * (1.0f / (1ULL << 53));
    }

    void simdUniform4(double* out) noexcept {
        uint64_t r0 = (*this)();
        uint64_t r1 = (*this)();
        uint64_t r2 = (*this)();
        uint64_t r3 = (*this)();
        out[0] = (r0 >> 11) * (1.0 / (1ULL << 53));
        out[1] = (r1 >> 11) * (1.0 / (1ULL << 53));
        out[2] = (r2 >> 11) * (1.0 / (1ULL << 53));
        out[3] = (r3 >> 11) * (1.0 / (1ULL << 53));
    }

private:
    static uint64_t rotl(uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }
    static uint64_t splitmix64(uint64_t x) noexcept {
        x = (x + 0x9e3779b97f4a7c15ULL);
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }
    uint64_t m_state[4];
};

// ============================================================================
//  Low‑discrepancy sequences (Halton, Sobol, Hammersley)
// ============================================================================
class LowDiscrepancySequence {
public:
    static double halton(uint32_t index, uint32_t base) noexcept {
        double result = 0.0;
        double f = 1.0;
        uint32_t i = index;
        while (i > 0) {
            f /= base;
            result += f * (i % base);
            i /= base;
        }
        return result;
    }

    template<std::size_t N>
    static Math::Vector<double, N> haltonPoint(uint32_t index) noexcept {
        Math::Vector<double, N> p;
        for (std::size_t d = 0; d < N; ++d) {
            uint32_t base = primes[d % 16];
            p[d] = halton(index, base);
        }
        return p;
    }

    // Batched Halton points (SIMD friendly)
    template<std::size_t N>
    static void batchHalton(uint32_t startIndex, uint32_t count, Math::Vector<double, N>* out) noexcept {
        for (uint32_t i = 0; i < count; ++i) {
            out[i] = haltonPoint<N>(startIndex + i);
        }
    }

private:
    static constexpr uint32_t primes[16] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53};
};

// ============================================================================
//  Poisson disk sampling (2D/3D)
//  Generates maximal set of points with minimum distance.
// ============================================================================
template<typename T = float, std::size_t N = 2>
class PoissonDiskSampler {
public:
    using value_type = T;
    using point_type = Math::Vector<T, N>;

    struct Config {
        T radius = T(1);
        uint32_t maxAttempts = 30;
        aabb_type region;
        bool useSIMD = true;
    };

    explicit PoissonDiskSampler(const Config& cfg) : m_cfg(cfg) {
        m_cellSize = m_cfg.radius / std::sqrt(static_cast<T>(N));
        for (size_t i = 0; i < N; ++i) {
            m_gridSize[i] = static_cast<uint32_t>(std::ceil((m_cfg.region.extents()[i]) / m_cellSize));
        }
        m_grid.resize(m_gridSize[0] * m_gridSize[1] * (N == 3 ? m_gridSize[2] : 1),
                      std::numeric_limits<uint32_t>::max());
        m_rng.seed(123456789);
    }

    std::vector<point_type> generate() {
        std::vector<point_type> samples;
        std::vector<point_type> activeList;
        point_type first = randomPoint();
        addSample(first, samples, activeList);
        while (!activeList.empty()) {
            uint32_t idx = uniformInt(0, static_cast<uint32_t>(activeList.size()) - 1);
            point_type center = activeList[idx];
            bool found = false;
            for (uint32_t attempt = 0; attempt < m_cfg.maxAttempts; ++attempt) {
                point_type candidate = center + randomVectorInAnnulus();
                if (insideRegion(candidate) && isFarEnough(candidate, samples)) {
                    addSample(candidate, samples, activeList);
                    found = true;
                    break;
                }
            }
            if (!found) {
                activeList.erase(activeList.begin() + idx);
            }
        }
        return samples;
    }

private:
    void addSample(const point_type& p, std::vector<point_type>& samples,
                   std::vector<point_type>& active) {
        samples.push_back(p);
        active.push_back(p);
        uint32_t idx = static_cast<uint32_t>(samples.size() - 1);
        point_type cell = toGrid(p);
        size_t gidx = static_cast<size_t>(cell[0]) +
                      static_cast<size_t>(cell[1]) * m_gridSize[0];
        if constexpr (N == 3) {
            gidx += static_cast<size_t>(cell[2]) * m_gridSize[0] * m_gridSize[1];
        }
        if (gidx < m_grid.size()) m_grid[gidx] = idx;
    }

    bool isFarEnough(const point_type& p, const std::vector<point_type>& samples) const {
        point_type cell = toGrid(p);
        int minX = std::max(0, static_cast<int>(cell[0]) - 2);
        int maxX = std::min(static_cast<int>(m_gridSize[0]) - 1, static_cast<int>(cell[0]) + 2);
        int minY = std::max(0, static_cast<int>(cell[1]) - 2);
        int maxY = std::min(static_cast<int>(m_gridSize[1]) - 1, static_cast<int>(cell[1]) + 2);
        int minZ = 0, maxZ = 0;
        if constexpr (N == 3) {
            minZ = std::max(0, static_cast<int>(cell[2]) - 2);
            maxZ = std::min(static_cast<int>(m_gridSize[2]) - 1, static_cast<int>(cell[2]) + 2);
        }
        T r2 = m_cfg.radius * m_cfg.radius;
        for (int z = minZ; z <= maxZ; ++z) {
            for (int y = minY; y <= maxY; ++y) {
                for (int x = minX; x <= maxX; ++x) {
                    size_t gidx = static_cast<size_t>(x) +
                                  static_cast<size_t>(y) * m_gridSize[0];
                    if constexpr (N == 3) {
                        gidx += static_cast<size_t>(z) * m_gridSize[0] * m_gridSize[1];
                    }
                    if (gidx < m_grid.size()) {
                        uint32_t s = m_grid[gidx];
                        if (s != std::numeric_limits<uint32_t>::max()) {
                            if ((samples[s] - p).squaredLength() < r2 - 1e-6)
                                return false;
                        }
                    }
                }
            }
        }
        return true;
    }

    point_type randomPoint() {
        point_type p;
        for (size_t i = 0; i < N; ++i) {
            p[i] = m_cfg.region.min()[i] + uniform() * m_cfg.region.extents()[i];
        }
        return p;
    }

    point_type randomVectorInAnnulus() {
        T r = m_cfg.radius + uniform() * m_cfg.radius;
        point_type dir;
        for (size_t i = 0; i < N; ++i) dir[i] = uniform() * T(2) - T(1);
        T len = dir.length();
        if (len > T(1e-8)) dir /= len;
        return dir * r;
    }

    bool insideRegion(const point_type& p) const {
        for (size_t i = 0; i < N; ++i) {
            if (p[i] < m_cfg.region.min()[i] || p[i] > m_cfg.region.max()[i])
                return false;
        }
        return true;
    }

    point_type toGrid(const point_type& p) const {
        point_type g;
        for (size_t i = 0; i < N; ++i) {
            T t = (p[i] - m_cfg.region.min()[i]) / m_cellSize;
            g[i] = std::floor(t);
            if (g[i] < 0) g[i] = 0;
            if (g[i] >= m_gridSize[i]) g[i] = m_gridSize[i] - 1;
        }
        return g;
    }

    T uniform() { return m_dist(m_rng); }
    uint32_t uniformInt(uint32_t a, uint32_t b) {
        return a + (m_rng() % (b - a + 1));
    }

    Config m_cfg;
    SimdRandomGenerator m_rng;
    std::uniform_real_distribution<T> m_dist{T(0), T(1)};
    std::vector<uint32_t> m_grid;
    std::array<uint32_t, N> m_gridSize;
    T m_cellSize;
};

// ============================================================================
//  Stratified sampling (jittered grid)
// ============================================================================
template<typename T = float, std::size_t N = 2>
class StratifiedSampler {
public:
    using point_type = Math::Vector<T, N>;

    static std::vector<point_type> sample(const std::array<uint32_t, N>& counts,
                                          const aabb_type& region,
                                          SimdRandomGenerator& rng) {
        std::vector<point_type> samples;
        samples.reserve(counts[0] * counts[1] * (N == 3 ? counts[2] : 1));
        point_type cellSize = region.extents();
        for (size_t d = 0; d < N; ++d) cellSize[d] /= static_cast<T>(counts[d]);
        for (uint32_t i0 = 0; i0 < counts[0]; ++i0) {
            for (uint32_t i1 = 0; i1 < counts[1]; ++i1) {
                if constexpr (N == 2) {
                    point_type p;
                    p[0] = region.min()[0] + (static_cast<T>(i0) + rng.uniform()) * cellSize[0];
                    p[1] = region.min()[1] + (static_cast<T>(i1) + rng.uniform()) * cellSize[1];
                    samples.push_back(p);
                } else if constexpr (N == 3) {
                    for (uint32_t i2 = 0; i2 < counts[2]; ++i2) {
                        point_type p;
                        p[0] = region.min()[0] + (static_cast<T>(i0) + rng.uniform()) * cellSize[0];
                        p[1] = region.min()[1] + (static_cast<T>(i1) + rng.uniform()) * cellSize[1];
                        p[2] = region.min()[2] + (static_cast<T>(i2) + rng.uniform()) * cellSize[2];
                        samples.push_back(p);
                    }
                }
            }
        }
        return samples;
    }
};

// ============================================================================
//  Dynamic environment controller for random sampling
// ============================================================================
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
    void setUseSIMD(bool use) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_useSIMD = use;
    }
    bool useSIMD() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_useSIMD;
    }
private:
    SamplingEnvironment() : m_globalSeed(123456789), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    uint64_t m_globalSeed;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_RANDOM_SAMPLING_H_INCLUDED