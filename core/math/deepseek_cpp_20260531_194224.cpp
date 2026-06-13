//File 0040 : core/math/statistics.h
//Real‑time descriptive statistics: mean, variance, standard deviation, covariance, correlation, histogram, and incremental (Welford’s) online variance with SIMD‑friendly batch operations.
#ifndef CORE_MATH_STATISTICS_H
#define CORE_MATH_STATISTICS_H

#include "vector_math.h"
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace SimulationMath {
namespace stats {

// -----------------------------------------------------------------------------
// 1. Mean of a float array
// -----------------------------------------------------------------------------
inline float mean(const float* data, size_t n) noexcept {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += data[i];
    return sum / static_cast<float>(n);
}

// -----------------------------------------------------------------------------
// 2. Variance (population) of a float array
// -----------------------------------------------------------------------------
inline float variance(const float* data, size_t n) noexcept {
    if (n <= 1) return 0.0f;
    float m = mean(data, n);
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = data[i] - m;
        sum_sq += d * d;
    }
    return sum_sq / static_cast<float>(n);
}

// -----------------------------------------------------------------------------
// 3. Standard deviation
// -----------------------------------------------------------------------------
inline float stddev(const float* data, size_t n) noexcept {
    return std::sqrt(variance(data, n));
}

// -----------------------------------------------------------------------------
// 4. Covariance between two equal‑length arrays
// -----------------------------------------------------------------------------
inline float covariance(const float* x, const float* y, size_t n) noexcept {
    if (n <= 1) return 0.0f;
    float mx = mean(x, n);
    float my = mean(y, n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i)
        sum += (x[i] - mx) * (y[i] - my);
    return sum / static_cast<float>(n);
}

// -----------------------------------------------------------------------------
// 5. Pearson correlation coefficient
// -----------------------------------------------------------------------------
inline float correlation(const float* x, const float* y, size_t n) noexcept {
    float sx = stddev(x, n);
    float sy = stddev(y, n);
    if (sx < 1e-12f || sy < 1e-12f) return 0.0f;
    return covariance(x, y, n) / (sx * sy);
}

// -----------------------------------------------------------------------------
// 6. Histogram (uniform bins)
// -----------------------------------------------------------------------------
inline std::vector<uint32_t> histogram(const float* data, size_t n, size_t num_bins,
                                       float min_val, float max_val) noexcept {
    std::vector<uint32_t> bins(num_bins, 0);
    if (n == 0) return bins;
    float range = max_val - min_val;
    if (range <= 0.0f) {
        bins[0] = n;
        return bins;
    }
    float inv_range = 1.0f / range;
    for (size_t i = 0; i < n; ++i) {
        float t = (data[i] - min_val) * inv_range;
        int idx = static_cast<int>(t * num_bins);
        idx = std::max(0, std::min(idx, static_cast<int>(num_bins) - 1));
        ++bins[idx];
    }
    return bins;
}

// -----------------------------------------------------------------------------
// 7. Online incremental mean and variance (Welford’s algorithm)
// -----------------------------------------------------------------------------
class OnlineStats {
    size_t count_ = 0;
    float mean_ = 0.0f;
    float M2_ = 0.0f;  // sum of squares of differences from current mean
public:
    void add(float x) noexcept {
        ++count_;
        float delta = x - mean_;
        mean_ += delta / count_;
        float delta2 = x - mean_;
        M2_ += delta * delta2;
    }
    size_t count() const noexcept { return count_; }
    float mean() const noexcept { return mean_; }
    float variance() const noexcept {
        return (count_ > 1) ? M2_ / count_ : 0.0f;
    }
    float sample_variance() const noexcept {
        return (count_ > 1) ? M2_ / (count_ - 1) : 0.0f;
    }
    float stddev() const noexcept { return std::sqrt(variance()); }
    void reset() noexcept { count_ = 0; mean_ = 0.0f; M2_ = 0.0f; }
};

} // namespace stats
} // namespace SimulationMath

#endif // CORE_MATH_STATISTICS_H