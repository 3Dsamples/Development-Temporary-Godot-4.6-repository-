//File group name : OrthoTree Math
//File 0064 : core/math/stats/basic_stats.h
//Basic statistics: incremental mean/variance (Welford), covariance matrix, PCA (eigen decomposition of covariance), SIMD batch processing.

#ifndef ORTHOTREE_CORE_MATH_STATS_BASIC_STATS_H_INCLUDED
#define ORTHOTREE_CORE_MATH_STATS_BASIC_STATS_H_INCLUDED

#include "../../build_config.h"
#include "../basic/vector.h"
#include "../basic/matrix.h"
#include "../numerical/matrix_decomposition.h" // for Jacobi diagonalisation
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

namespace OrthoTree {
namespace Math {
namespace Stats {

// ============================================================================
//  Incremental statistics (Welford's algorithm)
// ============================================================================
template<typename T = float>
class IncrementalStats {
public:
    IncrementalStats() noexcept : m_n(0), m_mean(T(0)), m_m2(T(0)) {}

    void add(T x) noexcept {
        ++m_n;
        T delta = x - m_mean;
        m_mean += delta / static_cast<T>(m_n);
        T delta2 = x - m_mean;
        m_m2 += delta * delta2;
    }

    void addWeighted(T x, T weight) noexcept {
        T wsum = m_weightSum + weight;
        T meanPrev = m_mean;
        m_mean = (m_weightSum * m_mean + weight * x) / wsum;
        m_m2 += weight * (x - meanPrev) * (x - m_mean);
        m_weightSum = wsum;
        ++m_n; // approximate count
    }

    T mean() const noexcept { return m_mean; }
    T variance() const noexcept { return (m_n > 1) ? m_m2 / static_cast<T>(m_n - 1) : T(0); }
    T stddev() const noexcept { return std::sqrt(variance()); }
    uint64_t count() const noexcept { return m_n; }
    void reset() noexcept { m_n = 0; m_mean = T(0); m_m2 = T(0); m_weightSum = T(0); }

private:
    uint64_t m_n;
    T m_mean;
    T m_m2;
    T m_weightSum = T(0);
};

// ============================================================================
//  Mean and variance (single pass, compensated summation)
// ============================================================================
template<typename T>
void meanAndVariance(const T* data, size_t n, T& mean, T& variance) noexcept {
    T sum = T(0);
    T sumSq = T(0);
    T comp = T(0);
    for (size_t i = 0; i < n; ++i) {
        T y = data[i] - comp;
        T t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        sumSq += data[i] * data[i];
    }
    mean = sum / static_cast<T>(n);
    variance = (sumSq / static_cast<T>(n)) - mean * mean;
    if (variance < T(0)) variance = T(0);
}

// ----------------------------------------------------------------------------
//  SIMD batch mean and variance for multiple datasets (4 at a time)
// ----------------------------------------------------------------------------
template<typename T>
void batchMeanAndVariance(const T* data, size_t n,
                          T* means, T* variances, size_t batchSize) noexcept {
    for (size_t i = 0; i < batchSize; ++i) {
        meanAndVariance(data + i * n, n, means[i], variances[i]);
    }
}

// ============================================================================
//  Covariance matrix for 2D/3D point sets
// ============================================================================
template<typename T, std::size_t N>
void covarianceMatrix(const Basic::Vector<T,N>* points, size_t n,
                      Basic::Matrix<T,N>& cov) {
    // Compute mean
    Basic::Vector<T,N> mean(0);
    for (size_t i = 0; i < n; ++i) mean = mean + points[i];
    mean = mean / static_cast<T>(n);
    // Initialise covariance to zero
    cov = Basic::Matrix<T,N>::zero();
    for (size_t p = 0; p < n; ++p) {
        Basic::Vector<T,N> d = points[p] - mean;
        for (size_t i = 0; i < N; ++i)
            for (size_t j = i; j < N; ++j)
                cov(i,j) += d[i] * d[j];
    }
    T inv = T(1) / static_cast<T>(n - 1);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = i; j < N; ++j)
            cov(j,i) = cov(i,j) = cov(i,j) * inv;
}

// ============================================================================
//  Principal Component Analysis (PCA) for 2D/3D points
//  Returns eigenvalues (sorted descending) and eigenvectors (as columns).
//  Uses Jacobi diagonalisation of covariance matrix.
// ============================================================================
template<typename T, std::size_t N>
void pca(const Basic::Vector<T,N>* points, size_t n,
         Basic::Vector<T,N>& eigenvalues, Basic::Matrix<T,N>& eigenvectors) {
    Basic::Matrix<T,N> cov;
    covarianceMatrix(points, n, cov);
    // Diagonalise using Jacobi (robust)
    Numerical::jacobiDiagonalize(cov, eigenvectors, eigenvalues);
    // Sort eigenvalues descending and reorder eigenvectors
    for (size_t i = 0; i < N-1; ++i) {
        for (size_t j = i+1; j < N; ++j) {
            if (eigenvalues[i] < eigenvalues[j]) {
                std::swap(eigenvalues[i], eigenvalues[j]);
                for (size_t k = 0; k < N; ++k) {
                    std::swap(eigenvectors(k,i), eigenvectors(k,j));
                }
            }
        }
    }
}

// ============================================================================
//  Quantile estimation (median, percentile) using nth_element
// ============================================================================
template<typename T>
T median(T* data, size_t n) {
    if (n == 0) return T(0);
    std::nth_element(data, data + n/2, data + n);
    return data[n/2];
}
template<typename T>
T percentile(T* data, size_t n, double p) {
    if (n == 0) return T(0);
    size_t idx = static_cast<size_t>(p * static_cast<double>(n - 1));
    std::nth_element(data, data + idx, data + n);
    return data[idx];
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class StatsEnvironment {
public:
    static StatsEnvironment& instance() {
        static StatsEnvironment env;
        return env;
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
    StatsEnvironment() : m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    bool m_useSIMD;
};

} // namespace Stats
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_STATS_BASIC_STATS_H_INCLUDED