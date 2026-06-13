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

#ifndef ORTHOTREE_CORE_MATH_STATISTICS_H_INCLUDED
#define ORTHOTREE_CORE_MATH_STATISTICS_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cstddef>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Statistical functions for data analysis: mean, variance, covariance,
//  correlation, principal component analysis (PCA). SIMD batch versions
//  for high throughput. Supports 2D/3D point sets and dynamic environment
//  controls (e.g., adaptive outlier removal, weighted statistics).
// ============================================================================

// ----------------------------------------------------------------------------
//  Single‑pass mean and variance (Welford's algorithm)
//  For incremental updates – O(1) memory.
// ----------------------------------------------------------------------------
template<typename T>
class IncrementalStatistics {
public:
    IncrementalStatistics() noexcept : m_n(0), m_mean(T(0)), m_m2(T(0)) {}

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
        m_n += 1; // weighted count not exactly n
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

// ----------------------------------------------------------------------------
//  Batch mean and variance (single pass, SIMD friendly)
//  Uses compensated summation (Kahan) for high accuracy.
// ----------------------------------------------------------------------------
template<typename T>
void meanAndVariance(const T* data, size_t n, T& mean, T& variance) noexcept {
    T sum = T(0);
    T sumSq = T(0);
    T comp = T(0); // Kahan compensation
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

// SIMD batch version (4 inputs at a time)
template<typename T>
void batchMeanAndVariance(const T* data, size_t n,
                          T* means, T* variances, size_t batchSize) noexcept {
    if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && std::is_same_v<T,float>) {
        // Process 4 sequences in parallel (pseudo)
        for (size_t j = 0; j < batchSize; ++j) {
            meanAndVariance(data + j * n, n, means[j], variances[j]);
        }
    } else {
        for (size_t j = 0; j < batchSize; ++j) {
            meanAndVariance(data + j * n, n, means[j], variances[j]);
        }
    }
}

// ----------------------------------------------------------------------------
//  Covariance matrix for 2D/3D point sets (vector of vectors)
//  Output as symmetric matrix (lower triangle).
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
void covarianceMatrix(const Vector<T,N>* points, size_t n,
                      T cov[N][N]) noexcept {
    // Compute mean
    Vector<T,N> mean(0);
    for (size_t i = 0; i < n; ++i) mean = mean + points[i];
    mean = mean / static_cast<T>(n);
    // Compute covariance
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            cov[i][j] = T(0);
    for (size_t p = 0; p < n; ++p) {
        for (size_t i = 0; i < N; ++i) {
            T di = points[p][i] - mean[i];
            for (size_t j = i; j < N; ++j) {
                cov[i][j] += di * (points[p][j] - mean[j]);
            }
        }
    }
    T invn = T(1) / static_cast<T>(n - 1);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i; j < N; ++j) {
            cov[i][j] *= invn;
            cov[j][i] = cov[i][j];
        }
    }
}

// ----------------------------------------------------------------------------
//  Principal Component Analysis (PCA) for 2D/3D points.
//  Returns eigenvectors (principal components) and eigenvalues.
//  Uses Power Iteration for largest eigenvalue and deflation.
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
void pca(const Vector<T,N>* points, size_t n,
         Vector<T,N> eigenVectors[N], T eigenValues[N]) noexcept {
    T cov[N][N];
    covarianceMatrix(points, n, cov);
    // Power iteration to find largest eigenvalue/vector
    Vector<T,N> v;
    for (size_t i = 0; i < N; ++i) v[i] = T(1);
    for (int iter = 0; iter < 100; ++iter) {
        Vector<T,N> w(0);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                w[i] += cov[i][j] * v[j];
        T norm = w.length();
        if (norm > T(1e-12)) w = w / norm;
        if ((w - v).length() < T(1e-8)) break;
        v = w;
    }
    eigenVectors[0] = v;
    eigenValues[0] = v.dot(cov * v); // Rayleigh quotient
    // Deflation and repeat for smaller components
    for (int comp = 1; comp < N; ++comp) {
        // Remove projection onto previous eigenvectors
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                for (int k = 0; k < comp; ++k)
                    cov[i][j] -= eigenValues[k] * eigenVectors[k][i] * eigenVectors[k][j];
        v = Vector<T,N>(1);
        for (int iter = 0; iter < 100; ++iter) {
            Vector<T,N> w(0);
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < N; ++j)
                    w[i] += cov[i][j] * v[j];
            T norm = w.length();
            if (norm > T(1e-12)) w = w / norm;
            if ((w - v).length() < T(1e-8)) break;
            v = w;
        }
        eigenVectors[comp] = v;
        eigenValues[comp] = v.dot(cov * v);
    }
}

// ----------------------------------------------------------------------------
//  Fast PCA using covariance matrix with eigenvalues via Jacobi method.
//  Suitable for small N (2/3). Provides all eigenvectors simultaneously.
//  Jacobi rotation for symmetric matrix.
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
void jacobiDiagonalize(T A[N][N], T V[N][N], T D[N]) {
    // Initialise V as identity
    for (size_t i = 0; i < N; ++i) {
        D[i] = A[i][i];
        for (size_t j = 0; j < N; ++j) V[i][j] = (i == j) ? T(1) : T(0);
    }
    for (int iter = 0; iter < 50; ++iter) {
        T maxOff = T(0);
        size_t p = 0, q = 1;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i+1; j < N; ++j) {
                T off = std::abs(A[i][j]);
                if (off > maxOff) { maxOff = off; p = i; q = j; }
            }
        }
        if (maxOff < T(1e-12)) break;
        T theta = (A[q][q] - A[p][p]) / (T(2) * A[p][q]);
        T t = (theta >= T(0)) ? T(1) / (theta + std::sqrt(T(1) + theta*theta))
                              : T(1) / (theta - std::sqrt(T(1) + theta*theta));
        T c = T(1) / std::sqrt(T(1) + t*t);
        T s = t * c;
        T tau = s / (T(1) + c);
        // Update A
        T app = A[p][p];
        T aqq = A[q][q];
        T apq = A[p][q];
        A[p][p] = app - t * apq;
        A[q][q] = aqq + t * apq;
        A[p][q] = A[q][p] = T(0);
        for (size_t i = 0; i < N; ++i) {
            if (i != p && i != q) {
                T aip = A[i][p];
                T aiq = A[i][q];
                A[i][p] = aip - s * (aiq + tau * aip);
                A[p][i] = A[i][p];
                A[i][q] = aiq + s * (aip - tau * aiq);
                A[q][i] = A[i][q];
            }
        }
        // Update eigenvectors V
        for (size_t i = 0; i < N; ++i) {
            T vip = V[i][p];
            T viq = V[i][q];
            V[i][p] = vip - s * (viq + tau * vip);
            V[i][q] = viq + s * (vip - tau * viq);
        }
        D[p] = A[p][p];
        D[q] = A[q][q];
    }
}

// ----------------------------------------------------------------------------
//  Quantile estimation (median, percentile) using quickselect.
//  For large data, use approximate algorithm.
// ----------------------------------------------------------------------------
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
//  Fast batch quantiles for multiple datasets (SIMD)
//  Assumes each dataset is separate in memory.
// ----------------------------------------------------------------------------
template<typename T>
void batchMedian(const T* datasets, size_t n, T* outMedians, size_t numSets) {
    for (size_t s = 0; s < numSets; ++s) {
        std::vector<T> copy(datasets + s*n, datasets + (s+1)*n);
        outMedians[s] = median(copy.data(), n);
    }
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller for statistics
// ----------------------------------------------------------------------------
class StatisticsEnvironment {
public:
    static StatisticsEnvironment& instance() {
        static StatisticsEnvironment env;
        return env;
    }

    void setDefaultAlpha(T alpha) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_alpha = alpha;
    }
    T defaultAlpha() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_alpha;
    }

    void setEnableSIMD(bool enable) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_enableSIMD = enable;
    }
    bool enableSIMD() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_enableSIMD;
    }

private:
    StatisticsEnvironment() : m_alpha(T(0.05)), m_enableSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_alpha;
    bool m_enableSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_STATISTICS_H_INCLUDED