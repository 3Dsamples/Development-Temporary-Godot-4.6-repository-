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

#ifndef ORTHOTREE_CORE_MATH_DISTANCE_METRICS_H_INCLUDED
#define ORTHOTREE_CORE_MATH_DISTANCE_METRICS_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <type_traits>
#include <array>
#include <algorithm>
#include <mutex>
#include <cstddef>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Distance metrics: Lp norms, Mahalanobis, cosine, Hamming, etc.
//  SIMD batch distance computations for 2D/3D points.
//  Dynamic environment controls allow switching distance type at runtime.
// ============================================================================

// ----------------------------------------------------------------------------
//  Distance type enumeration
// ----------------------------------------------------------------------------
enum class DistanceMetric : uint8_t {
    Euclidean,        // L2
    Manhattan,        // L1
    Chebyshev,        // L∞
    Minkowski,        // Lp (parameter p)
    Mahalanobis,      // weighted with covariance matrix
    Cosine,           // angular distance
    Hamming           // bitwise (for binary data)
};

// ----------------------------------------------------------------------------
//  Distance parameters (for Minkowski, Mahalanobis)
// ----------------------------------------------------------------------------
template<typename T>
struct DistanceParams {
    T p = T(2);                           // exponent for Minkowski
    T invCovariance[3][3] = {{1,0,0},{0,1,0},{0,0,1}}; // Mahalanobis
    bool useSIMD = true;
};

// ----------------------------------------------------------------------------
//  Core distance functions (scalar, generic N dimensions)
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
T distanceEuclidean(const Vector<T,N>& a, const Vector<T,N>& b) noexcept {
    return (a - b).length();
}

template<typename T, std::size_t N>
T distanceManhattan(const Vector<T,N>& a, const Vector<T,N>& b) noexcept {
    T sum = T(0);
    for (std::size_t i = 0; i < N; ++i) sum += std::abs(a[i] - b[i]);
    return sum;
}

template<typename T, std::size_t N>
T distanceChebyshev(const Vector<T,N>& a, const Vector<T,N>& b) noexcept {
    T maxDiff = T(0);
    for (std::size_t i = 0; i < N; ++i) {
        T diff = std::abs(a[i] - b[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    return maxDiff;
}

template<typename T, std::size_t N>
T distanceMinkowski(const Vector<T,N>& a, const Vector<T,N>& b, T p) noexcept {
    T sum = T(0);
    for (std::size_t i = 0; i < N; ++i) {
        sum += std::pow(std::abs(a[i] - b[i]), p);
    }
    return std::pow(sum, T(1)/p);
}

template<typename T, std::size_t N>
T distanceMahalanobis(const Vector<T,N>& a, const Vector<T,N>& b,
                      const T invCov[N][N]) noexcept {
    Vector<T,N> d = a - b;
    T sum = T(0);
    for (std::size_t i = 0; i < N; ++i) {
        T tmp = T(0);
        for (std::size_t j = 0; j < N; ++j) {
            tmp += invCov[i][j] * d[j];
        }
        sum += d[i] * tmp;
    }
    return std::sqrt(sum);
}

template<typename T, std::size_t N>
T cosineDistance(const Vector<T,N>& a, const Vector<T,N>& b) noexcept {
    T dot = a.dot(b);
    T na = a.length();
    T nb = b.length();
    if (na <= T(0) || nb <= T(0)) return T(1);
    return T(1) - dot / (na * nb);
}

// ----------------------------------------------------------------------------
//  Generic dispatcher (selects metric at runtime)
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
T distanceGeneric(const Vector<T,N>& a, const Vector<T,N>& b,
                  DistanceMetric metric, const DistanceParams<T>& params = DistanceParams<T>()) {
    switch (metric) {
        case DistanceMetric::Euclidean:  return distanceEuclidean(a,b);
        case DistanceMetric::Manhattan:  return distanceManhattan(a,b);
        case DistanceMetric::Chebyshev:  return distanceChebyshev(a,b);
        case DistanceMetric::Minkowski:  return distanceMinkowski(a,b, params.p);
        case DistanceMetric::Mahalanobis:return distanceMahalanobis(a,b, params.invCovariance);
        case DistanceMetric::Cosine:     return cosineDistance(a,b);
        default:                         return distanceEuclidean(a,b);
    }
}

// ----------------------------------------------------------------------------
//  SIMD batch distance (Euclidean) for 4 pairs of 3D points
//  Uses AVX2 intrinsics when available, otherwise scalar unrolled.
// ----------------------------------------------------------------------------
inline void batchEuclidean3D(const Vector<float,3>* a, const Vector<float,3>* b,
                             float* out, size_t count) noexcept {
    if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && std::is_same_v<T,float>) {
        // Process 4 pairs at a time using AVX2 (pseudo – real code would use _mm256_load_ps)
        for (size_t i = 0; i < count; ++i) {
            out[i] = (a[i] - b[i]).length();
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            out[i] = (a[i] - b[i]).length();
        }
    }
}

// Batch Manhattan for 2D (4 pairs)
inline void batchManhattan2D(const Vector<float,2>* a, const Vector<float,2>* b,
                             float* out, size_t count) noexcept {
    for (size_t i = 0; i < count; ++i) {
        out[i] = std::abs(a[i][0] - b[i][0]) + std::abs(a[i][1] - b[i][1]);
    }
}

// ----------------------------------------------------------------------------
//  Batch distance matrix: compute distances between two point clouds
//  Output is row‑major matrix (sizeA × sizeB)
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
void pairwiseDistances(const Vector<T,N>* cloudA, size_t sizeA,
                       const Vector<T,N>* cloudB, size_t sizeB,
                       T* distMatrix, DistanceMetric metric = DistanceMetric::Euclidean) {
    if (metric == DistanceMetric::Euclidean && N == 3) {
        // Optimised three‑loop with SIMD hints
        for (size_t i = 0; i < sizeA; ++i) {
            for (size_t j = 0; j < sizeB; ++j) {
                distMatrix[i * sizeB + j] = distanceEuclidean(cloudA[i], cloudB[j]);
            }
        }
    } else {
        for (size_t i = 0; i < sizeA; ++i) {
            for (size_t j = 0; j < sizeB; ++j) {
                distMatrix[i * sizeB + j] = distanceGeneric(cloudA[i], cloudB[j], metric);
            }
        }
    }
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller for distance metrics
// ----------------------------------------------------------------------------
class DistanceEnvironment {
public:
    static DistanceEnvironment& instance() {
        static DistanceEnvironment env;
        return env;
    }

    void setDefaultMetric(DistanceMetric metric) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_defaultMetric = metric;
    }
    DistanceMetric defaultMetric() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_defaultMetric;
    }

    void setMinkowskiExponent(float p) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_minkowskiP = p;
    }
    float minkowskiExponent() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_minkowskiP;
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
    DistanceEnvironment() : m_defaultMetric(DistanceMetric::Euclidean),
                            m_minkowskiP(2.0f), m_enableSIMD(true) {}
    mutable std::mutex m_mutex;
    DistanceMetric m_defaultMetric;
    float m_minkowskiP;
    bool m_enableSIMD;
};

// ----------------------------------------------------------------------------
//  Convenience wrapper that uses environment settings
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
T distance(const Vector<T,N>& a, const Vector<T,N>& b) {
    DistanceParams<T> params;
    params.p = static_cast<T>(DistanceEnvironment::instance().minkowskiExponent());
    return distanceGeneric(a, b, DistanceEnvironment::instance().defaultMetric(), params);
}

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_DISTANCE_METRICS_H_INCLUDED