//File group name : OrthoTree Math
//File 0065 : core/math/distance/metrics.h
//Distance metrics: Euclidean, Manhattan, Chebyshev, Minkowski, Mahalanobis, Cosine, Hamming. SIMD batch for L2 and L1.

#ifndef ORTHOTREE_CORE_MATH_DISTANCE_METRICS_H_INCLUDED
#define ORTHOTREE_CORE_MATH_DISTANCE_METRICS_H_INCLUDED

#include "../../build_config.h"
#include "../basic/vector.h"
#include "../basic/matrix.h"
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <cstdint>

namespace OrthoTree {
namespace Math {
namespace Distance {

// ============================================================================
//  Euclidean distance (L2)
// ============================================================================
template<typename T, std::size_t N>
T euclidean(const Basic::Vector<T,N>& a, const Basic::Vector<T,N>& b) noexcept {
    return (a - b).length();
}
template<typename T, std::size_t N>
T squaredEuclidean(const Basic::Vector<T,N>& a, const Basic::Vector<T,N>& b) noexcept {
    return (a - b).squaredLength();
}

// ----------------------------------------------------------------------------
//  SIMD batch Euclidean for 3D vectors (4 pairs)
// ----------------------------------------------------------------------------
inline void batchEuclidean3D(const Basic::Vector3f* a, const Basic::Vector3f* b,
                             float* out, size_t count) noexcept {
    if constexpr (std::is_same_v<T,float> && ORTHOTREE_SIMD_LEVEL >= 128) {
        for (size_t i = 0; i < count; ++i) out[i] = (a[i] - b[i]).length();
    } else {
        for (size_t i = 0; i < count; ++i) out[i] = (a[i] - b[i]).length();
    }
}

// ============================================================================
//  Manhattan distance (L1)
// ============================================================================
template<typename T, std::size_t N>
T manhattan(const Basic::Vector<T,N>& a, const Basic::Vector<T,N>& b) noexcept {
    T sum = T(0);
    for (std::size_t i = 0; i < N; ++i) sum += std::abs(a[i] - b[i]);
    return sum;
}

// ----------------------------------------------------------------------------
//  Batch Manhattan (2D, 4 pairs)
// ----------------------------------------------------------------------------
inline void batchManhattan2D(const Basic::Vector2f* a, const Basic::Vector2f* b,
                             float* out, size_t count) noexcept {
    for (size_t i = 0; i < count; ++i) {
        out[i] = std::abs(a[i][0] - b[i][0]) + std::abs(a[i][1] - b[i][1]);
    }
}

// ============================================================================
//  Chebyshev distance (L∞)
// ============================================================================
template<typename T, std::size_t N>
T chebyshev(const Basic::Vector<T,N>& a, const Basic::Vector<T,N>& b) noexcept {
    T maxDiff = T(0);
    for (std::size_t i = 0; i < N; ++i) {
        T d = std::abs(a[i] - b[i]);
        if (d > maxDiff) maxDiff = d;
    }
    return maxDiff;
}

// ============================================================================
//  Minkowski distance (Lp)
// ============================================================================
template<typename T, std::size_t N>
T minkowski(const Basic::Vector<T,N>& a, const Basic::Vector<T,N>& b, T p) noexcept {
    T sum = T(0);
    for (std::size_t i = 0; i < N; ++i) {
        sum += std::pow(std::abs(a[i] - b[i]), p);
    }
    return std::pow(sum, T(1)/p);
}

// ============================================================================
//  Mahalanobis distance (requires inverse covariance matrix)
// ============================================================================
template<typename T, std::size_t N>
T mahalanobis(const Basic::Vector<T,N>& a, const Basic::Vector<T,N>& b,
              const Basic::Matrix<T,N>& invCov) noexcept {
    Basic::Vector<T,N> d = a - b;
    T sum = T(0);
    for (std::size_t i = 0; i < N; ++i) {
        T tmp = T(0);
        for (std::size_t j = 0; j < N; ++j) {
            tmp += invCov(i,j) * d[j];
        }
        sum += d[i] * tmp;
    }
    return std::sqrt(sum);
}

// ============================================================================
//  Cosine distance (1 - cosine similarity)
// ============================================================================
template<typename T, std::size_t N>
T cosine(const Basic::Vector<T,N>& a, const Basic::Vector<T,N>& b) noexcept {
    T dot = a.dot(b);
    T na = a.length();
    T nb = b.length();
    if (na <= T(0) || nb <= T(0)) return T(1);
    return T(1) - dot / (na * nb);
}

// ============================================================================
//  Hamming distance (for integral types)
// ============================================================================
template<typename T>
T hamming(const T* a, const T* b, size_t len) noexcept {
    T dist = 0;
    for (size_t i = 0; i < len; ++i) {
        if (a[i] != b[i]) ++dist;
    }
    return dist;
}

// ============================================================================
//  Dynamic environment controller (choose default metric)
// ============================================================================
enum class DefaultMetric : uint8_t { Euclidean, Manhattan, Chebyshev, Cosine };
class DistanceEnvironment {
public:
    static DistanceEnvironment& instance() {
        static DistanceEnvironment env;
        return env;
    }
    void setDefaultMetric(DefaultMetric m) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_default = m;
    }
    DefaultMetric defaultMetric() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_default;
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
    DistanceEnvironment() : m_default(DefaultMetric::Euclidean), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    DefaultMetric m_default;
    bool m_useSIMD;
};

} // namespace Distance
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_DISTANCE_METRICS_H_INCLUDED