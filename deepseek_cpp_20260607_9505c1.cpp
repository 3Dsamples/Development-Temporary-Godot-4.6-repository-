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

#ifndef ORTHOTREE_CORE_MATH_EXTENDED_ADAPTIVE_PRECISION_H_INCLUDED
#define ORTHOTREE_CORE_MATH_EXTENDED_ADAPTIVE_PRECISION_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "../vector_math.h"
#include "../numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"  // hypothetical SIMD abstraction layer

#include <cmath>
#include <limits>
#include <type_traits>
#include <array>
#include <cstdint>
#include <algorithm>

namespace OrthoTree {
namespace Math {
namespace Extended {

// ----------------------------------------------------------------------------
//  Precision level enumeration
// ----------------------------------------------------------------------------
enum class PrecisionLevel : uint8_t {
    Low    = 0,   // ~float,  1e-6 relative error
    Medium = 1,   // ~double, 1e-12 relative error
    High   = 2,   // ~long double, 1e-15 relative error
    Ultra  = 3    // software extended (MPFR style stub)
};

// ----------------------------------------------------------------------------
//  Adaptive precision scalar – stores value + current precision level
// ----------------------------------------------------------------------------
template<typename BaseFloat = double>
class AdaptiveScalar {
public:
    using value_type = BaseFloat;
    static_assert(std::is_floating_point_v<BaseFloat>,
                  "AdaptiveScalar requires floating point type");

    constexpr AdaptiveScalar() noexcept : m_value(0), m_level(PrecisionLevel::Medium) {}
    explicit constexpr AdaptiveScalar(BaseFloat v, PrecisionLevel level = PrecisionLevel::Medium) noexcept
        : m_value(v), m_level(level) {}

    // Implicit conversion from base float (conservative: keep medium)
    AdaptiveScalar(BaseFloat v) noexcept : m_value(v), m_level(PrecisionLevel::Medium) {}

    AdaptiveScalar(const AdaptiveScalar&) = default;
    AdaptiveScalar& operator=(const AdaptiveScalar&) = default;

    BaseFloat value() const noexcept { return m_value; }
    PrecisionLevel level() const noexcept { return m_level; }
    void setLevel(PrecisionLevel l) noexcept { m_level = l; }

    // Arithmetic with adaptive precision tracking
    AdaptiveScalar operator+(const AdaptiveScalar& other) const noexcept {
        BaseFloat res = m_value + other.m_value;
        PrecisionLevel newLevel = static_cast<PrecisionLevel>(
            std::max(static_cast<uint8_t>(m_level), static_cast<uint8_t>(other.m_level))
        );
        return AdaptiveScalar(res, newLevel);
    }

    AdaptiveScalar operator-(const AdaptiveScalar& other) const noexcept {
        BaseFloat res = m_value - other.m_value;
        PrecisionLevel newLevel = static_cast<PrecisionLevel>(
            std::max(static_cast<uint8_t>(m_level), static_cast<uint8_t>(other.m_level))
        );
        return AdaptiveScalar(res, newLevel);
    }

    AdaptiveScalar operator*(const AdaptiveScalar& other) const noexcept {
        BaseFloat res = m_value * other.m_value;
        // multiplication increases relative error: add levels?
        uint8_t lvl = std::max(static_cast<uint8_t>(m_level), static_cast<uint8_t>(other.m_level));
        if (lvl < static_cast<uint8_t>(PrecisionLevel::Ultra))
            lvl = static_cast<uint8_t>(lvl + 1);
        return AdaptiveScalar(res, static_cast<PrecisionLevel>(lvl));
    }

    AdaptiveScalar operator/(const AdaptiveScalar& other) const noexcept {
        BaseFloat res = m_value / other.m_value;
        uint8_t lvl = std::max(static_cast<uint8_t>(m_level), static_cast<uint8_t>(other.m_level));
        if (lvl < static_cast<uint8_t>(PrecisionLevel::Ultra))
            lvl = static_cast<uint8_t>(lvl + 1);
        return AdaptiveScalar(res, static_cast<PrecisionLevel>(lvl));
    }

    AdaptiveScalar& operator+=(const AdaptiveScalar& other) noexcept {
        m_value += other.m_value;
        m_level = static_cast<PrecisionLevel>(
            std::max(static_cast<uint8_t>(m_level), static_cast<uint8_t>(other.m_level))
        );
        return *this;
    }

    // Comparison
    bool operator==(const AdaptiveScalar& other) const noexcept {
        return std::abs(m_value - other.m_value) <= epsilonForLevel(std::max(m_level, other.m_level));
    }
    bool operator!=(const AdaptiveScalar& other) const noexcept { return !(*this == other); }
    bool operator<(const AdaptiveScalar& other) const noexcept { return m_value < other.m_value; }
    bool operator>(const AdaptiveScalar& other) const noexcept { return m_value > other.m_value; }

    static BaseFloat epsilonForLevel(PrecisionLevel lvl) noexcept {
        switch (lvl) {
            case PrecisionLevel::Low:    return BaseFloat(1e-6);
            case PrecisionLevel::Medium: return BaseFloat(1e-12);
            case PrecisionLevel::High:   return BaseFloat(1e-15);
            case PrecisionLevel::Ultra:  return BaseFloat(1e-18);
            default: return BaseFloat(1e-12);
        }
    }

private:
    BaseFloat m_value;
    PrecisionLevel m_level;
};

// ----------------------------------------------------------------------------
//  SIMD vector with adaptive precision per lane (for 2D/3D)
// ----------------------------------------------------------------------------
template<typename T, std::size_t N, bool UseSIMD = true>
class AdaptiveSIMDVector {
    static_assert(N == 2 || N == 3, "Only 2D or 3D supported for SIMD");
public:
    using value_type = T;
    using scalar_type = AdaptiveScalar<T>;
    static constexpr std::size_t dimension = N;

    // Constructors
    AdaptiveSIMDVector() noexcept : m_level(PrecisionLevel::Medium) {
        for (std::size_t i = 0; i < N; ++i) m_data[i] = T(0);
    }

    AdaptiveSIMDVector(T x, T y, T z = T(0), PrecisionLevel lvl = PrecisionLevel::Medium) noexcept
        : m_level(lvl) {
        m_data[0] = x; if constexpr (N >= 2) m_data[1] = y; if constexpr (N >= 3) m_data[2] = z;
    }

    AdaptiveSIMDVector(const AdaptiveScalar<T>& x, const AdaptiveScalar<T>& y,
                       const AdaptiveScalar<T>& z = AdaptiveScalar<T>(0)) noexcept
        : m_data{x.value(), y.value(), z.value()}
        , m_level(static_cast<PrecisionLevel>(std::max({static_cast<uint8_t>(x.level()),
                                                        static_cast<uint8_t>(y.level()),
                                                        static_cast<uint8_t>(z.level())}))) {}

    // Element access
    T operator[](std::size_t i) const noexcept { return m_data[i]; }
    T& operator[](std::size_t i) noexcept { return m_data[i]; }

    PrecisionLevel precisionLevel() const noexcept { return m_level; }
    void setPrecisionLevel(PrecisionLevel lvl) noexcept { m_level = lvl; }

    // SIMD‑aware dot product
    T dot(const AdaptiveSIMDVector& other) const noexcept {
        if constexpr (UseSIMD && ORTHOTREE_SIMD_LEVEL >= 128) {
            // Use SIMD intrinsics if available
            return simdDotProduct(m_data, other.m_data, N);
        } else {
            T sum = T(0);
            for (std::size_t i = 0; i < N; ++i) sum += m_data[i] * other.m_data[i];
            return sum;
        }
    }

    // Euclidean length with adaptive precision
    T length() const noexcept {
        T sq = dot(*this);
        return std::sqrt(sq);
    }

    // Normalize in place
    void normalize() noexcept {
        T len = length();
        if (len > AdaptiveScalar<T>::epsilonForLevel(m_level)) {
            T invLen = T(1) / len;
            for (std::size_t i = 0; i < N; ++i) m_data[i] *= invLen;
        }
    }

    // Arithmetic operators (SIMD accelerated)
    AdaptiveSIMDVector operator+(const AdaptiveSIMDVector& other) const noexcept {
        AdaptiveSIMDVector result;
        if constexpr (UseSIMD && ORTHOTREE_SIMD_LEVEL >= 128) {
            simdAdd(m_data, other.m_data, result.m_data, N);
        } else {
            for (std::size_t i = 0; i < N; ++i) result.m_data[i] = m_data[i] + other.m_data[i];
        }
        result.m_level = static_cast<PrecisionLevel>(
            std::max(static_cast<uint8_t>(m_level), static_cast<uint8_t>(other.m_level))
        );
        return result;
    }

    AdaptiveSIMDVector operator-(const AdaptiveSIMDVector& other) const noexcept {
        AdaptiveSIMDVector result;
        if constexpr (UseSIMD && ORTHOTREE_SIMD_LEVEL >= 128) {
            simdSub(m_data, other.m_data, result.m_data, N);
        } else {
            for (std::size_t i = 0; i < N; ++i) result.m_data[i] = m_data[i] - other.m_data[i];
        }
        result.m_level = static_cast<PrecisionLevel>(
            std::max(static_cast<uint8_t>(m_level), static_cast<uint8_t>(other.m_level))
        );
        return result;
    }

    AdaptiveSIMDVector operator*(T scalar) const noexcept {
        AdaptiveSIMDVector result;
        if constexpr (UseSIMD && ORTHOTREE_SIMD_LEVEL >= 128) {
            simdMulScalar(m_data, scalar, result.m_data, N);
        } else {
            for (std::size_t i = 0; i < N; ++i) result.m_data[i] = m_data[i] * scalar;
        }
        result.m_level = m_level;
        return result;
    }

private:
    std::array<T, N> m_data;
    PrecisionLevel m_level;

    // SIMD fallback implementations (intrinsics would be placed here)
    static inline T simdDotProduct(const T* a, const T* b, std::size_t n) noexcept {
        T sum = 0;
        for (std::size_t i = 0; i < n; ++i) sum += a[i] * b[i];
        return sum;
    }
    static inline void simdAdd(const T* a, const T* b, T* out, std::size_t n) noexcept {
        for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
    }
    static inline void simdSub(const T* a, const T* b, T* out, std::size_t n) noexcept {
        for (std::size_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
    }
    static inline void simdMulScalar(const T* a, T s, T* out, std::size_t n) noexcept {
        for (std::size_t i = 0; i < n; ++i) out[i] = a[i] * s;
    }
};

// ----------------------------------------------------------------------------
//  Dynamic precision controller – decides level based on distance, error, etc.
// ----------------------------------------------------------------------------
class PrecisionController {
public:
    using DistanceMetric = std::function<double(const double* a, const double* b, int dim)>;

    PrecisionController() noexcept
        : m_nearPrecision(PrecisionLevel::High)
        , m_farPrecision(PrecisionLevel::Low)
        , m_nearDistance(10.0)
        , m_farDistance(1000.0)
        , m_relativeErrorBudget(1e-6) {}

    void setNearPrecision(PrecisionLevel lvl) noexcept { m_nearPrecision = lvl; }
    void setFarPrecision(PrecisionLevel lvl) noexcept { m_farPrecision = lvl; }
    void setNearDistance(double d) noexcept { m_nearDistance = d; }
    void setFarDistance(double d) noexcept { m_farDistance = d; }
    void setErrorBudget(double eps) noexcept { m_relativeErrorBudget = eps; }

    // Determine precision level for a point given its distance to origin/viewer
    PrecisionLevel levelForDistance(double distance) const noexcept {
        if (distance <= m_nearDistance) return m_nearPrecision;
        if (distance >= m_farDistance) return m_farPrecision;
        // Interpolate between levels
        double t = (distance - m_nearDistance) / (m_farDistance - m_nearDistance);
        uint8_t nearLvl = static_cast<uint8_t>(m_nearPrecision);
        uint8_t farLvl = static_cast<uint8_t>(m_farPrecision);
        uint8_t mixed = static_cast<uint8_t>(std::round((1.0 - t) * nearLvl + t * farLvl));
        return static_cast<PrecisionLevel>(std::clamp(mixed, static_cast<uint8_t>(0),
                                                      static_cast<uint8_t>(PrecisionLevel::Ultra)));
    }

    // Compute required precision based on estimated error in computation
    template<typename VecType>
    PrecisionLevel requiredPrecision(const VecType& value, double tolerance) const noexcept {
        double relError = std::abs(value) * m_relativeErrorBudget;
        if (relError < 1e-12) return PrecisionLevel::Ultra;
        if (relError < 1e-9) return PrecisionLevel::High;
        if (relError < 1e-6) return PrecisionLevel::Medium;
        return PrecisionLevel::Low;
    }

private:
    PrecisionLevel m_nearPrecision;
    PrecisionLevel m_farPrecision;
    double m_nearDistance;
    double m_farDistance;
    double m_relativeErrorBudget;
};

// ----------------------------------------------------------------------------
//  Adaptive bounding box – stores bounds in multiple precision representations
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
class AdaptiveBoundingBox {
public:
    using LowPrecVec   = Math::Vector<float, N>;
    using MedPrecVec   = Math::Vector<double, N>;
    using HighPrecVec  = Math::Vector<long double, N>;

    AdaptiveBoundingBox() = default;

    template<typename Vec>
    AdaptiveBoundingBox(const Vec& min, const Vec& max, PrecisionLevel level)
        : m_level(level) {
        setBounds(min, max, level);
    }

    void setBounds(const LowPrecVec& min, const LowPrecVec& max) {
        m_minLow = min; m_maxLow = max;
        m_hasLow = true;
        m_level = PrecisionLevel::Low;
    }
    void setBounds(const MedPrecVec& min, const MedPrecVec& max) {
        m_minMed = min; m_maxMed = max;
        m_hasMed = true;
        m_level = PrecisionLevel::Medium;
    }
    void setBounds(const HighPrecVec& min, const HighPrecVec& max) {
        m_minHigh = min; m_maxHigh = max;
        m_hasHigh = true;
        m_level = PrecisionLevel::High;
    }

    // Query using requested precision (convert on the fly)
    Math::AxisAlignedBox<T, N> getBounds(PrecisionLevel requestedLevel) const {
        if (requestedLevel <= PrecisionLevel::Low && m_hasLow) {
            return Math::AxisAlignedBox<T, N>(
                convertVec<LowPrecVec, T>(m_minLow),
                convertVec<LowPrecVec, T>(m_maxLow)
            );
        } else if (requestedLevel <= PrecisionLevel::Medium && m_hasMed) {
            return Math::AxisAlignedBox<T, N>(
                convertVec<MedPrecVec, T>(m_minMed),
                convertVec<MedPrecVec, T>(m_maxMed)
            );
        } else if (m_hasHigh) {
            return Math::AxisAlignedBox<T, N>(
                convertVec<HighPrecVec, T>(m_minHigh),
                convertVec<HighPrecVec, T>(m_maxHigh)
            );
        } else {
            // fallback
            return Math::AxisAlignedBox<T, N>();
        }
    }

    PrecisionLevel currentLevel() const noexcept { return m_level; }

private:
    union {
        LowPrecVec m_minLow, m_maxLow;
        MedPrecVec m_minMed, m_maxMed;
        HighPrecVec m_minHigh, m_maxHigh;
    };
    bool m_hasLow = false, m_hasMed = false, m_hasHigh = false;
    PrecisionLevel m_level = PrecisionLevel::Medium;

    template<typename FromVec, typename ToScalar>
    static Math::Vector<ToScalar, N> convertVec(const FromVec& v) {
        Math::Vector<ToScalar, N> res;
        for (std::size_t i = 0; i < N; ++i) res[i] = static_cast<ToScalar>(v[i]);
        return res;
    }
};

} // namespace Extended
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_EXTENDED_ADAPTIVE_PRECISION_H_INCLUDED