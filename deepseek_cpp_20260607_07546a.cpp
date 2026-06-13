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

#ifndef ORTHOTREE_CORE_MATH_EXTENDED_QUANTIZED_NUMERICS_H_INCLUDED
#define ORTHOTREE_CORE_MATH_EXTENDED_QUANTIZED_NUMERICS_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "../vector_math.h"
#include "../numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <array>
#include <algorithm>

namespace OrthoTree {
namespace Math {
namespace Extended {

// ============================================================================
//  Fixed‑point arithmetic (64‑bit integer, configurable fractional bits)
// ============================================================================
template<typename IntType = int64_t, uint8_t FractionalBits = 32>
class FixedPoint {
public:
    using value_type = IntType;
    static_assert(std::is_integral_v<IntType> && std::is_signed_v<IntType>,
                  "FixedPoint requires signed integer type");
    static constexpr uint8_t FRAC_BITS = FractionalBits;
    static constexpr IntType SCALE = IntType(1) << FRAC_BITS;
    static constexpr double SCALE_DBL = static_cast<double>(SCALE);

    constexpr FixedPoint() noexcept : m_raw(0) {}
    explicit constexpr FixedPoint(IntType raw) noexcept : m_raw(raw) {}
    constexpr FixedPoint(int v) noexcept : m_raw(static_cast<IntType>(v) * SCALE) {}
    constexpr FixedPoint(double v) noexcept : m_raw(static_cast<IntType>(std::round(v * SCALE_DBL))) {}
    constexpr FixedPoint(float v) noexcept : FixedPoint(static_cast<double>(v)) {}

    constexpr IntType raw() const noexcept { return m_raw; }
    constexpr double toDouble() const noexcept { return static_cast<double>(m_raw) / SCALE_DBL; }
    constexpr float toFloat() const noexcept { return static_cast<float>(toDouble()); }

    // Arithmetic with saturation (optional)
    FixedPoint operator+(const FixedPoint& other) const noexcept {
        IntType sum = m_raw + other.m_raw;
        return FixedPoint(sum);
    }
    FixedPoint operator-(const FixedPoint& other) const noexcept {
        IntType diff = m_raw - other.m_raw;
        return FixedPoint(diff);
    }
    FixedPoint operator*(const FixedPoint& other) const noexcept {
        // Multiply two fixed-point numbers: (a*SCALE)*(b*SCALE) / SCALE = a*b*SCALE
        // To avoid overflow, use 128-bit intermediate if available
        using DoubleInt = __int128_t;
        DoubleInt prod = static_cast<DoubleInt>(m_raw) * static_cast<DoubleInt>(other.m_raw);
        IntType result = static_cast<IntType>(prod / SCALE);
        return FixedPoint(result);
    }
    FixedPoint operator/(const FixedPoint& other) const noexcept {
        DoubleInt dividend = static_cast<DoubleInt>(m_raw) * SCALE;
        IntType result = static_cast<IntType>(dividend / other.m_raw);
        return FixedPoint(result);
    }

    FixedPoint& operator+=(const FixedPoint& other) noexcept { m_raw += other.m_raw; return *this; }
    FixedPoint& operator-=(const FixedPoint& other) noexcept { m_raw -= other.m_raw; return *this; }
    FixedPoint& operator*=(const FixedPoint& other) noexcept { *this = *this * other; return *this; }
    FixedPoint& operator/=(const FixedPoint& other) noexcept { *this = *this / other; return *this; }

    bool operator==(const FixedPoint& other) const noexcept { return m_raw == other.m_raw; }
    bool operator!=(const FixedPoint& other) const noexcept { return !(*this == other); }
    bool operator<(const FixedPoint& other) const noexcept { return m_raw < other.m_raw; }
    bool operator>(const FixedPoint& other) const noexcept { return m_raw > other.m_raw; }

    // Fast approximate functions (using integer arithmetic)
    FixedPoint sqrt() const noexcept {
        if (m_raw <= 0) return FixedPoint(0);
        // Newton's method for integer sqrt of scaled value
        IntType x = m_raw;
        IntType y = (x + SCALE) >> 1;
        for (int i = 0; i < 8; ++i) {
            y = (y + x / y) >> 1;
        }
        return FixedPoint(y);
    }

private:
    IntType m_raw;
};

// ============================================================================
//  Log‑scale arithmetic (for very large or very small numbers)
// ============================================================================
template<typename Base = double>
class LogScaleNumber {
public:
    using value_type = Base;
    static constexpr Base LOG_BASE = Base(10.0); // decade scale

    LogScaleNumber() noexcept : m_logValue(-std::numeric_limits<Base>::infinity()) {}
    explicit LogScaleNumber(Base linear) noexcept {
        if (linear > Base(0)) {
            m_logValue = std::log(linear) / std::log(LOG_BASE);
        } else if (linear == Base(0)) {
            m_logValue = -std::numeric_limits<Base>::infinity();
        } else {
            m_logValue = std::numeric_limits<Base>::quiet_NaN();
        }
    }
    LogScaleNumber(Base logValue, bool isLog) noexcept : m_logValue(logValue) {}

    static LogScaleNumber fromLog(Base logVal) noexcept { return LogScaleNumber(logVal, true); }

    Base toLinear() const noexcept {
        if (std::isinf(m_logValue) && m_logValue < 0) return Base(0);
        return std::pow(LOG_BASE, m_logValue);
    }

    LogScaleNumber operator+(const LogScaleNumber& other) const noexcept {
        // ln(a+b) = ln(a) + ln(1 + b/a)
        Base maxLog = std::max(m_logValue, other.m_logValue);
        Base minLog = std::min(m_logValue, other.m_logValue);
        Base diff = minLog - maxLog;
        // diff <= 0, so exp(diff) <= 1
        Base sumLog = maxLog + std::log1p(std::pow(LOG_BASE, diff));
        return LogScaleNumber(sumLog, true);
    }

    LogScaleNumber operator*(const LogScaleNumber& other) const noexcept {
        return LogScaleNumber(m_logValue + other.m_logValue, true);
    }

    LogScaleNumber operator/(const LogScaleNumber& other) const noexcept {
        return LogScaleNumber(m_logValue - other.m_logValue, true);
    }

    bool operator<(const LogScaleNumber& other) const noexcept { return m_logValue < other.m_logValue; }
    bool operator>(const LogScaleNumber& other) const noexcept { return m_logValue > other.m_logValue; }

    Base logValue() const noexcept { return m_logValue; }

private:
    Base m_logValue;
};

// ============================================================================
//  SIMD quantized vector for 2D/3D coordinates (8‑bit or 16‑bit per component)
// ============================================================================
template<typename T, std::size_t N, typename QuantInt = int16_t>
class QuantizedVector {
    static_assert(N == 2 || N == 3, "Only 2D or 3D");
    static_assert(std::is_integral_v<QuantInt> && std::is_signed_v<QuantInt>,
                  "QuantInt must be signed integer");
public:
    using value_type = T;
    using quant_type = QuantInt;
    static constexpr std::size_t dimension = N;

    QuantizedVector() noexcept : m_quant{{0}} {}
    QuantizedVector(const Math::Vector<T, N>& v, T minVal, T maxVal) noexcept {
        setFromFloat(v, minVal, maxVal);
    }

    void setFromFloat(const Math::Vector<T, N>& v, T minVal, T maxVal) noexcept {
        T range = maxVal - minVal;
        T invRange = T(1) / range;
        for (std::size_t i = 0; i < N; ++i) {
            T t = (v[i] - minVal) * invRange;
            t = Math::clamp(t, T(0), T(1));
            quant_type q = static_cast<quant_type>(t * static_cast<T>(std::numeric_limits<quant_type>::max()));
            m_quant[i] = q;
        }
    }

    Math::Vector<T, N> toFloat(T minVal, T maxVal) const noexcept {
        Math::Vector<T, N> result;
        T range = maxVal - minVal;
        T maxQuant = static_cast<T>(std::numeric_limits<quant_type>::max());
        for (std::size_t i = 0; i < N; ++i) {
            T t = static_cast<T>(m_quant[i]) / maxQuant;
            result[i] = minVal + t * range;
        }
        return result;
    }

    // SIMD batch conversion (for arrays of vectors)
    static void batchToFloat(const QuantizedVector* src, Math::Vector<T, N>* dst,
                             std::size_t count, T minVal, T maxVal) noexcept {
        T range = maxVal - minVal;
        T maxQuant = static_cast<T>(std::numeric_limits<quant_type>::max());
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && N == 3) {
            // Pseudo‑SIMD (real implementation would use aligned loads and intrinsics)
            for (std::size_t i = 0; i < count; ++i) {
                dst[i] = src[i].toFloat(minVal, maxVal);
            }
        } else {
            for (std::size_t i = 0; i < count; ++i) {
                dst[i] = src[i].toFloat(minVal, maxVal);
            }
        }
    }

private:
    std::array<quant_type, N> m_quant;
};

// ============================================================================
//  Dynamic range controller for quantization (adapts to data distribution)
// ============================================================================
template<typename T>
class AdaptiveQuantizationRange {
public:
    using vec_type = Math::Vector<T, 3>;

    AdaptiveQuantizationRange() noexcept
        : m_min(T(0)), m_max(T(1)), m_automatic(true), m_updateCounter(0) {}

    void setFixedRange(const vec_type& minBound, const vec_type& maxBound) noexcept {
        m_min = minBound;
        m_max = maxBound;
        m_automatic = false;
    }

    void update(const vec_type& point) noexcept {
        if (!m_automatic) return;
        ++m_updateCounter;
        m_min = m_min.componentWiseMin(point);
        m_max = m_max.componentWiseMax(point);
        if (m_updateCounter > 1000) {
            // Expand range slightly to avoid frequent re‑quantization
            vec_type margin = (m_max - m_min) * T(0.05);
            m_min = m_min - margin;
            m_max = m_max + margin;
            m_updateCounter = 0;
        }
    }

    vec_type min() const noexcept { return m_min; }
    vec_type max() const noexcept { return m_max; }

private:
    vec_type m_min, m_max;
    bool m_automatic;
    uint32_t m_updateCounter;
};

// ============================================================================
//  Block floating point (shared exponent for a group of numbers)
// ============================================================================
template<typename BaseInt = int16_t>
class BlockFloat {
public:
    using int_type = BaseInt;
    static constexpr int MAX_EXPONENT = 15;

    BlockFloat() noexcept : m_mantissa(0), m_exponent(0) {}
    BlockFloat(float value) noexcept {
        int exp;
        float mant = std::frexp(value, &exp);
        m_exponent = static_cast<int8_t>(Math::clamp(exp, -MAX_EXPONENT, MAX_EXPONENT));
        m_mantissa = static_cast<int_type>(mant * static_cast<float>(std::numeric_limits<int_type>::max()));
    }

    float toFloat() const noexcept {
        float mant = static_cast<float>(m_mantissa) / static_cast<float>(std::numeric_limits<int_type>::max());
        return std::ldexp(mant, m_exponent);
    }

    // Batch encode/decode with shared exponent
    static void encodeBlock(const float* src, BlockFloat* dst, std::size_t count, int8_t& sharedExp) noexcept {
        // Find maximum exponent among all
        int maxExp = -MAX_EXPONENT;
        for (std::size_t i = 0; i < count; ++i) {
            int exp;
            std::frexp(src[i], &exp);
            if (exp > maxExp) maxExp = exp;
        }
        sharedExp = static_cast<int8_t>(Math::clamp(maxExp, -MAX_EXPONENT, MAX_EXPONENT));
        for (std::size_t i = 0; i < count; ++i) {
            float mant = std::ldexp(src[i], -sharedExp);
            dst[i].m_mantissa = static_cast<int_type>(mant * static_cast<float>(std::numeric_limits<int_type>::max()));
            dst[i].m_exponent = sharedExp;
        }
    }

    static void decodeBlock(const BlockFloat* src, float* dst, std::size_t count) noexcept {
        for (std::size_t i = 0; i < count; ++i) {
            dst[i] = src[i].toFloat();
        }
    }

private:
    int_type m_mantissa;
    int8_t m_exponent;
};

// ============================================================================
//  Dynamic environment controller for quantized numerics (auto‑adjust precision)
// ============================================================================
template<typename T>
class QuantizedEnvironment {
public:
    using vec_type = Math::Vector<T, 3>;

    QuantizedEnvironment() noexcept
        : m_precisionMode(PrecisionMode::Adaptive)
        , m_targetError(T(1e-4))
        , m_currentError(T(0)) {}

    enum class PrecisionMode : uint8_t {
        Fixed,      // use fixed quantization range
        Adaptive,   // expand range based on observed points
        Conservative // always keep extra margin
    };

    void setMode(PrecisionMode mode) noexcept { m_precisionMode = mode; }
    void setTargetError(T error) noexcept { m_targetError = error; }

    // Compute required bits for given range and desired error
    uint8_t requiredBits(T range, T absoluteError) const noexcept {
        if (range <= T(0)) return 8;
        T relError = absoluteError / range;
        if (relError <= T(0)) return 32;
        uint8_t bits = static_cast<uint8_t>(std::ceil(std::log2(T(1) / relError)));
        return Math::clamp(bits, uint8_t(8), uint8_t(24));
    }

    // Update error estimate from recent quantization
    void updateErrorEstimate(T observedError) noexcept {
        m_currentError = m_currentError * T(0.9) + observedError * T(0.1);
    }

    T currentError() const noexcept { return m_currentError; }
    bool needsAdjustment() const noexcept { return m_currentError > m_targetError * T(1.2); }

private:
    PrecisionMode m_precisionMode;
    T m_targetError;
    T m_currentError;
};

} // namespace Extended
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_EXTENDED_QUANTIZED_NUMERICS_H_INCLUDED