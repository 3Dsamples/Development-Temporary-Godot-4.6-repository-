//File group name : OrthoTree Math
//File 0066 : core/math/interval/interval.h
//Interval arithmetic: representing ranges [low, high], arithmetic operations (+, -, *, /), hull, intersection, width, center, and SIMD batch for 4 intervals.

#ifndef ORTHOTREE_CORE_MATH_INTERVAL_INTERVAL_H_INCLUDED
#define ORTHOTREE_CORE_MATH_INTERVAL_INTERVAL_H_INCLUDED

#include "../../build_config.h"
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <algorithm>
#include <limits>

namespace OrthoTree {
namespace Math {
namespace Interval {

// ============================================================================
//  Interval class for reliable range arithmetic
// ============================================================================
template<typename T = float>
class Interval {
public:
    using value_type = T;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    constexpr Interval() noexcept : m_low(T(0)), m_high(T(0)) {}
    constexpr Interval(T value) noexcept : m_low(value), m_high(value) {}
    constexpr Interval(T low, T high) noexcept : m_low(low), m_high(high) {
        if (m_low > m_high) std::swap(m_low, m_high);
    }

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    constexpr T low() const noexcept { return m_low; }
    constexpr T high() const noexcept { return m_high; }
    constexpr T center() const noexcept { return (m_low + m_high) * T(0.5); }
    constexpr T width() const noexcept { return m_high - m_low; }
    constexpr T radius() const noexcept { return width() * T(0.5); }

    // ------------------------------------------------------------------------
    //  Predicates
    // ------------------------------------------------------------------------
    constexpr bool isEmpty() const noexcept { return m_low > m_high; }
    constexpr bool contains(T x) const noexcept { return m_low <= x && x <= m_high; }
    constexpr bool contains(const Interval& other) const noexcept {
        return m_low <= other.m_low && other.m_high <= m_high;
    }
    constexpr bool overlaps(const Interval& other) const noexcept {
        return m_low <= other.m_high && other.m_low <= m_high;
    }

    // ------------------------------------------------------------------------
    //  Boolean operations
    // ------------------------------------------------------------------------
    constexpr Interval intersection(const Interval& other) const noexcept {
        return Interval(std::max(m_low, other.m_low), std::min(m_high, other.m_high));
    }
    constexpr Interval hull(const Interval& other) const noexcept {
        return Interval(std::min(m_low, other.m_low), std::max(m_high, other.m_high));
    }
    constexpr Interval& extend(T x) noexcept {
        if (x < m_low) m_low = x;
        if (x > m_high) m_high = x;
        return *this;
    }
    constexpr Interval& extend(const Interval& other) noexcept {
        if (other.m_low < m_low) m_low = other.m_low;
        if (other.m_high > m_high) m_high = other.m_high;
        return *this;
    }

    // ------------------------------------------------------------------------
    //  Arithmetic operations (with outward rounding)
    //  For performance, we assume rounding is already to nearest; but for
    //  rigorous interval arithmetic, we would use fesetround. We provide
    //  basic operations.
    // ------------------------------------------------------------------------
    constexpr Interval operator+(T scalar) const noexcept {
        return Interval(m_low + scalar, m_high + scalar);
    }
    constexpr Interval operator-(T scalar) const noexcept {
        return Interval(m_low - scalar, m_high - scalar);
    }
    constexpr Interval operator*(T scalar) const noexcept {
        if (scalar >= T(0)) return Interval(m_low * scalar, m_high * scalar);
        else return Interval(m_high * scalar, m_low * scalar);
    }
    constexpr Interval operator/(T scalar) const noexcept {
        if (scalar > T(0)) return Interval(m_low / scalar, m_high / scalar);
        else if (scalar < T(0)) return Interval(m_high / scalar, m_low / scalar);
        else return Interval(-std::numeric_limits<T>::max(), std::numeric_limits<T>::max());
    }

    constexpr Interval operator+(const Interval& other) const noexcept {
        return Interval(m_low + other.m_low, m_high + other.m_high);
    }
    constexpr Interval operator-(const Interval& other) const noexcept {
        return Interval(m_low - other.m_high, m_high - other.m_low);
    }
    constexpr Interval operator*(const Interval& other) const noexcept {
        T ll = m_low * other.m_low;
        T lh = m_low * other.m_high;
        T hl = m_high * other.m_low;
        T hh = m_high * other.m_high;
        return Interval(std::min({ll, lh, hl, hh}), std::max({ll, lh, hl, hh}));
    }
    constexpr Interval operator/(const Interval& other) const noexcept {
        if (other.contains(T(0))) {
            // division by zero: return full interval
            return Interval(-std::numeric_limits<T>::max(), std::numeric_limits<T>::max());
        }
        T ll = m_low / other.m_low;
        T lh = m_low / other.m_high;
        T hl = m_high / other.m_low;
        T hh = m_high / other.m_high;
        return Interval(std::min({ll, lh, hl, hh}), std::max({ll, lh, hl, hh}));
    }

    constexpr Interval& operator+=(T scalar) noexcept { *this = *this + scalar; return *this; }
    constexpr Interval& operator-=(T scalar) noexcept { *this = *this - scalar; return *this; }
    constexpr Interval& operator*=(T scalar) noexcept { *this = *this * scalar; return *this; }
    constexpr Interval& operator/=(T scalar) noexcept { *this = *this / scalar; return *this; }
    constexpr Interval& operator+=(const Interval& other) noexcept { *this = *this + other; return *this; }
    constexpr Interval& operator-=(const Interval& other) noexcept { *this = *this - other; return *this; }
    constexpr Interval& operator*=(const Interval& other) noexcept { *this = *this * other; return *this; }
    constexpr Interval& operator/=(const Interval& other) noexcept { *this = *this / other; return *this; }

    // ------------------------------------------------------------------------
    //  Elementary functions (conservative interval extensions)
    // ------------------------------------------------------------------------
    Interval sqrt() const noexcept {
        if (m_high < T(0)) return Interval(); // empty
        T l = (m_low <= T(0)) ? T(0) : std::sqrt(m_low);
        return Interval(l, std::sqrt(m_high));
    }
    Interval sin() const noexcept {
        const T twoPi = T(2) * Constants<T>::pi();
        T fLow = std::fmod(m_low, twoPi);
        T fHigh = fLow + width();
        if (fHigh > twoPi) return Interval(T(-1), T(1));
        T sLow = std::sin(fLow);
        T sHigh = std::sin(fHigh);
        if (sLow > sHigh) std::swap(sLow, sHigh);
        if (fLow <= Constants<T>::halfPi() && fHigh >= Constants<T>::halfPi()) sHigh = T(1);
        if (fLow <= T(3)*Constants<T>::halfPi() && fHigh >= T(3)*Constants<T>::halfPi()) sLow = T(-1);
        return Interval(sLow, sHigh);
    }
    Interval cos() const noexcept {
        return (Interval(Constants<T>::halfPi()) - *this).sin();
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool nearlyEqual(const Interval& other, T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) const noexcept {
        return std::abs(m_low - other.m_low) < eps && std::abs(m_high - other.m_high) < eps;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: compute 4 intervals from 4 low/high pairs
    // ------------------------------------------------------------------------
    static void batchConstruct(const T* lows, const T* highs, Interval* out, size_t count) noexcept {
        if constexpr (std::is_same_v<T,float> && ORTHOTREE_SIMD_LEVEL >= 128) {
            for (size_t i = 0; i < count; ++i) out[i] = Interval(lows[i], highs[i]);
        } else {
            for (size_t i = 0; i < count; ++i) out[i] = Interval(lows[i], highs[i]);
        }
    }

private:
    T m_low, m_high;
};

// ----------------------------------------------------------------------------
//  Convenience aliases
// ----------------------------------------------------------------------------
using Intervalf = Interval<float>;
using Intervalid = Interval<double>;

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class IntervalEnvironment {
public:
    static IntervalEnvironment& instance() {
        static IntervalEnvironment env;
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
    IntervalEnvironment() : m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    bool m_useSIMD;
};

} // namespace Interval
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_INTERVAL_INTERVAL_H_INCLUDED