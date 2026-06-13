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
#pragma once
#ifndef ORTHOTREE_CORE_MATH_INTERVAL_ARITHMETIC_H_INCLUDED
#define ORTHOTREE_CORE_MATH_INTERVAL_ARITHMETIC_H_INCLUDED

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace OrthoTree::Math {

template<typename T>
class Interval {
public:
    static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
                  "Interval type must be numeric");
    
    using value_type = T;
    
    constexpr Interval() noexcept : m_low(T{0}), m_high(T{0}) {}
    constexpr Interval(T value) noexcept : m_low(value), m_high(value) {}
    constexpr Interval(T low, T high) noexcept : m_low(low), m_high(high) {
        if (m_low > m_high) std::swap(m_low, m_high);
    }
    
    constexpr T low() const noexcept { return m_low; }
    constexpr T high() const noexcept { return m_high; }
    
    constexpr T center() const noexcept { return (m_low + m_high) / T{2}; }
    constexpr T radius() const noexcept { return (m_high - m_low) / T{2}; }
    constexpr T width() const noexcept { return m_high - m_low; }
    
    constexpr bool empty() const noexcept { return m_low > m_high; }
    
    constexpr bool contains(T value) const noexcept {
        return m_low <= value && value <= m_high;
    }
    
    constexpr bool contains(const Interval& other) const noexcept {
        return m_low <= other.m_low && other.m_high <= m_high;
    }
    
    constexpr bool overlaps(const Interval& other) const noexcept {
        return m_low <= other.m_high && other.m_low <= m_high;
    }
    
    constexpr Interval intersection(const Interval& other) const noexcept {
        return Interval(std::max(m_low, other.m_low), std::min(m_high, other.m_high));
    }
    
    constexpr Interval hull(const Interval& other) const noexcept {
        return Interval(std::min(m_low, other.m_low), std::max(m_high, other.m_high));
    }
    
    constexpr Interval& extend(T value) noexcept {
        if (value < m_low) m_low = value;
        if (value > m_high) m_high = value;
        return *this;
    }
    
    constexpr Interval& extend(const Interval& other) noexcept {
        if (other.m_low < m_low) m_low = other.m_low;
        if (other.m_high > m_high) m_high = other.m_high;
        return *this;
    }
    
    constexpr Interval& operator+=(T value) noexcept {
        m_low += value;
        m_high += value;
        return *this;
    }
    
    constexpr Interval& operator-=(T value) noexcept {
        m_low -= value;
        m_high -= value;
        return *this;
    }
    
    constexpr Interval& operator*=(T value) noexcept {
        if (value >= T{0}) {
            m_low *= value;
            m_high *= value;
        } else {
            T tmp = m_low * value;
            m_low = m_high * value;
            m_high = tmp;
        }
        return *this;
    }
    
    constexpr Interval& operator/=(T value) noexcept {
        if (value == T{0}) return *this;
        T inv = T{1} / value;
        return *this *= inv;
    }
    
    constexpr Interval operator+(T value) const noexcept { return Interval(*this) += value; }
    constexpr Interval operator-(T value) const noexcept { return Interval(*this) -= value; }
    constexpr Interval operator*(T value) const noexcept { return Interval(*this) *= value; }
    constexpr Interval operator/(T value) const noexcept { return Interval(*this) /= value; }
    
    friend constexpr Interval operator+(T value, const Interval& interval) noexcept {
        return interval + value;
    }
    
    friend constexpr Interval operator-(T value, const Interval& interval) noexcept {
        return Interval(value) - interval;
    }
    
    friend constexpr Interval operator*(T value, const Interval& interval) noexcept {
        return interval * value;
    }
    
    constexpr Interval operator+(const Interval& other) const noexcept {
        return Interval(m_low + other.m_low, m_high + other.m_high);
    }
    
    constexpr Interval operator-(const Interval& other) const noexcept {
        return Interval(m_low - other.m_high, m_high - other.m_low);
    }
    
    constexpr Interval operator*(const Interval& other) const noexcept {
        T products[4] = {
            m_low * other.m_low, m_low * other.m_high,
            m_high * other.m_low, m_high * other.m_high
        };
        return Interval(
            std::min({products[0], products[1], products[2], products[3]}),
            std::max({products[0], products[1], products[2], products[3]})
        );
    }
    
    constexpr Interval operator/(const Interval& other) const noexcept {
        if (other.contains(T{0})) {
            T divPosInf = std::numeric_limits<T>::infinity();
            T divNegInf = -divPosInf;
            if (other.m_low < T{0} && other.m_high > T{0}) {
                return Interval(divNegInf, divPosInf);
            } else if (other.m_high == T{0}) {
                return Interval(divNegInf, m_high / other.m_low);
            } else {
                return Interval(m_low / other.m_high, divPosInf);
            }
        }
        T invLow = T{1} / other.m_low;
        T invHigh = T{1} / other.m_high;
        if (invLow > invHigh) std::swap(invLow, invHigh);
        return *this * Interval(invLow, invHigh);
    }
    
    constexpr Interval sqrt() const noexcept {
        if (m_high < T{0}) return Interval(T{0}, T{0});
        T sqrtLow = m_low <= T{0} ? T{0} : std::sqrt(m_low);
        return Interval(sqrtLow, std::sqrt(m_high));
    }
    
    constexpr Interval abs() const noexcept {
        if (m_low >= T{0}) return *this;
        if (m_high <= T{0}) return Interval(-m_high, -m_low);
        return Interval(T{0}, std::max(-m_low, m_high));
    }
    
    constexpr Interval pow(int exponent) const noexcept {
        if (exponent == 0) return Interval(T{1});
        if (exponent < 0) {
            Interval pos = *this;
            if (pos.contains(T{0})) return Interval(
                -std::numeric_limits<T>::infinity(),
                std::numeric_limits<T>::infinity()
            );
            Interval result = pos.pow(-exponent);
            return Interval(T{1} / result.m_high, T{1} / result.m_low);
        }
        if (exponent % 2 == 0) {
            T absLow = std::max(std::abs(m_low), std::abs(m_high));
            T pLow = T{0};
            T pHigh = std::pow(absLow, exponent);
            return Interval(pLow, pHigh);
        }
        return Interval(std::pow(m_low, exponent), std::pow(m_high, exponent));
    }
    
    constexpr Interval sin() const noexcept {
        const T twoPi = T{2} * T{3.14159265358979323846};
        T fLow = std::fmod(m_low, twoPi);
        T fHigh = fLow + width();
        if (fHigh > twoPi) {
            return Interval(T{-1}, T{1});
        }
        T sinLow = std::sin(fLow);
        T sinHigh = std::sin(fHigh);
        if (sinLow > sinHigh) std::swap(sinLow, sinHigh);
        if (fLow <= T{1.5707963267948966} && fHigh >= T{1.5707963267948966}) sinHigh = T{1};
        if (fLow <= T{4.71238898038469} && fHigh >= T{4.71238898038469}) sinLow = T{-1};
        return Interval(sinLow, sinHigh);
    }
    
    constexpr Interval cos() const noexcept {
        return (Interval(T{1.5707963267948966}) - *this).sin();
    }
    
    constexpr Interval tan() const noexcept {
        Interval result = sin() / cos();
        return result;
    }
    
    constexpr Interval log() const noexcept {
        if (m_high <= T{0}) return Interval(-std::numeric_limits<T>::infinity());
        T logLow = m_low <= T{0} ? -std::numeric_limits<T>::infinity() : std::log(m_low);
        return Interval(logLow, std::log(m_high));
    }
    
    constexpr Interval exp() const noexcept {
        return Interval(std::exp(m_low), std::exp(m_high));
    }
    
    constexpr Interval mag() const noexcept {
        return Interval(std::max(std::abs(m_low), std::abs(m_high)));
    }
    
    constexpr Interval mig() const noexcept {
        if (m_low > T{0}) return Interval(m_low);
        if (m_high < T{0}) return Interval(-m_high);
        return Interval(T{0});
    }
    
    constexpr bool operator==(const Interval& other) const noexcept {
        return m_low == other.m_low && m_high == other.m_high;
    }
    
    constexpr bool operator!=(const Interval& other) const noexcept {
        return !(*this == other);
    }
    
private:
    T m_low;
    T m_high;
};

template<typename T>
constexpr Interval<T> hull(const Interval<T>& a, const Interval<T>& b) noexcept {
    return a.hull(b);
}

template<typename T>
constexpr Interval<T> intersect(const Interval<T>& a, const Interval<T>& b) noexcept {
    return a.intersection(b);
}

template<typename T, typename... Args>
constexpr Interval<T> hull(const Interval<T>& first, const Args&... rest) noexcept {
    Interval<T> result = first;
    ((result.extend(rest)), ...);
    return result;
}

template<typename T>
using Interval1 = Interval<T>;

template<typename T, std::size_t N>
class Hyperrectangle {
public:
    using value_type = T;
    using size_type = std::size_t;
    
    static constexpr size_type dimension() noexcept { return N; }
    
    constexpr Hyperrectangle() noexcept : m_intervals{} {}
    constexpr Hyperrectangle(const Interval<T>& interval) noexcept {
        for (size_type i = 0; i < N; ++i) m_intervals[i] = interval;
    }
    constexpr Hyperrectangle(std::initializer_list<Interval<T>> init) noexcept {
        size_type i = 0;
        for (auto it = init.begin(); it != init.end() && i < N; ++it, ++i) {
            m_intervals[i] = *it;
        }
        for (; i < N; ++i) m_intervals[i] = Interval<T>(T{0});
    }
    
    constexpr Interval<T>& operator[](size_type idx) noexcept { return m_intervals[idx]; }
    constexpr const Interval<T>& operator[](size_type idx) const noexcept { return m_intervals[idx]; }
    
    constexpr bool contains(const std::array<T, N>& point) const noexcept {
        for (size_type i = 0; i < N; ++i) {
            if (!m_intervals[i].contains(point[i])) return false;
        }
        return true;
    }
    
    constexpr bool contains(const Hyperrectangle& other) const noexcept {
        for (size_type i = 0; i < N; ++i) {
            if (!m_intervals[i].contains(other.m_intervals[i])) return false;
        }
        return true;
    }
    
    constexpr bool overlaps(const Hyperrectangle& other) const noexcept {
        for (size_type i = 0; i < N; ++i) {
            if (!m_intervals[i].overlaps(other.m_intervals[i])) return false;
        }
        return true;
    }
    
    constexpr Hyperrectangle intersection(const Hyperrectangle& other) const noexcept {
        Hyperrectangle result;
        for (size_type i = 0; i < N; ++i) {
            result[i] = m_intervals[i].intersection(other[i]);
        }
        return result;
    }
    
    constexpr Hyperrectangle hull(const Hyperrectangle& other) const noexcept {
        Hyperrectangle result;
        for (size_type i = 0; i < N; ++i) {
            result[i] = m_intervals[i].hull(other[i]);
        }
        return result;
    }
    
    constexpr std::array<T, N> center() const noexcept {
        std::array<T, N> result;
        for (size_type i = 0; i < N; ++i) result[i] = m_intervals[i].center();
        return result;
    }
    
    constexpr T volume() const noexcept {
        T vol = T{1};
        for (size_type i = 0; i < N; ++i) vol *= m_intervals[i].width();
        return vol;
    }
    
    constexpr T squaredDiameter() const noexcept {
        T maxWidth = T{0};
        for (size_type i = 0; i < N; ++i) {
            T w = m_intervals[i].width();
            if (w > maxWidth) maxWidth = w;
        }
        return maxWidth;
    }
    
private:
    std::array<Interval<T>, N> m_intervals;
};

template<typename T, std::size_t N>
constexpr Hyperrectangle<T, N> hull(const Hyperrectangle<T, N>& a, const Hyperrectangle<T, N>& b) noexcept {
    return a.hull(b);
}

using Intervalf = Interval<float>;
using Intervald = Interval<double>;

} // namespace OrthoTree::Math

#endif // ORTHOTREE_CORE_MATH_INTERVAL_ARITHMETIC_H_INCLUDED