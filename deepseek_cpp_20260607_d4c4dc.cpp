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
#ifndef ORTHOTREE_CORE_MATH_VECTOR_MATH_H_INCLUDED
#define ORTHOTREE_CORE_MATH_VECTOR_MATH_H_INCLUDED

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace OrthoTree::Math {

template<typename T, std::size_t N>
class Vector {
public:
    using value_type = T;
    using size_type = std::size_t;
    using iterator = typename std::array<T, N>::iterator;
    using const_iterator = typename std::array<T, N>::const_iterator;
    
    static constexpr size_type dimension() noexcept { return N; }
    
    constexpr Vector() noexcept : m_data{} {}
    
    constexpr Vector(std::initializer_list<T> init) noexcept {
        size_type i = 0;
        for (auto it = init.begin(); it != init.end() && i < N; ++it, ++i) {
            m_data[i] = *it;
        }
        for (; i < N; ++i) m_data[i] = T{0};
    }
    
    template<typename... Args>
    constexpr Vector(Args... args) noexcept : m_data{static_cast<T>(args)...} {}
    
    constexpr Vector(const Vector&) = default;
    constexpr Vector(Vector&&) = default;
    constexpr Vector& operator=(const Vector&) = default;
    constexpr Vector& operator=(Vector&&) = default;
    
    constexpr T& operator[](size_type idx) noexcept { return m_data[idx]; }
    constexpr const T& operator[](size_type idx) const noexcept { return m_data[idx]; }
    
    constexpr T* data() noexcept { return m_data.data(); }
    constexpr const T* data() const noexcept { return m_data.data(); }
    
    constexpr size_type size() const noexcept { return N; }
    
    constexpr iterator begin() noexcept { return m_data.begin(); }
    constexpr const_iterator begin() const noexcept { return m_data.begin(); }
    constexpr const_iterator cbegin() const noexcept { return m_data.cbegin(); }
    constexpr iterator end() noexcept { return m_data.end(); }
    constexpr const_iterator end() const noexcept { return m_data.end(); }
    constexpr const_iterator cend() const noexcept { return m_data.cend(); }
    
    constexpr Vector operator-() const noexcept {
        Vector result;
        for (size_type i = 0; i < N; ++i) result[i] = -m_data[i];
        return result;
    }
    
    constexpr Vector& operator+=(const Vector& other) noexcept {
        for (size_type i = 0; i < N; ++i) m_data[i] += other[i];
        return *this;
    }
    
    constexpr Vector& operator-=(const Vector& other) noexcept {
        for (size_type i = 0; i < N; ++i) m_data[i] -= other[i];
        return *this;
    }
    
    constexpr Vector& operator*=(T scalar) noexcept {
        for (size_type i = 0; i < N; ++i) m_data[i] *= scalar;
        return *this;
    }
    
    constexpr Vector& operator/=(T scalar) noexcept {
        T inv = T{1} / scalar;
        for (size_type i = 0; i < N; ++i) m_data[i] *= inv;
        return *this;
    }
    
    constexpr Vector operator+(const Vector& other) const noexcept {
        Vector result = *this;
        result += other;
        return result;
    }
    
    constexpr Vector operator-(const Vector& other) const noexcept {
        Vector result = *this;
        result -= other;
        return result;
    }
    
    constexpr Vector operator*(T scalar) const noexcept {
        Vector result = *this;
        result *= scalar;
        return result;
    }
    
    constexpr Vector operator/(T scalar) const noexcept {
        Vector result = *this;
        result /= scalar;
        return result;
    }
    
    constexpr T dot(const Vector& other) const noexcept {
        T result = T{0};
        for (size_type i = 0; i < N; ++i) result += m_data[i] * other[i];
        return result;
    }
    
    constexpr T squaredLength() const noexcept {
        return dot(*this);
    }
    
    T length() const noexcept {
        return std::sqrt(squaredLength());
    }
    
    T normalize() noexcept {
        T len = length();
        if (len > T{0}) *this /= len;
        return len;
    }
    
    constexpr Vector normalized() const noexcept {
        Vector result = *this;
        result.normalize();
        return result;
    }
    
    constexpr T distanceTo(const Vector& other) const noexcept {
        return (*this - other).length();
    }
    
    constexpr T squaredDistanceTo(const Vector& other) const noexcept {
        return (*this - other).squaredLength();
    }
    
    constexpr T maxComponent() const noexcept {
        T maxVal = m_data[0];
        for (size_type i = 1; i < N; ++i) {
            if (m_data[i] > maxVal) maxVal = m_data[i];
        }
        return maxVal;
    }
    
    constexpr T minComponent() const noexcept {
        T minVal = m_data[0];
        for (size_type i = 1; i < N; ++i) {
            if (m_data[i] < minVal) minVal = m_data[i];
        }
        return minVal;
    }
    
    constexpr T componentSum() const noexcept {
        T sum = T{0};
        for (size_type i = 0; i < N; ++i) sum += m_data[i];
        return sum;
    }
    
    constexpr Vector abs() const noexcept {
        Vector result;
        for (size_type i = 0; i < N; ++i) {
            result[i] = m_data[i] >= T{0} ? m_data[i] : -m_data[i];
        }
        return result;
    }
    
    constexpr Vector componentWiseMin(const Vector& other) const noexcept {
        Vector result;
        for (size_type i = 0; i < N; ++i) {
            result[i] = m_data[i] < other[i] ? m_data[i] : other[i];
        }
        return result;
    }
    
    constexpr Vector componentWiseMax(const Vector& other) const noexcept {
        Vector result;
        for (size_type i = 0; i < N; ++i) {
            result[i] = m_data[i] > other[i] ? m_data[i] : other[i];
        }
        return result;
    }
    
    constexpr Vector clamp(const Vector& minVal, const Vector& maxVal) const noexcept {
        return componentWiseMax(minVal).componentWiseMin(maxVal);
    }
    
    template<typename U>
    constexpr Vector<U, N> cast() const noexcept {
        Vector<U, N> result;
        for (size_type i = 0; i < N; ++i) result[i] = static_cast<U>(m_data[i]);
        return result;
    }
    
private:
    std::array<T, N> m_data;
};

template<typename T, std::size_t N>
constexpr Vector<T, N> operator*(T scalar, const Vector<T, N>& vec) noexcept {
    return vec * scalar;
}

template<typename T, std::size_t N>
constexpr Vector<T, N> cross(const Vector<T, N>& a, const Vector<T, N>& b) noexcept {
    static_assert(N == 3, "Cross product defined only for 3D vectors");
    return Vector<T, 3>(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    );
}

template<typename T>
constexpr T cross(const Vector<T, 2>& a, const Vector<T, 2>& b) noexcept {
    return a[0] * b[1] - a[1] * b[0];
}

template<typename T, std::size_t N>
constexpr T tripleProduct(const Vector<T, N>& a, const Vector<T, N>& b, const Vector<T, N>& c) noexcept {
    static_assert(N == 3, "Triple product defined only for 3D vectors");
    return a.dot(cross(b, c));
}

template<typename T, std::size_t N>
constexpr T angleBetween(const Vector<T, N>& a, const Vector<T, N>& b) noexcept {
    T cosTheta = a.dot(b) / (a.length() * b.length());
    if (cosTheta > T{1}) cosTheta = T{1};
    if (cosTheta < T{-1}) cosTheta = T{-1};
    return std::acos(cosTheta);
}

template<typename T, std::size_t N>
constexpr Vector<T, N> lerp(const Vector<T, N>& a, const Vector<T, N>& b, T t) noexcept {
    return a + (b - a) * t;
}

template<typename T, std::size_t N>
constexpr Vector<T, N> reflect(const Vector<T, N>& incident, const Vector<T, N>& normal) noexcept {
    return incident - normal * (T{2} * incident.dot(normal));
}

template<typename T, std::size_t N>
constexpr Vector<T, N> refract(const Vector<T, N>& incident, const Vector<T, N>& normal, T eta) noexcept {
    T dotNI = incident.dot(normal);
    T k = T{1} - eta * eta * (T{1} - dotNI * dotNI);
    if (k < T{0}) return Vector<T, N>{};
    return incident * eta - normal * (eta * dotNI + std::sqrt(k));
}

template<typename T, std::size_t N>
constexpr Vector<T, N> projectOnto(const Vector<T, N>& a, const Vector<T, N>& b) noexcept {
    return b * (a.dot(b) / b.squaredLength());
}

template<typename T, std::size_t N>
constexpr Vector<T, N> rejectFrom(const Vector<T, N>& a, const Vector<T, N>& b) noexcept {
    return a - projectOnto(a, b);
}

template<typename T, std::size_t N>
constexpr bool isNull(const Vector<T, N>& v, T eps = T{1e-8}) noexcept {
    return v.squaredLength() <= eps * eps;
}

template<typename T, std::size_t N>
constexpr bool isOrthogonal(const Vector<T, N>& a, const Vector<T, N>& b, T eps = T{1e-8}) noexcept {
    return std::abs(a.dot(b)) <= eps;
}

template<typename T, std::size_t N>
constexpr bool isParallel(const Vector<T, N>& a, const Vector<T, N>& b, T eps = T{1e-8}) noexcept {
    T dotAA = a.squaredLength();
    T dotBB = b.squaredLength();
    T dotAB = a.dot(b);
    T crossNormSq = dotAA * dotBB - dotAB * dotAB;
    return crossNormSq <= eps * eps * dotAA * dotBB;
}

template<typename T>
using Vector2 = Vector<T, 2>;
template<typename T>
using Vector3 = Vector<T, 3>;
template<typename T>
using Vector4 = Vector<T, 4>;

using Vec2f = Vector<float, 2>;
using Vec3f = Vector<float, 3>;
using Vec4f = Vector<float, 4>;
using Vec2d = Vector<double, 2>;
using Vec3d = Vector<double, 3>;
using Vec4d = Vector<double, 4>;
using Vec2i = Vector<int, 2>;
using Vec3i = Vector<int, 3>;
using Vec4i = Vector<int, 4>;
using Vec2u = Vector<unsigned, 2>;
using Vec3u = Vector<unsigned, 3>;
using Vec4u = Vector<unsigned, 4>;

} // namespace OrthoTree::Math

#endif // ORTHOTREE_CORE_MATH_VECTOR_MATH_H_INCLUDED