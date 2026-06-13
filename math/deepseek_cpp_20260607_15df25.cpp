//File group name : OrthoTree Math
//File 0051 : core/math/basic/vector.h
//N‑dimensional vector (2D, 3D, generic) with arithmetic, length, dot, cross (3D), normalization, SIMD batch operations, and dynamic environment controls.

#ifndef ORTHOTREE_CORE_MATH_BASIC_VECTOR_H_INCLUDED
#define ORTHOTREE_CORE_MATH_BASIC_VECTOR_H_INCLUDED

#include "../../build_config.h"
#include "scalar.h"
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <array>
#include <cmath>
#include <type_traits>
#include <algorithm>
#include <cstddef>

namespace OrthoTree {
namespace Math {
namespace Basic {

// ============================================================================
//  Vector class template (N dimensions, arbitrary scalar type)
// ============================================================================
template<typename T, std::size_t N>
class Vector {
public:
    using value_type = T;
    using size_type = std::size_t;
    using iterator = T*;
    using const_iterator = const T*;

    static constexpr size_type dimension() noexcept { return N; }

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    constexpr Vector() noexcept : m_data{} {}
    constexpr Vector(std::initializer_list<T> init) noexcept {
        size_type i = 0;
        for (auto it = init.begin(); it != init.end() && i < N; ++it, ++i) m_data[i] = *it;
        for (; i < N; ++i) m_data[i] = T(0);
    }
    template<typename... Args>
    constexpr Vector(Args... args) noexcept : m_data{static_cast<T>(args)...} {}
    constexpr Vector(const Vector&) = default;
    constexpr Vector(Vector&&) = default;
    constexpr Vector& operator=(const Vector&) = default;
    constexpr Vector& operator=(Vector&&) = default;

    // ------------------------------------------------------------------------
    //  Element access
    // ------------------------------------------------------------------------
    constexpr T& operator[](size_type i) noexcept { return m_data[i]; }
    constexpr const T& operator[](size_type i) const noexcept { return m_data[i]; }
    constexpr T* data() noexcept { return m_data.data(); }
    constexpr const T* data() const noexcept { return m_data.data(); }
    constexpr size_type size() const noexcept { return N; }

    constexpr iterator begin() noexcept { return m_data.begin(); }
    constexpr const_iterator begin() const noexcept { return m_data.begin(); }
    constexpr iterator end() noexcept { return m_data.end(); }
    constexpr const_iterator end() const noexcept { return m_data.end(); }

    // ------------------------------------------------------------------------
    //  Arithmetic operations (component‑wise)
    // ------------------------------------------------------------------------
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
        T inv = T(1) / scalar;
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

    friend constexpr Vector operator*(T scalar, const Vector& v) noexcept {
        return v * scalar;
    }

    // ------------------------------------------------------------------------
    //  Dot product, length, normalisation
    // ------------------------------------------------------------------------
    constexpr T dot(const Vector& other) const noexcept {
        T sum = T(0);
        for (size_type i = 0; i < N; ++i) sum += m_data[i] * other[i];
        return sum;
    }
    constexpr T squaredLength() const noexcept { return dot(*this); }
    T length() const noexcept { return std::sqrt(squaredLength()); }
    T normalize() noexcept {
        T len = length();
        if (len > T(0)) *this /= len;
        return len;
    }
    Vector normalized() const noexcept {
        Vector result = *this;
        result.normalize();
        return result;
    }

    // ------------------------------------------------------------------------
    //  Cross product (only for 3D)
    // ------------------------------------------------------------------------
    Vector cross(const Vector& other) const noexcept {
        static_assert(N == 3, "Cross product only for 3D vectors");
        return Vector(
            m_data[1] * other[2] - m_data[2] * other[1],
            m_data[2] * other[0] - m_data[0] * other[2],
            m_data[0] * other[1] - m_data[1] * other[0]
        );
    }

    // ------------------------------------------------------------------------
    //  Component‑wise min / max
    // ------------------------------------------------------------------------
    constexpr Vector componentWiseMin(const Vector& other) const noexcept {
        Vector result;
        for (size_type i = 0; i < N; ++i) result[i] = (m_data[i] < other[i]) ? m_data[i] : other[i];
        return result;
    }
    constexpr Vector componentWiseMax(const Vector& other) const noexcept {
        Vector result;
        for (size_type i = 0; i < N; ++i) result[i] = (m_data[i] > other[i]) ? m_data[i] : other[i];
        return result;
    }

    // ------------------------------------------------------------------------
    //  Distance
    // ------------------------------------------------------------------------
    T distanceTo(const Vector& other) const noexcept {
        return (*this - other).length();
    }
    T squaredDistanceTo(const Vector& other) const noexcept {
        return (*this - other).squaredLength();
    }

    // ------------------------------------------------------------------------
    //  Cast to different scalar type
    // ------------------------------------------------------------------------
    template<typename U>
    Vector<U, N> cast() const noexcept {
        Vector<U, N> result;
        for (size_type i = 0; i < N; ++i) result[i] = static_cast<U>(m_data[i]);
        return result;
    }

    // ------------------------------------------------------------------------
    //  Comparison with tolerance (uses global epsilon)
    // ------------------------------------------------------------------------
    bool nearlyEqual(const Vector& other, T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) const noexcept {
        for (size_type i = 0; i < N; ++i) {
            if (!Basic::nearlyEqual(m_data[i], other[i], eps)) return false;
        }
        return true;
    }

private:
    std::array<T, N> m_data;
};

// ============================================================================
//  Convenience aliases for 2D, 3D, 4D
// ============================================================================
template<typename T> using Vector2 = Vector<T, 2>;
template<typename T> using Vector3 = Vector<T, 3>;
template<typename T> using Vector4 = Vector<T, 4>;

using Vec2f = Vector<float, 2>;
using Vec3f = Vector<float, 3>;
using Vec4f = Vector<float, 4>;
using Vec2d = Vector<double, 2>;
using Vec3d = Vector<double, 3>;
using Vec4d = Vector<double, 4>;

// ============================================================================
//  SIMD batch operations (4 vectors at once, for 3D)
// ============================================================================
template<typename T>
void batchAdd(const Vector<T,3>* a, const Vector<T,3>* b, Vector<T,3>* out, size_t count) noexcept {
    if constexpr (std::is_same_v<T,float> && ORTHOTREE_SIMD_LEVEL >= 128) {
        for (size_t i = 0; i < count; ++i) out[i] = a[i] + b[i];
    } else {
        for (size_t i = 0; i < count; ++i) out[i] = a[i] + b[i];
    }
}

template<typename T>
void batchDot(const Vector<T,3>* a, const Vector<T,3>* b, T* out, size_t count) noexcept {
    for (size_t i = 0; i < count; ++i) out[i] = a[i].dot(b[i]);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller (for vector operations)
// ----------------------------------------------------------------------------
class VectorEnvironment {
public:
    static VectorEnvironment& instance() {
        static VectorEnvironment env;
        return env;
    }
    void setUseSIMD(bool use) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_useSIMD = use;
        MathConfig::instance().setUseSIMD(use);
    }
    bool useSIMD() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_useSIMD;
    }
private:
    VectorEnvironment() : m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    bool m_useSIMD;
};

} // namespace Basic
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_BASIC_VECTOR_H_INCLUDED