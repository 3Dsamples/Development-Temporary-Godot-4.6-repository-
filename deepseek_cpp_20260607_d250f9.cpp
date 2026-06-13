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
#ifndef ORTHOTREE_CORE_MATH_QUATERNION_H_INCLUDED
#define ORTHOTREE_CORE_MATH_QUATERNION_H_INCLUDED

#include "vector_math.h"
#include <cmath>

namespace OrthoTree::Math {

template<typename T>
class Quaternion {
public:
    using value_type = T;
    
    constexpr Quaternion() noexcept : m_w(T{1}), m_x(T{0}), m_y(T{0}), m_z(T{0}) {}
    constexpr Quaternion(T w, T x, T y, T z) noexcept : m_w(w), m_x(x), m_y(y), m_z(z) {}
    constexpr Quaternion(T w, const Vector<T, 3>& v) noexcept : m_w(w), m_x(v[0]), m_y(v[1]), m_z(v[2]) {}
    
    constexpr Quaternion(const Quaternion&) = default;
    constexpr Quaternion(Quaternion&&) = default;
    constexpr Quaternion& operator=(const Quaternion&) = default;
    constexpr Quaternion& operator=(Quaternion&&) = default;
    
    constexpr T w() const noexcept { return m_w; }
    constexpr T x() const noexcept { return m_x; }
    constexpr T y() const noexcept { return m_y; }
    constexpr T z() const noexcept { return m_z; }
    constexpr Vector<T, 3> vec() const noexcept { return Vector<T, 3>(m_x, m_y, m_z); }
    
    constexpr void setW(T w) noexcept { m_w = w; }
    constexpr void setX(T x) noexcept { m_x = x; }
    constexpr void setY(T y) noexcept { m_y = y; }
    constexpr void setZ(T z) noexcept { m_z = z; }
    constexpr void set(T w, T x, T y, T z) noexcept { m_w = w; m_x = x; m_y = y; m_z = z; }
    constexpr void set(T w, const Vector<T, 3>& v) noexcept { m_w = w; m_x = v[0]; m_y = v[1]; m_z = v[2]; }
    
    constexpr T squaredNorm() const noexcept {
        return m_w * m_w + m_x * m_x + m_y * m_y + m_z * m_z;
    }
    
    T norm() const noexcept {
        return std::sqrt(squaredNorm());
    }
    
    constexpr Quaternion conjugate() const noexcept {
        return Quaternion(m_w, -m_x, -m_y, -m_z);
    }
    
    constexpr Quaternion inverse() const noexcept {
        T invSqNorm = T{1} / squaredNorm();
        return Quaternion(m_w * invSqNorm, -m_x * invSqNorm, -m_y * invSqNorm, -m_z * invSqNorm);
    }
    
    T normalize() noexcept {
        T n = norm();
        if (n > T{0}) {
            T invN = T{1} / n;
            m_w *= invN;
            m_x *= invN;
            m_y *= invN;
            m_z *= invN;
        }
        return n;
    }
    
    constexpr Quaternion normalized() const noexcept {
        Quaternion result = *this;
        result.normalize();
        return result;
    }
    
    constexpr Quaternion operator-() const noexcept {
        return Quaternion(-m_w, -m_x, -m_y, -m_z);
    }
    
    constexpr Quaternion operator+(const Quaternion& other) const noexcept {
        return Quaternion(
            m_w + other.m_w,
            m_x + other.m_x,
            m_y + other.m_y,
            m_z + other.m_z
        );
    }
    
    constexpr Quaternion operator-(const Quaternion& other) const noexcept {
        return Quaternion(
            m_w - other.m_w,
            m_x - other.m_x,
            m_y - other.m_y,
            m_z - other.m_z
        );
    }
    
    constexpr Quaternion operator*(T scalar) const noexcept {
        return Quaternion(m_w * scalar, m_x * scalar, m_y * scalar, m_z * scalar);
    }
    
    constexpr Quaternion operator*(const Quaternion& other) const noexcept {
        return Quaternion(
            m_w * other.m_w - m_x * other.m_x - m_y * other.m_y - m_z * other.m_z,
            m_w * other.m_x + m_x * other.m_w + m_y * other.m_z - m_z * other.m_y,
            m_w * other.m_y - m_x * other.m_z + m_y * other.m_w + m_z * other.m_x,
            m_w * other.m_z + m_x * other.m_y - m_y * other.m_x + m_z * other.m_w
        );
    }
    
    constexpr Quaternion& operator+=(const Quaternion& other) noexcept {
        *this = *this + other;
        return *this;
    }
    
    constexpr Quaternion& operator-=(const Quaternion& other) noexcept {
        *this = *this - other;
        return *this;
    }
    
    constexpr Quaternion& operator*=(T scalar) noexcept {
        *this = *this * scalar;
        return *this;
    }
    
    constexpr Quaternion& operator*=(const Quaternion& other) noexcept {
        *this = *this * other;
        return *this;
    }
    
    constexpr Vector<T, 3> rotate(const Vector<T, 3>& v) const noexcept {
        Quaternion p(T{0}, v);
        Quaternion r = (*this) * p * conjugate();
        return Vector<T, 3>(r.x(), r.y(), r.z());
    }
    
    constexpr Vector<T, 3> rotateInverse(const Vector<T, 3>& v) const noexcept {
        Quaternion p(T{0}, v);
        Quaternion r = conjugate() * p * (*this);
        return Vector<T, 3>(r.x(), r.y(), r.z());
    }
    
    static constexpr Quaternion identity() noexcept {
        return Quaternion(T{1}, T{0}, T{0}, T{0});
    }
    
    static Quaternion fromAxisAngle(const Vector<T, 3>& axis, T angle) noexcept {
        T halfAngle = angle * T{0.5};
        T s = std::sin(halfAngle);
        Vector<T, 3> n = axis.normalized();
        return Quaternion(std::cos(halfAngle), n * s);
    }
    
    static Quaternion fromEuler(T roll, T pitch, T yaw) noexcept {
        T cr = std::cos(roll * T{0.5});
        T sr = std::sin(roll * T{0.5});
        T cp = std::cos(pitch * T{0.5});
        T sp = std::sin(pitch * T{0.5});
        T cy = std::cos(yaw * T{0.5});
        T sy = std::sin(yaw * T{0.5});
        
        return Quaternion(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        );
    }
    
    static Quaternion fromTwoVectors(const Vector<T, 3>& u, const Vector<T, 3>& v) noexcept {
        Vector<T, 3> n = cross(u, v);
        T w = u.dot(v);
        T normN = n.length();
        if (normN > T{0}) n /= normN;
        T angle = std::atan2(normN, w);
        return fromAxisAngle(n, angle);
    }
    
    static Quaternion slerp(const Quaternion& a, const Quaternion& b, T t) noexcept {
        T cosTheta = a.w() * b.w() + a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
        
        Quaternion b2 = b;
        if (cosTheta < T{0}) {
            b2 = -b;
            cosTheta = -cosTheta;
        }
        
        if (cosTheta > T{0.999999}) {
            Quaternion result = a + (b2 - a) * t;
            result.normalize();
            return result;
        }
        
        T theta = std::acos(cosTheta);
        T sinTheta = std::sin(theta);
        T w1 = std::sin((T{1} - t) * theta) / sinTheta;
        T w2 = std::sin(t * theta) / sinTheta;
        
        return a * w1 + b2 * w2;
    }
    
    static Quaternion nlerp(const Quaternion& a, const Quaternion& b, T t) noexcept {
        T cosTheta = a.w() * b.w() + a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
        if (cosTheta < T{0}) {
            Quaternion result = (a * (T{1} - t) + (-b) * t);
            result.normalize();
            return result;
        }
        Quaternion result = a * (T{1} - t) + b * t;
        result.normalize();
        return result;
    }
    
    constexpr bool operator==(const Quaternion& other) const noexcept {
        return m_w == other.m_w && m_x == other.m_x && m_y == other.m_y && m_z == other.m_z;
    }
    
    constexpr bool operator!=(const Quaternion& other) const noexcept {
        return !(*this == other);
    }
    
    T roll() const noexcept {
        T sinr_cosp = T{2} * (m_w * m_x + m_y * m_z);
        T cosr_cosp = T{1} - T{2} * (m_x * m_x + m_y * m_y);
        return std::atan2(sinr_cosp, cosr_cosp);
    }
    
    T pitch() const noexcept {
        T sinp = T{2} * (m_w * m_y - m_z * m_x);
        if (std::abs(sinp) >= T{1}) {
            return std::copysign(T{3.14159265358979323846} / T{2}, sinp);
        }
        return std::asin(sinp);
    }
    
    T yaw() const noexcept {
        T siny_cosp = T{2} * (m_w * m_z + m_x * m_y);
        T cosy_cosp = T{1} - T{2} * (m_y * m_y + m_z * m_z);
        return std::atan2(siny_cosp, cosy_cosp);
    }
    
private:
    T m_w, m_x, m_y, m_z;
};

template<typename T>
constexpr Quaternion<T> operator*(T scalar, const Quaternion<T>& q) noexcept {
    return q * scalar;
}

using Quaternionf = Quaternion<float>;
using Quaterniond = Quaternion<double>;

} // namespace OrthoTree::Math

#endif // ORTHOTREE_CORE_MATH_QUATERNION_H_INCLUDED