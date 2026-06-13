//File group name : OrthoTree Math
//File 0069 : core/math/basic/quaternion.h
//Quaternion for 3D rotations: multiplication, conjugation, norm, rotation of vectors, conversion to/from matrix, Euler angles, spherical linear interpolation (SLERP), and SIMD batch operations.

#ifndef ORTHOTREE_CORE_MATH_BASIC_QUATERNION_H_INCLUDED
#define ORTHOTREE_CORE_MATH_BASIC_QUATERNION_H_INCLUDED

#include "../../build_config.h"
#include "scalar.h"
#include "vector.h"
#include "matrix.h"
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <limits>

namespace OrthoTree {
namespace Math {
namespace Basic {

// ============================================================================
//  Quaternion: w + x*i + y*j + z*k, representing rotation (unit quaternion).
// ============================================================================
template<typename T = float>
class Quaternion {
public:
    using value_type = T;
    using vector_type = Vector<T, 3>;
    using matrix_type = Matrix<T, 3>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    constexpr Quaternion() noexcept : m_w(T(1)), m_x(T(0)), m_y(T(0)), m_z(T(0)) {}
    constexpr Quaternion(T w, T x, T y, T z) noexcept : m_w(w), m_x(x), m_y(y), m_z(z) {}
    constexpr Quaternion(T w, const vector_type& v) noexcept : m_w(w), m_x(v[0]), m_y(v[1]), m_z(v[2]) {}

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    constexpr T w() const noexcept { return m_w; }
    constexpr T x() const noexcept { return m_x; }
    constexpr T y() const noexcept { return m_y; }
    constexpr T z() const noexcept { return m_z; }
    constexpr void setW(T w) noexcept { m_w = w; }
    constexpr void setX(T x) noexcept { m_x = x; }
    constexpr void setY(T y) noexcept { m_y = y; }
    constexpr void setZ(T z) noexcept { m_z = z; }
    constexpr vector_type vec() const noexcept { return vector_type(m_x, m_y, m_z); }

    // ------------------------------------------------------------------------
    //  Norm and normalisation
    // ------------------------------------------------------------------------
    constexpr T squaredNorm() const noexcept { return m_w*m_w + m_x*m_x + m_y*m_y + m_z*m_z; }
    T norm() const noexcept { return std::sqrt(squaredNorm()); }
    Quaternion normalized() const noexcept {
        T n = norm();
        if (n > T(0)) return Quaternion(m_w / n, m_x / n, m_y / n, m_z / n);
        return *this;
    }
    void normalize() noexcept { *this = normalized(); }

    // ------------------------------------------------------------------------
    //  Conjugate
    // ------------------------------------------------------------------------
    constexpr Quaternion conjugate() const noexcept { return Quaternion(m_w, -m_x, -m_y, -m_z); }

    // ------------------------------------------------------------------------
    //  Inverse (for unit quaternion, conjugate; otherwise conjugate / norm²)
    // ------------------------------------------------------------------------
    Quaternion inverse() const noexcept {
        T n2 = squaredNorm();
        if (n2 > T(0)) {
            T inv = T(1) / n2;
            return Quaternion(m_w * inv, -m_x * inv, -m_y * inv, -m_z * inv);
        }
        return *this;
    }

    // ------------------------------------------------------------------------
    //  Multiplication (Hamilton product)
    // ------------------------------------------------------------------------
    constexpr Quaternion operator*(const Quaternion& other) const noexcept {
        return Quaternion(
            m_w * other.m_w - m_x * other.m_x - m_y * other.m_y - m_z * other.m_z,
            m_w * other.m_x + m_x * other.m_w + m_y * other.m_z - m_z * other.m_y,
            m_w * other.m_y - m_x * other.m_z + m_y * other.m_w + m_z * other.m_x,
            m_w * other.m_z + m_x * other.m_y - m_y * other.m_x + m_z * other.m_w
        );
    }
    constexpr Quaternion& operator*=(const Quaternion& other) noexcept { *this = *this * other; return *this; }

    // ------------------------------------------------------------------------
    //  Rotate a vector (v' = q * v * q⁻¹)
    // ------------------------------------------------------------------------
    vector_type rotate(const vector_type& v) const noexcept {
        Quaternion p(T(0), v);
        Quaternion q = *this;
        Quaternion r = q * p * q.conjugate();
        return vector_type(r.x(), r.y(), r.z());
    }

    // ------------------------------------------------------------------------
    //  Inverse rotation (rotate by conjugate)
    // ------------------------------------------------------------------------
    vector_type rotateInverse(const vector_type& v) const noexcept {
        Quaternion p(T(0), v);
        Quaternion q = conjugate();
        Quaternion r = q * p * q.conjugate();
        return vector_type(r.x(), r.y(), r.z());
    }

    // ------------------------------------------------------------------------
    //  Convert to rotation matrix (3x3, column‑major)
    // ------------------------------------------------------------------------
    matrix_type toMatrix() const noexcept {
        T xx = m_x * m_x, yy = m_y * m_y, zz = m_z * m_z;
        T xy = m_x * m_y, xz = m_x * m_z, yz = m_y * m_z;
        T wx = m_w * m_x, wy = m_w * m_y, wz = m_w * m_z;
        matrix_type m;
        m(0,0) = T(1) - T(2)*(yy + zz);
        m(0,1) = T(2)*(xy - wz);
        m(0,2) = T(2)*(xz + wy);
        m(1,0) = T(2)*(xy + wz);
        m(1,1) = T(1) - T(2)*(xx + zz);
        m(1,2) = T(2)*(yz - wx);
        m(2,0) = T(2)*(xz - wy);
        m(2,1) = T(2)*(yz + wx);
        m(2,2) = T(1) - T(2)*(xx + yy);
        return m;
    }

    // ------------------------------------------------------------------------
    //  Construct from rotation matrix (assuming orthonormal)
    // ------------------------------------------------------------------------
    static Quaternion fromMatrix(const matrix_type& m) {
        T trace = m(0,0) + m(1,1) + m(2,2);
        if (trace > T(0)) {
            T s = T(0.5) / std::sqrt(trace + T(1));
            return Quaternion(T(0.25) / s,
                              (m(2,1) - m(1,2)) * s,
                              (m(0,2) - m(2,0)) * s,
                              (m(1,0) - m(0,1)) * s);
        } else if (m(0,0) > m(1,1) && m(0,0) > m(2,2)) {
            T s = T(2) * std::sqrt(T(1) + m(0,0) - m(1,1) - m(2,2));
            return Quaternion((m(2,1) - m(1,2)) / s,
                              T(0.25) * s,
                              (m(0,1) + m(1,0)) / s,
                              (m(0,2) + m(2,0)) / s);
        } else if (m(1,1) > m(2,2)) {
            T s = T(2) * std::sqrt(T(1) + m(1,1) - m(0,0) - m(2,2));
            return Quaternion((m(0,2) - m(2,0)) / s,
                              (m(0,1) + m(1,0)) / s,
                              T(0.25) * s,
                              (m(1,2) + m(2,1)) / s);
        } else {
            T s = T(2) * std::sqrt(T(1) + m(2,2) - m(0,0) - m(1,1));
            return Quaternion((m(1,0) - m(0,1)) / s,
                              (m(0,2) + m(2,0)) / s,
                              (m(1,2) + m(2,1)) / s,
                              T(0.25) * s);
        }
    }

    // ------------------------------------------------------------------------
    //  Rotation from axis and angle (unit axis)
    // ------------------------------------------------------------------------
    static Quaternion fromAxisAngle(const vector_type& axis, T angle) noexcept {
        T half = angle * T(0.5);
        T s = std::sin(half);
        vector_type a = axis.normalized();
        return Quaternion(std::cos(half), a[0] * s, a[1] * s, a[2] * s);
    }

    // ------------------------------------------------------------------------
    //  Rotation from Euler angles (ZYX order: yaw, pitch, roll)
    // ------------------------------------------------------------------------
    static Quaternion fromEuler(T roll, T pitch, T yaw) noexcept {
        T cr = std::cos(roll * T(0.5));
        T sr = std::sin(roll * T(0.5));
        T cp = std::cos(pitch * T(0.5));
        T sp = std::sin(pitch * T(0.5));
        T cy = std::cos(yaw * T(0.5));
        T sy = std::sin(yaw * T(0.5));
        return Quaternion(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        );
    }

    // ------------------------------------------------------------------------
    //  Spherical linear interpolation (SLERP)
    // ------------------------------------------------------------------------
    static Quaternion slerp(const Quaternion& a, const Quaternion& b, T t) noexcept {
        T dot = a.w()*b.w() + a.x()*b.x() + a.y()*b.y() + a.z()*b.z();
        Quaternion b2 = b;
        if (dot < T(0)) {
            b2 = -b;
            dot = -dot;
        }
        if (dot > T(0.9995)) {
            Quaternion r = a + (b2 - a) * t;
            return r.normalized();
        }
        T theta = std::acos(dot);
        T sinTheta = std::sin(theta);
        T w1 = std::sin((T(1)-t) * theta) / sinTheta;
        T w2 = std::sin(t * theta) / sinTheta;
        return Quaternion(
            a.w() * w1 + b2.w() * w2,
            a.x() * w1 + b2.x() * w2,
            a.y() * w1 + b2.y() * w2,
            a.z() * w1 + b2.z() * w2
        );
    }

    // ------------------------------------------------------------------------
    //  Identity
    // ------------------------------------------------------------------------
    static constexpr Quaternion identity() noexcept { return Quaternion(T(1), T(0), T(0), T(0)); }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool nearlyEqual(const Quaternion& other, T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) const noexcept {
        return std::abs(m_w - other.m_w) < eps && std::abs(m_x - other.m_x) < eps &&
               std::abs(m_y - other.m_y) < eps && std::abs(m_z - other.m_z) < eps;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: rotate 4 vectors by 4 quaternions (pairwise)
    // ------------------------------------------------------------------------
    static void batchRotate(const Quaternion* q, const vector_type* v, vector_type* out, size_t count) noexcept {
        if constexpr (std::is_same_v<T,float> && ORTHOTREE_SIMD_LEVEL >= 128) {
            for (size_t i = 0; i < count; ++i) {
                out[i] = q[i].rotate(v[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                out[i] = q[i].rotate(v[i]);
            }
        }
    }

private:
    T m_w, m_x, m_y, m_z;
};

// ----------------------------------------------------------------------------
//  Convenience aliases
// ----------------------------------------------------------------------------
using Quaternionf = Quaternion<float>;
using Quaterniond = Quaternion<double>;

// ----------------------------------------------------------------------------
//  Helper: create quaternion from two vectors (shortest arc)
// ----------------------------------------------------------------------------
template<typename T>
Quaternion<T> quaternionFromTwoVectors(const Basic::Vector<T,3>& from, const Basic::Vector<T,3>& to) {
    Basic::Vector<T,3> f = from.normalized();
    Basic::Vector<T,3> t = to.normalized();
    T dot = f.dot(t);
    if (dot > T(0.99999)) return Quaternion<T>::identity();
    if (dot < -T(0.99999)) {
        // 180 degree rotation: find any perpendicular axis
        Basic::Vector<T,3> axis = f.cross(Basic::Vector<T,3>(1,0,0));
        if (axis.squaredLength() < T(1e-6)) axis = f.cross(Basic::Vector<T,3>(0,1,0));
        axis = axis.normalized();
        return Quaternion<T>::fromAxisAngle(axis, Constants<T>::pi());
    }
    Basic::Vector<T,3> axis = f.cross(t).normalized();
    T angle = std::acos(dot);
    return Quaternion<T>::fromAxisAngle(axis, angle);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class QuaternionEnvironment {
public:
    static QuaternionEnvironment& instance() {
        static QuaternionEnvironment env;
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
    QuaternionEnvironment() : m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    bool m_useSIMD;
};

} // namespace Basic
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_BASIC_QUATERNION_H_INCLUDED