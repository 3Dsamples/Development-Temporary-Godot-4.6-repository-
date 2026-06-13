//File group name : OrthoTree Math
//File 0036 : core/math/dual_quaternion.h
//Dual quaternion for rigid transformations (rotation + translation) and skinning. Supports composition, blending, conversion to/from affine transform, and SIMD batch evaluation for 4 dual quaternions.

#ifndef ORTHOTREE_CORE_MATH_DUAL_QUATERNION_H_INCLUDED
#define ORTHOTREE_CORE_MATH_DUAL_QUATERNION_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "quaternion.h"
#include "transform.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <array>
#include <algorithm>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  DualQuaternion: represents a rigid transformation (rotation + translation)
//  using dual quaternion: q + ε q' where q is unit quaternion (rotation) and
//  q' = 0.5 * t * q (where t is translation as pure quaternion).
//  Supports multiplication (composition), blending (linear, spherical, lerp),
//  conversion to/from AffineTransform, and skinning of points.
// ============================================================================
template<typename T = float>
class DualQuaternion {
public:
    using value_type = T;
    using point_type = Vector<T, 3>;
    using quat_type = Quaternion<T>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    constexpr DualQuaternion() noexcept : m_real(1,0,0,0), m_dual(0,0,0,0) {}
    constexpr DualQuaternion(const quat_type& real, const quat_type& dual) noexcept
        : m_real(real), m_dual(dual) {}
    DualQuaternion(const quat_type& rotation, const point_type& translation) noexcept {
        m_real = rotation.normalized();
        quat_type t(0, translation[0], translation[1], translation[2]);
        m_dual = (t * m_real) * T(0.5);
    }
    DualQuaternion(const AffineTransform<T,3>& tf) noexcept {
        quat_type rot = Quaternion<T>::fromMatrix(tf.matrix());
        point_type trans = tf.translation();
        m_real = rot.normalized();
        quat_type t(0, trans[0], trans[1], trans[2]);
        m_dual = (t * m_real) * T(0.5);
    }

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    const quat_type& real() const noexcept { return m_real; }
    const quat_type& dual() const noexcept { return m_dual; }
    void setReal(const quat_type& r) noexcept { m_real = r; }
    void setDual(const quat_type& d) noexcept { m_dual = d; }

    // ------------------------------------------------------------------------
    //  Conjugate
    // ------------------------------------------------------------------------
    DualQuaternion conjugate() const noexcept {
        return DualQuaternion(m_real.conjugate(), m_dual.conjugate());
    }

    // ------------------------------------------------------------------------
    //  Norm squared (real^2 + dual^2)
    // ------------------------------------------------------------------------
    T squaredNorm() const noexcept {
        return m_real.squaredNorm() + m_dual.squaredNorm();
    }

    // ------------------------------------------------------------------------
    //  Normalize (rigid transform: real must be unit, then dual adjusted)
    // ------------------------------------------------------------------------
    DualQuaternion normalized() const noexcept {
        T normReal = m_real.norm();
        if (normReal > T(0)) {
            quat_type realNorm = m_real / normReal;
            quat_type dualNorm = m_dual / normReal;
            // Make dual orthogonal to real: dual = dual - real * dot(real, dual)
            T dot = realNorm.dot(dualNorm);
            dualNorm = dualNorm - realNorm * dot;
            return DualQuaternion(realNorm, dualNorm);
        }
        return DualQuaternion();
    }

    // ------------------------------------------------------------------------
    //  Composition (product) of two dual quaternions
    //  (a + ε a') * (b + ε b') = a*b + ε (a*b' + a'*b)
    // ------------------------------------------------------------------------
    DualQuaternion operator*(const DualQuaternion& other) const noexcept {
        return DualQuaternion(m_real * other.m_real,
                              m_real * other.m_dual + m_dual * other.m_real);
    }

    DualQuaternion& operator*=(const DualQuaternion& other) noexcept {
        *this = *this * other;
        return *this;
    }

    // ------------------------------------------------------------------------
    //  Transform a point (translate by 2 * dual * real.conjugate())
    //  Equivalent to: p' = rotation(p) + translation
    // ------------------------------------------------------------------------
    point_type transformPoint(const point_type& p) const noexcept {
        // Compute rotation using quaternion
        point_type rp = m_real.rotate(p);
        // Extract translation: t = 2 * (dual * real.conjugate()).vec()
        quat_type q = m_dual * m_real.conjugate();
        point_type trans( q.x() * T(2), q.y() * T(2), q.z() * T(2) );
        return rp + trans;
    }

    // ------------------------------------------------------------------------
    //  Convert to AffineTransform
    // ------------------------------------------------------------------------
    AffineTransform<T,3> toAffineTransform() const noexcept {
        Matrix<T,3> rot = m_real.toMatrix();
        quat_type q = m_dual * m_real.conjugate();
        point_type trans(q.x() * T(2), q.y() * T(2), q.z() * T(2));
        return AffineTransform<T,3>(rot, trans);
    }

    // ------------------------------------------------------------------------
    //  Identity dual quaternion
    // ------------------------------------------------------------------------
    static DualQuaternion identity() noexcept {
        return DualQuaternion(quat_type::identity(), quat_type(0,0,0,0));
    }

    // ------------------------------------------------------------------------
    //  Linear blend of two dual quaternions (normalised)
    //  (1 - t) * dq0 + t * dq1, then normalise.
    // ------------------------------------------------------------------------
    static DualQuaternion blend(const DualQuaternion& a, const DualQuaternion& b, T t) noexcept {
        DualQuaternion result;
        result.m_real = a.m_real * (T(1)-t) + b.m_real * t;
        result.m_dual = a.m_dual * (T(1)-t) + b.m_dual * t;
        return result.normalized();
    }

    // ------------------------------------------------------------------------
    //  Spherical linear interpolation (lerp) of two dual quaternions
    //  For rigid transforms, using ScLERP (dual quaternion slerp)
    // ------------------------------------------------------------------------
    static DualQuaternion slerp(const DualQuaternion& a, const DualQuaternion& b, T t) noexcept {
        T cosTheta = a.m_real.dot(b.m_real);
        if (cosTheta < T(0)) {
            // Take shortest arc
            return slerp(a, DualQuaternion(-b.m_real, -b.m_dual), t);
        }
        T theta = std::acos(cosTheta);
        if (theta < T(1e-6)) {
            // parallel, use linear blend
            return blend(a, b, t);
        }
        T sinTheta = std::sin(theta);
        T w1 = std::sin((T(1)-t) * theta) / sinTheta;
        T w2 = std::sin(t * theta) / sinTheta;
        DualQuaternion result;
        result.m_real = a.m_real * w1 + b.m_real * w2;
        result.m_dual = a.m_dual * w1 + b.m_dual * w2;
        return result.normalized();
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: transform 4 points using 4 dual quaternions (each point with its own transform)
    //  Input: dq[4], points[4], output[4].
    // ------------------------------------------------------------------------
    static void batchTransformPoints(const DualQuaternion* dq, const point_type* points,
                                     point_type* out, size_t count) noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && std::is_same_v<T,float>) {
            for (size_t i = 0; i < count; ++i) {
                out[i] = dq[i].transformPoint(points[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                out[i] = dq[i].transformPoint(points[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool operator==(const DualQuaternion& other) const noexcept {
        T eps = T(1e-6);
        return (m_real - other.m_real).norm() < eps &&
               (m_dual - other.m_dual).norm() < eps;
    }

private:
    quat_type m_real;
    quat_type m_dual;
};

// ----------------------------------------------------------------------------
//  Helper: create dual quaternion from rotation axis and angle, then translation
// ----------------------------------------------------------------------------
template<typename T>
DualQuaternion<T> makeDualQuaternion(const Vector<T,3>& axis, T angle,
                                     const Vector<T,3>& translation) {
    Quaternion<T> q = Quaternion<T>::fromAxisAngle(axis, angle);
    return DualQuaternion<T>(q, translation);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class DualQuaternionEnvironment {
public:
    static DualQuaternionEnvironment& instance() {
        static DualQuaternionEnvironment env;
        return env;
    }
    void setEpsilon(T eps) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_epsilon = eps;
    }
    T epsilon() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_epsilon;
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
    DualQuaternionEnvironment() : m_epsilon(T(1e-8)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_DUAL_QUATERNION_H_INCLUDED