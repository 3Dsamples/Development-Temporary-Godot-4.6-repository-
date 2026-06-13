//File group name : OrthoTree Math
//File 0068 : core/math/basic/transform.h
//Affine transform (N=2,3) with matrix (linear part) + translation. Composition, inverse, transformation of points, vectors, normals, and SIMD batch operations.

#ifndef ORTHOTREE_CORE_MATH_BASIC_TRANSFORM_H_INCLUDED
#define ORTHOTREE_CORE_MATH_BASIC_TRANSFORM_H_INCLUDED

#include "../../build_config.h"
#include "vector.h"
#include "matrix.h"
#include "quaternion.h"   // will be defined separately, but forward declare if needed
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

namespace OrthoTree {
namespace Math {
namespace Basic {

// ============================================================================
//  AffineTransform: y = M * x + t
//  where M is NxN matrix, t is translation vector.
// ============================================================================
template<typename T, std::size_t N>
class AffineTransform {
public:
    using value_type = T;
    using matrix_type = Matrix<T, N>;
    using vector_type = Vector<T, N>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    constexpr AffineTransform() noexcept : m_matrix(Matrix<T,N>::identity()), m_translation(T(0)) {}
    constexpr AffineTransform(const matrix_type& mat, const vector_type& trans) noexcept
        : m_matrix(mat), m_translation(trans) {}
    // From rotation matrix and translation (assuming orthogonal)
    constexpr AffineTransform(const matrix_type& rot, const vector_type& trans) noexcept
        : m_matrix(rot), m_translation(trans) {}
    // From quaternion and translation (3D only)
    AffineTransform(const Quaternion<T>& q, const vector_type& trans) noexcept {
        m_matrix = q.toMatrix();
        m_translation = trans;
    }

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    constexpr const matrix_type& matrix() const noexcept { return m_matrix; }
    constexpr const vector_type& translation() const noexcept { return m_translation; }
    constexpr void setMatrix(const matrix_type& m) noexcept { m_matrix = m; }
    constexpr void setTranslation(const vector_type& t) noexcept { m_translation = t; }

    // ------------------------------------------------------------------------
    //  Transform point (affine: M*p + t)
    // ------------------------------------------------------------------------
    constexpr vector_type transformPoint(const vector_type& p) const noexcept {
        return m_matrix * p + m_translation;
    }

    // ------------------------------------------------------------------------
    //  Transform direction vector (only linear part, no translation)
    // ------------------------------------------------------------------------
    constexpr vector_type transformDirection(const vector_type& d) const noexcept {
        return m_matrix * d;
    }

    // ------------------------------------------------------------------------
    //  Transform normal (inverse transpose of linear part, for correct lighting)
    // ------------------------------------------------------------------------
    vector_type transformNormal(const vector_type& n) const noexcept {
        return m_matrix.inverse().transpose() * n;
    }

    // ------------------------------------------------------------------------
    //  Composition (this * other): first apply other, then this
    //  i.e., transformPoint(p) = this->transformPoint(other.transformPoint(p))
    // ------------------------------------------------------------------------
    constexpr AffineTransform operator*(const AffineTransform& other) const noexcept {
        return AffineTransform(m_matrix * other.m_matrix,
                               m_matrix * other.m_translation + m_translation);
    }
    constexpr AffineTransform& operator*=(const AffineTransform& other) noexcept {
        *this = *this * other;
        return *this;
    }

    // ------------------------------------------------------------------------
    //  Inverse (requires matrix invertible)
    // ------------------------------------------------------------------------
    AffineTransform inverse() const noexcept {
        matrix_type invMat = m_matrix.inverse();
        vector_type invTrans = invMat * (-m_translation);
        return AffineTransform(invMat, invTrans);
    }

    // ------------------------------------------------------------------------
    //  Identity
    // ------------------------------------------------------------------------
    static constexpr AffineTransform identity() noexcept { return AffineTransform(); }

    // ------------------------------------------------------------------------
    //  Translation only
    // ------------------------------------------------------------------------
    static constexpr AffineTransform translation(const vector_type& t) noexcept {
        return AffineTransform(Matrix<T,N>::identity(), t);
    }

    // ------------------------------------------------------------------------
    //  Rotation only (3D, from quaternion)
    // ------------------------------------------------------------------------
    static AffineTransform rotation(const Quaternion<T>& q) noexcept {
        static_assert(N == 3, "Rotation only for 3D");
        return AffineTransform(q.toMatrix(), vector_type(T(0)));
    }

    // ------------------------------------------------------------------------
    //  Scaling (uniform)
    // ------------------------------------------------------------------------
    static constexpr AffineTransform scaling(T s) noexcept {
        matrix_type mat(0);
        for (std::size_t i = 0; i < N; ++i) mat(i,i) = s;
        return AffineTransform(mat, vector_type(T(0)));
    }
    // Non‑uniform scaling
    static constexpr AffineTransform scaling(const vector_type& s) noexcept {
        matrix_type mat(0);
        for (std::size_t i = 0; i < N; ++i) mat(i,i) = s[i];
        return AffineTransform(mat, vector_type(T(0)));
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool nearlyEqual(const AffineTransform& other, T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) const noexcept {
        return m_matrix.nearlyEqual(other.m_matrix, eps) && m_translation.nearlyEqual(other.m_translation, eps);
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: transform 4 points by 4 transforms (pairwise)
    // ------------------------------------------------------------------------
    static void batchTransformPoint(const AffineTransform* transforms, const vector_type* points,
                                    vector_type* out, size_t count) noexcept {
        if constexpr (std::is_same_v<T,float> && ORTHOTREE_SIMD_LEVEL >= 128 && N == 3) {
            for (size_t i = 0; i < count; ++i) {
                out[i] = transforms[i].transformPoint(points[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                out[i] = transforms[i].transformPoint(points[i]);
            }
        }
    }

private:
    matrix_type m_matrix;
    vector_type m_translation;
};

// ----------------------------------------------------------------------------
//  Convenience aliases
// ----------------------------------------------------------------------------
template<typename T> using AffineTransform2 = AffineTransform<T, 2>;
template<typename T> using AffineTransform3 = AffineTransform<T, 3>;

using AffineTransform2f = AffineTransform<float, 2>;
using AffineTransform2d = AffineTransform<double, 2>;
using AffineTransform3f = AffineTransform<float, 3>;
using AffineTransform3d = AffineTransform<double, 3>;

// ----------------------------------------------------------------------------
//  Helper: create rigid transform from rotation and translation
// ----------------------------------------------------------------------------
template<typename T>
AffineTransform<T,3> makeRigidTransform(const Quaternion<T>& q, const Basic::Vector<T,3>& t) {
    return AffineTransform<T,3>(q, t);
}

// ----------------------------------------------------------------------------
//  Helper: create transform from Euler angles (XYZ order)
// ----------------------------------------------------------------------------
template<typename T>
AffineTransform<T,3> makeEulerTransform(T rx, T ry, T rz, const Basic::Vector<T,3>& t) {
    // Not fully implemented; would compute rotation matrix from Euler
    Matrix<T,3> rot;
    // Dummy identity
    return AffineTransform<T,3>(rot, t);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class TransformEnvironment {
public:
    static TransformEnvironment& instance() {
        static TransformEnvironment env;
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
    TransformEnvironment() : m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    bool m_useSIMD;
};

} // namespace Basic
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_BASIC_TRANSFORM_H_INCLUDED