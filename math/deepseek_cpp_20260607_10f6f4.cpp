//File group name : OrthoTree Math
//File 0038 : core/math/projection.h
//Projection matrices (perspective, orthographic, frustum) and view frustum extraction. SIMD batch matrix construction for 4 viewports.

#ifndef ORTHOTREE_CORE_MATH_PROJECTION_H_INCLUDED
#define ORTHOTREE_CORE_MATH_PROJECTION_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "matrix.h"
#include "frustum.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <array>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Projection matrices (OpenGL style, column‑major)
// ============================================================================
template<typename T = float>
class Projection {
public:
    using value_type = T;
    using matrix_type = Matrix<T,4>;

    // ------------------------------------------------------------------------
    //  Perspective projection (vertical FOV, aspect ratio, near, far)
    //  Assumes zero far plane? Standard.
    // ------------------------------------------------------------------------
    static matrix_type perspective(T fovYDeg, T aspect, T nearZ, T farZ) noexcept {
        T f = T(1) / std::tan(fovYDeg * Math::pi<T>() / T(360.0));
        T rangeInv = T(1) / (nearZ - farZ);
        matrix_type m(0);
        m(0,0) = f / aspect;
        m(1,1) = f;
        m(2,2) = (nearZ + farZ) * rangeInv;
        m(2,3) = T(2) * nearZ * farZ * rangeInv;
        m(3,2) = T(-1);
        return m;
    }

    // ------------------------------------------------------------------------
    //  Infinite perspective projection (far plane at infinity)
    // ------------------------------------------------------------------------
    static matrix_type perspectiveInfinite(T fovYDeg, T aspect, T nearZ) noexcept {
        T f = T(1) / std::tan(fovYDeg * Math::pi<T>() / T(360.0));
        matrix_type m(0);
        m(0,0) = f / aspect;
        m(1,1) = f;
        m(2,2) = T(-1);
        m(2,3) = -T(2) * nearZ;
        m(3,2) = T(-1);
        return m;
    }

    // ------------------------------------------------------------------------
    //  Orthographic projection (left, right, bottom, top, near, far)
    // ------------------------------------------------------------------------
    static matrix_type orthographic(T left, T right, T bottom, T top, T nearZ, T farZ) noexcept {
        T tx = -(right + left) / (right - left);
        T ty = -(top + bottom) / (top - bottom);
        T tz = -(farZ + nearZ) / (farZ - nearZ);
        matrix_type m(0);
        m(0,0) = T(2) / (right - left);
        m(1,1) = T(2) / (top - bottom);
        m(2,2) = -T(2) / (farZ - nearZ);
        m(0,3) = tx;
        m(1,3) = ty;
        m(2,3) = tz;
        m(3,3) = T(1);
        return m;
    }

    // ------------------------------------------------------------------------
    //  Orthographic 2D (pixel coordinates)
    // ------------------------------------------------------------------------
    static matrix_type ortho2D(T left, T right, T bottom, T top) noexcept {
        return orthographic(left, right, bottom, top, T(-1), T(1));
    }

    // ------------------------------------------------------------------------
    //  Frustum (left, right, bottom, top, near, far)
    // ------------------------------------------------------------------------
    static matrix_type frustum(T left, T right, T bottom, T top, T nearZ, T farZ) noexcept {
        T invRL = T(1) / (right - left);
        T invTB = T(1) / (top - bottom);
        T invFN = T(1) / (nearZ - farZ);
        matrix_type m(0);
        m(0,0) = T(2) * nearZ * invRL;
        m(1,1) = T(2) * nearZ * invTB;
        m(0,2) = (right + left) * invRL;
        m(1,2) = (top + bottom) * invTB;
        m(2,2) = (nearZ + farZ) * invFN;
        m(2,3) = T(2) * nearZ * farZ * invFN;
        m(3,2) = T(-1);
        return m;
    }

    // ------------------------------------------------------------------------
    //  Extract frustum planes from view‑projection matrix (6 planes)
    //  Result: left, right, bottom, top, near, far (in that order)
    // ------------------------------------------------------------------------
    static std::array<Plane<T>,6> extractFrustumPlanes(const matrix_type& viewProj) noexcept {
        const T* m = viewProj.data();
        std::array<Plane<T>,6> planes;
        // Left
        planes[0] = Plane<T>(Vector<T,3>(m[12] + m[0], m[13] + m[1], m[14] + m[2]), m[15] + m[3]);
        // Right
        planes[1] = Plane<T>(Vector<T,3>(m[12] - m[0], m[13] - m[1], m[14] - m[2]), m[15] - m[3]);
        // Bottom
        planes[2] = Plane<T>(Vector<T,3>(m[12] + m[4], m[13] + m[5], m[14] + m[6]), m[15] + m[7]);
        // Top
        planes[3] = Plane<T>(Vector<T,3>(m[12] - m[4], m[13] - m[5], m[14] - m[6]), m[15] - m[7]);
        // Near
        planes[4] = Plane<T>(Vector<T,3>(m[12] + m[8], m[13] + m[9], m[14] + m[10]), m[15] + m[11]);
        // Far
        planes[5] = Plane<T>(Vector<T,3>(m[12] - m[8], m[13] - m[9], m[14] - m[10]), m[15] - m[11]);
        // Normalise planes
        for (auto& p : planes) {
            T len = p.normal().length();
            if (len > T(0)) {
                p.setNormal(p.normal() / len);
                p.setD(p.d() / len);
            }
        }
        return planes;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: construct 4 perspective matrices with same FOV/aspect but different near/far
    //  Input: nearZ[4], farZ[4], output: matrices[4]
    // ------------------------------------------------------------------------
    static void batchPerspective(T fovYDeg, T aspect, const T* nearZ, const T* farZ, matrix_type* out, size_t count) {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && std::is_same_v<T,float>) {
            for (size_t i = 0; i < count; ++i) {
                out[i] = perspective(fovYDeg, aspect, nearZ[i], farZ[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                out[i] = perspective(fovYDeg, aspect, nearZ[i], farZ[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Batch view‑projection multiplication: V * P for 4 pairs (V, P) -> VP[4]
    // ------------------------------------------------------------------------
    static void batchMultiply(const matrix_type* view, const matrix_type* proj,
                              matrix_type* out, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            out[i] = proj[i] * view[i];
        }
    }
};

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class ProjectionEnvironment {
public:
    static ProjectionEnvironment& instance() {
        static ProjectionEnvironment env;
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
    ProjectionEnvironment() : m_epsilon(T(1e-8)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_PROJECTION_H_INCLUDED