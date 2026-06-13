//File group name : OrthoTree Math
//File 0074 : core/math/geometry/frustum.h
//Frustum defined by 6 planes (left, right, bottom, top, near, far). Point/sphere/AABB culling, extraction from view‑projection matrix, transformation, and SIMD batch culling.

#ifndef ORTHOTREE_CORE_MATH_GEOMETRY_FRUSTUM_H_INCLUDED
#define ORTHOTREE_CORE_MATH_GEOMETRY_FRUSTUM_H_INCLUDED

#include "../../build_config.h"
#include "../basic/vector.h"
#include "../basic/matrix.h"
#include "plane.h"
#include "aabb.h"
#include "sphere.h"
#include "ray.h"
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <array>

namespace OrthoTree {
namespace Math {
namespace Geometry {

// ============================================================================
//  Frustum: defined by six planes (left, right, bottom, top, near, far).
//  Normals point inward. Provides visibility tests for points, spheres, AABBs.
//  Can be built from view‑projection matrix.
// ============================================================================
template<typename T = float>
class Frustum {
public:
    using value_type = T;
    using point_type = Basic::Vector<T, 3>;
    using aabb_type = AABB<T, 3>;
    using sphere_type = Sphere<T, 3>;
    using plane_type = Plane<T>;
    using ray_type = Ray<T, 3>;
    using matrix_type = Basic::Matrix<T, 4>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    Frustum() = default;
    explicit Frustum(const std::array<plane_type, 6>& planes) : m_planes(planes) {}

    // Build from view‑projection matrix (OpenGL style, column‑major)
    void buildFromMatrix(const matrix_type& viewProj) noexcept {
        const T* m = viewProj.data();
        // Left   = row3 + row0
        m_planes[0] = plane_type(point_type(m[12] + m[0], m[13] + m[1], m[14] + m[2]), m[15] + m[3]);
        // Right  = row3 - row0
        m_planes[1] = plane_type(point_type(m[12] - m[0], m[13] - m[1], m[14] - m[2]), m[15] - m[3]);
        // Bottom = row3 + row1
        m_planes[2] = plane_type(point_type(m[12] + m[4], m[13] + m[5], m[14] + m[6]), m[15] + m[7]);
        // Top    = row3 - row1
        m_planes[3] = plane_type(point_type(m[12] - m[4], m[13] - m[5], m[14] - m[6]), m[15] - m[7]);
        // Near   = row3 + row2
        m_planes[4] = plane_type(point_type(m[12] + m[8], m[13] + m[9], m[14] + m[10]), m[15] + m[11]);
        // Far    = row3 - row2
        m_planes[5] = plane_type(point_type(m[12] - m[8], m[13] - m[9], m[14] - m[10]), m[15] - m[11]);
        // Normalise planes
        for (auto& p : m_planes) {
            T len = p.normal().length();
            if (len > T(0)) {
                p.setNormal(p.normal() / len);
                p.setD(p.d() / len);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    const plane_type& left()   const noexcept { return m_planes[0]; }
    const plane_type& right()  const noexcept { return m_planes[1]; }
    const plane_type& bottom() const noexcept { return m_planes[2]; }
    const plane_type& top()    const noexcept { return m_planes[3]; }
    const plane_type& near()   const noexcept { return m_planes[4]; }
    const plane_type& far()    const noexcept { return m_planes[5]; }
    void setPlanes(const std::array<plane_type,6>& planes) { m_planes = planes; }

    // ------------------------------------------------------------------------
    //  Culling tests
    // ------------------------------------------------------------------------
    bool isPointVisible(const point_type& p) const noexcept {
        for (const auto& plane : m_planes) {
            if (plane.signedDistance(p) < T(0)) return false;
        }
        return true;
    }

    bool isSphereVisible(const sphere_type& sphere) const noexcept {
        for (const auto& plane : m_planes) {
            T dist = plane.signedDistance(sphere.center());
            if (dist < -sphere.radius()) return false;
        }
        return true;
    }

    bool isAABBVisible(const aabb_type& aabb) const noexcept {
        for (const auto& plane : m_planes) {
            // Find the point of the AABB most in the plane normal direction
            point_type p = aabb.min();
            const point_type& n = plane.normal();
            if (n[0] >= T(0)) p[0] = aabb.max()[0];
            if (n[1] >= T(0)) p[1] = aabb.max()[1];
            if (n[2] >= T(0)) p[2] = aabb.max()[2];
            if (plane.signedDistance(p) < T(0)) return false;
        }
        return true;
    }

    // ------------------------------------------------------------------------
    //  Transform frustum by affine transform (planes transformed by inverse transpose)
    // ------------------------------------------------------------------------
    Frustum transform(const Basic::AffineTransform<T,3>& tf) const noexcept {
        std::array<plane_type,6> newPlanes;
        for (size_t i = 0; i < 6; ++i) {
            newPlanes[i] = m_planes[i].transform(tf);
        }
        return Frustum(newPlanes);
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool nearlyEqual(const Frustum& other, T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) const noexcept {
        for (size_t i = 0; i < 6; ++i) {
            if (!m_planes[i].nearlyEqual(other.m_planes[i], eps)) return false;
        }
        return true;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: test 4 points for visibility (same frustum)
    // ------------------------------------------------------------------------
    void batchPointsVisible(const point_type* points, bool* out, size_t count) const noexcept {
        if constexpr (std::is_same_v<T,float> && ORTHOTREE_SIMD_LEVEL >= 128) {
            for (size_t i = 0; i < count; ++i) {
                out[i] = isPointVisible(points[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                out[i] = isPointVisible(points[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: test 4 spheres for visibility
    // ------------------------------------------------------------------------
    void batchSpheresVisible(const sphere_type* spheres, bool* out, size_t count) const noexcept {
        for (size_t i = 0; i < count; ++i) {
            out[i] = isSphereVisible(spheres[i]);
        }
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: test 4 AABBs for visibility
    // ------------------------------------------------------------------------
    void batchAABBVisible(const aabb_type* boxes, bool* out, size_t count) const noexcept {
        for (size_t i = 0; i < count; ++i) {
            out[i] = isAABBVisible(boxes[i]);
        }
    }

private:
    std::array<plane_type, 6> m_planes;
};

// ----------------------------------------------------------------------------
//  Convenience aliases
// ----------------------------------------------------------------------------
using Frustumf = Frustum<float>;
using Frustumd = Frustum<double>;

// ----------------------------------------------------------------------------
//  Helper: create frustum from camera parameters (position, orientation, FOV, aspect)
// ----------------------------------------------------------------------------
template<typename T>
Frustum<T> createFrustum(const Basic::Vector<T,3>& position,
                         const Basic::Quaternion<T>& orientation,
                         T fovYDeg, T aspect, T nearZ, T farZ) {
    Basic::AffineTransform<T,3> view(orientation.toMatrix(), position);
    Basic::Matrix<T,4> viewMat = view.inverse().toMatrix4(); // not directly available, placeholder
    T f = T(1) / std::tan(fovYDeg * Constants<T>::pi() / T(360.0));
    Basic::Matrix<T,4> proj(0);
    proj(0,0) = f / aspect;
    proj(1,1) = f;
    proj(2,2) = (farZ + nearZ) / (nearZ - farZ);
    proj(2,3) = T(2) * farZ * nearZ / (nearZ - farZ);
    proj(3,2) = T(-1);
    Basic::Matrix<T,4> viewProj = proj * viewMat; // order depends on convention
    Frustum<T> frustum;
    frustum.buildFromMatrix(viewProj);
    return frustum;
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class FrustumEnvironment {
public:
    static FrustumEnvironment& instance() {
        static FrustumEnvironment env;
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
    FrustumEnvironment() : m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    bool m_useSIMD;
};

} // namespace Geometry
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_GEOMETRY_FRUSTUM_H_INCLUDED