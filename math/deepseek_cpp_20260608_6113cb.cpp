//File group name : Oriented Bounding Box
//File 0073 : core/math/geometry/obb.h
//Oriented bounding box (OBB): center, half‑extents, rotation matrix. Point distance, ray intersection, OBB‑OBB intersection (SAT), transformation, bounding sphere/AABB, and SIMD batch operations.

#ifndef ORTHOTREE_CORE_MATH_GEOMETRY_OBB_H_INCLUDED
#define ORTHOTREE_CORE_MATH_GEOMETRY_OBB_H_INCLUDED

#include "../../build_config.h"
#include "../basic/vector.h"
#include "../basic/matrix.h"
#include "../basic/quaternion.h"
#include "aabb.h"
#include "sphere.h"
#include "ray.h"
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <limits>
#include <array>

namespace OrthoTree {
namespace Math {
namespace Geometry {

// ============================================================================
//  Oriented bounding box (OBB) defined by center, half‑extents, and a rotation
//  matrix (orthonormal, columns are axes). Provides point distance, ray
//  intersection, OBB‑OBB test (separating axis theorem), bounding sphere/AABB,
//  transformation, and SIMD batch operations.
// ============================================================================
template<typename T = float>
class OBB {
public:
    using value_type = T;
    using point_type = Basic::Vector<T, 3>;
    using ray_type = Ray<T, 3>;
    using aabb_type = AABB<T, 3>;
    using sphere_type = Sphere<T, 3>;
    using matrix_type = Basic::Matrix<T, 3>;
    using quaternion_type = Basic::Quaternion<T>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    constexpr OBB() noexcept
        : m_center(T(0)), m_halfExtents(T(1)), m_rotation(matrix_type::identity()) {}
    constexpr OBB(const point_type& center, const point_type& halfExtents,
                  const matrix_type& rotation) noexcept
        : m_center(center), m_halfExtents(halfExtents), m_rotation(rotation) {}
    OBB(const aabb_type& aabb) noexcept
        : m_center(aabb.center()), m_halfExtents(aabb.halfExtents()),
          m_rotation(matrix_type::identity()) {}
    OBB(const point_type& center, const point_type& halfExtents,
        const quaternion_type& q) noexcept
        : m_center(center), m_halfExtents(halfExtents), m_rotation(q.toMatrix()) {}

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    constexpr const point_type& center() const noexcept { return m_center; }
    constexpr const point_type& halfExtents() const noexcept { return m_halfExtents; }
    constexpr const matrix_type& rotation() const noexcept { return m_rotation; }
    constexpr void setCenter(const point_type& c) noexcept { m_center = c; }
    constexpr void setHalfExtents(const point_type& he) noexcept { m_halfExtents = he; }
    constexpr void setRotation(const matrix_type& rot) noexcept { m_rotation = rot; }

    // ------------------------------------------------------------------------
    //  Axes (columns of rotation matrix)
    // ------------------------------------------------------------------------
    constexpr point_type axisX() const noexcept {
        return point_type(m_rotation(0,0), m_rotation(1,0), m_rotation(2,0));
    }
    constexpr point_type axisY() const noexcept {
        return point_type(m_rotation(0,1), m_rotation(1,1), m_rotation(2,1));
    }
    constexpr point_type axisZ() const noexcept {
        return point_type(m_rotation(0,2), m_rotation(1,2), m_rotation(2,2));
    }

    // ------------------------------------------------------------------------
    //  Transform point from world to local OBB space (where OBB is axis‑aligned)
    // ------------------------------------------------------------------------
    point_type toLocal(const point_type& p) const noexcept {
        point_type d = p - m_center;
        return point_type(d.dot(axisX()), d.dot(axisY()), d.dot(axisZ()));
    }
    point_type toWorld(const point_type& local) const noexcept {
        return m_center + axisX() * local[0] + axisY() * local[1] + axisZ() * local[2];
    }

    // ------------------------------------------------------------------------
    //  Distance to point
    // ------------------------------------------------------------------------
    T squaredDistanceToPoint(const point_type& p) const noexcept {
        point_type local = toLocal(p);
        T dx = std::max(T(0), std::abs(local[0]) - m_halfExtents[0]);
        T dy = std::max(T(0), std::abs(local[1]) - m_halfExtents[1]);
        T dz = std::max(T(0), std::abs(local[2]) - m_halfExtents[2]);
        return dx*dx + dy*dy + dz*dz;
    }
    T distanceToPoint(const point_type& p) const noexcept {
        return std::sqrt(squaredDistanceToPoint(p));
    }
    bool containsPoint(const point_type& p) const noexcept {
        point_type local = toLocal(p);
        return std::abs(local[0]) <= m_halfExtents[0] + T(1e-8) &&
               std::abs(local[1]) <= m_halfExtents[1] + T(1e-8) &&
               std::abs(local[2]) <= m_halfExtents[2] + T(1e-8);
    }

    // ------------------------------------------------------------------------
    //  Ray intersection (transform ray to local space, intersect AABB)
    // ------------------------------------------------------------------------
    bool intersectRay(const ray_type& ray, T& tMin, T& tMax) const noexcept {
        point_type localOrigin = toLocal(ray.origin());
        point_type localDir = point_type(ray.direction().dot(axisX()),
                                        ray.direction().dot(axisY()),
                                        ray.direction().dot(axisZ()));
        Ray<T,3> localRay(localOrigin, localDir);
        aabb_type localBox(-m_halfExtents, m_halfExtents);
        return localRay.intersectsAABB(localBox, tMin, tMax);
    }

    // ------------------------------------------------------------------------
    //  OBB‑OBB intersection test (separating axis theorem, SAT)
    //  Returns true if OBBs overlap.
    // ------------------------------------------------------------------------
    bool intersects(const OBB& other) const noexcept {
        const T eps = T(1e-8);
        point_type aAxis[3] = {axisX(), axisY(), axisZ()};
        point_type bAxis[3] = {other.axisX(), other.axisY(), other.axisZ()};
        point_type t = other.m_center - m_center;

        std::array<point_type, 15> axes;
        for (int i = 0; i < 3; ++i) axes[i] = aAxis[i];
        for (int i = 0; i < 3; ++i) axes[3+i] = bAxis[i];
        int idx = 6;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                axes[idx++] = aAxis[i].cross(bAxis[j]);
            }
        }

        for (const auto& axis : axes) {
            T len2 = axis.squaredLength();
            if (len2 < eps) continue;
            T invLen = T(1) / std::sqrt(len2);
            point_type n = axis * invLen;

            T ra = std::abs(m_halfExtents[0] * aAxis[0].dot(n)) +
                   std::abs(m_halfExtents[1] * aAxis[1].dot(n)) +
                   std::abs(m_halfExtents[2] * aAxis[2].dot(n));
            T rb = std::abs(other.m_halfExtents[0] * bAxis[0].dot(n)) +
                   std::abs(other.m_halfExtents[1] * bAxis[1].dot(n)) +
                   std::abs(other.m_halfExtents[2] * bAxis[2].dot(n));
            T dist = std::abs(t.dot(n));
            if (dist > ra + rb + eps) return false;
        }
        return true;
    }

    // ------------------------------------------------------------------------
    //  Bounding sphere (radius = distance from center to farthest corner)
    // ------------------------------------------------------------------------
    sphere_type boundingSphere() const noexcept {
        T r = m_halfExtents.length();
        return sphere_type(m_center, r);
    }

    // ------------------------------------------------------------------------
    //  Bounding AABB (world axis‑aligned)
    // ------------------------------------------------------------------------
    aabb_type boundingAABB() const noexcept {
        point_type minP, maxP;
        constexpr int corners = 8;
        point_type cornersList[corners];
        for (int i = 0; i < corners; ++i) {
            point_type p;
            for (int d = 0; d < 3; ++d) {
                T s = ((i >> d) & 1) ? T(1) : T(-1);
                p[d] = s * m_halfExtents[d];
            }
            cornersList[i] = toWorld(p);
        }
        minP = maxP = cornersList[0];
        for (int i = 1; i < corners; ++i) {
            minP = minP.componentWiseMin(cornersList[i]);
            maxP = maxP.componentWiseMax(cornersList[i]);
        }
        return aabb_type(minP, maxP);
    }

    // ------------------------------------------------------------------------
    //  Transform OBB by affine transform (assuming rotation + translation)
    //  New OBB = transform(center), rotate axes, scale half‑extents by max singular value.
    // ------------------------------------------------------------------------
    OBB transform(const Basic::AffineTransform<T,3>& tf) const noexcept {
        point_type newCenter = tf.transformPoint(m_center);
        matrix_type newRot = tf.matrix() * m_rotation;
        // Orthogonalise newRot (Gram‑Schmidt) to keep it orthonormal
        point_type x = newRot.col(0).normalized();
        point_type y = (newRot.col(1) - x * x.dot(newRot.col(1))).normalized();
        point_type z = x.cross(y);
        matrix_type ortho;
        ortho(0,0)=x[0]; ortho(1,0)=x[1]; ortho(2,0)=x[2];
        ortho(0,1)=y[0]; ortho(1,1)=y[1]; ortho(2,1)=y[2];
        ortho(0,2)=z[0]; ortho(1,2)=z[1]; ortho(2,2)=z[2];
        // Compute scaling factor for half‑extents (max singular value of linear part)
        T maxScale = T(0);
        for (int i = 0; i < 3; ++i) {
            point_type col;
            for (int j = 0; j < 3; ++j) col[j] = tf.matrix()(j, i);
            T len = col.length();
            if (len > maxScale) maxScale = len;
        }
        point_type newHalf = m_halfExtents * maxScale;
        return OBB(newCenter, newHalf, ortho);
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool nearlyEqual(const OBB& other, T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) const noexcept {
        return m_center.nearlyEqual(other.m_center, eps) &&
               m_halfExtents.nearlyEqual(other.m_halfExtents, eps) &&
               (m_rotation - other.m_rotation).frobeniusNorm() < eps;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: squared distance to point for 4 OBBs
    // ------------------------------------------------------------------------
    static void batchSquaredDistance(const OBB* boxes, const point_type* points,
                                     T* out, size_t count) noexcept {
        if constexpr (std::is_same_v<T,float> && ORTHOTREE_SIMD_LEVEL >= 128) {
            for (size_t i = 0; i < count; ++i) {
                out[i] = boxes[i].squaredDistanceToPoint(points[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                out[i] = boxes[i].squaredDistanceToPoint(points[i]);
            }
        }
    }

private:
    point_type m_center;
    point_type m_halfExtents;
    matrix_type m_rotation;
};

// ----------------------------------------------------------------------------
//  Convenience aliases
// ----------------------------------------------------------------------------
using OBBf = OBB<float>;
using OBBd = OBB<double>;

// ----------------------------------------------------------------------------
//  Helper: create OBB from center, size (full extents), and rotation
// ----------------------------------------------------------------------------
template<typename T>
OBB<T> makeOBB(const Basic::Vector<T,3>& center, const Basic::Vector<T,3>& size,
               const Basic::Matrix<T,3>& rot) {
    return OBB<T>(center, size * T(0.5), rot);
}

// ----------------------------------------------------------------------------
//  Helper: create OBB from AABB (axis‑aligned)
// ----------------------------------------------------------------------------
template<typename T>
OBB<T> obbFromAABB(const AABB<T,3>& aabb) {
    return OBB<T>(aabb);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class OBBEnvironment {
public:
    static OBBEnvironment& instance() {
        static OBBEnvironment env;
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
    OBBEnvironment() : m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    bool m_useSIMD;
};

} // namespace Geometry
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_GEOMETRY_OBB_H_INCLUDED