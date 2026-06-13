//File group name : OrthoTree Math
//File 0022 : core/math/obb.h
//Oriented bounding box (OBB): center, half‑extents, rotation matrix. Distance queries, ray intersection, OBB‑OBB test (SAT), transformation, and SIMD batch operations.

#ifndef ORTHOTREE_CORE_MATH_OBB_H_INCLUDED
#define ORTHOTREE_CORE_MATH_OBB_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "matrix.h"
#include "ray_intersection.h"
#include "geometry_queries.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <array>
#include <limits>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Oriented bounding box (OBB): defined by center, half‑extents (positive),
//  and a rotation matrix (orthonormal, columns are axes). Provides
//  point distance, ray intersection, OBB‑OBB intersection (SAT),
//  transformation, and bounding sphere.
// ============================================================================
template<typename T = float>
class OBB {
public:
    using value_type = T;
    using point_type = Vector<T, 3>;
    using ray_type = Ray<T, 3>;
    using aabb_type = AxisAlignedBox<T, 3>;
    using matrix_type = Matrix<T, 3>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    OBB() noexcept : m_center(T(0)), m_halfExtents(T(1)), m_rotation(matrix_type::identity()) {}
    OBB(const point_type& center, const point_type& halfExtents, const matrix_type& rotation) noexcept
        : m_center(center), m_halfExtents(halfExtents), m_rotation(rotation) {}
    OBB(const aabb_type& aabb) noexcept
        : m_center(aabb.center()), m_halfExtents(aabb.halfExtents()), m_rotation(matrix_type::identity()) {}

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    const point_type& center() const noexcept { return m_center; }
    const point_type& halfExtents() const noexcept { return m_halfExtents; }
    const matrix_type& rotation() const noexcept { return m_rotation; }
    void setCenter(const point_type& c) noexcept { m_center = c; }
    void setHalfExtents(const point_type& he) noexcept { m_halfExtents = he; }
    void setRotation(const matrix_type& rot) noexcept { m_rotation = rot; }

    // ------------------------------------------------------------------------
    //  Axis access (columns of rotation matrix)
    // ------------------------------------------------------------------------
    point_type axisX() const noexcept { return point_type(m_rotation(0,0), m_rotation(1,0), m_rotation(2,0)); }
    point_type axisY() const noexcept { return point_type(m_rotation(0,1), m_rotation(1,1), m_rotation(2,1)); }
    point_type axisZ() const noexcept { return point_type(m_rotation(0,2), m_rotation(1,2), m_rotation(2,2)); }

    // ------------------------------------------------------------------------
    //  Transform point from world to local OBB space
    // ------------------------------------------------------------------------
    point_type toLocal(const point_type& p) const noexcept {
        point_type d = p - m_center;
        return point_type(d.dot(axisX()), d.dot(axisY()), d.dot(axisZ()));
    }

    // ------------------------------------------------------------------------
    //  Transform point from local to world space
    // ------------------------------------------------------------------------
    point_type toWorld(const point_type& p) const noexcept {
        return m_center + axisX() * p[0] + axisY() * p[1] + axisZ() * p[2];
    }

    // ------------------------------------------------------------------------
    //  Distance to point (squared, exact)
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

    // ------------------------------------------------------------------------
    //  Ray intersection (returns tMin, tMax, true if hit)
    //  Transform ray to OBB local space, then use AABB intersection.
    // ------------------------------------------------------------------------
    bool intersectRay(const ray_type& ray, T& tMin, T& tMax) const noexcept {
        // Transform ray to OBB local space
        point_type localOrigin = toLocal(ray.origin());
        point_type localDir = point_type(ray.direction().dot(axisX()),
                                        ray.direction().dot(axisY()),
                                        ray.direction().dot(axisZ()));
        aabb_type localBox(-m_halfExtents, m_halfExtents);
        Ray<T,3> localRay(localOrigin, localDir);
        return rayAABBIntersect(localRay, localBox, tMin, tMax);
    }

    // ------------------------------------------------------------------------
    //  OBB‑OBB intersection test (separating axis theorem)
    //  Returns true if OBBs overlap.
    // ------------------------------------------------------------------------
    bool intersects(const OBB& other) const noexcept {
        const T eps = T(1e-6);
        // 15 axes: 3 axes from each OBB + 9 cross products
        point_type aAxis[3] = {axisX(), axisY(), axisZ()};
        point_type bAxis[3] = {other.axisX(), other.axisY(), other.axisZ()};
        point_type t = other.m_center - m_center;

        std::array<point_type, 15> axes;
        for (int i = 0; i < 3; ++i) axes[i] = aAxis[i];
        for (int i = 0; i < 3; ++i) axes[3+i] = bAxis[i];
        int idx = 6;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                axes[idx++] = cross(aAxis[i], bAxis[j]);
            }
        }

        for (const auto& axis : axes) {
            T len2 = axis.squaredLength();
            if (len2 < eps) continue;
            T invLen = T(1) / std::sqrt(len2);
            point_type n = axis * invLen;
            T ra = std::abs(m_halfExtents[0] * dot(aAxis[0], n)) +
                   std::abs(m_halfExtents[1] * dot(aAxis[1], n)) +
                   std::abs(m_halfExtents[2] * dot(aAxis[2], n));
            T rb = std::abs(other.m_halfExtents[0] * dot(bAxis[0], n)) +
                   std::abs(other.m_halfExtents[1] * dot(bAxis[1], n)) +
                   std::abs(other.m_halfExtents[2] * dot(bAxis[2], n));
            T dist = std::abs(dot(t, n));
            if (dist > ra + rb + eps) return false;
        }
        return true;
    }

    // ------------------------------------------------------------------------
    //  Bounding sphere (radius = distance from center to farthest corner)
    // ------------------------------------------------------------------------
    Sphere<T,3> boundingSphere() const noexcept {
        T radius = m_halfExtents.length();
        return Sphere<T,3>(m_center, radius);
    }

    // ------------------------------------------------------------------------
    //  Bounding AABB (world axis aligned)
    // ------------------------------------------------------------------------
    aabb_type boundingAABB() const noexcept {
        point_type minP, maxP;
        for (int sign = 0; sign < 8; ++sign) {
            point_type p;
            for (int i = 0; i < 3; ++i) {
                T s = ((sign >> i) & 1) ? T(1) : T(-1);
                p += axis(i) * (m_halfExtents[i] * s);
            }
            p = m_center + p;
            if (sign == 0) { minP = maxP = p; }
            else {
                minP = minP.componentWiseMin(p);
                maxP = maxP.componentWiseMax(p);
            }
        }
        return aabb_type(minP, maxP);
    }

    // ------------------------------------------------------------------------
    //  Transform OBB by affine transform (rotation + translation)
    //  Note: scaling not uniformly supported; assumes rotation only or uniform scale.
    //  For general affine, OBB would become another OBB (rotate axes and scale half‑extents).
    // ------------------------------------------------------------------------
    OBB transform(const AffineTransform<T,3>& tf) const noexcept {
        point_type newCenter = tf.transform(m_center);
        matrix_type newRot = tf.matrix() * m_rotation;
        // Scale half‑extents: new half‑extent = old half‑extent * max singular value of transform's linear part
        point_type newHalf = m_halfExtents;
        T maxScale = T(0);
        for (int i = 0; i < 3; ++i) {
            point_type col;
            for (int j = 0; j < 3; ++j) col[j] = tf.matrix()(j, i);
            T len = col.length();
            if (len > maxScale) maxScale = len;
        }
        newHalf = newHalf * maxScale;
        return OBB(newCenter, newHalf, newRot);
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: ray intersection for 4 OBBs with the same ray
    // ------------------------------------------------------------------------
    static void batchRayIntersect(const OBB* boxes, const ray_type& ray,
                                  bool* hit, T* tMin, T* tMax, size_t count) noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && std::is_same_v<T,float>) {
            for (size_t i = 0; i < count; ++i) {
                hit[i] = boxes[i].intersectRay(ray, tMin[i], tMax[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                hit[i] = boxes[i].intersectRay(ray, tMin[i], tMax[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool operator==(const OBB& other) const noexcept {
        T eps = T(1e-6);
        return (m_center - other.m_center).length() < eps &&
               (m_halfExtents - other.m_halfExtents).length() < eps &&
               (m_rotation - other.m_rotation).frobeniusNorm() < eps;
    }

private:
    point_type m_center;
    point_type m_halfExtents;
    matrix_type m_rotation;
};

// ----------------------------------------------------------------------------
//  Helper: create OBB from center, size (full extents), and rotation
// ----------------------------------------------------------------------------
template<typename T>
OBB<T> makeOBB(const Vector<T,3>& center, const Vector<T,3>& size, const Matrix<T,3>& rot) {
    return OBB<T>(center, size * T(0.5), rot);
}

// ----------------------------------------------------------------------------
//  Helper: create OBB from AABB (identity rotation)
// ----------------------------------------------------------------------------
template<typename T>
OBB<T> obbFromAABB(const AxisAlignedBox<T,3>& aabb) {
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
    OBBEnvironment() : m_epsilon(T(1e-8)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_OBB_H_INCLUDED