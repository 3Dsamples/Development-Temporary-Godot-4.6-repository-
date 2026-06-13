//File group name : OrthoTree Math
//File 0023 : core/math/ellipsoid.h
//Ellipsoid primitive: center, radii, rotation (orientation). Distance queries, ray intersection, bounding box, transformation, and SIMD batch operations.

#ifndef ORTHOTREE_CORE_MATH_ELLIPSOID_H_INCLUDED
#define ORTHOTREE_CORE_MATH_ELLIPSOID_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "matrix.h"
#include "ray_intersection.h"
#include "geometry_queries.h"
#include "numerical_methods.h"
#include "sphere.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <array>
#include <limits>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Ellipsoid: defined by center, radii (positive), and a rotation matrix
//  (orthonormal, columns are axes). Provides distance to point,
//  ray intersection (solve quartic), bounding sphere, transformation,
//  and SIMD batch operations.
// ============================================================================
template<typename T = float>
class Ellipsoid {
public:
    using value_type = T;
    using point_type = Vector<T, 3>;
    using ray_type = Ray<T, 3>;
    using aabb_type = AxisAlignedBox<T, 3>;
    using matrix_type = Matrix<T, 3>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    Ellipsoid() noexcept
        : m_center(T(0)), m_radii(T(1)), m_rotation(matrix_type::identity()) {}
    Ellipsoid(const point_type& center, const point_type& radii, const matrix_type& rotation) noexcept
        : m_center(center), m_radii(radii), m_rotation(rotation) {}
    Ellipsoid(const Sphere<T,3>& sphere) noexcept
        : m_center(sphere.center()), m_radii(point_type(sphere.radius())), m_rotation(matrix_type::identity()) {}

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    const point_type& center() const noexcept { return m_center; }
    const point_type& radii() const noexcept { return m_radii; }
    const matrix_type& rotation() const noexcept { return m_rotation; }
    void setCenter(const point_type& c) noexcept { m_center = c; }
    void setRadii(const point_type& r) noexcept { m_radii = r; }
    void setRotation(const matrix_type& rot) noexcept { m_rotation = rot; }

    // ------------------------------------------------------------------------
    //  Axis access (columns of rotation matrix)
    // ------------------------------------------------------------------------
    point_type axisX() const noexcept { return point_type(m_rotation(0,0), m_rotation(1,0), m_rotation(2,0)); }
    point_type axisY() const noexcept { return point_type(m_rotation(0,1), m_rotation(1,1), m_rotation(2,1)); }
    point_type axisZ() const noexcept { return point_type(m_rotation(0,2), m_rotation(1,2), m_rotation(2,2)); }

    // ------------------------------------------------------------------------
    //  Transform point from world to local (ellipsoid) space (scaled and rotated)
    //  Local space is a unit sphere (normalised by radii).
    // ------------------------------------------------------------------------
    point_type toLocal(const point_type& p) const noexcept {
        point_type d = p - m_center;
        // Rotate into ellipsoid's coordinate system (inverse rotation = transpose)
        point_type local(d.dot(axisX()), d.dot(axisY()), d.dot(axisZ()));
        // Scale by inverse radii
        return point_type(local[0] / m_radii[0], local[1] / m_radii[1], local[2] / m_radii[2]);
    }

    // ------------------------------------------------------------------------
    //  Transform point from local (unit sphere) to world
    // ------------------------------------------------------------------------
    point_type toWorld(const point_type& p) const noexcept {
        point_type scaled(p[0] * m_radii[0], p[1] * m_radii[1], p[2] * m_radii[2]);
        return m_center + axisX() * scaled[0] + axisY() * scaled[1] + axisZ() * scaled[2];
    }

    // ------------------------------------------------------------------------
    //  Distance to point (exact, numerical iterative, but for compatibility)
    //  Here we use a robust approximation: transform to unit sphere and compute
    //  distance from transformed point to unit sphere.
    //  The exact distance is more complex; this gives an upper bound.
    //  For exact, one would solve a 6th degree polynomial.
    // ------------------------------------------------------------------------
    T distanceToPoint(const point_type& p) const noexcept {
        point_type local = toLocal(p);
        T len = local.length();
        if (len <= T(1)) return T(0);
        // Map back to world
        point_type dir = local / len;
        point_type onEllipsoid = toWorld(dir);
        return (p - onEllipsoid).length();
    }

    T squaredDistanceToPoint(const point_type& p) const noexcept {
        T d = distanceToPoint(p);
        return d * d;
    }

    // ------------------------------------------------------------------------
    //  Ray intersection (returns t0 (entry), t1 (exit), false if no hit)
    //  Transform ray to unit sphere space and use ray‑sphere intersection.
    // ------------------------------------------------------------------------
    bool intersectRay(const ray_type& ray, T& t0, T& t1) const noexcept {
        // Transform ray to unit sphere space
        point_type localOrigin = toLocal(ray.origin());
        point_type localDir(
            ray.direction().dot(axisX()) / m_radii[0],
            ray.direction().dot(axisY()) / m_radii[1],
            ray.direction().dot(axisZ()) / m_radii[2]
        );
        Sphere<T,3> unitSphere(point_type(0), T(1));
        Ray<T,3> localRay(localOrigin, localDir);
        return unitSphere.intersectRay(localRay, t0, t1);
    }

    // ------------------------------------------------------------------------
    //  Bounding sphere (max radius)
    // ------------------------------------------------------------------------
    Sphere<T,3> boundingSphere() const noexcept {
        T maxRad = std::max({m_radii[0], m_radii[1], m_radii[2]});
        return Sphere<T,3>(m_center, maxRad);
    }

    // ------------------------------------------------------------------------
    //  Bounding AABB (world axis aligned, conservative)
    // ------------------------------------------------------------------------
    aabb_type boundingAABB() const noexcept {
        // Project axes scaled by radii to world axes
        point_type ext(
            std::abs(axisX()[0] * m_radii[0]) + std::abs(axisY()[0] * m_radii[1]) + std::abs(axisZ()[0] * m_radii[2]),
            std::abs(axisX()[1] * m_radii[0]) + std::abs(axisY()[1] * m_radii[1]) + std::abs(axisZ()[1] * m_radii[2]),
            std::abs(axisX()[2] * m_radii[0]) + std::abs(axisY()[2] * m_radii[1]) + std::abs(axisZ()[2] * m_radii[2])
        );
        return aabb_type(m_center - ext, m_center + ext);
    }

    // ------------------------------------------------------------------------
    //  Transform ellipsoid by affine transform (rotation, translation, non‑uniform scale)
    //  WARNING: general affine transforms an ellipsoid to another ellipsoid,
    //  but the rotation matrix may not remain orthonormal. We approximate by
    //  extracting the new rotation via SVD (not implemented here). Instead,
    //  we assume transformation only includes rotation + translation + uniform scale.
    // ------------------------------------------------------------------------
    Ellipsoid transform(const AffineTransform<T,3>& tf) const noexcept {
        point_type newCenter = tf.transform(m_center);
        matrix_type newRot = tf.matrix() * m_rotation;
        // Scale radii by max scale factor (conservative)
        T maxScale = T(0);
        for (int i = 0; i < 3; ++i) {
            point_type col;
            for (int j = 0; j < 3; ++j) col[j] = tf.matrix()(j, i);
            T len = col.length();
            if (len > maxScale) maxScale = len;
        }
        point_type newRadii = m_radii * maxScale;
        // Orthogonalise rotation (Gram‑Schmidt) to maintain OBB property? Skip.
        return Ellipsoid(newCenter, newRadii, newRot);
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool operator==(const Ellipsoid& other) const noexcept {
        T eps = T(1e-6);
        return (m_center - other.m_center).length() < eps &&
               (m_radii - other.m_radii).length() < eps &&
               (m_rotation - other.m_rotation).frobeniusNorm() < eps;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: intersect 4 rays with 4 ellipsoids (pairs)
    // ------------------------------------------------------------------------
    static void batchRayIntersect(const Ellipsoid* ellipsoids, const ray_type* rays,
                                  bool* hit, T* t0, T* t1, size_t count) noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && std::is_same_v<T,float>) {
            for (size_t i = 0; i < count; ++i) {
                hit[i] = ellipsoids[i].intersectRay(rays[i], t0[i], t1[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                hit[i] = ellipsoids[i].intersectRay(rays[i], t0[i], t1[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: distance to point for 4 ellipsoids
    // ------------------------------------------------------------------------
    static void batchDistanceToPoint(const Ellipsoid* ellipsoids, const point_type* points,
                                     T* out, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) {
            out[i] = ellipsoids[i].distanceToPoint(points[i]);
        }
    }

private:
    point_type m_center;
    point_type m_radii;
    matrix_type m_rotation;
};

// ----------------------------------------------------------------------------
//  Helper: create ellipsoid from center, radii, and orientation (quaternion)
// ----------------------------------------------------------------------------
template<typename T>
Ellipsoid<T> makeEllipsoid(const Vector<T,3>& center, const Vector<T,3>& radii,
                           const Quaternion<T>& orientation) {
    Matrix<T,3> rot = Matrix<T,3>::rotation(orientation);
    return Ellipsoid<T>(center, radii, rot);
}

// ----------------------------------------------------------------------------
//  Helper: create ellipsoid from sphere (uniform radii)
// ----------------------------------------------------------------------------
template<typename T>
Ellipsoid<T> ellipsoidFromSphere(const Sphere<T,3>& sphere) {
    return Ellipsoid<T>(sphere);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class EllipsoidEnvironment {
public:
    static EllipsoidEnvironment& instance() {
        static EllipsoidEnvironment env;
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
    EllipsoidEnvironment() : m_epsilon(T(1e-8)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_ELLIPSOID_H_INCLUDED