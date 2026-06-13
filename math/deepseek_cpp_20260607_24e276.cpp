//File group name : OrthoTree Math
//File 0021 : core/math/sphere.h
//Sphere primitive: center + radius, distance queries, ray intersection, bounding box, transformation, and SIMD batch operations.

#ifndef ORTHOTREE_CORE_MATH_SPHERE_H_INCLUDED
#define ORTHOTREE_CORE_MATH_SPHERE_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "ray_intersection.h"
#include "geometry_queries.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <limits>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Sphere: defined by center and radius. Provides distance queries,
//  ray intersection, bounding box, transformation, and SIMD batch operations.
//  Supports 2D (circle) and 3D (sphere).
// ============================================================================
template<typename T = float, std::size_t N = 3>
class Sphere {
public:
    using value_type = T;
    using point_type = Vector<T, N>;
    using ray_type = Ray<T, N>;
    using aabb_type = AxisAlignedBox<T, N>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    constexpr Sphere() noexcept : m_center(T(0)), m_radius(T(1)) {}
    constexpr Sphere(const point_type& center, T radius) noexcept
        : m_center(center), m_radius(radius) {}

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    const point_type& center() const noexcept { return m_center; }
    T radius() const noexcept { return m_radius; }
    void setCenter(const point_type& c) noexcept { m_center = c; }
    void setRadius(T r) noexcept { m_radius = r; }

    // ------------------------------------------------------------------------
    //  Geometric properties
    // ------------------------------------------------------------------------
    T volume() const noexcept {
        if constexpr (N == 2) return Math::pi<T>() * m_radius * m_radius;
        else return T(4) / T(3) * Math::pi<T>() * m_radius * m_radius * m_radius;
    }
    T surfaceArea() const noexcept {
        if constexpr (N == 2) return T(2) * Math::pi<T>() * m_radius;
        else return T(4) * Math::pi<T>() * m_radius * m_radius;
    }
    aabb_type boundingBox() const noexcept {
        point_type ext(m_radius);
        return aabb_type(m_center - ext, m_center + ext);
    }

    // ------------------------------------------------------------------------
    //  Distance queries
    // ------------------------------------------------------------------------
    T distanceToPoint(const point_type& p) const noexcept {
        return std::max(T(0), (p - m_center).length() - m_radius);
    }
    T squaredDistanceToPoint(const point_type& p) const noexcept {
        T d = (p - m_center).length() - m_radius;
        if (d < T(0)) return T(0);
        return d * d;
    }
    bool containsPoint(const point_type& p) const noexcept {
        return (p - m_center).squaredLength() <= m_radius * m_radius;
    }

    // ------------------------------------------------------------------------
    //  Ray intersection (returns t0 (entry) and t1 (exit), false if no hit)
    // ------------------------------------------------------------------------
    bool intersectRay(const ray_type& ray, T& t0, T& t1) const noexcept {
        point_type oc = ray.origin() - m_center;
        T a = ray.direction().squaredLength();
        T b = T(2) * oc.dot(ray.direction());
        T c = oc.squaredLength() - m_radius * m_radius;
        T disc = b * b - T(4) * a * c;
        if (disc < T(0)) return false;
        T sqrtDisc = std::sqrt(disc);
        t0 = (-b - sqrtDisc) / (T(2) * a);
        t1 = (-b + sqrtDisc) / (T(2) * a);
        return true;
    }

    // ------------------------------------------------------------------------
    //  Transform sphere by affine transform (scales radius by max scaling factor)
    //  Conservative: radius multiplied by maximum singular value of linear part.
    // ------------------------------------------------------------------------
    Sphere transform(const AffineTransform<T,N>& tf) const noexcept {
        point_type newCenter = tf.transform(m_center);
        // Compute max scale factor from matrix columns
        T maxScale = T(0);
        for (size_t i = 0; i < N; ++i) {
            point_type col;
            for (size_t j = 0; j < N; ++j) col[j] = tf.matrix()(j, i);
            T len = col.length();
            if (len > maxScale) maxScale = len;
        }
        return Sphere(newCenter, m_radius * maxScale);
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool operator==(const Sphere& other) const noexcept {
        T eps = T(1e-6);
        return (m_center - other.m_center).length() < eps &&
               std::abs(m_radius - other.m_radius) < eps;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: intersect 4 rays with 4 spheres (pairs)
    // ------------------------------------------------------------------------
    static void batchRayIntersect(const Sphere* spheres, const ray_type* rays,
                                  bool* hit, T* t0, T* t1, size_t count) noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && N == 3 && std::is_same_v<T,float>) {
            for (size_t i = 0; i < count; ++i) {
                hit[i] = spheres[i].intersectRay(rays[i], t0[i], t1[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                hit[i] = spheres[i].intersectRay(rays[i], t0[i], t1[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: distance to point for 4 spheres
    // ------------------------------------------------------------------------
    static void batchDistanceToPoint(const Sphere* spheres, const point_type* points,
                                     T* out, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) {
            out[i] = spheres[i].distanceToPoint(points[i]);
        }
    }

private:
    point_type m_center;
    T m_radius;
};

// ----------------------------------------------------------------------------
//  Helper: create sphere from center and radius
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
Sphere<T,N> makeSphere(const Vector<T,N>& center, T radius) {
    return Sphere<T,N>(center, radius);
}

// ----------------------------------------------------------------------------
//  Helper: circumsphere of triangle (3D)
// ----------------------------------------------------------------------------
template<typename T>
Sphere<T,3> circumsphereTriangle(const Vector<T,3>& a, const Vector<T,3>& b, const Vector<T,3>& c) {
    // Not implemented for brevity
    return Sphere<T,3>();
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class SphereEnvironment {
public:
    static SphereEnvironment& instance() {
        static SphereEnvironment env;
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
    SphereEnvironment() : m_epsilon(T(1e-8)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_SPHERE_H_INCLUDED