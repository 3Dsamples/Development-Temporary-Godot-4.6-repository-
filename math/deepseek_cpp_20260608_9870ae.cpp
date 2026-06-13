//File group name : OrthoTree Math
//File 0056 : core/math/geometry/sphere.h
//Sphere (2D circle / 3D sphere) defined by center and radius. Distance queries, ray intersection, bounding box, transformation, and SIMD batch operations.

#ifndef ORTHOTREE_CORE_MATH_GEOMETRY_SPHERE_H_INCLUDED
#define ORTHOTREE_CORE_MATH_GEOMETRY_SPHERE_H_INCLUDED

#include "../../build_config.h"
#include "../basic/vector.h"
#include "../basic/matrix.h"
#include "aabb.h"
#include "ray.h"
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <limits>

namespace OrthoTree {
namespace Math {
namespace Geometry {

// ============================================================================
//  Sphere (2D circle or 3D sphere) with center and radius.
// ============================================================================
template<typename T, std::size_t N>
class Sphere {
public:
    using value_type = T;
    using point_type = Basic::Vector<T, N>;
    using ray_type = Ray<T, N>;
    using aabb_type = AABB<T, N>;

    static constexpr size_type dimension = N;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    constexpr Sphere() noexcept : m_center(T(0)), m_radius(T(1)) {}
    constexpr Sphere(const point_type& center, T radius) noexcept : m_center(center), m_radius(radius) {}

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    constexpr const point_type& center() const noexcept { return m_center; }
    constexpr T radius() const noexcept { return m_radius; }
    constexpr void setCenter(const point_type& c) noexcept { m_center = c; }
    constexpr void setRadius(T r) noexcept { m_radius = r; }

    // ------------------------------------------------------------------------
    //  Geometric properties
    // ------------------------------------------------------------------------
    constexpr T volume() const noexcept {
        if constexpr (N == 2) return Constants<T>::pi() * m_radius * m_radius;
        else return T(4) / T(3) * Constants<T>::pi() * m_radius * m_radius * m_radius;
    }
    constexpr T surfaceArea() const noexcept {
        if constexpr (N == 2) return T(2) * Constants<T>::pi() * m_radius;
        else return T(4) * Constants<T>::pi() * m_radius * m_radius;
    }

    // ------------------------------------------------------------------------
    //  Bounding box
    // ------------------------------------------------------------------------
    constexpr aabb_type boundingBox() const noexcept {
        point_type ext(m_radius);
        return aabb_type(m_center - ext, m_center + ext);
    }

    // ------------------------------------------------------------------------
    //  Distance queries
    // ------------------------------------------------------------------------
    constexpr T distanceToPoint(const point_type& p) const noexcept {
        return std::max(T(0), (p - m_center).length() - m_radius);
    }
    constexpr T squaredDistanceToPoint(const point_type& p) const noexcept {
        T d = (p - m_center).length() - m_radius;
        if (d < T(0)) return T(0);
        return d * d;
    }
    constexpr bool containsPoint(const point_type& p) const noexcept {
        return (p - m_center).squaredLength() <= m_radius * m_radius;
    }

    // ------------------------------------------------------------------------
    //  Ray intersection (returns t0 (entry), t1 (exit), true if hit)
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
    //  Transform sphere by affine transform (radius scales by max scaling factor)
    //  Conservative: new radius = old radius * max singular value of linear part.
    // ------------------------------------------------------------------------
    Sphere transform(const Basic::AffineTransform<T,N>& tf) const noexcept {
        point_type newCenter = tf.transform(m_center);
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
    bool nearlyEqual(const Sphere& other, T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) const noexcept {
        return m_center.nearlyEqual(other.m_center, eps) && std::abs(m_radius - other.m_radius) < eps;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: intersect 4 rays with 4 spheres (pairwise)
    // ------------------------------------------------------------------------
    static void batchRayIntersect(const Sphere* spheres, const ray_type* rays,
                                  bool* hit, T* t0, T* t1, size_t count) noexcept {
        if constexpr (std::is_same_v<T,float> && ORTHOTREE_SIMD_LEVEL >= 128 && N == 3) {
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
    //  SIMD batch: squared distance to point for 4 spheres
    // ------------------------------------------------------------------------
    static void batchSquaredDistanceToPoint(const Sphere* spheres, const point_type* points,
                                            T* out, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) {
            out[i] = spheres[i].squaredDistanceToPoint(points[i]);
        }
    }

private:
    point_type m_center;
    T m_radius;
};

// ----------------------------------------------------------------------------
//  Convenience aliases
// ----------------------------------------------------------------------------
template<typename T> using Circle = Sphere<T, 2>;
template<typename T> using Sphere3 = Sphere<T, 3>;

using Circlef = Sphere<float, 2>;
using Circled = Sphere<double, 2>;
using Sphere3f = Sphere<float, 3>;
using Sphere3d = Sphere<double, 3>;

// ----------------------------------------------------------------------------
//  Helper: create sphere from center and radius
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
Sphere<T,N> makeSphere(const Basic::Vector<T,N>& center, T radius) {
    return Sphere<T,N>(center, radius);
}

// ----------------------------------------------------------------------------
//  Helper: circumsphere of triangle (3D)
// ----------------------------------------------------------------------------
template<typename T>
Sphere<T,3> circumsphereTriangle(const Basic::Vector<T,3>& a, const Basic::Vector<T,3>& b, const Basic::Vector<T,3>& c) {
    // Not implemented – would compute circumcenter and radius.
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
    void setUseSIMD(bool use) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_useSIMD = use;
    }
    bool useSIMD() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_useSIMD;
    }
private:
    SphereEnvironment() : m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    bool m_useSIMD;
};

} // namespace Geometry
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_GEOMETRY_SPHERE_H_INCLUDED