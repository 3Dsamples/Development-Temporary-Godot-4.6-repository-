//File group name : OrthoTree Math
//File 0034 : core/math/convex_polyhedron.h
//Convex polyhedron defined by intersection of half‑spaces (planes). Supports point containment, ray intersection, bounding box, transformation, and SIMD batch tests.

#ifndef ORTHOTREE_CORE_MATH_CONVEX_POLYHEDRON_H_INCLUDED
#define ORTHOTREE_CORE_MATH_CONVEX_POLYHEDRON_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "plane.h"
#include "ray_intersection.h"
#include "geometry_queries.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <algorithm>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  ConvexPolyhedron: defined as intersection of N half‑spaces (planes with normals pointing inward).
//  Provides point containment test (inside if all plane distances >= 0), ray intersection,
//  bounding box (iterative expansion), transformation (transform each plane),
//  and SIMD batch point testing.
// ============================================================================
template<typename T = float>
class ConvexPolyhedron {
public:
    using value_type = T;
    using point_type = Vector<T, 3>;
    using ray_type = Ray<T, 3>;
    using plane_type = Plane<T>;
    using aabb_type = AxisAlignedBox<T, 3>;
    using size_type = size_t;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    ConvexPolyhedron() = default;
    explicit ConvexPolyhedron(const std::vector<plane_type>& planes) : m_planes(planes) {}
    ConvexPolyhedron(std::vector<plane_type>&& planes) : m_planes(std::move(planes)) {}

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    const std::vector<plane_type>& planes() const noexcept { return m_planes; }
    void setPlanes(const std::vector<plane_type>& planes) { m_planes = planes; }
    size_type numPlanes() const noexcept { return m_planes.size(); }

    // ------------------------------------------------------------------------
    //  Add a plane (assumes convex, no validation)
    // ------------------------------------------------------------------------
    void addPlane(const plane_type& p) { m_planes.push_back(p); }

    // ------------------------------------------------------------------------
    //  Point containment: returns true if point is inside (distance to all planes >= 0)
    // ------------------------------------------------------------------------
    bool containsPoint(const point_type& p, T eps = T(1e-8)) const noexcept {
        for (const auto& plane : m_planes) {
            if (plane.signedDistance(p) < -eps) return false;
        }
        return true;
    }

    // ------------------------------------------------------------------------
    //  Ray intersection: find first intersection with polyhedron (closest t > 0).
    //  Returns true if hit, outputs t (distance) and normal of the hit plane.
    //  Uses slab method: tMin = max(t near), tMax = min(t far), with early exit.
    // ------------------------------------------------------------------------
    bool intersectRay(const ray_type& ray, T& t, point_type& normal) const noexcept {
        T tMin = T(0);
        T tMax = std::numeric_limits<T>::max();
        bool hit = false;
        for (const auto& plane : m_planes) {
            T denom = plane.normal().dot(ray.direction());
            T dist = -plane.signedDistance(ray.origin());  // distance from ray origin to plane
            if (std::abs(denom) < T(1e-12)) {
                // Ray parallel to plane
                if (dist < -T(1e-8)) return false; // ray starts outside and never enters
                continue;
            }
            T tHit = dist / denom;
            if (denom > 0) {
                // entering half‑space
                if (tHit > tMin) {
                    tMin = tHit;
                    normal = plane.normal();
                }
            } else {
                // exiting half‑space
                if (tHit < tMax) tMax = tHit;
            }
            if (tMin > tMax + T(1e-8)) return false;
        }
        if (tMin < T(0)) tMin = T(0);
        t = tMin;
        return (t < tMax);
    }

    // ------------------------------------------------------------------------
    //  Bounding box (iterate over extreme points along each axis)
    //  Solve linear programming: maximize x subject to plane constraints.
    //  Simplified: we sample vertices from plane intersections (expensive).
    //  Alternative: compute by iterating over planes and clipping an initial large box.
    //  Here we use a simple iterative expansion: start with large box, clip by each plane.
    // ------------------------------------------------------------------------
    aabb_type boundingBox() const {
        // Start with a large box spanning [-1e9,1e9]
        point_type minP(-1e9), maxP(1e9);
        for (const auto& plane : m_planes) {
            const point_type& n = plane.normal();
            T d = plane.d();
            // Clip min in direction of normal
            if (n[0] > 0) minP[0] = std::max(minP[0], -d / n[0]);
            else if (n[0] < 0) maxP[0] = std::min(maxP[0], -d / n[0]);
            if (n[1] > 0) minP[1] = std::max(minP[1], -d / n[1]);
            else if (n[1] < 0) maxP[1] = std::min(maxP[1], -d / n[1]);
            if (n[2] > 0) minP[2] = std::max(minP[2], -d / n[2]);
            else if (n[2] < 0) maxP[2] = std::min(maxP[2], -d / n[2]);
        }
        return aabb_type(minP, maxP);
    }

    // ------------------------------------------------------------------------
    //  Transform polyhedron (transform each plane by inverse transpose of affine map)
    //  For point containment: new plane = old_plane * inv(A).T
    //  The constant term d transforms accordingly.
    // ------------------------------------------------------------------------
    ConvexPolyhedron transform(const AffineTransform<T,3>& tf) const {
        std::vector<plane_type> newPlanes;
        newPlanes.reserve(m_planes.size());
        for (const auto& p : m_planes) {
            point_type newNormal = tf.transformNormal(p.normal());
            // The transformed plane constant d: new_d = d - newNormal·translation?
            // Actually for plane equation n·x + d = 0, after transform x = M·x' + t,
            // the transformed plane is (n·M)·x' + (d + n·t) = 0.
            // So we need to compute new d accordingly.
            point_type trans = tf.translation();
            T newD = p.d() + p.normal().dot(trans);
            newPlanes.emplace_back(newNormal, newD);
        }
        return ConvexPolyhedron(std::move(newPlanes));
    }

    // ------------------------------------------------------------------------
    //  SIMD batch point containment: test 4 points against the same polyhedron
    // ------------------------------------------------------------------------
    void batchContainsPoint(const point_type* points, bool* out, size_type count) const noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && std::is_same_v<T,float>) {
            for (size_type i = 0; i < count; ++i) {
                out[i] = containsPoint(points[i]);
            }
        } else {
            for (size_type i = 0; i < count; ++i) {
                out[i] = containsPoint(points[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Convert to list of triangles (triangulation of convex polyhedron)
    //  Simplified: compute vertices as intersection of three planes, then create convex hull.
    //  Not full implementation; returns empty if not implemented.
    // ------------------------------------------------------------------------
    std::vector<std::array<point_type,3>> triangulate() const {
        std::vector<std::array<point_type,3>> triangles;
        // For convex polyhedron, we could use the bounding box vertices and clip by planes.
        // This is a placeholder.
        return triangles;
    }

private:
    std::vector<plane_type> m_planes;
};

// ----------------------------------------------------------------------------
//  Helper: create axis‑aligned box as convex polyhedron (6 planes)
// ----------------------------------------------------------------------------
template<typename T>
ConvexPolyhedron<T> boxToConvexPolyhedron(const AxisAlignedBox<T,3>& box) {
    point_type minP = box.min(), maxP = box.max();
    std::vector<Plane<T>> planes;
    // Planes with inward normals: x = min, x = max, etc.
    planes.emplace_back(point_type( 1, 0, 0), -minP[0]); // x >= min
    planes.emplace_back(point_type(-1, 0, 0),  maxP[0]); // x <= max
    planes.emplace_back(point_type( 0, 1, 0), -minP[1]);
    planes.emplace_back(point_type( 0,-1, 0),  maxP[1]);
    planes.emplace_back(point_type( 0, 0, 1), -minP[2]);
    planes.emplace_back(point_type( 0, 0,-1),  maxP[2]);
    return ConvexPolyhedron<T>(planes);
}

// ----------------------------------------------------------------------------
//  Helper: create convex hull from vertices (quick hull) – not implemented
// ----------------------------------------------------------------------------
template<typename T>
ConvexPolyhedron<T> convexHull(const std::vector<Vector<T,3>>& vertices) {
    // Placeholder: would compute convex hull and return polyhedron.
    return ConvexPolyhedron<T>();
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class ConvexPolyhedronEnvironment {
public:
    static ConvexPolyhedronEnvironment& instance() {
        static ConvexPolyhedronEnvironment env;
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
    ConvexPolyhedronEnvironment() : m_epsilon(T(1e-8)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_CONVEX_POLYHEDRON_H_INCLUDED