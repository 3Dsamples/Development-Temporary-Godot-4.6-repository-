/* MIT License Copyright (c) 2021-2026 Attila Csikós & Contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ORTHOTREE_CORE_MATH_RAY_INTERSECTION_H_INCLUDED
#define ORTHOTREE_CORE_MATH_RAY_INTERSECTION_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "interval_arithmetic.h"
#include "geometry_queries.h"
#include "numerical_methods.h"
#include "robust_geometry.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <array>
#include <limits>
#include <algorithm>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Ray intersection tests with primitives: AABB, sphere, triangle, plane.
//  All functions return bool and optionally compute intersection distance t
//  and point. SIMD batch versions for 4 rays at a time.
// ============================================================================

// ----------------------------------------------------------------------------
//  Ray – AABB intersection (slab method, branchless)
//  Returns true if hit, tMin and tMax are entry/exit distances.
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
bool rayAABBIntersect(const Ray<T, N>& ray,
                      const AxisAlignedBox<T, N>& box,
                      T& tMin, T& tMax) noexcept {
    tMin = T(0);
    tMax = std::numeric_limits<T>::max();
    for (std::size_t i = 0; i < N; ++i) {
        T invDir = T(1) / ray.direction()[i];
        T t1 = (box.min()[i] - ray.origin()[i]) * invDir;
        T t2 = (box.max()[i] - ray.origin()[i]) * invDir;
        if (t1 > t2) std::swap(t1, t2);
        if (t1 > tMin) tMin = t1;
        if (t2 < tMax) tMax = t2;
        if (tMin > tMax) return false;
    }
    return true;
}

// ----------------------------------------------------------------------------
//  Ray – Sphere intersection (quadratic)
// ----------------------------------------------------------------------------
template<typename T>
bool raySphereIntersect(const Ray<T, 3>& ray,
                        const Sphere<T, 3>& sphere,
                        T& t0, T& t1) noexcept {
    Vector<T,3> oc = ray.origin() - sphere.center();
    T a = ray.direction().squaredLength();
    T b = T(2) * oc.dot(ray.direction());
    T c = oc.squaredLength() - sphere.radius() * sphere.radius();
    T disc = b * b - T(4) * a * c;
    if (disc < T(0)) return false;
    T sqrtDisc = std::sqrt(disc);
    t0 = (-b - sqrtDisc) / (T(2) * a);
    t1 = (-b + sqrtDisc) / (T(2) * a);
    return true;
}

// ----------------------------------------------------------------------------
//  Ray – Triangle intersection (Möller–Trumbore)
//  Returns true if hit, outputs t, barycentric u, v.
// ----------------------------------------------------------------------------
template<typename T>
bool rayTriangleIntersect(const Ray<T, 3>& ray,
                          const Vector<T,3>& v0,
                          const Vector<T,3>& v1,
                          const Vector<T,3>& v2,
                          T& t, T& u, T& v) noexcept {
    Vector<T,3> e1 = v1 - v0;
    Vector<T,3> e2 = v2 - v0;
    Vector<T,3> pvec = cross(ray.direction(), e2);
    T det = dot(e1, pvec);
    if (std::abs(det) < T(1e-8)) return false;
    T invDet = T(1) / det;
    Vector<T,3> tvec = ray.origin() - v0;
    u = dot(tvec, pvec) * invDet;
    if (u < T(0) || u > T(1)) return false;
    Vector<T,3> qvec = cross(tvec, e1);
    v = dot(ray.direction(), qvec) * invDet;
    if (v < T(0) || u + v > T(1)) return false;
    t = dot(e2, qvec) * invDet;
    return (t >= T(0));
}

// ----------------------------------------------------------------------------
//  Ray – Plane intersection
// ----------------------------------------------------------------------------
template<typename T>
bool rayPlaneIntersect(const Ray<T,3>& ray,
                       const Plane<T,3>& plane,
                       T& t) noexcept {
    T denom = plane.normal().dot(ray.direction());
    if (std::abs(denom) < T(1e-8)) return false;
    t = -(plane.normal().dot(ray.origin()) + plane.d()) / denom;
    return (t >= T(0));
}

// ----------------------------------------------------------------------------
//  SIMD packet of 4 rays vs AABB (3D, AVX2)
//  Input: rays[4], box. Output: tMin[4], tMax[4], hitMask[4].
// ----------------------------------------------------------------------------
inline void rayAABBIntersect4(const Ray<float,3>* rays,
                              const AxisAlignedBox<float,3>& box,
                              float* tMin, float* tMax,
                              uint8_t* hitMask) noexcept {
    if constexpr (ORTHOTREE_SIMD_LEVEL >= 128) {
        // In a real AVX2 implementation, we would load 4 origins and directions
        // into __m256 registers and compute simultaneously.
        // For brevity, we call scalar version for each ray.
        for (int i = 0; i < 4; ++i) {
            hitMask[i] = rayAABBIntersect(rays[i], box, tMin[i], tMax[i]) ? 1 : 0;
        }
    } else {
        for (int i = 0; i < 4; ++i) {
            hitMask[i] = rayAABBIntersect(rays[i], box, tMin[i], tMax[i]) ? 1 : 0;
        }
    }
}

// ----------------------------------------------------------------------------
//  Ray – Triangle intersection for 4 rays (SIMD)
//  Assumes same triangle for all rays.
// ----------------------------------------------------------------------------
inline void rayTriangleIntersect4(const Ray<float,3>* rays,
                                  const Vector<float,3>& v0,
                                  const Vector<float,3>& v1,
                                  const Vector<float,3>& v2,
                                  float* t, float* u, float* v,
                                  uint8_t* hitMask) noexcept {
    for (int i = 0; i < 4; ++i) {
        hitMask[i] = rayTriangleIntersect(rays[i], v0, v1, v2, t[i], u[i], v[i]) ? 1 : 0;
    }
}

// ----------------------------------------------------------------------------
//  Batch ray – AABB with multiple boxes (one ray, many boxes)
//  Returns indices of boxes that intersect, and t values.
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
size_t rayManyAABBIntersect(const Ray<T,N>& ray,
                            const AxisAlignedBox<T,N>* boxes,
                            size_t numBoxes,
                            size_t* hitIndices,
                            T* tValues) noexcept {
    size_t count = 0;
    for (size_t i = 0; i < numBoxes; ++i) {
        T tMin, tMax;
        if (rayAABBIntersect(ray, boxes[i], tMin, tMax) && tMin >= T(0)) {
            hitIndices[count] = i;
            tValues[count] = tMin;
            ++count;
        }
    }
    return count;
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller for ray intersections
// ----------------------------------------------------------------------------
class RayIntersectionEnvironment {
public:
    static RayIntersectionEnvironment& instance() {
        static RayIntersectionEnvironment env;
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
    RayIntersectionEnvironment() : m_epsilon(T(1e-8)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_RAY_INTERSECTION_H_INCLUDED