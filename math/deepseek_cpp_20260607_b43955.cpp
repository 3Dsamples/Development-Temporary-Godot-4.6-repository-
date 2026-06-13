//File group name : OrthoTree Math
//File 0035 : core/math/sdf.h
//Signed distance functions (SDF) for primitives (sphere, box, torus, cylinder, cone, etc.) and operations (union, smooth union, intersection, subtraction, transformation). Provides ray marching, bounding box, and SIMD batch evaluation.

#ifndef ORTHOTREE_CORE_MATH_SDF_H_INCLUDED
#define ORTHOTREE_CORE_MATH_SDF_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "ray_intersection.h"
#include "geometry_queries.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <functional>
#include <memory>
#include <algorithm>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Signed distance function (SDF) base class.
//  Provides virtual evaluation, bounding box, and ray marching.
// ============================================================================
template<typename T = float>
class SDF {
public:
    using value_type = T;
    using point_type = Vector<T, 3>;
    using ray_type = Ray<T, 3>;
    using aabb_type = AxisAlignedBox<T, 3>;

    virtual ~SDF() = default;
    virtual T eval(const point_type& p) const = 0;
    virtual aabb_type boundingBox() const = 0;

    // Ray marching (sphere tracing) to find surface intersection.
    // Returns true if hit, outputs t (distance) and hit point.
    bool rayMarch(const ray_type& ray, T maxDist, T& t, point_type& point, T eps = T(1e-5), int maxIter = 200) const {
        t = T(0);
        for (int iter = 0; iter < maxIter; ++iter) {
            point = ray.origin() + ray.direction() * t;
            T d = eval(point);
            if (d < eps) return true;
            if (d > maxDist - t) return false;
            t += d;
            if (t > maxDist) return false;
        }
        return false;
    }

    // SIMD batch: evaluate 4 points at once (if derived class supports)
    virtual void batchEval(const point_type* points, T* out, size_t count) const {
        for (size_t i = 0; i < count; ++i) out[i] = eval(points[i]);
    }
};

// ============================================================================
//  Primitive SDFs
// ============================================================================
template<typename T>
class SphereSDF : public SDF<T> {
public:
    SphereSDF(const point_type& center, T radius) : m_center(center), m_radius(radius) {}
    T eval(const point_type& p) const override { return (p - m_center).length() - m_radius; }
    aabb_type boundingBox() const override {
        point_type ext(m_radius);
        return aabb_type(m_center - ext, m_center + ext);
    }
private:
    point_type m_center;
    T m_radius;
};

template<typename T>
class BoxSDF : public SDF<T> {
public:
    BoxSDF(const point_type& center, const point_type& halfExtents) : m_center(center), m_half(halfExtents) {}
    T eval(const point_type& p) const override {
        point_type d = (p - m_center).abs() - m_half;
        T outside = d.maxComponent();
        if (outside > T(0)) return outside;
        return std::max({d[0], d[1], d[2]});
    }
    aabb_type boundingBox() const override {
        return aabb_type(m_center - m_half, m_center + m_half);
    }
private:
    point_type m_center, m_half;
};

template<typename T>
class TorusSDF : public SDF<T> {
public:
    TorusSDF(const point_type& center, T majorR, T minorR, const point_type& axis = point_type(0,0,1))
        : m_center(center), m_major(majorR), m_minor(minorR), m_axis(axis.normalized()) {}
    T eval(const point_type& p) const override {
        point_type d = p - m_center;
        // Project onto plane perpendicular to axis
        T proj = d.dot(m_axis);
        point_type radial = d - m_axis * proj;
        T q = radial.length() - m_major;
        return std::sqrt(q*q + proj*proj) - m_minor;
    }
    aabb_type boundingBox() const override {
        T r = m_major + m_minor;
        point_type ext(r);
        return aabb_type(m_center - ext, m_center + ext);
    }
private:
    point_type m_center, m_axis;
    T m_major, m_minor;
};

template<typename T>
class CylinderSDF : public SDF<T> {
public:
    CylinderSDF(const point_type& center, const point_type& axis, T height, T radius)
        : m_center(center), m_axis(axis.normalized()), m_height(height), m_radius(radius) {}
    T eval(const point_type& p) const override {
        point_type d = p - m_center;
        T proj = d.dot(m_axis);
        point_type radial = d - m_axis * proj;
        T axialDist = std::abs(proj) - m_height * T(0.5);
        T radialDist = radial.length() - m_radius;
        return std::max(axialDist, radialDist);
    }
    aabb_type boundingBox() const override {
        point_type half = m_axis * (m_height * T(0.5));
        point_type minP = m_center - half, maxP = m_center + half;
        point_type rad(m_radius);
        minP = minP - rad;
        maxP = maxP + rad;
        return aabb_type(minP, maxP);
    }
private:
    point_type m_center, m_axis;
    T m_height, m_radius;
};

// ============================================================================
//  SDF operations (union, smooth union, intersection, subtraction)
// ============================================================================
template<typename T>
class UnionSDF : public SDF<T> {
public:
    UnionSDF(std::shared_ptr<SDF<T>> a, std::shared_ptr<SDF<T>> b) : m_a(a), m_b(b) {}
    T eval(const point_type& p) const override { return std::min(m_a->eval(p), m_b->eval(p)); }
    aabb_type boundingBox() const override { return m_a->boundingBox().hull(m_b->boundingBox()); }
private:
    std::shared_ptr<SDF<T>> m_a, m_b;
};

template<typename T>
class SmoothUnionSDF : public SDF<T> {
public:
    SmoothUnionSDF(std::shared_ptr<SDF<T>> a, std::shared_ptr<SDF<T>> b, T k) : m_a(a), m_b(b), m_k(k) {}
    T eval(const point_type& p) const override {
        T d1 = m_a->eval(p), d2 = m_b->eval(p);
        T h = std::max(T(0), std::min(T(1), (d2 - d1) / m_k + T(0.5)));
        return lerp(d2, d1, h) - m_k * h * (T(1)-h);
    }
    aabb_type boundingBox() const override { return m_a->boundingBox().hull(m_b->boundingBox()); }
private:
    std::shared_ptr<SDF<T>> m_a, m_b;
    T m_k;
};

template<typename T>
class IntersectionSDF : public SDF<T> {
public:
    IntersectionSDF(std::shared_ptr<SDF<T>> a, std::shared_ptr<SDF<T>> b) : m_a(a), m_b(b) {}
    T eval(const point_type& p) const override { return std::max(m_a->eval(p), m_b->eval(p)); }
    aabb_type boundingBox() const override { return m_a->boundingBox().intersect(m_b->boundingBox()); }
private:
    std::shared_ptr<SDF<T>> m_a, m_b;
};

template<typename T>
class SubtractionSDF : public SDF<T> {
public:
    SubtractionSDF(std::shared_ptr<SDF<T>> a, std::shared_ptr<SDF<T>> b) : m_a(a), m_b(b) {}
    T eval(const point_type& p) const override { return std::max(m_a->eval(p), -m_b->eval(p)); }
    aabb_type boundingBox() const override { return m_a->boundingBox(); }
private:
    std::shared_ptr<SDF<T>> m_a, m_b;
};

template<typename T>
class TransformSDF : public SDF<T> {
public:
    TransformSDF(std::shared_ptr<SDF<T>> sdf, const AffineTransform<T,3>& tf)
        : m_sdf(sdf), m_invTf(tf.inverse()) {}
    T eval(const point_type& p) const override {
        point_type local = m_invTf.transform(p);
        return m_sdf->eval(local);
    }
    aabb_type boundingBox() const override {
        return m_sdf->boundingBox().transform(m_invTf.inverse()); // simplified
    }
private:
    std::shared_ptr<SDF<T>> m_sdf;
    AffineTransform<T,3> m_invTf;
};

// ----------------------------------------------------------------------------
//  SIMD batch SDF evaluation for multiple points (using virtual dispatch)
//  This is a helper that calls virtual eval sequentially.
//  Derived classes could override batchEval for better performance.
// ----------------------------------------------------------------------------
template<typename T>
void batchEvalSDF(const SDF<T>* sdf, const typename SDF<T>::point_type* points,
                  T* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = sdf->eval(points[i]);
    }
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class SDFEnvironment {
public:
    static SDFEnvironment& instance() {
        static SDFEnvironment env;
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
    SDFEnvironment() : m_epsilon(T(1e-8)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_SDF_H_INCLUDED