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

#ifndef ORTHOTREE_CORE_MATH_EXTENDED_CURVED_SPACE_METRICS_H_INCLUDED
#define ORTHOTREE_CORE_MATH_EXTENDED_CURVED_SPACE_METRICS_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "../vector_math.h"
#include "../quaternion.h"
#include "../transform.h"
#include "../numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <array>
#include <cstdint>
#include <limits>

namespace OrthoTree {
namespace Math {
namespace Extended {

// ============================================================================
//  Curved space types: spherical, toroidal, hyperbolic, cylindrical
// ============================================================================
enum class CurvedSpaceType : uint8_t {
    Euclidean = 0,
    Spherical,
    Toroidal,
    Hyperbolic,
    Cylindrical
};

// ============================================================================
//  Metric tensor interface (for Riemannian geometry)
// ============================================================================
template<typename T, std::size_t N>
class MetricTensor {
public:
    using vec_type = Math::Vector<T, N>;
    using matrix_type = std::array<std::array<T, N>, N>;

    virtual ~MetricTensor() = default;
    virtual T distance(const vec_type& p, const vec_type& q) const = 0;
    virtual vec_type geodesic(const vec_type& p, const vec_type& dir, T t) const = 0;
    virtual matrix_type metricAt(const vec_type& p) const = 0;
};

// ============================================================================
//  Spherical space (radius R, 2D surface of sphere or 3D spherical coordinates)
// ============================================================================
template<typename T, std::size_t N>
class SphericalMetric : public MetricTensor<T, N> {
    static_assert(N == 2 || N == 3, "SphericalMetric: only 2D (sphere surface) or 3D (spherical space)");
public:
    using vec_type = Math::Vector<T, N>;

    explicit SphericalMetric(T radius = T(1)) noexcept : m_radius(radius) {}

    // Great-circle distance (2D) or geodesic in 3D spherical space
    T distance(const vec_type& p, const vec_type& q) const override {
        if constexpr (N == 2) {
            // p and q are (theta, phi) in radians
            T dtheta = p[0] - q[0];
            T dphi   = p[1] - q[1];
            T a = std::sin(dtheta/T(2)) * std::sin(dtheta/T(2)) +
                  std::cos(p[0]) * std::cos(q[0]) * std::sin(dphi/T(2)) * std::sin(dphi/T(2));
            T centralAngle = T(2) * std::asin(std::sqrt(std::max(T(0), std::min(T(1), a))));
            return m_radius * centralAngle;
        } else {
            // 3D spherical space: use standard Euclidean distance in spherical coordinates? Actually metric is:
            // ds^2 = dr^2 + r^2 dtheta^2 + r^2 sin^2 theta dphi^2. But here we treat p and q as (r,theta,phi)
            // For simplicity, we convert to Cartesian and compute chord length, then central angle.
            vec_type cartP = sphericalToCartesian(p);
            vec_type cartQ = sphericalToCartesian(q);
            T dot = cartP.dot(cartQ);
            T cosTheta = dot / (m_radius * m_radius);
            cosTheta = Math::clamp(cosTheta, T(-1), T(1));
            T angle = std::acos(cosTheta);
            return m_radius * angle;
        }
    }

    vec_type geodesic(const vec_type& p, const vec_type& dir, T t) const override {
        // For sphere: move along great circle. Not trivial; simplified: linear in angular space
        vec_type result = p + dir * t;
        if constexpr (N == 2) {
            // wrap theta
            result[0] = std::fmod(result[0], T(2 * Math::pi<T>()));
            if (result[0] < 0) result[0] += T(2 * Math::pi<T>());
            result[1] = std::fmod(result[1], T(2 * Math::pi<T>()));
            if (result[1] < 0) result[1] += T(2 * Math::pi<T>());
        }
        return result;
    }

    typename MetricTensor<T, N>::matrix_type metricAt(const vec_type& p) const override {
        typename MetricTensor<T, N>::matrix_type g;
        if constexpr (N == 2) {
            T sinTheta = std::sin(p[0]);
            g[0][0] = m_radius * m_radius;
            g[0][1] = T(0);
            g[1][0] = T(0);
            g[1][1] = m_radius * m_radius * sinTheta * sinTheta;
        } else {
            T r = p[0];
            T theta = p[1];
            T sinTheta = std::sin(theta);
            g[0][0] = T(1);
            g[0][1] = g[0][2] = T(0);
            g[1][0] = T(0);
            g[1][1] = r * r;
            g[1][2] = T(0);
            g[2][0] = T(0);
            g[2][1] = T(0);
            g[2][2] = r * r * sinTheta * sinTheta;
        }
        return g;
    }

    T radius() const noexcept { return m_radius; }

private:
    vec_type sphericalToCartesian(const vec_type& sph) const {
        if constexpr (N == 2) {
            // (theta, phi) -> (x,y,z) on sphere of radius m_radius
            T theta = sph[0];
            T phi   = sph[1];
            T x = m_radius * std::sin(theta) * std::cos(phi);
            T y = m_radius * std::sin(theta) * std::sin(phi);
            T z = m_radius * std::cos(theta);
            return vec_type(x, y, z);
        } else {
            // (r, theta, phi) -> (x,y,z)
            T r     = sph[0];
            T theta = sph[1];
            T phi   = sph[2];
            T x = r * std::sin(theta) * std::cos(phi);
            T y = r * std::sin(theta) * std::sin(phi);
            T z = r * std::cos(theta);
            return vec_type(x, y, z);
        }
    }

    T m_radius;
};

// ============================================================================
//  Toroidal space (2D torus or 3D torus)
// ============================================================================
template<typename T, std::size_t N>
class ToroidalMetric : public MetricTensor<T, N> {
    static_assert(N == 2 || N == 3, "ToroidalMetric: 2D or 3D torus");
public:
    using vec_type = Math::Vector<T, N>;

    ToroidalMetric(const vec_type& radii) noexcept : m_radii(radii) {}

    T distance(const vec_type& p, const vec_type& q) const override {
        T sumSq = T(0);
        for (std::size_t i = 0; i < N; ++i) {
            T delta = std::abs(p[i] - q[i]);
            T period = m_radii[i] * T(2 * Math::pi<T>());
            delta = std::min(delta, period - delta);
            sumSq += delta * delta;
        }
        return std::sqrt(sumSq);
    }

    vec_type geodesic(const vec_type& p, const vec_type& dir, T t) const override {
        vec_type result;
        for (std::size_t i = 0; i < N; ++i) {
            T period = m_radii[i] * T(2 * Math::pi<T>());
            T val = p[i] + dir[i] * t;
            val = std::fmod(val, period);
            if (val < 0) val += period;
            result[i] = val;
        }
        return result;
    }

    typename MetricTensor<T, N>::matrix_type metricAt(const vec_type&) const override {
        // Flat metric on torus (induced from Euclidean)
        typename MetricTensor<T, N>::matrix_type g;
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j)
                g[i][j] = (i == j) ? T(1) : T(0);
        return g;
    }

    vec_type radii() const noexcept { return m_radii; }

private:
    vec_type m_radii;
};

// ============================================================================
//  Hyperbolic space (Poincaré ball model, 2D/3D)
// ============================================================================
template<typename T, std::size_t N>
class HyperbolicMetric : public MetricTensor<T, N> {
    static_assert(N == 2 || N == 3, "HyperbolicMetric: 2D or 3D Poincaré ball");
public:
    using vec_type = Math::Vector<T, N>;

    explicit HyperbolicMetric(T curvature = T(-1)) noexcept : m_curvature(curvature) {}

    T distance(const vec_type& p, const vec_type& q) const override {
        // Poincaré ball metric: d(p,q) = acosh(1 + 2*|p-q|^2/((1-|p|^2)(1-|q|^2)))
        T p2 = p.squaredLength();
        T q2 = q.squaredLength();
        if (p2 >= T(1) || q2 >= T(1)) return std::numeric_limits<T>::max(); // outside disk
        T diff2 = (p - q).squaredLength();
        T denom = (T(1) - p2) * (T(1) - q2);
        if (denom <= T(0)) return std::numeric_limits<T>::max();
        T arg = T(1) + T(2) * diff2 / denom;
        if (arg < T(1)) arg = T(1);
        return std::acosh(arg) / std::sqrt(-m_curvature);
    }

    vec_type geodesic(const vec_type& p, const vec_type& dir, T t) const override {
        // Simplified exponential map (not exact, but approximation)
        vec_type result = p + dir * t;
        // Clamp to ball
        T norm = result.length();
        if (norm >= T(1)) {
            result = result * (T(0.9999) / norm);
        }
        return result;
    }

    typename MetricTensor<T, N>::matrix_type metricAt(const vec_type& p) const override {
        // g_ij = (4/(1-|p|^2)^2) * delta_ij
        T p2 = p.squaredLength();
        T factor = T(4) / ((T(1) - p2) * (T(1) - p2));
        typename MetricTensor<T, N>::matrix_type g;
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j)
                g[i][j] = (i == j) ? factor : T(0);
        return g;
    }

private:
    T m_curvature;
};

// ============================================================================
//  Cylindrical space (2D: (r, theta) with flat theta direction)
// ============================================================================
template<typename T, std::size_t N>
class CylindricalMetric : public MetricTensor<T, N> {
    static_assert(N == 2 || N == 3, "CylindricalMetric: 2D (r,theta) or 3D (r,theta,z)");
public:
    using vec_type = Math::Vector<T, N>;

    CylindricalMetric(T radius = T(1)) noexcept : m_radius(radius) {}

    T distance(const vec_type& p, const vec_type& q) const override {
        if constexpr (N == 2) {
            // p = (r1, theta1), q = (r2, theta2)
            T dr = p[0] - q[0];
            T dtheta = std::abs(p[1] - q[1]);
            T period = T(2 * Math::pi<T>());
            dtheta = std::min(dtheta, period - dtheta);
            T dsq = dr * dr + (m_radius * m_radius) * dtheta * dtheta;
            return std::sqrt(dsq);
        } else {
            T dr = p[0] - q[0];
            T dtheta = std::abs(p[1] - q[1]);
            T period = T(2 * Math::pi<T>());
            dtheta = std::min(dtheta, period - dtheta);
            T dz = p[2] - q[2];
            T dsq = dr * dr + (m_radius * m_radius) * dtheta * dtheta + dz * dz;
            return std::sqrt(dsq);
        }
    }

    vec_type geodesic(const vec_type& p, const vec_type& dir, T t) const override {
        vec_type result;
        result[0] = p[0] + dir[0] * t;
        if (result[0] < T(0)) result[0] = T(0); // radius non‑negative
        T angle = p[1] + dir[1] * t;
        angle = std::fmod(angle, T(2 * Math::pi<T>()));
        if (angle < 0) angle += T(2 * Math::pi<T>());
        result[1] = angle;
        if constexpr (N == 3) result[2] = p[2] + dir[2] * t;
        return result;
    }

    typename MetricTensor<T, N>::matrix_type metricAt(const vec_type& p) const override {
        typename MetricTensor<T, N>::matrix_type g;
        g[0][0] = T(1);
        g[0][1] = g[1][0] = T(0);
        if constexpr (N == 2) {
            g[1][1] = m_radius * m_radius;
        } else {
            g[1][1] = m_radius * m_radius;
            g[2][2] = T(1);
            g[0][2] = g[2][0] = g[1][2] = g[2][1] = T(0);
        }
        return g;
    }

private:
    T m_radius;
};

// ============================================================================
//  Dynamic environment controller for curved spaces
// ============================================================================
template<typename T, std::size_t N>
class CurvedSpaceController {
public:
    using vec_type = Math::Vector<T, N>;
    using metric_ptr = std::unique_ptr<MetricTensor<T, N>>;

    CurvedSpaceController() : m_type(CurvedSpaceType::Euclidean) {
        m_metrics[static_cast<uint8_t>(CurvedSpaceType::Euclidean)] = nullptr;
    }

    void setSpace(CurvedSpaceType type, const vec_type& params = vec_type(T(1))) {
        m_type = type;
        switch (type) {
            case CurvedSpaceType::Spherical:
                m_metrics[static_cast<uint8_t>(type)] = std::make_unique<SphericalMetric<T, N>>(params[0]);
                break;
            case CurvedSpaceType::Toroidal:
                m_metrics[static_cast<uint8_t>(type)] = std::make_unique<ToroidalMetric<T, N>>(params);
                break;
            case CurvedSpaceType::Hyperbolic:
                m_metrics[static_cast<uint8_t>(type)] = std::make_unique<HyperbolicMetric<T, N>>(params[0]);
                break;
            case CurvedSpaceType::Cylindrical:
                m_metrics[static_cast<uint8_t>(type)] = std::make_unique<CylindricalMetric<T, N>>(params[0]);
                break;
            default:
                m_metrics[static_cast<uint8_t>(type)] = nullptr;
                break;
        }
    }

    T distance(const vec_type& p, const vec_type& q) const {
        auto* metric = m_metrics[static_cast<uint8_t>(m_type)].get();
        if (metric) return metric->distance(p, q);
        // Euclidean fallback
        return (p - q).length();
    }

    vec_type geodesic(const vec_type& p, const vec_type& dir, T t) const {
        auto* metric = m_metrics[static_cast<uint8_t>(m_type)].get();
        if (metric) return metric->geodesic(p, dir, t);
        return p + dir * t;
    }

    // Convert from curved coordinates to Cartesian (embedding)
    vec_type toCartesian(const vec_type& curved) const {
        switch (m_type) {
            case CurvedSpaceType::Spherical: {
                if constexpr (N == 2) {
                    T theta = curved[0], phi = curved[1];
                    T r = (dynamic_cast<SphericalMetric<T, N>*>(m_metrics[static_cast<uint8_t>(m_type)].get()))->radius();
                    return vec_type(r * std::sin(theta) * std::cos(phi),
                                    r * std::sin(theta) * std::sin(phi),
                                    r * std::cos(theta));
                } else {
                    // 3D spherical -> Cartesian
                    T r = curved[0], theta = curved[1], phi = curved[2];
                    return vec_type(r * std::sin(theta) * std::cos(phi),
                                    r * std::sin(theta) * std::sin(phi),
                                    r * std::cos(theta));
                }
            }
            default:
                return curved;
        }
    }

    CurvedSpaceType currentSpace() const noexcept { return m_type; }

private:
    CurvedSpaceType m_type;
    std::array<metric_ptr, 4> m_metrics; // 4 types
};

// ============================================================================
//  Ray intersection in curved spaces (example: sphere intersection)
// ============================================================================
template<typename T>
bool raySphereIntersect(const Math::Ray<T, 3>& ray, const Math::Sphere<T, 3>& sphere, T& t) noexcept {
    Math::Vector<T, 3> oc = ray.origin() - sphere.center();
    T a = ray.direction().squaredLength();
    T b = T(2) * oc.dot(ray.direction());
    T c = oc.squaredLength() - sphere.radius() * sphere.radius();
    T disc = b * b - T(4) * a * c;
    if (disc < T(0)) return false;
    T sqrtDisc = std::sqrt(disc);
    T t0 = (-b - sqrtDisc) / (T(2) * a);
    T t1 = (-b + sqrtDisc) / (T(2) * a);
    t = (t0 >= T(0)) ? t0 : t1;
    if (t < T(0)) return false;
    return true;
}

// SIMD batch ray‑sphere intersection (for 4 rays)
inline void raySphereIntersect4(const Math::Ray<float, 3>* rays,
                                const Math::Sphere<float, 3>& sphere,
                                bool* hitMask, float* tOut) noexcept {
    if constexpr (ORTHOTREE_SIMD_LEVEL >= 128) {
        // Placeholder for real SIMD implementation (e.g., using SSE/AVX)
        for (int i = 0; i < 4; ++i) {
            hitMask[i] = raySphereIntersect(rays[i], sphere, tOut[i]);
        }
    } else {
        for (int i = 0; i < 4; ++i) {
            hitMask[i] = raySphereIntersect(rays[i], sphere, tOut[i]);
        }
    }
}

} // namespace Extended
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_EXTENDED_CURVED_SPACE_METRICS_H_INCLUDED