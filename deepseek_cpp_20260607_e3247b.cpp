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
#pragma once
#ifndef ORTHOTREE_CORE_MATH_GEOMETRY_QUERIES_H_INCLUDED
#define ORTHOTREE_CORE_MATH_GEOMETRY_QUERIES_H_INCLUDED

#include "vector_math.h"
#include "interval_arithmetic.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace OrthoTree::Math {

template<typename T, std::size_t N>
class AxisAlignedBox {
public:
    using value_type = T;
    using size_type = std::size_t;
    
    static constexpr size_type dimension() noexcept { return N; }
    
    constexpr AxisAlignedBox() noexcept
        : m_min(Vector<T, N>(std::numeric_limits<T>::max())),
          m_max(Vector<T, N>(std::numeric_limits<T>::lowest())) {}
    
    constexpr AxisAlignedBox(const Vector<T, N>& min, const Vector<T, N>& max) noexcept
        : m_min(min), m_max(max) {}
    
    constexpr AxisAlignedBox(const Vector<T, N>& center, T radius) noexcept
        : m_min(center - Vector<T, N>(radius)),
          m_max(center + Vector<T, N>(radius)) {}
    
    constexpr const Vector<T, N>& min() const noexcept { return m_min; }
    constexpr const Vector<T, N>& max() const noexcept { return m_max; }
    constexpr void setMin(const Vector<T, N>& min) noexcept { m_min = min; }
    constexpr void setMax(const Vector<T, N>& max) noexcept { m_max = max; }
    
    constexpr Vector<T, N> center() const noexcept {
        return (m_min + m_max) * T{0.5};
    }
    
    constexpr Vector<T, N> halfExtents() const noexcept {
        return (m_max - m_min) * T{0.5};
    }
    
    constexpr Vector<T, N> extents() const noexcept {
        return m_max - m_min;
    }
    
    constexpr T volume() const noexcept {
        T vol = T{1};
        for (size_type i = 0; i < N; ++i) {
            vol *= (m_max[i] - m_min[i]);
        }
        return vol;
    }
    
    constexpr T surfaceArea() const noexcept {
        if constexpr (N == 2) {
            return (m_max[0] - m_min[0]) * (m_max[1] - m_min[1]);
        } else if constexpr (N == 3) {
            Vector<T, 3> ext = extents();
            return T{2} * (ext[0] * ext[1] + ext[0] * ext[2] + ext[1] * ext[2]);
        }
        return T{0};
    }
    
    constexpr bool isEmpty() const noexcept {
        for (size_type i = 0; i < N; ++i) {
            if (m_min[i] > m_max[i]) return true;
        }
        return false;
    }
    
    constexpr bool contains(const Vector<T, N>& point) const noexcept {
        for (size_type i = 0; i < N; ++i) {
            if (point[i] < m_min[i] || point[i] > m_max[i]) return false;
        }
        return true;
    }
    
    constexpr bool contains(const AxisAlignedBox& other) const noexcept {
        for (size_type i = 0; i < N; ++i) {
            if (other.m_min[i] < m_min[i] || other.m_max[i] > m_max[i]) return false;
        }
        return true;
    }
    
    constexpr bool overlaps(const AxisAlignedBox& other) const noexcept {
        for (size_type i = 0; i < N; ++i) {
            if (m_min[i] > other.m_max[i] || other.m_min[i] > m_max[i]) return false;
        }
        return true;
    }
    
    constexpr AxisAlignedBox intersect(const AxisAlignedBox& other) const noexcept {
        return AxisAlignedBox(
            m_min.componentWiseMax(other.m_min),
            m_max.componentWiseMin(other.m_max)
        );
    }
    
    constexpr AxisAlignedBox hull(const AxisAlignedBox& other) const noexcept {
        return AxisAlignedBox(
            m_min.componentWiseMin(other.m_min),
            m_max.componentWiseMax(other.m_max)
        );
    }
    
    constexpr AxisAlignedBox& extend(const Vector<T, N>& point) noexcept {
        m_min = m_min.componentWiseMin(point);
        m_max = m_max.componentWiseMax(point);
        return *this;
    }
    
    constexpr AxisAlignedBox& extend(const AxisAlignedBox& other) noexcept {
        m_min = m_min.componentWiseMin(other.m_min);
        m_max = m_max.componentWiseMax(other.m_max);
        return *this;
    }
    
    constexpr AxisAlignedBox transform(const AffineTransform<T, N>& transform) const noexcept {
        Vector<T, N> corners[1 << N];
        for (size_type i = 0; i < (1 << N); ++i) {
            Vector<T, N> corner;
            for (size_type d = 0; d < N; ++d) {
                corner[d] = (i & (1 << d)) ? m_max[d] : m_min[d];
            }
            corners[i] = transform.transform(corner);
        }
        AxisAlignedBox result;
        for (size_type i = 0; i < (1 << N); ++i) {
            result.extend(corners[i]);
        }
        return result;
    }
    
    constexpr T distanceTo(const Vector<T, N>& point) const noexcept {
        T sqDist = T{0};
        for (size_type i = 0; i < N; ++i) {
            if (point[i] < m_min[i]) {
                T d = m_min[i] - point[i];
                sqDist += d * d;
            } else if (point[i] > m_max[i]) {
                T d = point[i] - m_max[i];
                sqDist += d * d;
            }
        }
        return std::sqrt(sqDist);
    }
    
    constexpr T squaredDistanceTo(const Vector<T, N>& point) const noexcept {
        T sqDist = T{0};
        for (size_type i = 0; i < N; ++i) {
            if (point[i] < m_min[i]) {
                T d = m_min[i] - point[i];
                sqDist += d * d;
            } else if (point[i] > m_max[i]) {
                T d = point[i] - m_max[i];
                sqDist += d * d;
            }
        }
        return sqDist;
    }
    
    constexpr Vector<T, N> closestPoint(const Vector<T, N>& point) const noexcept {
        Vector<T, N> result;
        for (size_type i = 0; i < N; ++i) {
            result[i] = std::clamp(point[i], m_min[i], m_max[i]);
        }
        return result;
    }
    
    constexpr Interval<T> projectOnAxis(const Vector<T, N>& axis) const noexcept {
        T dotMin = std::numeric_limits<T>::max();
        T dotMax = std::numeric_limits<T>::lowest();
        for (size_type i = 0; i < (1 << N); ++i) {
            Vector<T, N> corner;
            for (size_type d = 0; d < N; ++d) {
                corner[d] = (i & (1 << d)) ? m_max[d] : m_min[d];
            }
            T dot = corner.dot(axis);
            if (dot < dotMin) dotMin = dot;
            if (dot > dotMax) dotMax = dot;
        }
        return Interval<T>(dotMin, dotMax);
    }
    
    constexpr bool intersectRay(const Vector<T, N>& origin, const Vector<T, N>& direction,
                                 T& tMin, T& tMax) const noexcept {
        tMin = T{0};
        tMax = std::numeric_limits<T>::max();
        
        for (size_type i = 0; i < N; ++i) {
            T invDir = T{1} / direction[i];
            T t1 = (m_min[i] - origin[i]) * invDir;
            T t2 = (m_max[i] - origin[i]) * invDir;
            if (t1 > t2) std::swap(t1, t2);
            tMin = std::max(tMin, t1);
            tMax = std::min(tMax, t2);
            if (tMin > tMax) return false;
        }
        return true;
    }
    
    constexpr std::optional<Vector<T, N>> intersectRay(const Vector<T, N>& origin,
                                                        const Vector<T, N>& direction) const noexcept {
        T tMin, tMax;
        if (!intersectRay(origin, direction, tMin, tMax)) return std::nullopt;
        return origin + direction * tMin;
    }
    
private:
    Vector<T, N> m_min;
    Vector<T, N> m_max;
};

template<typename T, std::size_t N>
class Sphere {
public:
    using value_type = T;
    using size_type = std::size_t;
    
    static constexpr size_type dimension() noexcept { return N; }
    
    constexpr Sphere() noexcept : m_center(), m_radius(T{0}) {}
    constexpr Sphere(const Vector<T, N>& center, T radius) noexcept
        : m_center(center), m_radius(radius) {}
    
    constexpr const Vector<T, N>& center() const noexcept { return m_center; }
    constexpr T radius() const noexcept { return m_radius; }
    constexpr void setCenter(const Vector<T, N>& center) noexcept { m_center = center; }
    constexpr void setRadius(T radius) noexcept { m_radius = radius; }
    
    constexpr T volume() const noexcept {
        if constexpr (N == 2) return T{3.14159265358979323846} * m_radius * m_radius;
        else if constexpr (N == 3) return T{4.1887902047863909846} * m_radius * m_radius * m_radius;
        return T{0};
    }
    
    constexpr T surfaceArea() const noexcept {
        if constexpr (N == 2) return T{6.2831853071795864769} * m_radius;
        else if constexpr (N == 3) return T{12.566370614359172954} * m_radius * m_radius;
        return T{0};
    }
    
    constexpr bool contains(const Vector<T, N>& point) const noexcept {
        return m_center.squaredDistanceTo(point) <= m_radius * m_radius;
    }
    
    constexpr bool contains(const Sphere& other) const noexcept {
        return m_center.distanceTo(other.m_center) + other.m_radius <= m_radius;
    }
    
    constexpr bool overlaps(const Sphere& other) const noexcept {
        return m_center.distanceTo(other.m_center) <= m_radius + other.m_radius;
    }
    
    constexpr AxisAlignedBox<T, N> boundingBox() const noexcept {
        Vector<T, N> ext(m_radius);
        return AxisAlignedBox<T, N>(m_center - ext, m_center + ext);
    }
    
    constexpr Sphere hull(const Sphere& other) const noexcept {
        Vector<T, N> dir = other.m_center - m_center;
        T dist = dir.length();
        if (dist + other.m_radius <= m_radius) return *this;
        if (dist + m_radius <= other.m_radius) return other;
        T newRadius = (dist + m_radius + other.m_radius) * T{0.5};
        Vector<T, N> newCenter = m_center + dir * ((newRadius - m_radius) / dist);
        return Sphere(newCenter, newRadius);
    }
    
    constexpr bool intersectRay(const Vector<T, N>& origin, const Vector<T, N>& direction,
                                 T& t0, T& t1) const noexcept {
        Vector<T, N> oc = origin - m_center;
        T a = direction.squaredLength();
        T b = T{2} * oc.dot(direction);
        T c = oc.squaredLength() - m_radius * m_radius;
        T disc = b * b - T{4} * a * c;
        if (disc < T{0}) return false;
        T sqrtDisc = std::sqrt(disc);
        t0 = (-b - sqrtDisc) / (T{2} * a);
        t1 = (-b + sqrtDisc) / (T{2} * a);
        return true;
    }
    
    constexpr std::optional<Vector<T, N>> intersectRay(const Vector<T, N>& origin,
                                                        const Vector<T, N>& direction) const noexcept {
        T t0, t1;
        if (!intersectRay(origin, direction, t0, t1)) return std::nullopt;
        T t = (t0 >= T{0}) ? t0 : t1;
        if (t < T{0}) return std::nullopt;
        return origin + direction * t;
    }
    
private:
    Vector<T, N> m_center;
    T m_radius;
};

template<typename T, std::size_t N>
class Ray {
public:
    using value_type = T;
    using size_type = std::size_t;
    
    static constexpr size_type dimension() noexcept { return N; }
    
    constexpr Ray() noexcept : m_origin(), m_direction() {}
    constexpr Ray(const Vector<T, N>& origin, const Vector<T, N>& direction) noexcept
        : m_origin(origin), m_direction(direction) {}
    
    constexpr const Vector<T, N>& origin() const noexcept { return m_origin; }
    constexpr const Vector<T, N>& direction() const noexcept { return m_direction; }
    constexpr void setOrigin(const Vector<T, N>& origin) noexcept { m_origin = origin; }
    constexpr void setDirection(const Vector<T, N>& direction) noexcept { m_direction = direction; }
    
    constexpr Vector<T, N> pointAt(T t) const noexcept {
        return m_origin + m_direction * t;
    }
};

template<typename T, std::size_t N>
class Plane {
public:
    using value_type = T;
    using size_type = std::size_t;
    
    static constexpr size_type dimension() noexcept { return N; }
    
    constexpr Plane() noexcept : m_normal(), m_d(T{0}) {}
    constexpr Plane(const Vector<T, N>& normal, T d) noexcept
        : m_normal(normal.normalized()), m_d(d) {}
    constexpr Plane(const Vector<T, N>& normal, const Vector<T, N>& point) noexcept
        : m_normal(normal.normalized()), m_d(-m_normal.dot(point)) {}
    constexpr Plane(const Vector<T, N>& p1, const Vector<T, N>& p2, const Vector<T, N>& p3) noexcept {
        static_assert(N == 3, "Three-point plane requires 3D");
        m_normal = cross(p2 - p1, p3 - p1).normalized();
        m_d = -m_normal.dot(p1);
    }
    
    constexpr const Vector<T, N>& normal() const noexcept { return m_normal; }
    constexpr T d() const noexcept { return m_d; }
    constexpr void setNormal(const Vector<T, N>& normal) noexcept { m_normal = normal.normalized(); }
    constexpr void setD(T d) noexcept { m_d = d; }
    
    constexpr T signedDistance(const Vector<T, N>& point) const noexcept {
        return m_normal.dot(point) + m_d;
    }
    
    constexpr T distance(const Vector<T, N>& point) const noexcept {
        return std::abs(signedDistance(point));
    }
    
    constexpr Vector<T, N> project(const Vector<T, N>& point) const noexcept {
        return point - m_normal * signedDistance(point);
    }
    
    constexpr std::optional<Vector<T, N>> intersect(const Ray<T, N>& ray) const noexcept {
        T denom = m_normal.dot(ray.direction());
        if (std::abs(denom) < std::numeric_limits<T>::epsilon()) return std::nullopt;
        T t = -(m_normal.dot(ray.origin()) + m_d) / denom;
        if (t < T{0}) return std::nullopt;
        return ray.pointAt(t);
    }
    
    constexpr int classify(const Vector<T, N>& point, T eps = T{1e-8}) const noexcept {
        T dist = signedDistance(point);
        if (dist > eps) return 1;
        if (dist < -eps) return -1;
        return 0;
    }
    
private:
    Vector<T, N> m_normal;
    T m_d;
};

template<typename T>
using AABB2 = AxisAlignedBox<T, 2>;
template<typename T>
using AABB3 = AxisAlignedBox<T, 3>;

using AABB2f = AxisAlignedBox<float, 2>;
using AABB3f = AxisAlignedBox<float, 3>;
using AABB2d = AxisAlignedBox<double, 2>;
using AABB3d = AxisAlignedBox<double, 3>;

using Sphere2f = Sphere<float, 2>;
using Sphere3f = Sphere<float, 3>;
using Sphere2d = Sphere<double, 2>;
using Sphere3d = Sphere<double, 3>;

using Ray2f = Ray<float, 2>;
using Ray3f = Ray<float, 3>;
using Ray2d = Ray<double, 2>;
using Ray3d = Ray<double, 3>;

using Plane2f = Plane<float, 2>;
using Plane3f = Plane<float, 3>;
using Plane2d = Plane<double, 2>;
using Plane3d = Plane<double, 3>;

} // namespace OrthoTree::Math

#endif // ORTHOTREE_CORE_MATH_GEOMETRY_QUERIES_H_INCLUDED