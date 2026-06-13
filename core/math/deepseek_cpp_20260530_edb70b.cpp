// File 0010 : core/math/plane.h
// 3D plane representation (normal + distance), construction, distance, intersection with ray/AABB/sphere, and transformation.

#pragma once

#include "vec3.h"
#include "mat4.h"
#include <cmath>
#include <limits>

namespace wp {

template <typename T>
struct plane {
    vec3<T> normal;   // unit length normal
    T       d;        // signed distance from origin: normal·point + d = 0

    constexpr plane() noexcept : normal(T(0), T(1), T(0)), d(T(0)) {}
    constexpr plane(const vec3<T>& n, T dist) noexcept : normal(n), d(dist) {}
    constexpr plane(const vec3<T>& p0, const vec3<T>& p1, const vec3<T>& p2) noexcept {
        normal = normalize(cross(p1 - p0, p2 - p0));
        d = -dot(normal, p0);
    }
    constexpr plane(const vec3<T>& point, const vec3<T>& norm) noexcept : normal(norm), d(-dot(norm, point)) {}
    template <typename U> constexpr explicit plane(const plane<U>& o) noexcept : normal(o.normal), d(static_cast<T>(o.d)) {}

    constexpr T distance(const vec3<T>& point) const noexcept { return dot(normal, point) + d; }
    constexpr bool is_on_plane(const vec3<T>& point, T epsilon = MathConst<T>::epsilon) const noexcept { return std::abs(distance(point)) <= epsilon; }
    constexpr int classify(const vec3<T>& point) const noexcept {
        T dist = distance(point);
        if (dist > MathConst<T>::epsilon) return 1;
        if (dist < -MathConst<T>::epsilon) return -1;
        return 0;
    }
    constexpr vec3<T> project(const vec3<T>& point) const noexcept { return point - normal * distance(point); }
    constexpr vec3<T> reflect(const vec3<T>& point) const noexcept { return point - normal * T(2) * distance(point); }

    constexpr bool intersect_ray(const vec3<T>& origin, const vec3<T>& dir, T& t) const noexcept {
        T denom = dot(normal, dir);
        if (std::abs(denom) < MathConst<T>::epsilon) return false;
        t = -(dot(normal, origin) + d) / denom;
        return t >= T(0);
    }
    constexpr bool intersect_ray_segment(const vec3<T>& from, const vec3<T>& to, T& t) const noexcept {
        vec3<T> dir = to - from;
        T denom = dot(normal, dir);
        if (std::abs(denom) < MathConst<T>::epsilon) return false;
        t = -(dot(normal, from) + d) / denom;
        return t >= T(0) && t <= T(1);
    }

    constexpr plane normalized() const noexcept {
        T len = length(normal);
        if (len < MathConst<T>::epsilon) return *this;
        T inv = T(1) / len;
        return plane(normal * inv, d * inv);
    }

    constexpr plane transform(const mat4<T>& m) const noexcept {
        mat4<T> inv_trans = transpose(inverse(m));
        vec4<T> p(normal.x, normal.y, normal.z, -d);
        vec4<T> np = mul(inv_trans, p);
        return plane(vec3<T>(np.x, np.y, np.z), -np.w);
    }

    constexpr bool operator==(const plane& o) const noexcept { return normal == o.normal && d == o.d; }
    constexpr bool operator!=(const plane& o) const noexcept { return !(*this == o); }
};

// Intersection test helpers
template <typename T> constexpr bool intersect(const plane<T>& p, const struct aabb<T>& box) noexcept;
template <typename T> constexpr bool intersect(const plane<T>& p, const struct sphere<T>& s) noexcept;
template <typename T> constexpr bool intersect(const plane<T>& p, const struct ray<T>& r, T& t) noexcept;

using planef = plane<float>;
using planed = plane<double>;

} // namespace wp