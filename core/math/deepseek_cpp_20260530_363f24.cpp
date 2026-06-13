// File 0012 : core/math/sphere.h
// Sphere representation (center + radius), intersection tests with ray, aabb, plane, and point containment.

#pragma once

#include "vec3.h"
#include "aabb.h"   // for aabb intersection
#include "plane.h"  // for plane intersection
#include "ray.h"    // forward declaration of ray<T>
#include <cmath>
#include <limits>

namespace wp {

template <typename T> struct ray;

template <typename T>
struct alignas(sizeof(T) * 4) sphere {
    vec3<T> center;
    T       radius;

    constexpr sphere() noexcept : center(T(0)), radius(T(1)) {}
    constexpr sphere(const vec3<T>& c, T r) noexcept : center(c), radius(r) {}
    template <typename U> constexpr explicit sphere(const sphere<U>& o) noexcept : center(o.center), radius(static_cast<T>(o.radius)) {}

    constexpr T area() const noexcept { return T(4) * pi<T> * radius * radius; }
    constexpr T volume() const noexcept { return T(4) / T(3) * pi<T> * radius * radius * radius; }
    constexpr bool contains(const vec3<T>& p) const noexcept { return distance_sq(p, center) <= radius * radius; }
    constexpr T distance(const vec3<T>& p) const noexcept { return std::max(T(0), wp::distance(p, center) - radius); }

    constexpr bool intersects(const aabb<T>& box) const noexcept {
        vec3<T> closest = box.closest_point(center);
        return distance_sq(center, closest) <= radius * radius;
    }
    constexpr bool intersects(const plane<T>& p) const noexcept {
        T dist = p.distance(center);
        return std::abs(dist) <= radius;
    }
    constexpr bool intersects(const sphere& s) const noexcept {
        return distance_sq(center, s.center) <= (radius + s.radius) * (radius + s.radius);
    }

    // Ray intersection (declaration, defined in ray.h)
    template <typename U> constexpr bool intersect_ray(const ray<U>& r, U& t) const noexcept;

    constexpr sphere transform(const mat4<T>& m) const noexcept {
        vec4<T> c = mul(m, vec4<T>(center.x, center.y, center.z, T(1)));
        T uniform_scale = std::cbrt(determinant(mat3<T>(m.m00, m.m01, m.m02, m.m10, m.m11, m.m12, m.m20, m.m21, m.m22)));
        return sphere(vec3<T>(c.x, c.y, c.z) / c.w, radius * uniform_scale);
    }

    constexpr bool operator==(const sphere& o) const noexcept { return center == o.center && radius == o.radius; }
    constexpr bool operator!=(const sphere& o) const noexcept { return !(*this == o); }
};

// Intersection non-member functions
template <typename T> constexpr bool intersect(const aabb<T>& box, const sphere<T>& s) { return s.intersects(box); }
template <typename T> constexpr bool intersect(const sphere<T>& s, const aabb<T>& box) { return s.intersects(box); }
template <typename T> constexpr bool intersect(const sphere<T>& a, const sphere<T>& b) { return a.intersects(b); }
template <typename T> constexpr bool intersect(const plane<T>& p, const sphere<T>& s) { return s.intersects(p); }
template <typename T> constexpr bool intersect(const sphere<T>& s, const plane<T>& p) { return s.intersects(p); }

using spheref = sphere<float>;
using sphered = sphere<double>;

} // namespace wp