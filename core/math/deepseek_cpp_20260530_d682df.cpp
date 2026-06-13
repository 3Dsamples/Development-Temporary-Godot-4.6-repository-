// File 0014 : core/math/capsule.h
// Capsule primitive defined by a line segment (a,b) and a radius; distance, containment, intersection with sphere/aabb/ray, and transformation.

#pragma once

#include "vec3.h"
#include "sphere.h"
#include "aabb.h"
#include "ray.h"            // for ray<T> and intersect_ray_capsule free function
#include <algorithm>
#include <cmath>

namespace wp {

template <typename T>
struct capsule {
    vec3<T> a, b;        // endpoints of the inner segment
    T       radius;

    constexpr capsule() noexcept : a(T(0), T(-1), T(0)), b(T(0), T(1), T(0)), radius(T(0.5)) {}
    constexpr capsule(const vec3<T>& p0, const vec3<T>& p1, T r) noexcept : a(p0), b(p1), radius(r) {}
    template <typename U> constexpr explicit capsule(const capsule<U>& o) noexcept : a(o.a), b(o.b), radius(static_cast<T>(o.radius)) {}

    constexpr vec3<T> segment_vector() const noexcept { return b - a; }
    constexpr T       segment_length() const noexcept { return wp::length(segment_vector()); }
    constexpr vec3<T> segment_direction() const noexcept { return normalize(segment_vector()); }
    constexpr vec3<T> center() const noexcept { return (a + b) * T(0.5); }

    // Closest point on segment to a point
    constexpr vec3<T> closest_point_segment(const vec3<T>& p) const noexcept {
        vec3<T> ab = b - a;
        T t = dot(p - a, ab) / dot(ab, ab);
        t = clamp(t, T(0), T(1));
        return a + ab * t;
    }

    // Closest point on the capsule surface to a point (including interior)
    constexpr vec3<T> closest_point(const vec3<T>& p) const noexcept {
        vec3<T> seg_pt = closest_point_segment(p);
        vec3<T> delta = p - seg_pt;
        T dist_sq = length_sq(delta);
        if (dist_sq <= radius * radius) return seg_pt;
        return seg_pt + delta * (radius / std::sqrt(dist_sq));
    }

    // Distance from point to capsule surface
    constexpr T distance(const vec3<T>& p) const noexcept {
        return std::max(T(0), wp::distance(p, closest_point_segment(p)) - radius);
    }

    // Containment
    constexpr bool contains(const vec3<T>& p) const noexcept {
        return wp::distance(p, closest_point_segment(p)) <= radius;
    }

    // Bounding sphere
    constexpr sphere<T> bounding_sphere() const noexcept {
        return sphere<T>(center(), segment_length() * T(0.5) + radius);
    }

    // Bounding AABB
    constexpr aabb<T> bounding_aabb() const noexcept {
        // Two spheres at a and b plus radius, take union
        aabb<T> box;
        box.expand(a - vec3<T>(radius));
        box.expand(a + vec3<T>(radius));
        box.expand(b - vec3<T>(radius));
        box.expand(b + vec3<T>(radius));
        return box;
    }

    // Intersection tests
    constexpr bool intersects(const sphere<T>& s) const noexcept {
        return distance(s.center) <= s.radius;
    }
    constexpr bool intersects(const aabb<T>& box) const noexcept {
        // approximate with bounding sphere first, then more precise?
        // Use closest point to box center and check distance
        vec3<T> cp = closest_point_segment(box.center());
        T dist = box.distance(cp);
        return dist <= radius;
    }
    constexpr bool intersects(const capsule& other) const noexcept {
        // distance between two segments vs radius sum
        T dist = segment_segment_distance(a, b, other.a, other.b);
        return dist <= radius + other.radius;
    }

    // Ray intersection
    template <typename U>
    constexpr bool intersect_ray(const ray<U>& r, U& t) const noexcept {
        return intersect_ray_capsule(r, a, b, radius, t);
    }

    // Transform (assumes uniform scale)
    constexpr capsule transform(const mat4<T>& m) const noexcept {
        vec4<T> va = mul(m, vec4<T>(a.x, a.y, a.z, T(1)));
        vec4<T> vb = mul(m, vec4<T>(b.x, b.y, b.z, T(1)));
        T scale = std::cbrt(determinant(mat3<T>(m.m00, m.m01, m.m02, m.m10, m.m11, m.m12, m.m20, m.m21, m.m22)));
        return capsule(vec3<T>(va.x, va.y, va.z) / va.w,
                       vec3<T>(vb.x, vb.y, vb.z) / vb.w,
                       radius * scale);
    }

    constexpr bool operator==(const capsule& o) const noexcept { return a == o.a && b == o.b && radius == o.radius; }
    constexpr bool operator!=(const capsule& o) const noexcept { return !(*this == o); }

private:
    // Helper: distance between two line segments (closest points)
    static T segment_segment_distance(const vec3<T>& a0, const vec3<T>& a1,
                                      const vec3<T>& b0, const vec3<T>& b1) noexcept {
        vec3<T> d1 = a1 - a0;
        vec3<T> d2 = b1 - b0;
        vec3<T> r = a0 - b0;
        T a11 = dot(d1, d1);
        T a12 = dot(d1, d2);
        T a22 = dot(d2, d2);
        T b1 = dot(r, d1);
        T b2 = dot(r, d2);
        T det = a11 * a22 - a12 * a12;
        T t1, t2;
        if (det < MathConst<T>::epsilon) {
            t1 = T(0);
            t2 = (a12 > T(0) ? -b2 / a22 : -b1 / a11);
        } else {
            t1 = (a12 * b2 - a22 * b1) / det;
            t2 = (a12 * b1 - a11 * b2) / det;
        }
        t1 = clamp(t1, T(0), T(1));
        t2 = clamp(t2, T(0), T(1));
        vec3<T> p1 = a0 + d1 * t1;
        vec3<T> p2 = b0 + d2 * t2;
        return wp::distance(p1, p2);
    }
};

using capsulef = capsule<float>;
using capsuled = capsule<double>;

} // namespace wp