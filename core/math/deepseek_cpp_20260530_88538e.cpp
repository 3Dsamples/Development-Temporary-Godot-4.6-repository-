// File 0011 : core/math/aabb.h
// Axis-Aligned Bounding Box (min/max) with construction, merge, intersection tests, volume/center/extents, and transformations.

#pragma once

#include "vec3.h"
#include "mat4.h"
#include "plane.h"      // forward declared, now available
#include "sphere.h"     // will be defined later, forward declare
#include "ray.h"        // will be defined later, forward declare
#include <algorithm>
#include <limits>

namespace wp {

// Forward declarations for intersection (defined in respective headers)
template <typename T> struct sphere;
template <typename T> struct ray;

template <typename T>
struct alignas(sizeof(T) * 6) aabb {
    vec3<T> min;
    vec3<T> max;

    constexpr aabb() noexcept : min(max_val<T>, max_val<T>, max_val<T>), max(min_val<T>, min_val<T>, min_val<T>) {}
    constexpr aabb(const vec3<T>& mn, const vec3<T>& mx) noexcept : min(mn), max(mx) {}
    template <typename U> constexpr explicit aabb(const aabb<U>& o) noexcept : min(o.min), max(o.max) {}

    constexpr vec3<T> center() const noexcept { return (min + max) * T(0.5); }
    constexpr vec3<T> extent() const noexcept { return (max - min) * T(0.5); }
    constexpr vec3<T> size() const noexcept { return max - min; }
    constexpr T volume() const noexcept { vec3<T> s = size(); return s.x * s.y * s.z; }
    constexpr T surface_area() const noexcept { vec3<T> s = size(); return T(2) * (s.x*s.y + s.y*s.z + s.z*s.x); }
    constexpr bool is_empty() const noexcept { return min.x > max.x || min.y > max.y || min.z > max.z; }
    constexpr bool is_valid() const noexcept { return min.x <= max.x && min.y <= max.y && min.z <= max.z; }

    constexpr vec3<T> corner(int i) const noexcept {
        return vec3<T>(
            (i & 1) ? max.x : min.x,
            (i & 2) ? max.y : min.y,
            (i & 4) ? max.z : min.z
        );
    }
    constexpr vec3<T> closest_point(const vec3<T>& p) const noexcept {
        return vec3<T>(clamp(p.x, min.x, max.x), clamp(p.y, min.y, max.y), clamp(p.z, min.z, max.z));
    }
    constexpr T distance_sq(const vec3<T>& p) const noexcept { return length_sq(p - closest_point(p)); }
    constexpr T distance(const vec3<T>& p) const noexcept { return std::sqrt(distance_sq(p)); }

    constexpr bool contains(const vec3<T>& p) const noexcept { return p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y && p.z >= min.z && p.z <= max.z; }
    constexpr bool contains(const aabb& other) const noexcept { return min.x <= other.min.x && max.x >= other.max.x && min.y <= other.min.y && max.y >= other.max.y && min.z <= other.min.z && max.z >= other.max.z; }
    constexpr bool intersects(const aabb& other) const noexcept { return min.x <= other.max.x && max.x >= other.min.x && min.y <= other.max.y && max.y >= other.min.y && min.z <= other.max.z && max.z >= other.min.z; }

    constexpr aabb& expand(const vec3<T>& p) noexcept {
        min = wp::min(min, p);
        max = wp::max(max, p);
        return *this;
    }
    constexpr aabb& expand(const aabb& other) noexcept {
        min = wp::min(min, other.min);
        max = wp::max(max, other.max);
        return *this;
    }
    constexpr aabb expanded(const vec3<T>& p) const noexcept { aabb r = *this; r.expand(p); return r; }
    constexpr aabb expanded(const aabb& other) const noexcept { aabb r = *this; r.expand(other); return r; }

    constexpr aabb& grow(T amount) noexcept {
        min -= vec3<T>(amount);
        max += vec3<T>(amount);
        return *this;
    }
    constexpr aabb grown(T amount) const noexcept { aabb r = *this; r.grow(amount); return r; }

    constexpr aabb intersection(const aabb& other) const noexcept {
        return aabb(wp::max(min, other.min), wp::min(max, other.max));
    }

    constexpr bool intersect_ray(const ray<T>& r, T& t_min, T& t_max) const noexcept;
    constexpr bool intersect_plane(const plane<T>& p) const noexcept;
    constexpr bool intersect_sphere(const sphere<T>& s) const noexcept;

    constexpr aabb transform(const mat4<T>& m) const noexcept {
        // transform all 8 corners and recompute min/max
        vec3<T> corners[8] = {
            vec3<T>(min.x, min.y, min.z),
            vec3<T>(max.x, min.y, min.z),
            vec3<T>(min.x, max.y, min.z),
            vec3<T>(max.x, max.y, min.z),
            vec3<T>(min.x, min.y, max.z),
            vec3<T>(max.x, min.y, max.z),
            vec3<T>(min.x, max.y, max.z),
            vec3<T>(max.x, max.y, max.z)
        };
        aabb result;
        for (int i = 0; i < 8; ++i) {
            vec4<T> p = mul(m, vec4<T>(corners[i].x, corners[i].y, corners[i].z, T(1)));
            if (p.w != T(0)) p = p / p.w;
            result.expand(vec3<T>(p.x, p.y, p.z));
        }
        return result;
    }

    constexpr bool operator==(const aabb& o) const noexcept { return min == o.min && max == o.max; }
    constexpr bool operator!=(const aabb& o) const noexcept { return !(*this == o); }
};

template <typename T> constexpr aabb<T> merge(const aabb<T>& a, const aabb<T>& b) noexcept { return aabb<T>(wp::min(a.min, b.min), wp::max(a.max, b.max)); }
template <typename T> constexpr aabb<T> from_center_extent(const vec3<T>& center, const vec3<T>& extent) noexcept { return aabb<T>(center - extent, center + extent); }
template <typename T> constexpr aabb<T> from_points(const vec3<T>* points, size_t count) noexcept {
    aabb<T> box;
    for (size_t i = 0; i < count; ++i) box.expand(points[i]);
    return box;
}
template <typename T> constexpr aabb<T> unit_aabb() noexcept { return aabb<T>(vec3<T>(-T(0.5)), vec3<T>(T(0.5))); }

// Intersection declarations (defined in respective files)
template <typename T> constexpr bool intersect(const aabb<T>& box, const ray<T>& r, T& tmin, T& tmax);
template <typename T> constexpr bool intersect(const aabb<T>& box, const plane<T>& p);
template <typename T> constexpr bool intersect(const aabb<T>& box, const sphere<T>& s);

using aabbf = aabb<float>;
using aabbd = aabb<double>;

} // namespace wp