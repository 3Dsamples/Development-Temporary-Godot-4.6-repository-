// File 0018 : core/math/triangle.h
// Triangle primitive with area, normal, barycentrics, point containment, closest point, and distance functions.

#pragma once

#include "vec3.h"
#include "ray.h"               // for ray<T> and intersect_ray_triangle
#include <algorithm>
#include <cmath>
#include <limits>

namespace wp {

template <typename T>
struct triangle {
    vec3<T> v0, v1, v2;

    constexpr triangle() noexcept : v0(T(0)), v1(T(1), T(0), T(0)), v2(T(0), T(1), T(0)) {}
    constexpr triangle(const vec3<T>& a, const vec3<T>& b, const vec3<T>& c) noexcept : v0(a), v1(b), v2(c) {}
    template <typename U> constexpr explicit triangle(const triangle<U>& o) noexcept : v0(o.v0), v1(o.v1), v2(o.v2) {}

    // Area (absolute value)
    T area() const noexcept { return length(cross(v1 - v0, v2 - v0)) * T(0.5); }
    // Non‑normalised face normal (magnitude = 2 * area)
    vec3<T> normal() const noexcept { return cross(v1 - v0, v2 - v0); }
    // Unit normal (assuming non‑degenerate)
    vec3<T> unit_normal() const noexcept { return normalize(normal()); }
    // Centroid
    vec3<T> centroid() const noexcept { return (v0 + v1 + v2) / T(3); }
    // Edge vectors
    vec3<T> edge0() const noexcept { return v1 - v0; }
    vec3<T> edge1() const noexcept { return v2 - v0; }
    vec3<T> edge2() const noexcept { return v2 - v1; }

    // Compute barycentric coordinates for a point (assuming point is coplanar)
    void barycentric(const vec3<T>& p, T& u, T& v, T& w) const noexcept {
        vec3<T> v0p = p - v0;
        vec3<T> v0v1 = v1 - v0;
        vec3<T> v0v2 = v2 - v0;
        T d00 = dot(v0v1, v0v1);
        T d01 = dot(v0v1, v0v2);
        T d11 = dot(v0v2, v0v2);
        T d20 = dot(v0p, v0v1);
        T d21 = dot(v0p, v0v2);
        T denom = d00 * d11 - d01 * d01;
        if (std::abs(denom) < MathConst<T>::epsilon) {
            // degenerate triangle, fallback
            u = v = w = T(1)/T(3);
            return;
        }
        v = (d11 * d20 - d01 * d21) / denom;
        w = (d00 * d21 - d01 * d20) / denom;
        u = T(1) - v - w;
    }

    // Check if point is inside the triangle (including boundary) using barycentrics
    bool contains(const vec3<T>& p, T epsilon = MathConst<T>::epsilon) const noexcept {
        T u, v, w;
        barycentric(p, u, v, w);
        return (u >= -epsilon && v >= -epsilon && w >= -epsilon);
    }

    // Closest point on triangle to an arbitrary point
    vec3<T> closest_point(const vec3<T>& p) const noexcept {
        // Uses projection onto the plane and then inside‑outside test
        vec3<T> n = normal();
        T denom = dot(n, n);
        if (denom < MathConst<T>::epsilon) return v0; // degenerate
        // Project point onto plane
        T t = dot(n, v0 - p) / denom;
        vec3<T> proj = p + n * t;   // now p is projected onto plane
        // Check if inside triangle using barycentrics
        T u, v, w;
        barycentric(proj, u, v, w);
        if (u >= T(0) && v >= T(0) && w >= T(0)) return proj; // inside
        // Otherwise closest point is on an edge
        vec3<T> closest_pt;
        T min_dist_sq = max_val<T>;
        // edge 0-1
        vec3<T> e01 = v1 - v0;
        T t01 = clamp(dot(p - v0, e01) / dot(e01, e01), T(0), T(1));
        vec3<T> p01 = v0 + e01 * t01;
        T d2 = length_sq(p - p01);
        if (d2 < min_dist_sq) { min_dist_sq = d2; closest_pt = p01; }
        // edge 1-2
        vec3<T> e12 = v2 - v1;
        T t12 = clamp(dot(p - v1, e12) / dot(e12, e12), T(0), T(1));
        vec3<T> p12 = v1 + e12 * t12;
        d2 = length_sq(p - p12);
        if (d2 < min_dist_sq) { min_dist_sq = d2; closest_pt = p12; }
        // edge 2-0
        vec3<T> e20 = v0 - v2;
        T t20 = clamp(dot(p - v2, e20) / dot(e20, e20), T(0), T(1));
        vec3<T> p20 = v2 + e20 * t20;
        d2 = length_sq(p - p20);
        if (d2 < min_dist_sq) { min_dist_sq = d2; closest_pt = p20; }
        return closest_pt;
    }

    // Distance from point to triangle
    T distance(const vec3<T>& p) const noexcept { return length(p - closest_point(p)); }
    T distance_sq(const vec3<T>& p) const noexcept { return length_sq(p - closest_point(p)); }

    // Ray intersection (convenience, uses ray.h free function)
    template <typename U>
    bool intersect_ray(const ray<U>& r, U& t, U& u, U& v) const noexcept {
        return intersect_ray_triangle(r, v0, v1, v2, t, u, v);
    }
    template <typename U>
    bool intersect_ray(const ray<U>& r, U& t) const noexcept { U u, v; return intersect_ray(r, t, u, v); }
};

// Non‑member convenience
template <typename T>
constexpr triangle<T> make_triangle(const vec3<T>& a, const vec3<T>& b, const vec3<T>& c) noexcept { return triangle<T>(a, b, c); }

using trianglef = triangle<float>;
using triangled = triangle<double>;

} // namespace wp