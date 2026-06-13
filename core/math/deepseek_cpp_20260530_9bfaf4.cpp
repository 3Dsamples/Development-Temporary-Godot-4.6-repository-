// File 0013 : core/math/ray.h
// Ray representation (origin + direction) with intersection tests against plane, sphere, aabb, triangle, and capsule.

#pragma once

#include "vec3.h"
#include "constants.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace wp {

template <typename T>
struct ray {
    vec3<T> origin;
    vec3<T> direction;   // may be unnormalised; direction = to - from

    constexpr ray() noexcept : origin(T(0)), direction(T(0), T(0), T(1)) {}
    constexpr ray(const vec3<T>& o, const vec3<T>& d) noexcept : origin(o), direction(d) {}

    constexpr vec3<T> point_at(T t) const noexcept { return origin + direction * t; }
    constexpr vec3<T> normalized_dir() const noexcept { return normalize(direction); }
    constexpr ray normalized() const noexcept { return ray(origin, normalize(direction)); }
    constexpr T length() const noexcept { return wp::length(direction); }
    constexpr T length_sq() const noexcept { return wp::length_sq(direction); }

    constexpr vec3<T> closest_point(const vec3<T>& p) const noexcept {
        T t = dot(direction, p - origin) / length_sq(direction);
        t = clamp(t, T(0), T(1));
        return point_at(t);
    }
};

// Forward declare other primitives
template <typename T> struct plane;
template <typename T> struct aabb;
template <typename T> struct sphere;

// ---------- Intersection functions ----------
// They require the full definitions of the other types, which are included after this struct.

// Ray-Plane
template <typename T>
constexpr bool intersect_ray_plane(const ray<T>& r, const plane<T>& p, T& t) noexcept {
    T denom = dot(p.normal, r.direction);
    if (std::abs(denom) < MathConst<T>::epsilon) return false;
    t = -(dot(p.normal, r.origin) + p.d) / denom;
    return t >= T(0);
}

// Ray-Sphere
template <typename T>
constexpr bool intersect_ray_sphere(const ray<T>& r, const sphere<T>& s, T& t) noexcept {
    vec3<T> oc = r.origin - s.center;
    T a = length_sq(r.direction);
    T b = T(2) * dot(oc, r.direction);
    T c = length_sq(oc) - s.radius * s.radius;
    T discriminant = b * b - T(4) * a * c;
    if (discriminant < T(0)) return false;
    discriminant = std::sqrt(discriminant);
    T t0 = (-b - discriminant) / (T(2) * a);
    T t1 = (-b + discriminant) / (T(2) * a);
    if (t0 > t1) std::swap(t0, t1);
    if (t0 >= T(0)) { t = t0; return true; }
    if (t1 >= T(0)) { t = t1; return true; }
    return false;
}

// Ray-AABB (slab method)
template <typename T>
constexpr bool intersect_ray_aabb(const ray<T>& r, const aabb<T>& box, T& t_min, T& t_max) noexcept {
    T tmin = min_val<T>, tmax = max_val<T>;
    for (int i = 0; i < 3; ++i) {
        if (std::abs(r.direction[i]) < MathConst<T>::epsilon) {
            if (r.origin[i] < box.min[i] || r.origin[i] > box.max[i]) return false;
        } else {
            T inv_d = T(1) / r.direction[i];
            T t1 = (box.min[i] - r.origin[i]) * inv_d;
            T t2 = (box.max[i] - r.origin[i]) * inv_d;
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
            if (tmin > tmax) return false;
        }
    }
    t_min = tmin;
    t_max = tmax;
    return true;
}

// Ray-Triangle (Möller–Trumbore)
template <typename T>
constexpr bool intersect_ray_triangle(const ray<T>& r, const vec3<T>& v0, const vec3<T>& v1, const vec3<T>& v2,
                                      T& t, T& u, T& v) noexcept {
    vec3<T> e1 = v1 - v0;
    vec3<T> e2 = v2 - v0;
    vec3<T> h = cross(r.direction, e2);
    T a = dot(e1, h);
    if (std::abs(a) < MathConst<T>::epsilon) return false;
    T f = T(1) / a;
    vec3<T> s = r.origin - v0;
    u = f * dot(s, h);
    if (u < T(0) || u > T(1)) return false;
    vec3<T> q = cross(s, e1);
    v = f * dot(r.direction, q);
    if (v < T(0) || u + v > T(1)) return false;
    t = f * dot(e2, q);
    return t >= T(0);
}
template <typename T>
constexpr bool intersect_ray_triangle(const ray<T>& r, const vec3<T>& v0, const vec3<T>& v1, const vec3<T>& v2, T& t) noexcept {
    T u, v;
    return intersect_ray_triangle(r, v0, v1, v2, t, u, v);
}

// Ray-Capsule (infinite cylinder between two spheres, then caps)
template <typename T>
constexpr bool intersect_ray_capsule(const ray<T>& r, const vec3<T>& p0, const vec3<T>& p1, T radius, T& t) noexcept {
    // Segment vector and line parameters
    vec3<T> ba = p1 - p0;
    T ba_len2 = length_sq(ba);
    if (ba_len2 < MathConst<T>::epsilon) {
        // Degenerate to sphere
        return intersect_ray_sphere(r, sphere<T>(p0, radius), t);
    }

    // Quadratic for infinite cylinder
    vec3<T> oc = r.origin - p0;
    T baba = ba_len2;
    T bard = dot(ba, r.direction);
    T baoc = dot(ba, oc);
    T k2 = baba - bard * bard;
    T k1 = baba * dot(oc, r.direction) - baoc * bard;
    T k0 = baba * length_sq(oc) - baoc * baoc - radius * radius * baba;
    T h = k1 * k1 - k2 * k0;
    if (h < T(0)) return false;
    h = std::sqrt(h);
    T t_candidate = std::numeric_limits<T>::max();
    bool found = false;
    // Evaluate the two solutions
    for (int i = 0; i < 2; ++i) {
        T s = (i == 0) ? (k1 - h) / k2 : (k1 + h) / k2;
        if (s >= T(0)) {
            T y = baoc + s * bard;
            if (y >= T(0) && y <= baba) {
                if (s < t_candidate) { t_candidate = s; found = true; }
            } else {
                // Check sphere caps at t where segment parameter is 0 or 1
                // Cap at p0
                if (y < T(0)) {
                    T t0;
                    if (intersect_ray_sphere(r, sphere<T>(p0, radius), t0) && t0 < t_candidate) {
                        t_candidate = t0; found = true;
                    }
                }
                // Cap at p1
                if (y > baba) {
                    T t1;
                    if (intersect_ray_sphere(r, sphere<T>(p1, radius), t1) && t1 < t_candidate) {
                        t_candidate = t1; found = true;
                    }
                }
            }
        }
    }
    if (found) { t = t_candidate; }
    return found;
}

// ---------- Intersection non-member functions that utilize the primitive classes ----------
// These are placed here because they require both ray and the primitives fully defined.

// plane<T>::intersect_ray (defined inline, since plane.h only declared)
template <typename T>
constexpr bool plane<T>::intersect_ray(const ray<T>& r, T& t) const noexcept {
    return intersect_ray_plane(r, *this, t);
}

// sphere<T>::intersect_ray
template <typename T>
constexpr bool sphere<T>::intersect_ray(const ray<T>& r, T& t) const noexcept {
    return intersect_ray_sphere(r, *this, t);
}

// aabb<T>::intersect_ray
template <typename T>
constexpr bool aabb<T>::intersect_ray(const ray<T>& r, T& t_min, T& t_max) const noexcept {
    return intersect_ray_aabb(r, *this, t_min, t_max);
}

// Convenience overload for aabb returning only first hit
template <typename T>
constexpr bool intersect_ray_aabb(const ray<T>& r, const aabb<T>& box, T& t) noexcept {
    T tmin, tmax;
    if (!intersect_ray_aabb(r, box, tmin, tmax)) return false;
    if (tmin >= T(0)) { t = tmin; return true; }
    if (tmax >= T(0)) { t = tmax; return true; }
    return false;
}

// Intersection with plane and aabb (already declared in plane.h but defined here)
template <typename T>
constexpr bool intersect(const plane<T>& p, const aabb<T>& box) noexcept {
    // Check all eight corners against plane; if all on same side, no intersection.
    int side = 0;
    for (int i = 0; i < 8; ++i) {
        int s = p.classify(box.corner(i));
        if (s > 0) side |= 1;
        else if (s < 0) side |= 2;
        if (side == 3) return true; // both positive and negative points
    }
    return false;
}
template <typename T>
constexpr bool intersect(const aabb<T>& box, const plane<T>& p) noexcept { return intersect(p, box); }

// Intersection sphere/plane (already declared in sphere.h but implemented here)
template <typename T>
constexpr bool intersect(const plane<T>& p, const sphere<T>& s) noexcept { return s.intersects(p); }
template <typename T>
constexpr bool intersect(const sphere<T>& s, const plane<T>& p) noexcept { return s.intersects(p); }

// Intersection sphere/sphere and sphere/aabb implemented in sphere.h; we can still provide a unified intersect free function here.
// (Already declared in sphere.h via non-member functions; they are fine.)

// Ray normalization utility
template <typename T>
constexpr ray<T> normalized_ray(const vec3<T>& from, const vec3<T>& to) noexcept {
    return ray<T>(from, normalize(to - from));
}

using rayf = ray<float>;
using rayd = ray<double>;

} // namespace wp