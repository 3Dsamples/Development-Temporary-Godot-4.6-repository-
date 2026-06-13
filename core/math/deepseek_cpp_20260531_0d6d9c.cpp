// File 0046 : core/math/intersection2d.h
// 2D intersection tests: segment‑segment, point‑in‑convex polygon, circle‑aabb, circle‑circle.

#pragma once

#include "vec2.h"
#include "constants.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace wp {

// ── Check if two segments (p1,p2) and (q1,q2) intersect ─────────────
// Returns true and stores intersection parameter t (for p segment) if they cross.
// If collinear, returns false.
template <typename T>
bool segment_intersection(const vec2<T>& p1, const vec2<T>& p2,
                          const vec2<T>& q1, const vec2<T>& q2,
                          T& t, T& u) {
    vec2<T> r = p2 - p1;
    vec2<T> s = q2 - q1;
    T cross_rs = cross(r, s);
    if (std::abs(cross_rs) < epsilon<T>) return false; // parallel or collinear
    vec2<T> pq = q1 - p1;
    T t_val = cross(pq, s) / cross_rs;
    T u_val = cross(pq, r) / cross_rs;
    if (t_val >= T(0) && t_val <= T(1) && u_val >= T(0) && u_val <= T(1)) {
        t = t_val;
        u = u_val;
        return true;
    }
    return false;
}

// Convenience overload without t,u
template <typename T>
bool segment_intersects(const vec2<T>& p1, const vec2<T>& p2,
                        const vec2<T>& q1, const vec2<T>& q2) {
    T t, u;
    return segment_intersection(p1, p2, q1, q2, t, u);
}

// ── Point in convex polygon (ray casting or winding? We'll do ray casting) ──
// Works for any simple polygon (not necessarily convex).
template <typename T>
bool point_in_polygon(const vec2<T>& point, const std::vector<vec2<T>>& polygon) {
    int n = static_cast<int>(polygon.size());
    if (n < 3) return false;
    bool inside = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        const vec2<T>& pi = polygon[i];
        const vec2<T>& pj = polygon[j];
        // Check if edge crosses the horizontal ray at point.y
        if (((pi.y > point.y) != (pj.y > point.y)) &&
            (point.x < (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x)) {
            inside = !inside;
        }
    }
    return inside;
}

// ── Point in axis‑aligned rectangle (from rect2.h, but standalone) ──
template <typename T>
constexpr bool point_in_rect(const vec2<T>& p, const vec2<T>& rect_min, const vec2<T>& rect_max) noexcept {
    return p.x >= rect_min.x && p.x <= rect_max.x &&
           p.y >= rect_min.y && p.y <= rect_max.y;
}

// ── Circle‑circle intersection ──────────────────────────────────────
template <typename T>
constexpr bool circle_intersects(const vec2<T>& c1, T r1, const vec2<T>& c2, T r2) noexcept {
    T dist_sq = length_sq(c1 - c2);
    T sum_r = r1 + r2;
    return dist_sq <= sum_r * sum_r;
}

// ── Circle‑axis aligned rectangle intersection ──────────────────────
template <typename T>
constexpr bool circle_rect_intersects(const vec2<T>& circle_center, T radius,
                                      const vec2<T>& rect_min, const vec2<T>& rect_max) noexcept {
    // Closest point on rectangle to circle center
    T closest_x = clamp(circle_center.x, rect_min.x, rect_max.x);
    T closest_y = clamp(circle_center.y, rect_min.y, rect_max.y);
    T dist_sq = (circle_center.x - closest_x) * (circle_center.x - closest_x) +
                (circle_center.y - closest_y) * (circle_center.y - closest_y);
    return dist_sq <= radius * radius;
}

// ── Closest point on segment (2D) ──────────────────────────────────
template <typename T>
vec2<T> closest_point_on_segment(const vec2<T>& p, const vec2<T>& a, const vec2<T>& b) noexcept {
    vec2<T> ab = b - a;
    T t = dot(p - a, ab) / dot(ab, ab);
    t = clamp(t, T(0), T(1));
    return a + ab * t;
}

// ── Distance from point to segment (2D) ────────────────────────────
template <typename T>
T point_segment_distance(const vec2<T>& p, const vec2<T>& a, const vec2<T>& b) noexcept {
    vec2<T> closest = closest_point_on_segment(p, a, b);
    return distance(p, closest);
}

} // namespace wp