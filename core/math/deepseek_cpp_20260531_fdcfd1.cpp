//File 0029 : core/math/geometry_primitives.h
//3D geometry primitives: AABB, Sphere, Plane, Ray, Triangle, and fast intersection tests (sphere‑AABB, ray‑AABB, ray‑triangle, plane‑sphere, etc.) using SIMD vector_math.
#ifndef CORE_MATH_GEOMETRY_PRIMITIVES_H
#define CORE_MATH_GEOMETRY_PRIMITIVES_H

#include "vector_math.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace SimulationMath {
namespace geometry {

// -----------------------------------------------------------------------------
// 1. Axis‑Aligned Bounding Box (min, max)
// -----------------------------------------------------------------------------
struct AABB {
    DirectX::XMVECTOR min;
    DirectX::XMVECTOR max;

    AABB() noexcept : min(vector_math::replicate(std::numeric_limits<float>::max())),
                      max(vector_math::replicate(-std::numeric_limits<float>::max())) {}

    explicit AABB(DirectX::FXMVECTOR p) noexcept : min(p), max(p) {}

    AABB(DirectX::FXMVECTOR _min, DirectX::FXMVECTOR _max) noexcept : min(_min), max(_max) {}

    DirectX::XMVECTOR center() const noexcept { return vector_math::scale(vector_math::add(min, max), 0.5f); }
    DirectX::XMVECTOR extent() const noexcept { return vector_math::scale(vector_math::sub(max, min), 0.5f); }
    float surface_area() const noexcept {
        DirectX::XMVECTOR e = extent();
        float x = vector_math::get_x(e), y = vector_math::get_y(e), z = vector_math::get_z(e);
        return 2.0f * (x*y + x*z + y*z);
    }
    float volume() const noexcept {
        DirectX::XMVECTOR e = extent();
        return 8.0f * vector_math::get_x(e) * vector_math::get_y(e) * vector_math::get_z(e);
    }

    void extend(DirectX::FXMVECTOR p) noexcept {
        min = DirectX::XMVectorMin(min, p);
        max = DirectX::XMVectorMax(max, p);
    }
    void extend(const AABB& other) noexcept {
        min = DirectX::XMVectorMin(min, other.min);
        max = DirectX::XMVectorMax(max, other.max);
    }
    bool overlaps(const AABB& other) const noexcept {
        DirectX::XMVECTOR cmp = DirectX::XMVectorOr(
            DirectX::XMVectorGreater(min, other.max),
            DirectX::XMVectorGreater(other.min, max));
        return !(DirectX::XMVector4Equal(cmp, DirectX::XMVectorTrueInt()));
    }
    bool contains(DirectX::FXMVECTOR p) const noexcept {
        return DirectX::XMVector4Equal(DirectX::XMVectorAnd(
            DirectX::XMVectorGreaterOrEqual(p, min),
            DirectX::XMVectorLessOrEqual(p, max)),
            DirectX::XMVectorTrueInt());
    }
};

// -----------------------------------------------------------------------------
// 2. Sphere
// -----------------------------------------------------------------------------
struct Sphere {
    DirectX::XMVECTOR center;
    float radius;

    Sphere() noexcept : center(vector_math::zero()), radius(0.0f) {}
    Sphere(DirectX::FXMVECTOR c, float r) noexcept : center(c), radius(r) {}

    bool contains(DirectX::FXMVECTOR p) const noexcept {
        return vector_math::length_sq3_scalar(vector_math::sub(p, center)) <= radius * radius;
    }
};

// -----------------------------------------------------------------------------
// 3. Plane (normal * point = d)
// -----------------------------------------------------------------------------
struct Plane {
    DirectX::XMVECTOR normal;  // unit length
    float d;

    Plane() noexcept : normal(DirectX::XMVectorSet(0,1,0,0)), d(0.0f) {}
    Plane(DirectX::FXMVECTOR n, float _d) noexcept : normal(vector_math::normalize3(n)), d(_d) {}
    Plane(DirectX::FXMVECTOR n, DirectX::FXMVECTOR point) noexcept {
        normal = vector_math::normalize3(n);
        d = vector_math::dot3_scalar(normal, point);
    }

    float signed_distance(DirectX::FXMVECTOR p) const noexcept {
        return vector_math::dot3_scalar(normal, p) - d;
    }
    DirectX::XMVECTOR closest_point(DirectX::FXMVECTOR p) const noexcept {
        float dist = signed_distance(p);
        return DirectX::XMVectorSubtract(p, DirectX::XMVectorScale(normal, dist));
    }
};

// -----------------------------------------------------------------------------
// 4. Ray (origin, direction, t_min, t_max)
// -----------------------------------------------------------------------------
struct Ray {
    DirectX::XMVECTOR origin;
    DirectX::XMVECTOR direction;  // unit length
    float t_min;
    float t_max;

    Ray() noexcept : origin(vector_math::zero()), direction(DirectX::XMVectorSet(0,0,1,0)), t_min(0.0f), t_max(1e10f) {}
    Ray(DirectX::FXMVECTOR o, DirectX::FXMVECTOR d, float tmin = 0.0f, float tmax = 1e10f) noexcept
        : origin(o), direction(vector_math::normalize3(d)), t_min(tmin), t_max(tmax) {}

    DirectX::XMVECTOR point_at(float t) const noexcept {
        return DirectX::XMVectorAdd(origin, DirectX::XMVectorScale(direction, t));
    }
};

// -----------------------------------------------------------------------------
// 5. Triangle
// -----------------------------------------------------------------------------
struct Triangle {
    DirectX::XMVECTOR v0, v1, v2;

    Triangle() noexcept = default;
    Triangle(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b, DirectX::FXMVECTOR c) noexcept : v0(a), v1(b), v2(c) {}

    DirectX::XMVECTOR normal() const noexcept {
        DirectX::XMVECTOR e1 = DirectX::XMVectorSubtract(v1, v0);
        DirectX::XMVECTOR e2 = DirectX::XMVectorSubtract(v2, v0);
        return vector_math::normalize3(vector_math::cross3(e1, e2));
    }
    float area() const noexcept {
        DirectX::XMVECTOR e1 = DirectX::XMVectorSubtract(v1, v0);
        DirectX::XMVECTOR e2 = DirectX::XMVectorSubtract(v2, v0);
        return 0.5f * vector_math::length3_scalar(vector_math::cross3(e1, e2));
    }
};

// -----------------------------------------------------------------------------
// 6. Intersection tests
// -----------------------------------------------------------------------------

// Sphere vs AABB – returns true if overlap or touch
inline bool sphere_aabb_intersect(const Sphere& s, const AABB& box) noexcept {
    DirectX::XMVECTOR closest = DirectX::XMVectorMax(box.min, DirectX::XMVectorMin(s.center, box.max));
    float dist_sq = vector_math::length_sq3_scalar(DirectX::XMVectorSubtract(closest, s.center));
    return dist_sq <= s.radius * s.radius;
}

// AABB vs AABB
inline bool aabb_aabb_intersect(const AABB& a, const AABB& b) noexcept {
    return a.overlaps(b);
}

// Ray vs AABB (slabs method, returns t_enter and t_exit in *t if hit, else false)
inline bool ray_aabb_intersect(const Ray& ray, const AABB& box, float& t_enter, float& t_exit) noexcept {
    DirectX::XMVECTOR inv_dir = vector_math::recip(ray.direction);
    DirectX::XMVECTOR t0 = DirectX::XMVectorMultiply(DirectX::XMVectorSubtract(box.min, ray.origin), inv_dir);
    DirectX::XMVECTOR t1 = DirectX::XMVectorMultiply(DirectX::XMVectorSubtract(box.max, ray.origin), inv_dir);
    DirectX::XMVECTOR tmin = DirectX::XMVectorMin(t0, t1);
    DirectX::XMVECTOR tmax = DirectX::XMVectorMax(t0, t1);
    float enter = std::max({vector_math::get_x(tmin), vector_math::get_y(tmin), vector_math::get_z(tmin)});
    float exit  = std::min({vector_math::get_x(tmax), vector_math::get_y(tmax), vector_math::get_z(tmax)});
    if (enter <= exit && exit >= ray.t_min && enter <= ray.t_max) {
        t_enter = std::max(enter, ray.t_min);
        t_exit  = std::min(exit,  ray.t_max);
        return true;
    }
    return false;
}

// Ray vs Triangle (Möller–Trumbore)
inline bool ray_triangle_intersect(const Ray& ray, const Triangle& tri, float& t, float& u, float& v) noexcept {
    DirectX::XMVECTOR e1 = DirectX::XMVectorSubtract(tri.v1, tri.v0);
    DirectX::XMVECTOR e2 = DirectX::XMVectorSubtract(tri.v2, tri.v0);
    DirectX::XMVECTOR pvec = vector_math::cross3(ray.direction, e2);
    float det = vector_math::dot3_scalar(e1, pvec);
    if (std::abs(det) < 1e-8f) return false;
    float inv_det = 1.0f / det;
    DirectX::XMVECTOR tvec = DirectX::XMVectorSubtract(ray.origin, tri.v0);
    u = vector_math::dot3_scalar(tvec, pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;
    DirectX::XMVECTOR qvec = vector_math::cross3(tvec, e1);
    v = vector_math::dot3_scalar(ray.direction, qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;
    t = vector_math::dot3_scalar(e2, qvec) * inv_det;
    if (t >= ray.t_min && t <= ray.t_max) return true;
    return false;
}

// Plane vs Sphere
inline bool plane_sphere_intersect(const Plane& plane, const Sphere& sphere) noexcept {
    return std::abs(plane.signed_distance(sphere.center)) <= sphere.radius;
}

// Sphere vs Sphere
inline bool sphere_sphere_intersect(const Sphere& a, const Sphere& b) noexcept {
    float dist_sq = vector_math::length_sq3_scalar(DirectX::XMVectorSubtract(b.center, a.center));
    float r = a.radius + b.radius;
    return dist_sq <= r * r;
}

// Closest point on AABB to a given point
inline DirectX::XMVECTOR closest_point_aabb(DirectX::FXMVECTOR point, const AABB& box) noexcept {
    return DirectX::XMVectorMax(box.min, DirectX::XMVectorMin(point, box.max));
}

// Closest point on plane
inline DirectX::XMVECTOR closest_point_plane(DirectX::FXMVECTOR point, const Plane& plane) noexcept {
    return plane.closest_point(point);
}

// Closest point on triangle
inline DirectX::XMVECTOR closest_point_triangle(DirectX::FXMVECTOR p, const Triangle& tri) noexcept {
    DirectX::XMVECTOR ab = DirectX::XMVectorSubtract(tri.v1, tri.v0);
    DirectX::XMVECTOR ac = DirectX::XMVectorSubtract(tri.v2, tri.v0);
    DirectX::XMVECTOR ap = DirectX::XMVectorSubtract(p, tri.v0);
    float d1 = vector_math::dot3_scalar(ab, ap);
    float d2 = vector_math::dot3_scalar(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return tri.v0;
    DirectX::XMVECTOR bp = DirectX::XMVectorSubtract(p, tri.v1);
    float d3 = vector_math::dot3_scalar(ab, bp);
    float d4 = vector_math::dot3_scalar(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return tri.v1;
    float vc = d1*d4 - d3*d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        return DirectX::XMVectorAdd(tri.v0, DirectX::XMVectorScale(ab, v));
    }
    DirectX::XMVECTOR cp = DirectX::XMVectorSubtract(p, tri.v2);
    float d5 = vector_math::dot3_scalar(ab, cp);
    float d6 = vector_math::dot3_scalar(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return tri.v2;
    float vb = d5*d2 - d1*d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        return DirectX::XMVectorAdd(tri.v0, DirectX::XMVectorScale(ac, w));
    }
    float va = d3*d6 - d5*d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return DirectX::XMVectorAdd(tri.v1, DirectX::XMVectorScale(DirectX::XMVectorSubtract(tri.v2, tri.v1), w));
    }
    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return DirectX::XMVectorAdd(DirectX::XMVectorAdd(tri.v0, DirectX::XMVectorScale(ab, v)),
                                 DirectX::XMVectorScale(ac, w));
}

} // namespace geometry
} // namespace SimulationMath

#endif // CORE_MATH_GEOMETRY_PRIMITIVES_H