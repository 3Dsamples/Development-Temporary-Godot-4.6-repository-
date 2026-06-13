// system name : onetbb-warp
// File 0010 : core/math/geometry.h
// Description : Geometric primitives, intersections, distances, and projections.

#ifndef __TBB_WARP_CORE_MATH_GEOMETRY_H
#define __TBB_WARP_CORE_MATH_GEOMETRY_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/vector4.h"
#include "core/math/matrix3.h"
#include "core/math/matrix4.h"
#include "core/math/quaternion.h"
#include <cmath>
#include <type_traits>
#include <array>
#include <vector>
#include <algorithm>
#include <limits>
#include <optional>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Plane (3D)
// ============================================================

template<typename T>
struct plane {
    vector3<T> normal;
    T distance; // signed distance from origin: dot(normal, point) + distance = 0

    constexpr plane() noexcept : normal(T(0),T(1),T(0)), distance(T(0)) {}
    constexpr plane(const vector3<T>& n, T d) noexcept : normal(n), distance(d) {}
    constexpr plane(const vector3<T>& point_on_plane, const vector3<T>& n) noexcept
        : normal(normalize(n)), distance(-dot(normal, point_on_plane)) {}
    constexpr plane(const vector3<T>& p0, const vector3<T>& p1, const vector3<T>& p2) noexcept
        : normal(normalize(cross(p1-p0, p2-p0))), distance(-dot(normal, p0)) {}

    constexpr T signed_distance(const vector3<T>& point) const noexcept {
        return dot(normal, point) + distance;
    }
    constexpr bool is_front_facing(const vector3<T>& direction) const noexcept {
        return dot(normal, direction) < T(0);
    }
    constexpr vector3<T> project(const vector3<T>& point) const noexcept {
        return point - normal * signed_distance(point);
    }
};

// ============================================================
// Ray
// ============================================================

template<typename T>
struct ray {
    vector3<T> origin;
    vector3<T> direction; // assumed normalized

    constexpr ray() noexcept : origin(), direction(T(0),T(0),T(1)) {}
    constexpr ray(const vector3<T>& o, const vector3<T>& d) noexcept : origin(o), direction(normalize(d)) {}

    constexpr vector3<T> point_at(T t) const noexcept { return origin + direction * t; }
};

// ============================================================
// Line segment
// ============================================================

template<typename T>
struct segment3 {
    vector3<T> a, b;
    constexpr segment3() noexcept : a(), b() {}
    constexpr segment3(const vector3<T>& a_, const vector3<T>& b_) noexcept : a(a_), b(b_) {}
    constexpr vector3<T> direction() const noexcept { return b - a; }
    constexpr T length() const noexcept { return distance(a, b); }
    constexpr T length_sq() const noexcept { return distance_sq(a, b); }
    constexpr vector3<T> point_at(T t) const noexcept { return a + (b - a) * t; }
};

// ============================================================
// Sphere
// ============================================================

template<typename T>
struct sphere {
    vector3<T> center;
    T radius;
    constexpr sphere() noexcept : center(), radius(T(1)) {}
    constexpr sphere(const vector3<T>& c, T r) noexcept : center(c), radius(r) {}
    constexpr bool contains(const vector3<T>& point) const noexcept { return distance_sq(point, center) <= radius*radius; }
};

// ============================================================
// AABB (axis‑aligned bounding box)
// ============================================================

template<typename T>
struct aabb {
    vector3<T> min;
    vector3<T> max;

    constexpr aabb() noexcept : min(T(0)), max(T(0)) {}
    constexpr aabb(const vector3<T>& mn, const vector3<T>& mx) noexcept : min(mn), max(mx) {
        if (mn.x > mx.x) std::swap(min.x, max.x);
        if (mn.y > mx.y) std::swap(min.y, max.y);
        if (mn.z > mx.z) std::swap(min.z, max.z);
    }

    constexpr vector3<T> center() const noexcept { return (min + max) * T(0.5); }
    constexpr vector3<T> extents() const noexcept { return (max - min) * T(0.5); }
    constexpr vector3<T> size() const noexcept { return max - min; }
    constexpr T volume() const noexcept { auto s=size(); return s.x*s.y*s.z; }
    constexpr T surface_area() const noexcept { auto s=size(); return T(2)*(s.x*s.y + s.y*s.z + s.z*s.x); }

    constexpr bool contains(const vector3<T>& point) const noexcept {
        return point.x>=min.x && point.x<=max.x && point.y>=min.y && point.y<=max.y && point.z>=min.z && point.z<=max.z;
    }

    constexpr aabb& expand(const vector3<T>& point) noexcept {
        min.x = std::min(min.x, point.x); min.y = std::min(min.y, point.y); min.z = std::min(min.z, point.z);
        max.x = std::max(max.x, point.x); max.y = std::max(max.y, point.y); max.z = std::max(max.z, point.z);
        return *this;
    }

    constexpr aabb& expand(const aabb& other) noexcept {
        min = math::min(min, other.min); max = math::max(max, other.max);
        return *this;
    }

    constexpr bool intersects(const aabb& other) const noexcept {
        return min.x <= other.max.x && max.x >= other.min.x &&
               min.y <= other.max.y && max.y >= other.min.y &&
               min.z <= other.max.z && max.z >= other.min.z;
    }

    constexpr aabb intersection(const aabb& other) const noexcept {
        return aabb(math::max(min, other.min), math::min(max, other.max));
    }

    constexpr aabb transformed(const matrix4<T>& m) const noexcept {
        std::array<vector3<T>,8> corners = {{
            {min.x,min.y,min.z},{max.x,min.y,min.z},{min.x,max.y,min.z},{max.x,max.y,min.z},
            {min.x,min.y,max.z},{max.x,min.y,max.z},{min.x,max.y,max.z},{max.x,max.y,max.z}
        }};
        aabb result(transform_point(m, corners[0]), transform_point(m, corners[0]));
        for (int i=1; i<8; ++i) result.expand(transform_point(m, corners[i]));
        return result;
    }
};

// ============================================================
// OBB (oriented bounding box)
// ============================================================

template<typename T>
struct obb {
    vector3<T> center;
    vector3<T> half_extents;
    matrix3<T> orientation;

    constexpr obb() noexcept : center(), half_extents(T(1)), orientation() {}
    constexpr obb(const vector3<T>& c, const vector3<T>& he, const matrix3<T>& rot) noexcept
        : center(c), half_extents(he), orientation(rot) {}

    constexpr bool contains(const vector3<T>& point) const noexcept {
        vector3<T> local = transpose(orientation) * (point - center);
        return std::abs(local.x) <= half_extents.x &&
               std::abs(local.y) <= half_extents.y &&
               std::abs(local.z) <= half_extents.z;
    }
};

// ============================================================
// Triangle (3D)
// ============================================================

template<typename T>
struct triangle3 {
    vector3<T> v0, v1, v2;

    constexpr triangle3() noexcept : v0(), v1(), v2() {}
    constexpr triangle3(const vector3<T>& a, const vector3<T>& b, const vector3<T>& c) noexcept : v0(a), v1(b), v2(c) {}

    constexpr vector3<T> normal() const noexcept { return normalize(cross(v1-v0, v2-v0)); }
    constexpr T area() const noexcept { return T(0.5) * length(cross(v1-v0, v2-v0)); }
    constexpr vector3<T> centroid() const noexcept { return (v0 + v1 + v2) / T(3); }

    constexpr vector3<T> barycentric(const vector3<T>& point) const noexcept {
        vector3<T> v0v1 = v1 - v0, v0v2 = v2 - v0, v0p = point - v0;
        T d00 = dot(v0v1, v0v1), d01 = dot(v0v1, v0v2), d11 = dot(v0v2, v0v2);
        T d20 = dot(v0p, v0v1), d21 = dot(v0p, v0v2);
        T denom = d00*d11 - d01*d01;
        if (std::abs(denom) < T(FLOAT_EPSILON)) return vector3<T>(T(-1));
        T v = (d11*d20 - d01*d21) / denom;
        T w = (d00*d21 - d01*d20) / denom;
        return vector3<T>(T(1)-v-w, v, w);
    }

    constexpr bool contains(const vector3<T>& point) const noexcept {
        auto bc = barycentric(point);
        return bc.x >= T(0) && bc.y >= T(0) && bc.z >= T(0);
    }
};

// ============================================================
// Intersection tests
// ============================================================

// Ray‑plane intersection
template<typename T>
std::optional<T> intersect(const ray<T>& r, const plane<T>& p) noexcept {
    T nd = dot(p.normal, r.direction);
    if (std::abs(nd) < T(FLOAT_EPSILON)) return std::nullopt;
    T t = -(dot(p.normal, r.origin) + p.distance) / nd;
    if (t < T(0)) return std::nullopt;
    return t;
}

// Ray‑sphere intersection
template<typename T>
std::optional<std::pair<T,T>> intersect(const ray<T>& r, const sphere<T>& s) noexcept {
    vector3<T> oc = r.origin - s.center;
    T a = dot(r.direction, r.direction);
    T b = T(2) * dot(oc, r.direction);
    T c = dot(oc, oc) - s.radius*s.radius;
    T disc = b*b - T(4)*a*c;
    if (disc < T(0)) return std::nullopt;
    T sqrt_disc = std::sqrt(disc);
    T t0 = (-b - sqrt_disc) / (T(2)*a);
    T t1 = (-b + sqrt_disc) / (T(2)*a);
    if (t0 > t1) std::swap(t0, t1);
    if (t1 < T(0)) return std::nullopt;
    if (t0 < T(0)) t0 = T(0);
    return std::make_pair(t0, t1);
}

// Ray‑AABB intersection (slab method)
template<typename T>
std::optional<std::pair<T,T>> intersect(const ray<T>& r, const aabb<T>& box) noexcept {
    vector3<T> inv_dir(T(1)/r.direction.x, T(1)/r.direction.y, T(1)/r.direction.z);
    T t1 = (box.min.x - r.origin.x) * inv_dir.x;
    T t2 = (box.max.x - r.origin.x) * inv_dir.x;
    T t_min = std::min(t1, t2);
    T t_max = std::max(t1, t2);
    t1 = (box.min.y - r.origin.y) * inv_dir.y;
    t2 = (box.max.y - r.origin.y) * inv_dir.y;
    t_min = std::max(t_min, std::min(t1, t2));
    t_max = std::min(t_max, std::max(t1, t2));
    t1 = (box.min.z - r.origin.z) * inv_dir.z;
    t2 = (box.max.z - r.origin.z) * inv_dir.z;
    t_min = std::max(t_min, std::min(t1, t2));
    t_max = std::min(t_max, std::max(t1, t2));
    if (t_max >= t_min && t_max >= T(0)) {
        if (t_min < T(0)) t_min = T(0);
        return std::make_pair(t_min, t_max);
    }
    return std::nullopt;
}

// Ray‑triangle intersection (Möller‑Trumbore)
template<typename T>
std::optional<T> intersect(const ray<T>& r, const triangle3<T>& tri) noexcept {
    vector3<T> edge1 = tri.v1 - tri.v0;
    vector3<T> edge2 = tri.v2 - tri.v0;
    vector3<T> h = cross(r.direction, edge2);
    T a = dot(edge1, h);
    if (std::abs(a) < T(FLOAT_EPSILON)) return std::nullopt;
    T f = T(1) / a;
    vector3<T> s = r.origin - tri.v0;
    T u = f * dot(s, h);
    if (u < T(0) || u > T(1)) return std::nullopt;
    vector3<T> q = cross(s, edge1);
    T v = f * dot(r.direction, q);
    if (v < T(0) || u+v > T(1)) return std::nullopt;
    T t = f * dot(edge2, q);
    if (t > T(FLOAT_EPSILON)) return t;
    return std::nullopt;
}

// Sphere‑sphere intersection
template<typename T>
constexpr bool intersect(const sphere<T>& a, const sphere<T>& b) noexcept {
    return distance_sq(a.center, b.center) <= (a.radius+b.radius)*(a.radius+b.radius);
}

// Sphere‑AABB intersection
template<typename T>
constexpr bool intersect(const sphere<T>& s, const aabb<T>& box) noexcept {
    T sq_dist = T(0);
    for (int i=0; i<3; ++i) {
        T v = s.center[i];
        if (v < box.min[i]) sq_dist += (box.min[i]-v)*(box.min[i]-v);
        else if (v > box.max[i]) sq_dist += (v-box.max[i])*(v-box.max[i]);
    }
    return sq_dist <= s.radius*s.radius;
}

// AABB‑AABB intersection
template<typename T>
constexpr bool intersect(const aabb<T>& a, const aabb<T>& b) noexcept {
    return a.intersects(b);
}

// OBB‑OBB intersection (separating axis theorem)
template<typename T>
bool intersect(const obb<T>& a, const obb<T>& b) noexcept {
    matrix3<T> R = transpose(a.orientation) * b.orientation;
    matrix3<T> abs_R = abs(R);
    vector3<T> t = transpose(a.orientation) * (b.center - a.center);
    T ra, rb;
    for (int i=0; i<3; ++i) {
        ra = a.half_extents[i];
        rb = b.half_extents[0]*abs_R(i,0) + b.half_extents[1]*abs_R(i,1) + b.half_extents[2]*abs_R(i,2);
        if (std::abs(t[i]) > ra + rb) return false;
    }
    for (int i=0; i<3; ++i) {
        ra = a.half_extents[0]*abs_R(0,i) + a.half_extents[1]*abs_R(1,i) + a.half_extents[2]*abs_R(2,i);
        rb = b.half_extents[i];
        T val = t[0]*R(0,i) + t[1]*R(1,i) + t[2]*R(2,i);
        if (std::abs(val) > ra + rb) return false;
    }
    return true;
}

// ============================================================
// Closest point on primitives
// ============================================================

template<typename T>
constexpr vector3<T> closest_point(const plane<T>& p, const vector3<T>& point) noexcept {
    return p.project(point);
}

template<typename T>
constexpr vector3<T> closest_point(const segment3<T>& seg, const vector3<T>& point) noexcept {
    vector3<T> ab = seg.b - seg.a;
    T t = dot(point - seg.a, ab);
    if (t <= T(0)) return seg.a;
    T denom = dot(ab, ab);
    if (t >= denom) return seg.b;
    return seg.a + ab * (t / denom);
}

template<typename T>
constexpr vector3<T> closest_point(const aabb<T>& box, const vector3<T>& point) noexcept {
    return clamp(point, box.min, box.max);
}

template<typename T>
constexpr vector3<T> closest_point(const triangle3<T>& tri, const vector3<T>& point) noexcept {
    vector3<T> ab = tri.v1 - tri.v0, ac = tri.v2 - tri.v0, ap = point - tri.v0;
    T d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= T(0) && d2 <= T(0)) return tri.v0;
    vector3<T> bp = point - tri.v1;
    T d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= T(0) && d4 <= d3) return tri.v1;
    T vc = d1*d4 - d3*d2;
    if (vc <= T(0) && d1 >= T(0) && d3 <= T(0)) {
        T v = d1 / (d1 - d3);
        return tri.v0 + ab * v;
    }
    vector3<T> cp = point - tri.v2;
    T d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= T(0) && d5 <= d6) return tri.v2;
    T vb = d5*d2 - d1*d6;
    if (vb <= T(0) && d2 >= T(0) && d6 <= T(0)) {
        T w = d2 / (d2 - d6);
        return tri.v0 + ac * w;
    }
    T va = d3*d6 - d5*d4;
    if (va <= T(0) && (d4 - d3) >= T(0) && (d5 - d6) >= T(0)) {
        T w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return tri.v1 + (tri.v2 - tri.v1) * w;
    }
    T denom = T(1) / (va + vb + vc);
    T v = vb * denom, w = vc * denom;
    return tri.v0 + ab * v + ac * w;
}

// ============================================================
// Distance between primitives
// ============================================================

template<typename T>
constexpr T distance(const vector3<T>& point, const plane<T>& p) noexcept {
    return std::abs(p.signed_distance(point));
}

template<typename T>
T distance(const vector3<T>& point, const segment3<T>& seg) noexcept {
    return distance(point, closest_point(seg, point));
}

template<typename T>
constexpr T distance(const vector3<T>& point, const aabb<T>& box) noexcept {
    return distance(point, closest_point(box, point));
}

template<typename T>
T distance(const vector3<T>& point, const triangle3<T>& tri) noexcept {
    return distance(point, closest_point(tri, point));
}

// ============================================================
// Volume of a tetrahedron
// ============================================================

template<typename T>
constexpr T tetrahedron_volume(const vector3<T>& a, const vector3<T>& b,
                               const vector3<T>& c, const vector3<T>& d) noexcept {
    return std::abs(dot(cross(b-a, c-a), d-a)) / T(6);
}

// ============================================================
// Signed volume of a mesh (for closed manifold)
// ============================================================

template<typename T>
T mesh_volume(const std::vector<vector3<T>>& vertices, const std::vector<std::array<int,3>>& faces) noexcept {
    T vol = T(0);
    for (const auto& f : faces) {
        const auto& a = vertices[f[0]], b = vertices[f[1]], c = vertices[f[2]];
        vol += dot(cross(a, b), c);
    }
    return std::abs(vol) / T(6);
}

// ============================================================
// Bounding sphere from points (Ritter's algorithm)
// ============================================================

template<typename T>
sphere<T> bounding_sphere(const std::vector<vector3<T>>& points) noexcept {
    if (points.empty()) return sphere<T>(vector3<T>(T(0)), T(0));
    vector3<T> center = points[0];
    T radius = T(0);
    for (std::size_t i=1; i<points.size(); ++i) {
        T dist = distance(center, points[i]);
        if (dist > radius) {
            T new_radius = (radius + dist) * T(0.5);
            center = center + (points[i] - center) * ((new_radius - radius) / dist);
            radius = new_radius;
        }
    }
    return sphere<T>(center, radius);
}

// ============================================================
// Bounding AABB from points
// ============================================================

template<typename T>
aabb<T> bounding_aabb(const std::vector<vector3<T>>& points) noexcept {
    if (points.empty()) return aabb<T>();
    aabb<T> box(points[0], points[0]);
    for (const auto& p : points) box.expand(p);
    return box;
}

// ============================================================
// Frustum (view frustum from view‑projection matrix)
// ============================================================

template<typename T>
struct frustum {
    plane<T> planes[6]; // left, right, bottom, top, near, far

    constexpr frustum() noexcept = default;

    frustum(const matrix4<T>& vp) noexcept {
        planes[0] = plane<T>(vector3<T>(vp(3,0)+vp(0,0), vp(3,1)+vp(0,1), vp(3,2)+vp(0,2)), vp(3,3)+vp(0,3));
        planes[1] = plane<T>(vector3<T>(vp(3,0)-vp(0,0), vp(3,1)-vp(0,1), vp(3,2)-vp(0,2)), vp(3,3)-vp(0,3));
        planes[2] = plane<T>(vector3<T>(vp(3,0)+vp(1,0), vp(3,1)+vp(1,1), vp(3,2)+vp(1,2)), vp(3,3)+vp(1,3));
        planes[3] = plane<T>(vector3<T>(vp(3,0)-vp(1,0), vp(3,1)-vp(1,1), vp(3,2)-vp(1,2)), vp(3,3)-vp(1,3));
        planes[4] = plane<T>(vector3<T>(vp(3,0)+vp(2,0), vp(3,1)+vp(2,1), vp(3,2)+vp(2,2)), vp(3,3)+vp(2,3));
        planes[5] = plane<T>(vector3<T>(vp(3,0)-vp(2,0), vp(3,1)-vp(2,1), vp(3,2)-vp(2,2)), vp(3,3)-vp(2,3));
        for (int i=0; i<6; ++i) {
            T len = length(planes[i].normal);
            if (len > T(FLOAT_EPSILON)) { planes[i].normal /= len; planes[i].distance /= len; }
        }
    }

    bool contains(const vector3<T>& point) const noexcept {
        for (int i=0; i<6; ++i) if (planes[i].signed_distance(point) < T(0)) return false;
        return true;
    }

    bool intersects(const aabb<T>& box) const noexcept {
        for (int i=0; i<6; ++i) {
            vector3<T> p = box.min;
            if (planes[i].normal.x >= T(0)) p.x = box.max.x;
            if (planes[i].normal.y >= T(0)) p.y = box.max.y;
            if (planes[i].normal.z >= T(0)) p.z = box.max.z;
            if (planes[i].signed_distance(p) < T(0)) return false;
        }
        return true;
    }
};

// ============================================================
// Type aliases
// ============================================================

template<typename T> using plane_t = plane<T>;
template<typename T> using ray_t = ray<T>;
template<typename T> using segment3_t = segment3<T>;
template<typename T> using sphere_t = sphere<T>;
template<typename T> using aabb_t = aabb<T>;
template<typename T> using obb_t = obb<T>;
template<typename T> using triangle3_t = triangle3<T>;
template<typename T> using frustum_t = frustum<T>;

using planef = plane<float>;
using rayyf = ray<float>;
using segment3f = segment3<float>;
using spheref = sphere<float>;
using aabbf = aabb<float>;
using obbf = obb<float>;
using triangle3f = triangle3<float>;
using frustumf = frustum<float>;

using planed = plane<double>;
using rayd = ray<double>;
using segment3d = segment3<double>;
using sphered = sphere<double>;
using aabbd = aabb<double>;
using obbd = obb<double>;
using triangle3d = triangle3<double>;
using frustumd = frustum<double>;

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_GEOMETRY_H