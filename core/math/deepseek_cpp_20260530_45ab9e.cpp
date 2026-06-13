// File 0016 : core/math/obb.h
// Oriented Bounding Box (center + half‑extents + orientation matrix) with intersection, distance, and transformation.

#pragma once

#include "vec3.h"
#include "mat3.h"
#include "mat4.h"
#include "aabb.h"
#include "sphere.h"
#include "ray.h"
#include "plane.h"
#include "capsule.h"
#include <algorithm>
#include <cmath>

namespace wp {

template <typename T>
struct obb {
    vec3<T>    center;
    vec3<T>    half_extents;   // local‑space half‑lengths along each axis
    mat3<T>    orientation;    // rotation matrix mapping local → world

    constexpr obb() noexcept : center(T(0)), half_extents(T(1)), orientation(identity3<T>()) {}
    constexpr obb(const vec3<T>& c, const vec3<T>& he, const mat3<T>& rot) noexcept : center(c), half_extents(he), orientation(rot) {}
    template <typename U> constexpr explicit obb(const obb<U>& o) noexcept : center(o.center), half_extents(o.half_extents), orientation(o.orientation) {}

    // Local ↔ world conversion
    constexpr vec3<T> local_to_world(const vec3<T>& local) const noexcept { return center + mul(orientation, local); }
    constexpr vec3<T> world_to_local(const vec3<T>& world) const noexcept { return mul(transpose(orientation), world - center); }

    // Closest point on OBB (in world)
    constexpr vec3<T> closest_point(const vec3<T>& p) const noexcept {
        vec3<T> local_pt = world_to_local(p);
        // Clamp to box
        vec3<T> clamped = vec3<T>(clamp(local_pt.x, -half_extents.x, half_extents.x),
                                  clamp(local_pt.y, -half_extents.y, half_extents.y),
                                  clamp(local_pt.z, -half_extents.z, half_extents.z));
        return local_to_world(clamped);
    }

    // Distance from point to OBB surface
    constexpr T distance(const vec3<T>& p) const noexcept { return length(p - closest_point(p)); }
    constexpr T distance_sq(const vec3<T>& p) const noexcept { return length_sq(p - closest_point(p)); }

    // Containment
    constexpr bool contains(const vec3<T>& p) const noexcept {
        vec3<T> local = world_to_local(p);
        return std::abs(local.x) <= half_extents.x &&
               std::abs(local.y) <= half_extents.y &&
               std::abs(local.z) <= half_extents.z;
    }

    // Bounding AABB
    constexpr aabb<T> bounding_aabb() const noexcept {
        // Transform OBB extents into world AABB by projecting absolute extents onto axes
        mat3<T> abs_rot(
            std::abs(orientation.m00), std::abs(orientation.m01), std::abs(orientation.m02),
            std::abs(orientation.m10), std::abs(orientation.m11), std::abs(orientation.m12),
            std::abs(orientation.m20), std::abs(orientation.m21), std::abs(orientation.m22)
        );
        vec3<T> world_extent = mul(abs_rot, half_extents);
        return aabb<T>(center - world_extent, center + world_extent);
    }

    // Bounding sphere
    constexpr sphere<T> bounding_sphere() const noexcept {
        return sphere<T>(center, length(half_extents));
    }

    // Corners (8 corners in world space)
    std::array<vec3<T>, 8> corners() const noexcept {
        std::array<vec3<T>, 8> c;
        for (int i = 0; i < 8; ++i) {
            vec3<T> local_corner(
                (i & 1) ? half_extents.x : -half_extents.x,
                (i & 2) ? half_extents.y : -half_extents.y,
                (i & 4) ? half_extents.z : -half_extents.z
            );
            c[i] = local_to_world(local_corner);
        }
        return c;
    }

    // Intersection tests (using Separating Axis Theorem or approximated by bounding sphere/AABB)
    constexpr bool intersects(const aabb<T>& box) const noexcept {
        // Use SAT with OBB axes and AABB face normals (world axes)
        // Implement full SAT
        return sat_obb_aabb(*this, box);
    }
    constexpr bool intersects(const sphere<T>& s) const noexcept {
        // Closest point on OBB to sphere center, check distance
        return distance_sq(s.center) <= s.radius * s.radius;
    }
    constexpr bool intersects(const plane<T>& p) const noexcept {
        // Compute effective radius of OBB along plane normal
        vec3<T> abs_n = abs(p.normal); // in world
        T r = dot(mul(orientation, half_extents), abs_n); // projection length
        T dist = p.distance(center);
        return std::abs(dist) <= r;
    }
    template <typename U>
    constexpr bool intersect_ray(const ray<U>& r, U& t) const noexcept {
        // Ray-OBB using slab test in OBB local space
        vec3<T> local_origin = world_to_local(r.origin);
        vec3<T> local_dir = mul(transpose(orientation), r.direction);
        T tmin = min_val<T>, tmax = max_val<T>;
        for (int i = 0; i < 3; ++i) {
            if (std::abs(local_dir[i]) < MathConst<T>::epsilon) {
                if (local_origin[i] < -half_extents[i] || local_origin[i] > half_extents[i])
                    return false;
            } else {
                T inv_d = T(1) / local_dir[i];
                T t1 = (-half_extents[i] - local_origin[i]) * inv_d;
                T t2 = ( half_extents[i] - local_origin[i]) * inv_d;
                if (t1 > t2) std::swap(t1, t2);
                tmin = std::max(tmin, t1);
                tmax = std::min(tmax, t2);
                if (tmin > tmax) return false;
            }
        }
        if (tmin >= T(0)) { t = static_cast<U>(tmin); return true; }
        if (tmax >= T(0)) { t = static_cast<U>(tmax); return true; }
        return false;
    }

    // Transform
    constexpr obb transform(const mat4<T>& m) const noexcept {
        mat3<T> rot = upper_left_3x3(m);
        obb result;
        result.center = vec3<T>(m.m03, m.m13, m.m23) + mul(rot, center); // Actually multiply full transform: m * (center,1)
        // Decompose scale from rotation (if non-uniform, OBB will become a parallelepiped; we approximate)
        vec3<T> sx = length(rot.col(0));
        vec3<T> sy = length(rot.col(1));
        vec3<T> sz = length(rot.col(2));
        mat3<T> rot_unnorm = mat3<T>(rot.col(0)/sx, rot.col(1)/sy, rot.col(2)/sz);
        result.orientation = mul(rot_unnorm, orientation); // orientation = R * old_orient
        result.half_extents = mul(orientation, half_extents); // wait: we need to compute new extents from scaling
        // Actually, new extents = scale * half_extents, orientation = R*old_orient
        result.half_extents = vec3<T>(sx.x * half_extents.x, sy.y * half_extents.y, sz.z * half_extents.z);
        result.orientation = mul(rot_unnorm, orientation);
        // Re-orthonormalize orientation (optional)
        result.orientation = gram_schmidt(result.orientation);
        return result;
    }

    constexpr bool operator==(const obb& o) const noexcept { return center == o.center && half_extents == o.half_extents && orientation == o.orientation; }
    constexpr bool operator!=(const obb& o) const noexcept { return !(*this == o); }

private:
    // SAT test between OBB and AABB
    static constexpr bool sat_obb_aabb(const obb& o, const aabb<T>& b) {
        // Separate on OBB axes
        for (int i = 0; i < 3; ++i) {
            vec3<T> axis = o.orientation.col(i);
            if (get_separation(o, b, axis)) return false;
        }
        // Separate on world axes (AABB face normals)
        for (int i = 0; i < 3; ++i) {
            vec3<T> axis = vec3<T>(T(0));
            axis[i] = T(1);
            if (get_separation(o, b, axis)) return false;
        }
        // Separate on cross products of OBB axes and world axes (9 more)
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                vec3<T> axis = cross(o.orientation.col(i), vec3<T>((j==0)?T(1):T(0),(j==1)?T(1):T(0),(j==2)?T(1):T(0)));
                if (length_sq(axis) < MathConst<T>::epsilon) continue;
                if (get_separation(o, b, axis)) return false;
            }
        }
        return true; // no separating axis => intersecting
    }

    // Helper: test separation along axis
    static constexpr bool get_separation(const obb& o, const aabb<T>& b, const vec3<T>& axis) {
        // Project OBB
        T o_min, o_max;
        project_obb(o, axis, o_min, o_max);
        // Project AABB
        T b_min, b_max;
        project_aabb(b, axis, b_min, b_max);
        if (o_max < b_min || b_max < o_min) return true;
        return false;
    }

    static void project_obb(const obb& o, const vec3<T>& axis, T& minv, T& maxv) {
        T center_proj = dot(axis, o.center);
        // effective radius along axis
        T r = std::abs(dot(axis, mul(o.orientation.col(0), o.half_extents.x)))
            + std::abs(dot(axis, mul(o.orientation.col(1), o.half_extents.y)))
            + std::abs(dot(axis, mul(o.orientation.col(2), o.half_extents.z)));
        minv = center_proj - r;
        maxv = center_proj + r;
    }

    static void project_aabb(const aabb<T>& b, const vec3<T>& axis, T& minv, T& maxv) {
        // Compute AABB projection: center + sum of extents * |axis·face_axis|
        vec3<T> center_b = b.center();
        vec3<T> extent_b = b.extent();
        T r = std::abs(axis.x) * extent_b.x + std::abs(axis.y) * extent_b.y + std::abs(axis.z) * extent_b.z;
        T c = dot(axis, center_b);
        minv = c - r;
        maxv = c + r;
    }
};

// Convenience type aliases
using obbf = obb<float>;
using obbd = obb<double>;

} // namespace wp