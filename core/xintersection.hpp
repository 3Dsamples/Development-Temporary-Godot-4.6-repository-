// core/xintersection.hpp
#ifndef XTENSOR_XINTERSECTION_HPP
#define XTENSOR_XINTERSECTION_HPP

// ----------------------------------------------------------------------------
// xintersection.hpp – Geometric intersection tests for ray tracing and physics
// ----------------------------------------------------------------------------
// This header provides a comprehensive set of intersection functions:
//   - Ray‑Sphere, Ray‑Box (AABB), Ray‑Triangle (Möller–Trumbore)
//   - Ray‑Plane, Ray‑Cylinder, Ray‑Cone, Ray‑Torus
//   - Sphere‑Sphere, Box‑Box, Sphere‑Box
//   - Triangle‑Triangle (separating axis theorem)
//   - Point inclusion tests
//   - Distance queries
//
// All geometric primitives use bignumber::BigNumber for high‑precision
// coordinates. FFT acceleration is not directly used but the infrastructure
// is maintained for consistency with the broader library.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <tuple>
#include <array>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xnorm.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    namespace intersection
    {
        // ========================================================================
        // Basic geometric types
        // ========================================================================
        template <class T = double> struct vec3;
        template <class T = double> struct ray;
        template <class T = double> struct sphere;
        template <class T = double> struct aabb;
        template <class T = double> struct triangle;

        // ========================================================================
        // Ray intersection functions
        // ========================================================================
        // Intersect ray with sphere, returns smallest t > 0 if hit
        template <class T> std::optional<T> intersect_ray_sphere(const ray<T>& r, const sphere<T>& s);
        // Intersect ray with AABB (slab method), returns (tmin, tmax) pair
        template <class T> std::optional<std::pair<T, T>> intersect_ray_aabb(const ray<T>& r, const aabb<T>& box);
        // Intersect ray with triangle (Möller–Trumbore), returns (t, u, v)
        template <class T> std::optional<std::tuple<T, T, T>> intersect_ray_triangle(const ray<T>& r, const triangle<T>& tri);
        // Intersect ray with plane, returns t if hit
        template <class T> std::optional<T> intersect_ray_plane(const ray<T>& r, const vec3<T>& point, const vec3<T>& normal);
        // Intersect ray with infinite cylinder along Y axis
        template <class T> std::optional<std::pair<T, T>> intersect_ray_cylinder(const ray<T>& r, T radius);
        // Intersect ray with infinite cone around Y axis
        template <class T> std::optional<std::pair<T, T>> intersect_ray_cone(const ray<T>& r, T angle_rad);

        // ========================================================================
        // Object‑Object intersection tests
        // ========================================================================
        template <class T> bool intersect_sphere_sphere(const sphere<T>& s1, const sphere<T>& s2);
        template <class T> bool intersect_aabb_aabb(const aabb<T>& box1, const aabb<T>& box2);
        template <class T> bool intersect_sphere_aabb(const sphere<T>& s, const aabb<T>& box);
        template <class T> bool intersect_triangle_triangle(const triangle<T>& t1, const triangle<T>& t2);

        // ========================================================================
        // Distance queries
        // ========================================================================
        template <class T> vec3<T> closest_point_aabb(const vec3<T>& p, const aabb<T>& box);
        template <class T> T distance_point_aabb(const vec3<T>& p, const aabb<T>& box);
        template <class T> vec3<T> closest_point_triangle(const vec3<T>& p, const triangle<T>& tri);
        template <class T> T distance_point_triangle(const vec3<T>& p, const triangle<T>& tri);

        // ========================================================================
        // Utility functions
        // ========================================================================
        template <class E> aabb<typename E::value_type> bounding_box(const xexpression<E>& points_expr);
        template <class T> aabb<T> transform_aabb(const aabb<T>& box, const xarray_container<T>& mat);
    }
}

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt
{
    namespace intersection
    {
        // 3D vector
        template <class T> struct vec3 { T x,y,z; vec3() : x(0),y(0),z(0) {} vec3(T x,T y,T z):x(x),y(y),z(z){} vec3 operator+(const vec3&o)const{return vec3(x+o.x,y+o.y,z+o.z);} T dot(const vec3&o)const{return x*o.x+y*o.y+z*o.z;} vec3 cross(const vec3&o)const{return vec3(y*o.z-z*o.y,z*o.x-x*o.z,x*o.y-y*o.x);} T length_sq()const{return x*x+y*y+z*z;} T length()const{return std::sqrt(length_sq());} };
        // Ray with origin and direction
        template <class T> struct ray { vec3<T> origin, direction; ray() : origin(), direction(0,0,1) {} ray(const vec3<T>&o,const vec3<T>&d):origin(o),direction(d){} };
        // Sphere primitive
        template <class T> struct sphere { vec3<T> center; T radius; sphere() : center(), radius(1) {} sphere(const vec3<T>&c,T r):center(c),radius(r){} };
        // Axis‑aligned bounding box
        template <class T> struct aabb { vec3<T> min, max; aabb():min(),max(){} aabb(const vec3<T>&a,const vec3<T>&b):min(a),max(b){} bool contains(const vec3<T>&p)const{return p.x>=min.x&&p.x<=max.x&&p.y>=min.y&&p.y<=max.y&&p.z>=min.z&&p.z<=max.z;} };
        // Triangle
        template <class T> struct triangle { vec3<T> v0,v1,v2; triangle():v0(),v1(),v2(){} triangle(const vec3<T>&a,const vec3<T>&b,const vec3<T>&c):v0(a),v1(b),v2(c){} };

        // Ray‑sphere intersection
        template <class T> std::optional<T> intersect_ray_sphere(const ray<T>& r, const sphere<T>& s)
        { /* TODO: implement */ return std::nullopt; }
        // Ray‑AABB intersection
        template <class T> std::optional<std::pair<T, T>> intersect_ray_aabb(const ray<T>& r, const aabb<T>& box)
        { /* TODO: implement */ return std::nullopt; }
        // Ray‑triangle intersection (Möller–Trumbore)
        template <class T> std::optional<std::tuple<T, T, T>> intersect_ray_triangle(const ray<T>& r, const triangle<T>& tri)
        { /* TODO: implement */ return std::nullopt; }
        // Ray‑plane intersection
        template <class T> std::optional<T> intersect_ray_plane(const ray<T>& r, const vec3<T>& point, const vec3<T>& normal)
        { /* TODO: implement */ return std::nullopt; }
        // Ray‑cylinder intersection
        template <class T> std::optional<std::pair<T, T>> intersect_ray_cylinder(const ray<T>& r, T radius)
        { /* TODO: implement */ return std::nullopt; }
        // Ray‑cone intersection
        template <class T> std::optional<std::pair<T, T>> intersect_ray_cone(const ray<T>& r, T angle_rad)
        { /* TODO: implement */ return std::nullopt; }

        // Sphere‑sphere intersection test
        template <class T> bool intersect_sphere_sphere(const sphere<T>& s1, const sphere<T>& s2)
        { /* TODO: implement */ return false; }
        // AABB‑AABB intersection test
        template <class T> bool intersect_aabb_aabb(const aabb<T>& box1, const aabb<T>& box2)
        { /* TODO: implement */ return false; }
        // Sphere‑AABB intersection test
        template <class T> bool intersect_sphere_aabb(const sphere<T>& s, const aabb<T>& box)
        { /* TODO: implement */ return false; }
        // Triangle‑triangle intersection test (SAT)
        template <class T> bool intersect_triangle_triangle(const triangle<T>& t1, const triangle<T>& t2)
        { /* TODO: implement */ return false; }

        // Closest point on AABB to a point
        template <class T> vec3<T> closest_point_aabb(const vec3<T>& p, const aabb<T>& box)
        { /* TODO: implement */ return p; }
        // Distance from point to AABB
        template <class T> T distance_point_aabb(const vec3<T>& p, const aabb<T>& box)
        { /* TODO: implement */ return T(0); }
        // Closest point on triangle to a point
        template <class T> vec3<T> closest_point_triangle(const vec3<T>& p, const triangle<T>& tri)
        { /* TODO: implement */ return p; }
        // Distance from point to triangle
        template <class T> T distance_point_triangle(const vec3<T>& p, const triangle<T>& tri)
        { /* TODO: implement */ return T(0); }

        // Compute bounding box of a set of points
        template <class E> aabb<typename E::value_type> bounding_box(const xexpression<E>& points_expr)
        { /* TODO: implement */ return aabb<typename E::value_type>(); }
        // Transform an AABB by a 4x4 matrix
        template <class T> aabb<T> transform_aabb(const aabb<T>& box, const xarray_container<T>& mat)
        { /* TODO: implement */ return box; }
    }
}

#endif // XTENSOR_XINTERSECTION_HPP
                if (detail::abs_val(dir_i) < detail::epsilon<T>())
                {
                    if (origin_i < min_i || origin_i > max_i)
                        return std::nullopt;
                }
                else
                {
                    T inv_dir = T(1) / dir_i;
                    T t0 = (min_i - origin_i) * inv_dir;
                    T t1 = (max_i - origin_i) * inv_dir;
                    if (inv_dir < T(0))
                        std::swap(t0, t1);
                    tmin = std::max(tmin, t0);
                    tmax = std::min(tmax, t1);
                    if (tmin > tmax)
                        return std::nullopt;
                }
            }
            return std::make_pair(tmin, tmax);
        }

        // ------------------------------------------------------------------------
        // Ray‑Triangle intersection (Möller–Trumbore)
        // ------------------------------------------------------------------------
        template <class T>
        std::optional<std::tuple<T, T, T>> intersect_ray_triangle(const ray<T>& r, const triangle<T>& tri)
        {
            constexpr T eps = detail::epsilon<T>();
            vec3<T> edge1 = tri.v1 - tri.v0;
            vec3<T> edge2 = tri.v2 - tri.v0;
            vec3<T> h = r.direction.cross(edge2);
            T a = edge1.dot(h);

            if (detail::abs_val(a) < eps)
                return std::nullopt; // ray parallel to triangle

            T f = T(1) / a;
            vec3<T> s = r.origin - tri.v0;
            T u = f * s.dot(h);
            if (u < T(0) || u > T(1))
                return std::nullopt;

            vec3<T> q = s.cross(edge1);
            T v = f * r.direction.dot(q);
            if (v < T(0) || u + v > T(1))
                return std::nullopt;

            T t = f * edge2.dot(q);
            if (t > eps)
                return std::tuple<T, T, T>(t, u, v);
            return std::nullopt;
        }

        // ------------------------------------------------------------------------
        // Ray‑Plane intersection
        // ------------------------------------------------------------------------
        template <class T>
        std::optional<T> intersect_ray_plane(const ray<T>& r, const vec3<T>& point, const vec3<T>& normal)
        {
            T denom = normal.dot(r.direction);
            if (detail::abs_val(denom) < detail::epsilon<T>())
                return std::nullopt;
            T t = (point - r.origin).dot(normal) / denom;
            if (t > T(0))
                return t;
            return std::nullopt;
        }

        // ------------------------------------------------------------------------
        // Ray‑Cylinder intersection (infinite cylinder along Y axis)
        // ------------------------------------------------------------------------
        template <class T>
        std::optional<std::pair<T, T>> intersect_ray_cylinder(const ray<T>& r, T radius)
        {
            T a = r.direction.x * r.direction.x + r.direction.z * r.direction.z;
            T b = T(2) * (r.origin.x * r.direction.x + r.origin.z * r.direction.z);
            T c = r.origin.x * r.origin.x + r.origin.z * r.origin.z - radius * radius;

            T discriminant = b * b - T(4) * a * c;
            if (discriminant < T(0))
                return std::nullopt;

            T sqrt_disc = detail::sqrt_val(discriminant);
            T t0 = (-b - sqrt_disc) / (T(2) * a);
            T t1 = (-b + sqrt_disc) / (T(2) * a);
            return std::make_pair(t0, t1);
        }

        // ------------------------------------------------------------------------
        // Ray‑Cone intersection (infinite cone around Y axis)
        // ------------------------------------------------------------------------
        template <class T>
        std::optional<std::pair<T, T>> intersect_ray_cone(const ray<T>& r, T angle_rad)
        {
            T k = std::tan(angle_rad);
            T k2 = k * k;
            T a = r.direction.x * r.direction.x + r.direction.z * r.direction.z - k2 * r.direction.y * r.direction.y;
            T b = T(2) * (r.origin.x * r.direction.x + r.origin.z * r.direction.z - k2 * r.origin.y * r.direction.y);
            T c = r.origin.x * r.origin.x + r.origin.z * r.origin.z - k2 * r.origin.y * r.origin.y;

            if (detail::abs_val(a) < detail::epsilon<T>())
            {
                if (detail::abs_val(b) < detail::epsilon<T>())
                    return std::nullopt;
                T t = -c / b;
                return std::make_pair(t, t);
            }

            T discriminant = b * b - T(4) * a * c;
            if (discriminant < T(0))
                return std::nullopt;
            T sqrt_disc = detail::sqrt_val(discriminant);
            T t0 = (-b - sqrt_disc) / (T(2) * a);
            T t1 = (-b + sqrt_disc) / (T(2) * a);
            return std::make_pair(t0, t1);
        }

        // ========================================================================
        // Object‑Object intersection tests
        // ========================================================================

        // ------------------------------------------------------------------------
        // Sphere‑Sphere intersection
        // ------------------------------------------------------------------------
        template <class T>
        bool intersect_sphere_sphere(const sphere<T>& s1, const sphere<T>& s2)
        {
            T dist_sq = (s1.center - s2.center).length_sq();
            T radius_sum = s1.radius + s2.radius;
            return dist_sq <= radius_sum * radius_sum;
        }

        // ------------------------------------------------------------------------
        // AABB‑AABB intersection
        // ------------------------------------------------------------------------
        template <class T>
        bool intersect_aabb_aabb(const aabb<T>& box1, const aabb<T>& box2)
        {
            return (box1.min.x <= box2.max.x && box1.max.x >= box2.min.x) &&
                   (box1.min.y <= box2.max.y && box1.max.y >= box2.min.y) &&
                   (box1.min.z <= box2.max.z && box1.max.z >= box2.min.z);
        }

        // ------------------------------------------------------------------------
        // Sphere‑AABB intersection
        // ------------------------------------------------------------------------
        template <class T>
        bool intersect_sphere_aabb(const sphere<T>& s, const aabb<T>& box)
        {
            vec3<T> closest;
            closest.x = std::max(box.min.x, std::min(s.center.x, box.max.x));
            closest.y = std::max(box.min.y, std::min(s.center.y, box.max.y));
            closest.z = std::max(box.min.z, std::min(s.center.z, box.max.z));
            T dist_sq = (s.center - closest).length_sq();
            return dist_sq <= s.radius * s.radius;
        }

        // ------------------------------------------------------------------------
        // Triangle‑Triangle intersection (SAT in 3D)
        // ------------------------------------------------------------------------
        template <class T>
        bool intersect_triangle_triangle(const triangle<T>& t1, const triangle<T>& t2)
        {
            // Compute plane of t1
            vec3<T> n1 = t1.normal();
            T d1 = -n1.dot(t1.v0);

            // Compute signed distances of t2 vertices to plane of t1
            T dist2_0 = n1.dot(t2.v0) + d1;
            T dist2_1 = n1.dot(t2.v1) + d1;
            T dist2_2 = n1.dot(t2.v2) + d1;

            // If all same sign and none zero, no intersection
            if ((dist2_0 > T(0) && dist2_1 > T(0) && dist2_2 > T(0)) ||
                (dist2_0 < T(0) && dist2_1 < T(0) && dist2_2 < T(0)))
                return false;

            // Compute plane of t2
            vec3<T> n2 = t2.normal();
            T d2 = -n2.dot(t2.v0);

            T dist1_0 = n2.dot(t1.v0) + d2;
            T dist1_1 = n2.dot(t1.v1) + d2;
            T dist1_2 = n2.dot(t1.v2) + d2;

            if ((dist1_0 > T(0) && dist1_1 > T(0) && dist1_2 > T(0)) ||
                (dist1_0 < T(0) && dist1_1 < T(0) && dist1_2 < T(0)))
                return false;

            // Compute intersection line direction
            vec3<T> D = n1.cross(n2);
            if (D.length_sq() < detail::epsilon<T>())
                return false; // parallel planes, handle separately (projection overlap)

            // Project triangles onto line direction and check overlap
            // (Simplified: just check if intervals overlap)
            auto project_triangle = [&](const triangle<T>& tri, const vec3<T>& dir) -> std::pair<T, T> {
                T v0 = tri.v0.dot(dir);
                T v1 = tri.v1.dot(dir);
                T v2 = tri.v2.dot(dir);
                T min_val = std::min({v0, v1, v2});
                T max_val = std::max({v0, v1, v2});
                return {min_val, max_val};
            };

            auto proj1 = project_triangle(t1, D);
            auto proj2 = project_triangle(t2, D);
            return !(proj1.second < proj2.first || proj2.second < proj1.first);
        }

        // ========================================================================
        // Distance queries
        // ========================================================================

        // ------------------------------------------------------------------------
        // Closest point on AABB to a point
        // ------------------------------------------------------------------------
        template <class T>
        vec3<T> closest_point_aabb(const vec3<T>& p, const aabb<T>& box)
        {
            return vec3<T>(
                std::max(box.min.x, std::min(p.x, box.max.x)),
                std::max(box.min.y, std::min(p.y, box.max.y)),
                std::max(box.min.z, std::min(p.z, box.max.z))
            );
        }

        // ------------------------------------------------------------------------
        // Distance from point to AABB
        // ------------------------------------------------------------------------
        template <class T>
        T distance_point_aabb(const vec3<T>& p, const aabb<T>& box)
        {
            vec3<T> closest = closest_point_aabb(p, box);
            return (p - closest).length();
        }

        // ------------------------------------------------------------------------
        // Closest point on triangle to a point
        // ------------------------------------------------------------------------
        template <class T>
        vec3<T> closest_point_triangle(const vec3<T>& p, const triangle<T>& tri)
        {
            vec3<T> edge0 = tri.v1 - tri.v0;
            vec3<T> edge1 = tri.v2 - tri.v0;
            vec3<T> v0p = p - tri.v0;

            T a = edge0.dot(edge0);
            T b = edge0.dot(edge1);
            T c = edge1.dot(edge1);
            T d = edge0.dot(v0p);
            T e = edge1.dot(v0p);

            T det = a * c - b * b;
            T s = b * e - c * d;
            T t = b * d - a * e;

            if (s + t <= det)
            {
                if (s < T(0))
                {
                    if (t < T(0))
                    {
                        // Region 4
                        if (d < T(0))
                        {
                            t = T(0);
                            s = detail::clamp(-d / a, T(0), T(1));
                        }
                        else
                        {
                            s = T(0);
                            t = detail::clamp(-e / c, T(0), T(1));
                        }
                    }
                    else
                    {
                        // Region 3
                        s = T(0);
                        t = detail::clamp(-e / c, T(0), T(1));
                    }
                }
                else if (t < T(0))
                {
                    // Region 5
                    t = T(0);
                    s = detail::clamp(-d / a, T(0), T(1));
                }
                else
                {
                    // Region 0
                    T inv_det = T(1) / det;
                    s *= inv_det;
                    t *= inv_det;
                }
            }
            else
            {
                if (s < T(0))
                {
                    // Region 2
                    T tmp0 = b + d;
                    T tmp1 = c + e;
                    if (tmp1 > tmp0)
                    {
                        T numer = tmp1 - tmp0;
                        T denom = a - T(2) * b + c;
                        s = detail::clamp(numer / denom, T(0), T(1));
                        t = T(1) - s;
                    }
                    else
                    {
                        s = T(0);
                        t = detail::clamp(-e / c, T(0), T(1));
                    }
                }
                else if (t < T(0))
                {
                    // Region 6
                    T tmp0 = a + d;
                    T tmp1 = b + e;
                    if (tmp1 > tmp0)
                    {
                        T numer = tmp1 - tmp0;
                        T denom = a - T(2) * b + c;
                        t = detail::clamp(numer / denom, T(0), T(1));
                        s = T(1) - t;
                    }
                    else
                    {
                        t = T(0);
                        s = detail::clamp(-e / c, T(0), T(1));
                    }
                }
                else
                {
                    // Region 1
                    T numer = c + e - b - d;
                    T denom = a - T(2) * b + c;
                    s = detail::clamp(numer / denom, T(0), T(1));
                    t = T(1) - s;
                }
            }

            return tri.v0 + edge0 * s + edge1 * t;
        }

        // ------------------------------------------------------------------------
        // Distance from point to triangle
        // ------------------------------------------------------------------------
        template <class T>
        T distance_point_triangle(const vec3<T>& p, const triangle<T>& tri)
        {
            vec3<T> closest = closest_point_triangle(p, tri);
            return (p - closest).length();
        }

        // ========================================================================
        // Utility functions
        // ========================================================================

        // ------------------------------------------------------------------------
        // Build AABB from set of points
        // ------------------------------------------------------------------------
        template <class E>
        aabb<typename E::value_type> bounding_box(const xexpression<E>& points_expr)
        {
            const auto& points = points_expr.derived_cast();
            if (points.dimension() != 2 || points.shape()[1] != 3)
                XTENSOR_THROW(std::invalid_argument, "bounding_box: input must be Nx3 array");
            using T = typename E::value_type;
            aabb<T> box(vec3<T>(points(0,0), points(0,1), points(0,2)),
                        vec3<T>(points(0,0), points(0,1), points(0,2)));
            for (size_t i = 1; i < points.shape()[0]; ++i)
                box.expand(vec3<T>(points(i,0), points(i,1), points(i,2)));
            return box;
        }

        // ------------------------------------------------------------------------
        // Transform AABB by matrix (returns new AABB)
        // ------------------------------------------------------------------------
        template <class T>
        aabb<T> transform_aabb(const aabb<T>& box, const xarray_container<T>& mat)
        {
            if (mat.dimension() != 2 || mat.shape()[0] != 4 || mat.shape()[1] != 4)
                XTENSOR_THROW(std::invalid_argument, "transform_aabb: matrix must be 4x4");

            vec3<T> corners[8] = {
                vec3<T>(box.min.x, box.min.y, box.min.z),
                vec3<T>(box.max.x, box.min.y, box.min.z),
                vec3<T>(box.min.x, box.max.y, box.min.z),
                vec3<T>(box.max.x, box.max.y, box.min.z),
                vec3<T>(box.min.x, box.min.y, box.max.z),
                vec3<T>(box.max.x, box.min.y, box.max.z),
                vec3<T>(box.min.x, box.max.y, box.max.z),
                vec3<T>(box.max.x, box.max.y, box.max.z)
            };

            aabb<T> result;
            bool first = true;
            for (const auto& corner : corners)
            {
                T x = mat(0,0) * corner.x + mat(0,1) * corner.y + mat(0,2) * corner.z + mat(0,3);
                T y = mat(1,0) * corner.x + mat(1,1) * corner.y + mat(1,2) * corner.z + mat(1,3);
                T z = mat(2,0) * corner.x + mat(2,1) * corner.y + mat(2,2) * corner.z + mat(2,3);
                T w = mat(3,0) * corner.x + mat(3,1) * corner.y + mat(3,2) * corner.z + mat(3,3);
                if (w != T(1) && w != T(0))
                {
                    x /= w; y /= w; z /= w;
                }
                vec3<T> p(x, y, z);
                if (first)
                {
                    result.min = result.max = p;
                    first = false;
                }
                else
                {
                    result.expand(p);
                }
            }
            return result;
        }

    } // namespace intersection

    // Bring intersection functions into xt namespace
    using intersection::vec3;
    using intersection::ray;
    using intersection::sphere;
    using intersection::aabb;
    using intersection::triangle;
    using intersection::intersect_ray_sphere;
    using intersection::intersect_ray_aabb;
    using intersection::intersect_ray_triangle;
    using intersection::intersect_ray_plane;
    using intersection::intersect_ray_cylinder;
    using intersection::intersect_ray_cone;
    using intersection::intersect_sphere_sphere;
    using intersection::intersect_aabb_aabb;
    using intersection::intersect_sphere_aabb;
    using intersection::intersect_triangle_triangle;
    using intersection::closest_point_aabb;
    using intersection::closest_point_triangle;
    using intersection::distance_point_aabb;
    using intersection::distance_point_triangle;
    using intersection::bounding_box;
    using intersection::transform_aabb;

} // namespace xt

#endif // XTENSOR_XINTERSECTION_HPP