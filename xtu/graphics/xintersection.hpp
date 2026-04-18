// math/xintersection.hpp

#ifndef XTENSOR_XINTERSECTION_HPP
#define XTENSOR_XINTERSECTION_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "xnorm.hpp"
#include "xlinalg.hpp"

#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <limits>
#include <type_traits>
#include <functional>
#include <tuple>
#include <optional>
#include <cstdint>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace intersection
        {
            // --------------------------------------------------------------------
            // Basic geometric types
            // --------------------------------------------------------------------
            template <class T = double>
            struct Vector3
            {
                T x, y, z;
                Vector3(T x_ = 0, T y_ = 0, T z_ = 0) : x(x_), y(y_), z(z_) {}
                
                Vector3 operator+(const Vector3& v) const { return {x+v.x, y+v.y, z+v.z}; }
                Vector3 operator-(const Vector3& v) const { return {x-v.x, y-v.y, z-v.z}; }
                Vector3 operator*(T s) const { return {x*s, y*s, z*s}; }
                Vector3 operator/(T s) const { return {x/s, y/s, z/s}; }
                Vector3 operator-() const { return {-x, -y, -z}; }
                
                T dot(const Vector3& v) const { return x*v.x + y*v.y + z*v.z; }
                Vector3 cross(const Vector3& v) const
                {
                    return {y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x};
                }
                T length() const { return std::sqrt(dot(*this)); }
                T length_sq() const { return dot(*this); }
                Vector3 normalized() const
                {
                    T l = length();
                    if (l > 0) return *this / l;
                    return *this;
                }
                T& operator[](size_t i)
                {
                    if (i == 0) return x;
                    if (i == 1) return y;
                    return z;
                }
                const T& operator[](size_t i) const
                {
                    if (i == 0) return x;
                    if (i == 1) return y;
                    return z;
                }
            };
            
            template <class T = double>
            struct Vector2
            {
                T x, y;
                Vector2(T x_ = 0, T y_ = 0) : x(x_), y(y_) {}
                Vector2 operator+(const Vector2& v) const { return {x+v.x, y+v.y}; }
                Vector2 operator-(const Vector2& v) const { return {x-v.x, y-v.y}; }
                Vector2 operator*(T s) const { return {x*s, y*s}; }
                T dot(const Vector2& v) const { return x*v.x + y*v.y; }
                T cross(const Vector2& v) const { return x*v.y - y*v.x; }
                T length() const { return std::sqrt(dot(*this)); }
                Vector2 normalized() const
                {
                    T l = length();
                    if (l > 0) return *this / l;
                    return *this;
                }
            };
            
            template <class T = double>
            struct Ray
            {
                Vector3<T> origin;
                Vector3<T> direction;  // assumed normalized
            
                Ray(const Vector3<T>& o, const Vector3<T>& d)
                    : origin(o), direction(d.normalized()) {}
                    
                Vector3<T> point_at(T t) const { return origin + direction * t; }
            };
            
            template <class T = double>
            struct BoundingBox3
            {
                Vector3<T> min, max;
                BoundingBox3() : min(std::numeric_limits<T>::max()),
                                 max(std::numeric_limits<T>::lowest()) {}
                BoundingBox3(const Vector3<T>& min_, const Vector3<T>& max_)
                    : min(min_), max(max_) {}
                    
                void extend(const Vector3<T>& p)
                {
                    min.x = std::min(min.x, p.x);
                    min.y = std::min(min.y, p.y);
                    min.z = std::min(min.z, p.z);
                    max.x = std::max(max.x, p.x);
                    max.y = std::max(max.y, p.y);
                    max.z = std::max(max.z, p.z);
                }
                
                Vector3<T> center() const { return (min + max) * 0.5; }
                Vector3<T> extents() const { return (max - min) * 0.5; }
                bool contains(const Vector3<T>& p) const
                {
                    return p.x >= min.x && p.x <= max.x &&
                           p.y >= min.y && p.y <= max.y &&
                           p.z >= min.z && p.z <= max.z;
                }
            };
            
            template <class T = double>
            struct Triangle
            {
                Vector3<T> v0, v1, v2;
                Triangle() = default;
                Triangle(const Vector3<T>& a, const Vector3<T>& b, const Vector3<T>& c)
                    : v0(a), v1(b), v2(c) {}
                    
                Vector3<T> normal() const { return (v1 - v0).cross(v2 - v0).normalized(); }
                Vector3<T> center() const { return (v0 + v1 + v2) / 3.0; }
                T area() const { return (v1 - v0).cross(v2 - v0).length() * 0.5; }
            };
            
            template <class T = double>
            struct Sphere
            {
                Vector3<T> center;
                T radius;
                Sphere(const Vector3<T>& c, T r) : center(c), radius(r) {}
            };
            
            template <class T = double>
            struct Plane
            {
                Vector3<T> normal;
                T d; // distance from origin: normal.dot(point) + d = 0
                Plane() = default;
                Plane(const Vector3<T>& n, T d_) : normal(n.normalized()), d(d_) {}
                Plane(const Vector3<T>& point, const Vector3<T>& n)
                    : normal(n.normalized()), d(-normal.dot(point)) {}
                    
                T signed_distance(const Vector3<T>& p) const { return normal.dot(p) + d; }
            };
            
            // --------------------------------------------------------------------
            // Intersection result structures
            // --------------------------------------------------------------------
            template <class T = double>
            struct HitInfo
            {
                bool hit = false;
                T t = std::numeric_limits<T>::max();
                Vector3<T> point;
                Vector3<T> normal;
                Vector2<T> uv;
                size_t primitive_id = 0;
                void* material = nullptr;
            };
            
            // --------------------------------------------------------------------
            // Ray-Sphere intersection
            // --------------------------------------------------------------------
            template <class T>
            inline bool intersect_ray_sphere(const Ray<T>& ray, const Sphere<T>& sphere,
                                             T& t, T t_min = 0.001, T t_max = std::numeric_limits<T>::max())
            {
                Vector3<T> oc = ray.origin - sphere.center;
                T a = ray.direction.dot(ray.direction);
                T b = 2.0 * oc.dot(ray.direction);
                T c = oc.dot(oc) - sphere.radius * sphere.radius;
                T discriminant = b*b - 4.0*a*c;
                if (discriminant < 0) return false;
                T sqrt_d = std::sqrt(discriminant);
                T t1 = (-b - sqrt_d) / (2.0*a);
                T t2 = (-b + sqrt_d) / (2.0*a);
                t = t1;
                if (t < t_min) t = t2;
                if (t < t_min || t > t_max) return false;
                return true;
            }
            
            template <class T>
            inline bool intersect_ray_sphere(const Ray<T>& ray, const Sphere<T>& sphere, HitInfo<T>& hit,
                                             T t_min = 0.001, T t_max = std::numeric_limits<T>::max())
            {
                T t;
                if (!intersect_ray_sphere(ray, sphere, t, t_min, t_max)) return false;
                hit.hit = true;
                hit.t = t;
                hit.point = ray.point_at(t);
                hit.normal = (hit.point - sphere.center).normalized();
                return true;
            }
            
            // --------------------------------------------------------------------
            // Ray-Plane intersection
            // --------------------------------------------------------------------
            template <class T>
            inline bool intersect_ray_plane(const Ray<T>& ray, const Plane<T>& plane,
                                            T& t, T t_min = 0.001, T t_max = std::numeric_limits<T>::max())
            {
                T denom = plane.normal.dot(ray.direction);
                if (std::abs(denom) < 1e-8) return false;
                t = -(plane.normal.dot(ray.origin) + plane.d) / denom;
                return t >= t_min && t <= t_max;
            }
            
            template <class T>
            inline bool intersect_ray_plane(const Ray<T>& ray, const Plane<T>& plane, HitInfo<T>& hit,
                                            T t_min = 0.001, T t_max = std::numeric_limits<T>::max())
            {
                T t;
                if (!intersect_ray_plane(ray, plane, t, t_min, t_max)) return false;
                hit.hit = true;
                hit.t = t;
                hit.point = ray.point_at(t);
                hit.normal = plane.normal;
                return true;
            }
            
            // --------------------------------------------------------------------
            // Ray-Triangle intersection (Möller–Trumbore)
            // --------------------------------------------------------------------
            template <class T>
            inline bool intersect_ray_triangle(const Ray<T>& ray, const Triangle<T>& tri,
                                               T& t, T& u, T& v,
                                               T t_min = 0.001, T t_max = std::numeric_limits<T>::max())
            {
                Vector3<T> edge1 = tri.v1 - tri.v0;
                Vector3<T> edge2 = tri.v2 - tri.v0;
                Vector3<T> h = ray.direction.cross(edge2);
                T a = edge1.dot(h);
                if (std::abs(a) < 1e-8) return false;
                T f = 1.0 / a;
                Vector3<T> s = ray.origin - tri.v0;
                u = f * s.dot(h);
                if (u < 0.0 || u > 1.0) return false;
                Vector3<T> q = s.cross(edge1);
                v = f * ray.direction.dot(q);
                if (v < 0.0 || u + v > 1.0) return false;
                t = f * edge2.dot(q);
                return t >= t_min && t <= t_max;
            }
            
            template <class T>
            inline bool intersect_ray_triangle(const Ray<T>& ray, const Triangle<T>& tri, HitInfo<T>& hit,
                                               T t_min = 0.001, T t_max = std::numeric_limits<T>::max())
            {
                T t, u, v;
                if (!intersect_ray_triangle(ray, tri, t, u, v, t_min, t_max)) return false;
                hit.hit = true;
                hit.t = t;
                hit.point = ray.point_at(t);
                hit.normal = tri.normal();
                hit.uv = Vector2<T>(u, v);
                return true;
            }
            
            // --------------------------------------------------------------------
            // Ray-AABB intersection (slab method)
            // --------------------------------------------------------------------
            template <class T>
            inline bool intersect_ray_aabb(const Ray<T>& ray, const BoundingBox3<T>& box,
                                           T& t_min_out, T& t_max_out,
                                           T t_min = 0.001, T t_max = std::numeric_limits<T>::max())
            {
                T t1 = (box.min.x - ray.origin.x) / ray.direction.x;
                T t2 = (box.max.x - ray.origin.x) / ray.direction.x;
                T t_near = std::min(t1, t2);
                T t_far  = std::max(t1, t2);
                
                t1 = (box.min.y - ray.origin.y) / ray.direction.y;
                t2 = (box.max.y - ray.origin.y) / ray.direction.y;
                t_near = std::max(t_near, std::min(t1, t2));
                t_far  = std::min(t_far,  std::max(t1, t2));
                
                t1 = (box.min.z - ray.origin.z) / ray.direction.z;
                t2 = (box.max.z - ray.origin.z) / ray.direction.z;
                t_near = std::max(t_near, std::min(t1, t2));
                t_far  = std::min(t_far,  std::max(t1, t2));
                
                if (t_near > t_far || t_far < t_min || t_near > t_max) return false;
                t_min_out = t_near;
                t_max_out = t_far;
                return true;
            }
            
            template <class T>
            inline bool intersect_ray_aabb(const Ray<T>& ray, const BoundingBox3<T>& box,
                                           T t_min = 0.001, T t_max = std::numeric_limits<T>::max())
            {
                T t1, t2;
                return intersect_ray_aabb(ray, box, t1, t2, t_min, t_max);
            }
            
            // --------------------------------------------------------------------
            // Sphere-Sphere intersection
            // --------------------------------------------------------------------
            template <class T>
            inline bool intersect_sphere_sphere(const Sphere<T>& s1, const Sphere<T>& s2)
            {
                T dist_sq = (s1.center - s2.center).length_sq();
                T radius_sum = s1.radius + s2.radius;
                return dist_sq <= radius_sum * radius_sum;
            }
            
            // --------------------------------------------------------------------
            // Sphere-AABB intersection
            // --------------------------------------------------------------------
            template <class T>
            inline bool intersect_sphere_aabb(const Sphere<T>& sphere, const BoundingBox3<T>& box)
            {
                T dist_sq = 0.0;
                for (int i = 0; i < 3; ++i)
                {
                    T v = sphere.center[i];
                    if (v < box.min[i]) dist_sq += (box.min[i] - v) * (box.min[i] - v);
                    if (v > box.max[i]) dist_sq += (v - box.max[i]) * (v - box.max[i]);
                }
                return dist_sq <= sphere.radius * sphere.radius;
            }
            
            // --------------------------------------------------------------------
            // AABB-AABB intersection
            // --------------------------------------------------------------------
            template <class T>
            inline bool intersect_aabb_aabb(const BoundingBox3<T>& a, const BoundingBox3<T>& b)
            {
                return (a.min.x <= b.max.x && a.max.x >= b.min.x) &&
                       (a.min.y <= b.max.y && a.max.y >= b.min.y) &&
                       (a.min.z <= b.max.z && a.max.z >= b.min.z);
            }
            
            // --------------------------------------------------------------------
            // Triangle-AABB intersection (separating axis theorem)
            // --------------------------------------------------------------------
            template <class T>
            inline bool intersect_triangle_aabb(const Triangle<T>& tri, const BoundingBox3<T>& box)
            {
                // Translate triangle so box is centered at origin
                Vector3<T> center = box.center();
                Vector3<T> ext = box.extents();
                Vector3<T> v0 = tri.v0 - center;
                Vector3<T> v1 = tri.v1 - center;
                Vector3<T> v2 = tri.v2 - center;
                
                // Test axes: box normals
                if (!overlap_on_axis(v0, v1, v2, Vector3<T>(1,0,0), ext.x)) return false;
                if (!overlap_on_axis(v0, v1, v2, Vector3<T>(0,1,0), ext.y)) return false;
                if (!overlap_on_axis(v0, v1, v2, Vector3<T>(0,0,1), ext.z)) return false;
                
                // Test triangle normal
                Vector3<T> tri_normal = (v1 - v0).cross(v2 - v0);
                if (!overlap_on_axis(v0, v1, v2, tri_normal, std::abs(tri_normal.dot(ext)))) return false;
                
                // Test edge cross products
                Vector3<T> edges[3] = {v1 - v0, v2 - v1, v0 - v2};
                Vector3<T> box_normals[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        Vector3<T> axis = edges[i].cross(box_normals[j]);
                        T proj_ext = std::abs(axis.x) * ext.x + std::abs(axis.y) * ext.y + std::abs(axis.z) * ext.z;
                        if (!overlap_on_axis(v0, v1, v2, axis, proj_ext)) return false;
                    }
                }
                return true;
            }
            
        private:
            template <class T>
            static bool overlap_on_axis(const Vector3<T>& v0, const Vector3<T>& v1, const Vector3<T>& v2,
                                        const Vector3<T>& axis, T box_proj)
            {
                T p0 = v0.dot(axis);
                T p1 = v1.dot(axis);
                T p2 = v2.dot(axis);
                T min_p = std::min({p0, p1, p2});
                T max_p = std::max({p0, p1, p2});
                return !(min_p > box_proj || max_p < -box_proj);
            }
            
        public:
            // --------------------------------------------------------------------
            // Point in triangle (2D)
            // --------------------------------------------------------------------
            template <class T>
            inline bool point_in_triangle_2d(const Vector2<T>& p, const Vector2<T>& a,
                                             const Vector2<T>& b, const Vector2<T>& c)
            {
                Vector2<T> v0 = c - a;
                Vector2<T> v1 = b - a;
                Vector2<T> v2 = p - a;
                T dot00 = v0.dot(v0);
                T dot01 = v0.dot(v1);
                T dot02 = v0.dot(v2);
                T dot11 = v1.dot(v1);
                T dot12 = v1.dot(v2);
                T inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
                T u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
                T v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
                return (u >= 0) && (v >= 0) && (u + v <= 1);
            }
            
            // --------------------------------------------------------------------
            // Point in polygon (ray casting)
            // --------------------------------------------------------------------
            template <class T>
            inline bool point_in_polygon_2d(const Vector2<T>& p, const std::vector<Vector2<T>>& poly)
            {
                bool inside = false;
                size_t n = poly.size();
                for (size_t i = 0, j = n - 1; i < n; j = i++)
                {
                    const auto& pi = poly[i];
                    const auto& pj = poly[j];
                    if (((pi.y > p.y) != (pj.y > p.y)) &&
                        (p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y) + pi.x))
                        inside = !inside;
                }
                return inside;
            }
            
            // --------------------------------------------------------------------
            // Line segment intersection (2D)
            // --------------------------------------------------------------------
            template <class T>
            inline bool intersect_segment_segment_2d(const Vector2<T>& a1, const Vector2<T>& a2,
                                                     const Vector2<T>& b1, const Vector2<T>& b2,
                                                     Vector2<T>& out_intersection,
                                                     T eps = 1e-8)
            {
                Vector2<T> r = a2 - a1;
                Vector2<T> s = b2 - b1;
                T rxs = r.cross(s);
                Vector2<T> qp = b1 - a1;
                T qpxr = qp.cross(r);
                if (std::abs(rxs) < eps)
                {
                    if (std::abs(qpxr) < eps) // collinear
                    {
                        T t0 = qp.dot(r) / r.dot(r);
                        T t1 = t0 + s.dot(r) / r.dot(r);
                        if (std::min(t0, t1) <= 1.0 && std::max(t0, t1) >= 0.0)
                        {
                            out_intersection = a1 + r * std::max(0.0, std::min(t0, t1));
                            return true;
                        }
                    }
                    return false;
                }
                T t = qp.cross(s) / rxs;
                T u = qp.cross(r) / rxs;
                if (t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0)
                {
                    out_intersection = a1 + r * t;
                    return true;
                }
                return false;
            }
            
            // --------------------------------------------------------------------
            // Ray-Mesh intersection (naive O(N))
            // --------------------------------------------------------------------
            template <class T>
            inline bool intersect_ray_mesh(const Ray<T>& ray,
                                           const std::vector<Vector3<T>>& vertices,
                                           const std::vector<std::array<size_t, 3>>& triangles,
                                           HitInfo<T>& hit,
                                           T t_min = 0.001, T t_max = std::numeric_limits<T>::max())
            {
                bool any_hit = false;
                hit.t = t_max;
                for (size_t i = 0; i < triangles.size(); ++i)
                {
                    const auto& tri = triangles[i];
                    Triangle<T> t(vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]);
                    HitInfo<T> temp_hit;
                    if (intersect_ray_triangle(ray, t, temp_hit, t_min, hit.t))
                    {
                        if (temp_hit.t < hit.t)
                        {
                            hit = temp_hit;
                            hit.primitive_id = i;
                            any_hit = true;
                        }
                    }
                }
                return any_hit;
            }
            
            // --------------------------------------------------------------------
            // Ray casting against multiple spheres
            // --------------------------------------------------------------------
            template <class T>
            inline bool intersect_ray_spheres(const Ray<T>& ray,
                                              const std::vector<Sphere<T>>& spheres,
                                              HitInfo<T>& hit,
                                              T t_min = 0.001, T t_max = std::numeric_limits<T>::max())
            {
                bool any_hit = false;
                hit.t = t_max;
                for (size_t i = 0; i < spheres.size(); ++i)
                {
                    HitInfo<T> temp_hit;
                    if (intersect_ray_sphere(ray, spheres[i], temp_hit, t_min, hit.t))
                    {
                        if (temp_hit.t < hit.t)
                        {
                            hit = temp_hit;
                            hit.primitive_id = i;
                            any_hit = true;
                        }
                    }
                }
                return any_hit;
            }
            
            // --------------------------------------------------------------------
            // BVH Node (simple binary tree) and ray traversal
            // --------------------------------------------------------------------
            template <class T>
            struct BVHNode
            {
                BoundingBox3<T> bbox;
                size_t left_child = 0;
                size_t right_child = 0;
                size_t first_prim = 0;
                size_t prim_count = 0;
                bool is_leaf() const { return prim_count > 0; }
            };
            
            template <class T>
            inline bool intersect_ray_bvh(const Ray<T>& ray,
                                          const std::vector<BVHNode<T>>& nodes,
                                          const std::vector<Triangle<T>>& triangles,
                                          HitInfo<T>& hit,
                                          T t_min = 0.001, T t_max = std::numeric_limits<T>::max())
            {
                if (nodes.empty()) return false;
                bool any_hit = false;
                hit.t = t_max;
                std::vector<size_t> stack;
                stack.push_back(0);
                while (!stack.empty())
                {
                    size_t idx = stack.back();
                    stack.pop_back();
                    const BVHNode<T>& node = nodes[idx];
                    T t1, t2;
                    if (!intersect_ray_aabb(ray, node.bbox, t1, t2, t_min, hit.t)) continue;
                    if (node.is_leaf())
                    {
                        for (size_t i = 0; i < node.prim_count; ++i)
                        {
                            size_t prim_id = node.first_prim + i;
                            HitInfo<T> temp_hit;
                            if (intersect_ray_triangle(ray, triangles[prim_id], temp_hit, t_min, hit.t))
                            {
                                if (temp_hit.t < hit.t)
                                {
                                    hit = temp_hit;
                                    hit.primitive_id = prim_id;
                                    any_hit = true;
                                }
                            }
                        }
                    }
                    else
                    {
                        stack.push_back(node.left_child);
                        stack.push_back(node.right_child);
                    }
                }
                return any_hit;
            }
            
            // --------------------------------------------------------------------
            // Closest point on line segment
            // --------------------------------------------------------------------
            template <class T>
            inline Vector3<T> closest_point_on_segment(const Vector3<T>& p, const Vector3<T>& a, const Vector3<T>& b)
            {
                Vector3<T> ab = b - a;
                T t = (p - a).dot(ab) / ab.dot(ab);
                t = std::clamp(t, T(0), T(1));
                return a + ab * t;
            }
            
            // Distance from point to line segment
            template <class T>
            inline T distance_point_segment(const Vector3<T>& p, const Vector3<T>& a, const Vector3<T>& b)
            {
                Vector3<T> closest = closest_point_on_segment(p, a, b);
                return (p - closest).length();
            }
            
            // --------------------------------------------------------------------
            // Closest point on triangle
            // --------------------------------------------------------------------
            template <class T>
            inline Vector3<T> closest_point_on_triangle(const Vector3<T>& p, const Triangle<T>& tri)
            {
                Vector3<T> ab = tri.v1 - tri.v0;
                Vector3<T> ac = tri.v2 - tri.v0;
                Vector3<T> ap = p - tri.v0;
                T d1 = ab.dot(ap);
                T d2 = ac.dot(ap);
                if (d1 <= 0 && d2 <= 0) return tri.v0;
                
                Vector3<T> bp = p - tri.v1;
                T d3 = ab.dot(bp);
                T d4 = ac.dot(bp);
                if (d3 >= 0 && d4 <= d3) return tri.v1;
                
                T vc = d1*d4 - d3*d2;
                if (vc <= 0 && d1 >= 0 && d3 <= 0)
                {
                    T v = d1 / (d1 - d3);
                    return tri.v0 + ab * v;
                }
                
                Vector3<T> cp = p - tri.v2;
                T d5 = ab.dot(cp);
                T d6 = ac.dot(cp);
                if (d6 >= 0 && d5 <= d6) return tri.v2;
                
                T vb = d5*d2 - d1*d6;
                if (vb <= 0 && d2 >= 0 && d6 <= 0)
                {
                    T w = d2 / (d2 - d6);
                    return tri.v0 + ac * w;
                }
                
                T va = d3*d6 - d5*d4;
                if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
                {
                    T w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                    return tri.v1 + (tri.v2 - tri.v1) * w;
                }
                
                T denom = 1.0 / (va + vb + vc);
                T v = vb * denom;
                T w = vc * denom;
                return tri.v0 + ab * v + ac * w;
            }
            
            // Distance from point to triangle
            template <class T>
            inline T distance_point_triangle(const Vector3<T>& p, const Triangle<T>& tri)
            {
                Vector3<T> closest = closest_point_on_triangle(p, tri);
                return (p - closest).length();
            }
            
            // --------------------------------------------------------------------
            // Frustum culling
            // --------------------------------------------------------------------
            template <class T>
            struct Frustum
            {
                Plane<T> planes[6]; // left, right, bottom, top, near, far
            };
            
            template <class T>
            inline bool frustum_contains_sphere(const Frustum<T>& frustum, const Sphere<T>& sphere)
            {
                for (int i = 0; i < 6; ++i)
                {
                    if (frustum.planes[i].signed_distance(sphere.center) < -sphere.radius)
                        return false;
                }
                return true;
            }
            
            template <class T>
            inline bool frustum_contains_aabb(const Frustum<T>& frustum, const BoundingBox3<T>& box)
            {
                Vector3<T> center = box.center();
                Vector3<T> ext = box.extents();
                for (int i = 0; i < 6; ++i)
                {
                    const Plane<T>& p = frustum.planes[i];
                    T r = ext.x * std::abs(p.normal.x) +
                          ext.y * std::abs(p.normal.y) +
                          ext.z * std::abs(p.normal.z);
                    if (p.signed_distance(center) < -r)
                        return false;
                }
                return true;
            }
            
            // --------------------------------------------------------------------
            // Intersection of two moving spheres (continuous collision)
            // --------------------------------------------------------------------
            template <class T>
            inline bool intersect_moving_spheres(const Sphere<T>& s0, const Vector3<T>& v0,
                                                 const Sphere<T>& s1, const Vector3<T>& v1,
                                                 T& t, T t_max = 1.0)
            {
                Vector3<T> dv = v1 - v0;
                Vector3<T> dc = s1.center - s0.center;
                T r = s0.radius + s1.radius;
                T a = dv.dot(dv);
                if (a < 1e-8) return (dc.length_sq() <= r*r);
                T b = 2.0 * dc.dot(dv);
                T c = dc.dot(dc) - r*r;
                T disc = b*b - 4.0*a*c;
                if (disc < 0) return false;
                T sqrt_disc = std::sqrt(disc);
                T t1 = (-b - sqrt_disc) / (2.0*a);
                T t2 = (-b + sqrt_disc) / (2.0*a);
                if (t1 > t2) std::swap(t1, t2);
                if (t1 < 0) { t1 = t2; if (t1 < 0) return false; }
                if (t1 > t_max) return false;
                t = t1;
                return true;
            }
            
            // --------------------------------------------------------------------
            // Utility: create orthonormal basis from a single vector
            // --------------------------------------------------------------------
            template <class T>
            inline void orthonormal_basis(const Vector3<T>& n, Vector3<T>& t, Vector3<T>& b)
            {
                if (std::abs(n.x) > std::abs(n.y))
                    t = Vector3<T>{n.z, 0, -n.x}.normalized();
                else
                    t = Vector3<T>{0, -n.z, n.y}.normalized();
                b = n.cross(t);
            }
            
        } // namespace intersection
        
        // Bring intersection namespace into xt
        using intersection::Vector2;
        using intersection::Vector3;
        using intersection::Ray;
        using intersection::BoundingBox3;
        using intersection::Triangle;
        using intersection::Sphere;
        using intersection::Plane;
        using intersection::HitInfo;
        using intersection::intersect_ray_sphere;
        using intersection::intersect_ray_plane;
        using intersection::intersect_ray_triangle;
        using intersection::intersect_ray_aabb;
        using intersection::intersect_sphere_sphere;
        using intersection::intersect_sphere_aabb;
        using intersection::intersect_aabb_aabb;
        using intersection::intersect_triangle_aabb;
        using intersection::point_in_triangle_2d;
        using intersection::point_in_polygon_2d;
        using intersection::intersect_segment_segment_2d;
        using intersection::intersect_ray_mesh;
        using intersection::intersect_ray_spheres;
        using intersection::intersect_ray_bvh;
        using intersection::closest_point_on_segment;
        using intersection::distance_point_segment;
        using intersection::closest_point_on_triangle;
        using intersection::distance_point_triangle;
        using intersection::frustum_contains_sphere;
        using intersection::frustum_contains_aabb;
        using intersection::intersect_moving_spheres;
        using intersection::orthonormal_basis;
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XINTERSECTION_HPP

// math/xintersection.hpp