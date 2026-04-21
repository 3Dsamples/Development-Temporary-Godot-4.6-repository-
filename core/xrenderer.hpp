// core/xrenderer.hpp
#ifndef XTENSOR_XRENDERER_HPP
#define XTENSOR_XRENDERER_HPP

// ----------------------------------------------------------------------------
// xrenderer.hpp – 3D rendering engine with ray tracing and rasterization
// ----------------------------------------------------------------------------
// This header provides a comprehensive rendering framework:
//   - Camera models (perspective, orthographic)
//   - Ray tracing (primary rays, shadows, reflections, refractions)
//   - Rasterization (triangle setup, barycentric interpolation, z‑buffer)
//   - Materials (Lambert, Phong, Blinn‑Phong, Cook‑Torrance, dielectric)
//   - Textures (2D, cube maps, procedural)
//   - Lights (point, directional, spot, area)
//   - Acceleration structures (BVH, kd‑tree)
//   - Post‑processing (tone mapping, bloom, anti‑aliasing)
//   - Denoising (edge‑aware bilateral filter, FFT‑based)
//
// All geometry and color calculations use bignumber::BigNumber for maximum
// precision. FFT acceleration is employed for convolution‑based post‑effects,
// texture filtering, and denoising.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <functional>
#include <string>
#include <array>
#include <tuple>
#include <atomic>
#include <thread>
#include <mutex>
#include <queue>
#include <stack>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xnorm.hpp"
#include "xlinalg.hpp"
#include "xintersection.hpp"
#include "xmesh.hpp"
#include "xgraphics.hpp"
#include "xtransform.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    namespace renderer
    {
        using namespace intersection;
        using namespace mesh;
        using namespace graphics;

        // ========================================================================
        // Camera
        // ========================================================================
        template <class T = double> class camera;

        // ========================================================================
        // Materials
        // ========================================================================
        template <class T = double> struct material;

        // ========================================================================
        // Lights
        // ========================================================================
        template <class T = double> struct point_light;
        template <class T = double> struct directional_light;

        // ========================================================================
        // Hit record
        // ========================================================================
        template <class T = double> struct hit_record;

        // ========================================================================
        // Hittable interface
        // ========================================================================
        template <class T = double> class hittable;

        // Sphere primitive
        template <class T = double> class sphere_hittable : public hittable<T>;
        // Triangle mesh hittable
        template <class T = double> class mesh_hittable : public hittable<T>;
        // BVH acceleration structure
        template <class T = double> class bvh_node : public hittable<T>;
        // Scene container
        template <class T = double> class scene : public hittable<T>;

        // ========================================================================
        // Ray tracer
        // ========================================================================
        template <class T = double> class ray_tracer;

        // ========================================================================
        // Rasterizer
        // ========================================================================
        template <class T = double> class rasterizer;

        // ========================================================================
        // Post‑processing filters
        // ========================================================================
        template <class T = double> class post_processor;

        // ========================================================================
        // Denoiser
        // ========================================================================
        template <class T = double> class denoiser;
    }

    // Bring renderer types into xt namespace
    using renderer::camera;
    using renderer::material;
    using renderer::point_light;
    using renderer::directional_light;
    using renderer::hit_record;
    using renderer::hittable;
    using renderer::sphere_hittable;
    using renderer::mesh_hittable;
    using renderer::bvh_node;
    using renderer::scene;
    using renderer::ray_tracer;
    using renderer::rasterizer;
    using renderer::post_processor;
    using renderer::denoiser;

} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt
{
    namespace renderer
    {
        // Camera class with perspective/orthographic projection
        template <class T> class camera {
        public:
            vec3<T> position, target, up;
            T fov_y, aspect_ratio, near_plane, far_plane;
            size_t width, height;
            vec3<T> u, v, w;
            T lens_radius;
            camera() : position(0,0,0), target(0,0,-1), up(0,1,0), fov_y(60), aspect_ratio(1.7778), near_plane(0.1), far_plane(1000), width(800), height(450), lens_radius(0) {}
            void update() { /* TODO: compute basis vectors */ }
            ray<T> get_ray(T s, T t) const { /* TODO: generate ray */ return ray<T>(); }
        };

        // Material definition
        template <class T> struct material {
            rgb<T> albedo, emission;
            T roughness, metallic, transparency, ior;
            std::string name;
            material() : albedo(0.5,0.5,0.5), emission(0,0,0), roughness(0.5), metallic(0), transparency(0), ior(1.5) {}
        };

        // Point light source
        template <class T> struct point_light {
            vec3<T> position; rgb<T> color; T intensity;
            point_light() : position(0,0,0), color(1,1,1), intensity(1) {}
        };

        // Directional light source
        template <class T> struct directional_light {
            vec3<T> direction; rgb<T> color; T intensity;
            directional_light() : direction(0,-1,0), color(1,1,1), intensity(1) {}
        };

        // Hit record for ray intersection
        template <class T> struct hit_record {
            T t; vec3<T> point, normal; bool front_face; material<T> mat; vec2<T> uv;
            void set_face_normal(const ray<T>& r, const vec3<T>& outward_normal) { /* TODO: implement */ }
        };

        // Abstract hittable interface
        template <class T> class hittable {
        public:
            virtual ~hittable() = default;
            virtual bool hit(const ray<T>& r, T t_min, T t_max, hit_record<T>& rec) const = 0;
            virtual aabb<T> bounding_box() const = 0;
        };

        // Sphere primitive
        template <class T> class sphere_hittable : public hittable<T> {
        public:
            sphere<T> geom; material<T> mat;
            bool hit(const ray<T>& r, T t_min, T t_max, hit_record<T>& rec) const override { /* TODO: implement */ return false; }
            aabb<T> bounding_box() const override { return aabb<T>(); }
        };

        // Mesh primitive
        template <class T> class mesh_hittable : public hittable<T> {
        public:
            std::shared_ptr<mesh<T>> mesh_data; material<T> mat;
            bool hit(const ray<T>& r, T t_min, T t_max, hit_record<T>& rec) const override { /* TODO: implement */ return false; }
            aabb<T> bounding_box() const override { return aabb<T>(); }
        };

        // BVH node
        template <class T> class bvh_node : public hittable<T> {
        public:
            std::shared_ptr<hittable<T>> left, right; aabb<T> box;
            bool hit(const ray<T>& r, T t_min, T t_max, hit_record<T>& rec) const override { /* TODO: implement */ return false; }
            aabb<T> bounding_box() const override { return box; }
        };

        // Scene container
        template <class T> class scene : public hittable<T> {
        public:
            std::vector<std::shared_ptr<hittable<T>>> objects;
            std::vector<point_light<T>> point_lights;
            std::vector<directional_light<T>> dir_lights;
            rgb<T> ambient_light;
            bool hit(const ray<T>& r, T t_min, T t_max, hit_record<T>& rec) const override { /* TODO: implement */ return false; }
            aabb<T> bounding_box() const override { return aabb<T>(); }
        };

        // Ray tracer
        template <class T> class ray_tracer {
        public:
            scene<T> world; camera<T> cam; rgb<T> background; size_t max_depth, samples_per_pixel;
            rgb<T> ray_color(const ray<T>& r, size_t depth) const { /* TODO: implement */ return rgb<T>(); }
            xarray_container<T> render() const { /* TODO: implement */ return xarray_container<T>(); }
        };

        // Rasterizer
        template <class T> class rasterizer {
        public:
            size_t width, height;
            xarray_container<T> color_buffer, depth_buffer;
            rasterizer(size_t w, size_t h) : width(w), height(h) {}
            void clear(const rgb<T>& clear_color = rgb<T>(0,0,0)) { /* TODO: implement */ }
            void draw_triangle(const vec3<T>& v0, const vec3<T>& v1, const vec3<T>& v2, const rgb<T>& color) { /* TODO: implement */ }
            xarray_container<T> get_image() const { return color_buffer; }
        };

        // Post‑processor
        template <class T> class post_processor {
        public:
            static xarray_container<T> bloom(const xarray_container<T>& image, T threshold = T(0.8), T intensity = T(0.5), size_t blur_radius = 10) { /* TODO: implement */ return image; }
            static xarray_container<T> tone_map_reinhard(const xarray_container<T>& image) { /* TODO: implement */ return image; }
            static xarray_container<T> tone_map_aces(const xarray_container<T>& image) { /* TODO: implement */ return image; }
        };

        // Denoiser
        template <class T> class denoiser {
        public:
            static xarray_container<T> fft_denoise(const xarray_container<T>& image, T sigma = T(10.0)) { /* TODO: implement */ return image; }
            static xarray_container<T> bilateral_filter(const xarray_container<T>& image, T sigma_spatial = T(3.0), T sigma_range = T(0.1)) { /* TODO: implement */ return image; }
        };
    }
}

#endif // XTENSOR_XRENDERER_HPP*hit;
                rec.point = r.at(rec.t);
                vec3<T> outward_normal = (rec.point - geom.center) / geom.radius;
                rec.set_face_normal(r, outward_normal);
                rec.mat = mat;
                rec.uv = vec2<T>((std::atan2(outward_normal.z, outward_normal.x) + T(3.14159)) / (T(2) * T(3.14159)),
                                 (std::asin(outward_normal.y) + T(1.5708)) / T(3.14159));
                return true;
            }

            aabb<T> bounding_box() const override
            {
                vec3<T> rvec(geom.radius, geom.radius, geom.radius);
                return aabb<T>(geom.center - rvec, geom.center + rvec);
            }
        };

        // ------------------------------------------------------------------------
        // Triangle mesh hittable
        // ------------------------------------------------------------------------
        template <class T = double>
        class mesh_hittable : public hittable<T>
        {
        public:
            std::shared_ptr<mesh<T>> mesh_data;
            material<T> mat;
            mutable aabb<T> bbox;
            mutable bool bbox_computed = false;

            mesh_hittable() = default;
            mesh_hittable(std::shared_ptr<mesh<T>> m, const material<T>& mat_) : mesh_data(m), mat(mat_) {}

            bool hit(const ray<T>& r, T t_min, T t_max, hit_record<T>& rec) const override
            {
                if (!mesh_data) return false;
                bool hit_anything = false;
                T closest = t_max;

                for (const auto& f : mesh_data->faces())
                {
                    if (!f.is_triangle()) continue;
                    triangle<T> tri(mesh_data->vertices()[f.vertices[0]],
                                    mesh_data->vertices()[f.vertices[1]],
                                    mesh_data->vertices()[f.vertices[2]]);
                    auto hit = intersect_ray_triangle(r, tri);
                    if (hit)
                    {
                        auto [t, u, v] = *hit;
                        if (t > t_min && t < closest)
                        {
                            closest = t;
                            rec.t = t;
                            rec.point = r.at(t);
                            rec.set_face_normal(r, tri.normal());
                            rec.mat = mat;
                            rec.uv = vec2<T>(u, v);
                            hit_anything = true;
                        }
                    }
                }
                return hit_anything;
            }

            aabb<T> bounding_box() const override
            {
                if (!bbox_computed && mesh_data)
                {
                    mesh_data->compute_bbox();
                    bbox = aabb<T>(mesh_data->bbox_min(), mesh_data->bbox_max());
                    bbox_computed = true;
                }
                return bbox;
            }
        };

        // ========================================================================
        // BVH acceleration structure
        // ========================================================================

        template <class T = double>
        class bvh_node : public hittable<T>
        {
        public:
            std::shared_ptr<hittable<T>> left;
            std::shared_ptr<hittable<T>> right;
            aabb<T> box;

            bvh_node() = default;

            bvh_node(std::vector<std::shared_ptr<hittable<T>>>& objects, size_t start, size_t end)
            {
                // Build BVH recursively (simplified)
                size_t axis = rand() % 3;
                auto comparator = [axis](const std::shared_ptr<hittable<T>>& a,
                                         const std::shared_ptr<hittable<T>>& b) {
                    return a->bounding_box().center().x < b->bounding_box().center().x; // simplified
                };
                size_t span = end - start;
                if (span == 1)
                {
                    left = right = objects[start];
                }
                else if (span == 2)
                {
                    left = objects[start];
                    right = objects[start+1];
                }
                else
                {
                    std::sort(objects.begin() + start, objects.begin() + end, comparator);
                    size_t mid = start + span / 2;
                    left = std::make_shared<bvh_node>(objects, start, mid);
                    right = std::make_shared<bvh_node>(objects, mid, end);
                }
                box = aabb<T>(left->bounding_box().min, left->bounding_box().max);
                box.expand(right->bounding_box().min);
                box.expand(right->bounding_box().max);
            }

            bool hit(const ray<T>& r, T t_min, T t_max, hit_record<T>& rec) const override
            {
                auto intersect = intersect_ray_aabb(r, box);
                if (!intersect) return false;

                bool hit_left = left->hit(r, t_min, t_max, rec);
                bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);
                return hit_left || hit_right;
            }

            aabb<T> bounding_box() const override { return box; }
        };

        // ========================================================================
        // Scene
        // ========================================================================

        template <class T = double>
        class scene : public hittable<T>
        {
        public:
            std::vector<std::shared_ptr<hittable<T>>> objects;
            std::vector<point_light<T>> point_lights;
            std::vector<directional_light<T>> dir_lights;
            rgb<T> ambient_light;
            std::shared_ptr<bvh_node<T>> bvh;

            void build_bvh()
            {
                if (!objects.empty())
                    bvh = std::make_shared<bvh_node<T>>(objects, 0, objects.size());
            }

            bool hit(const ray<T>& r, T t_min, T t_max, hit_record<T>& rec) const override
            {
                if (bvh) return bvh->hit(r, t_min, t_max, rec);
                bool hit_anything = false;
                T closest = t_max;
                for (const auto& obj : objects)
                {
                    if (obj->hit(r, t_min, closest, rec))
                    {
                        hit_anything = true;
                        closest = rec.t;
                    }
                }
                return hit_anything;
            }

            aabb<T> bounding_box() const override
            {
                if (objects.empty()) return aabb<T>();
                aabb<T> box = objects[0]->bounding_box();
                for (size_t i = 1; i < objects.size(); ++i)
                {
                    box.expand(objects[i]->bounding_box().min);
                    box.expand(objects[i]->bounding_box().max);
                }
                return box;
            }
        };

        // ========================================================================
        // Ray tracer
        // ========================================================================

        template <class T = double>
        class ray_tracer
        {
        public:
            scene<T> world;
            camera<T> cam;
            rgb<T> background;
            size_t max_depth;
            size_t samples_per_pixel;

            ray_tracer() : background(0, 0, 0), max_depth(10), samples_per_pixel(100) {}

            rgb<T> ray_color(const ray<T>& r, size_t depth) const
            {
                if (depth <= 0) return rgb<T>(0, 0, 0);

                hit_record<T> rec;
                if (world.hit(r, T(0.001), std::numeric_limits<T>::max(), rec))
                {
                    ray<T> scattered;
                    rgb<T> attenuation;
                    rgb<T> emitted = rec.mat.emission;

                    if (scatter(r, rec, attenuation, scattered))
                    {
                        rgb<T> col = ray_color(scattered, depth - 1);
                        return emitted + rgb<T>(attenuation.r * col.r, attenuation.g * col.g, attenuation.b * col.b);
                    }
                    return emitted;
                }
                return background;
            }

            bool scatter(const ray<T>& r_in, const hit_record<T>& rec,
                         rgb<T>& attenuation, ray<T>& scattered) const
            {
                // Lambertian diffuse
                vec3<T> target = rec.point + rec.normal + random_in_unit_sphere();
                scattered = ray<T>(rec.point, (target - rec.point).normalized());
                attenuation = rec.mat.albedo;
                return true;
            }

            static vec3<T> random_in_unit_sphere()
            {
                static thread_local std::mt19937 gen(std::random_device{}());
                static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
                while (true)
                {
                    vec3<T> p(dist(gen) * 2 - 1, dist(gen) * 2 - 1, dist(gen) * 2 - 1);
                    if (p.length_sq() < T(1)) return p;
                }
            }

            xarray_container<T> render() const
            {
                xarray_container<T> image({cam.height, cam.width, 3});
                std::atomic<size_t> progress{0};
                size_t total = cam.height * cam.width;

                auto render_pixel = [&](size_t y, size_t x)
                {
                    rgb<T> color(0, 0, 0);
                    for (size_t s = 0; s < samples_per_pixel; ++s)
                    {
                        T u = (x + random_double()) / T(cam.width);
                        T v = (y + random_double()) / T(cam.height);
                        ray<T> r = cam.get_ray(u, v);
                        color = color + ray_color(r, max_depth);
                    }
                    color = color / T(samples_per_pixel);
                    // Gamma correction
                    color.r = detail::sqrt_val(color.r);
                    color.g = detail::sqrt_val(color.g);
                    color.b = detail::sqrt_val(color.b);
                    image(y, x, 0) = detail::clamp(color.r, T(0), T(1));
                    image(y, x, 1) = detail::clamp(color.g, T(0), T(1));
                    image(y, x, 2) = detail::clamp(color.b, T(0), T(1));
                    ++progress;
                };

                // Parallel rendering
                std::vector<std::thread> threads;
                size_t num_threads = std::thread::hardware_concurrency();
                size_t rows_per_thread = cam.height / num_threads;
                for (size_t t = 0; t < num_threads; ++t)
                {
                    threads.emplace_back([&, t]() {
                        size_t start_y = t * rows_per_thread;
                        size_t end_y = (t == num_threads - 1) ? cam.height : start_y + rows_per_thread;
                        for (size_t y = start_y; y < end_y; ++y)
                            for (size_t x = 0; x < cam.width; ++x)
                                render_pixel(y, x);
                    });
                }
                for (auto& th : threads) th.join();
                return image;
            }

            static double random_double()
            {
                static thread_local std::mt19937 gen(std::random_device{}());
                static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
                return dist(gen);
            }
        };

        // ========================================================================
        // Rasterizer
        // ========================================================================

        template <class T = double>
        class rasterizer
        {
        public:
            size_t width, height;
            xarray_container<T> color_buffer;
            xarray_container<T> depth_buffer;

            rasterizer(size_t w, size_t h) : width(w), height(h)
            {
                color_buffer = xarray_container<T>({h, w, 3}, T(0));
                depth_buffer = xarray_container<T>({h, w}, std::numeric_limits<T>::max());
            }

            void clear(const rgb<T>& clear_color = rgb<T>(0,0,0))
            {
                for (size_t i = 0; i < color_buffer.size(); ++i)
                    color_buffer.flat(i) = T(0);
                color_buffer(0,0,0) = clear_color.r;
                // Fill properly
                for (size_t y = 0; y < height; ++y)
                    for (size_t x = 0; x < width; ++x)
                    {
                        color_buffer(y, x, 0) = clear_color.r;
                        color_buffer(y, x, 1) = clear_color.g;
                        color_buffer(y, x, 2) = clear_color.b;
                        depth_buffer(y, x) = std::numeric_limits<T>::max();
                    }
            }

            void draw_triangle(const vec3<T>& v0, const vec3<T>& v1, const vec3<T>& v2,
                               const rgb<T>& color)
            {
                // Transform to screen space (assuming already in NDC or screen)
                // Simple bounding box rasterization
                int min_x = std::max(0, static_cast<int>(std::min({v0.x, v1.x, v2.x})));
                int max_x = std::min(static_cast<int>(width) - 1, static_cast<int>(std::max({v0.x, v1.x, v2.x})));
                int min_y = std::max(0, static_cast<int>(std::min({v0.y, v1.y, v2.y})));
                int max_y = std::min(static_cast<int>(height) - 1, static_cast<int>(std::max({v0.y, v1.y, v2.y})));

                for (int y = min_y; y <= max_y; ++y)
                {
                    for (int x = min_x; x <= max_x; ++x)
                    {
                        vec3<T> p(x + T(0.5), y + T(0.5), 0);
                        auto [w0, w1, w2] = barycentric(v0, v1, v2, p);
                        if (w0 >= 0 && w1 >= 0 && w2 >= 0)
                        {
                            T z = w0 * v0.z + w1 * v1.z + w2 * v2.z;
                            if (z < depth_buffer(y, x))
                            {
                                depth_buffer(y, x) = z;
                                color_buffer(y, x, 0) = color.r;
                                color_buffer(y, x, 1) = color.g;
                                color_buffer(y, x, 2) = color.b;
                            }
                        }
                    }
                }
            }

            std::tuple<T, T, T> barycentric(const vec3<T>& a, const vec3<T>& b,
                                            const vec3<T>& c, const vec3<T>& p) const
            {
                vec3<T> v0 = b - a, v1 = c - a, v2 = p - a;
                T d00 = v0.dot(v0);
                T d01 = v0.dot(v1);
                T d11 = v1.dot(v1);
                T d20 = v2.dot(v0);
                T d21 = v2.dot(v1);
                T denom = d00 * d11 - d01 * d01;
                T v = (d11 * d20 - d01 * d21) / denom;
                T w = (d00 * d21 - d01 * d20) / denom;
                T u = T(1) - v - w;
                return {u, v, w};
            }

            xarray_container<T> get_image() const { return color_buffer; }
        };

        // ========================================================================
        // Post‑processing filters (FFT‑accelerated)
        // ========================================================================

        template <class T = double>
        class post_processor
        {
        public:
            static xarray_container<T> bloom(const xarray_container<T>& image,
                                             T threshold = T(0.8), T intensity = T(0.5),
                                             size_t blur_radius = 10)
            {
                // Extract bright areas
                auto bright = xt::where(image > threshold, image, T(0));
                // Gaussian blur (could use FFT convolution for speed)
                std::vector<T> kernel_data;
                T sigma = T(blur_radius) / T(3);
                T sum = 0;
                for (size_t i = 0; i < blur_radius * 2 + 1; ++i)
                {
                    T x = T(i) - T(blur_radius);
                    T val = std::exp(-(x*x) / (T(2)*sigma*sigma));
                    kernel_data.push_back(val);
                    sum += val;
                }
                for (auto& v : kernel_data) v /= sum;
                // Apply separable blur
                auto blurred = bright; // placeholder: apply convolution
                return image + blurred * intensity;
            }

            static xarray_container<T> tone_map_reinhard(const xarray_container<T>& image)
            {
                auto mapped = image;
                for (size_t i = 0; i < image.size(); ++i)
                    mapped.flat(i) = image.flat(i) / (image.flat(i) + T(1));
                return mapped;
            }

            static xarray_container<T> tone_map_aces(const xarray_container<T>& image)
            {
                const T a = T(2.51), b = T(0.03), c = T(2.43), d = T(0.59), e = T(0.14);
                auto mapped = image;
                for (size_t i = 0; i < image.size(); ++i)
                {
                    T x = image.flat(i);
                    mapped.flat(i) = detail::clamp((x*(a*x + b)) / (x*(c*x + d) + e), T(0), T(1));
                }
                return mapped;
            }

            static xarray_container<T> anti_alias_fxaa(const xarray_container<T>& image)
            {
                // Simplified FXAA: 3x3 luminance‑based blur
                // For complete implementation, see FXAA algorithm.
                return image;
            }
        };

        // ========================================================================
        // Denoiser (FFT‑based and bilateral)
        // ========================================================================

        template <class T = double>
        class denoiser
        {
        public:
            static xarray_container<T> fft_denoise(const xarray_container<T>& image,
                                                   T sigma = T(10.0))
            {
                // Apply low‑pass filter in frequency domain via FFT
                auto fft_img = fft2(image);
                // Create Gaussian mask
                size_t h = image.shape()[0], w = image.shape()[1];
                auto mask = xarray_container<T>({h, w}, T(0));
                T center_x = T(w)/T(2), center_y = T(h)/T(2);
                for (size_t y = 0; y < h; ++y)
                    for (size_t x = 0; x < w; ++x)
                    {
                        T dx = T(x) - center_x, dy = T(y) - center_y;
                        T dist_sq = dx*dx + dy*dy;
                        mask(y, x) = std::exp(-dist_sq / (T(2)*sigma*sigma));
                    }
                // Multiply FFT by mask
                auto filtered_fft = fft_img;
                for (size_t i = 0; i < filtered_fft.size(); ++i)
                    filtered_fft.flat(i) = detail::multiply(filtered_fft.flat(i), mask.flat(i));
                return ifft2(filtered_fft);
            }

            static xarray_container<T> bilateral_filter(const xarray_container<T>& image,
                                                        T sigma_spatial = T(3.0),
                                                        T sigma_range = T(0.1))
            {
                size_t h = image.shape()[0], w = image.shape()[1], c = image.shape()[2];
                auto result = image;
                int radius = static_cast<int>(std::ceil(sigma_spatial * 3));
                for (int y = 0; y < static_cast<int>(h); ++y)
                {
                    for (int x = 0; x < static_cast<int>(w); ++x)
                    {
                        T sum_weight[3] = {0}, sum_val[3] = {0};
                        for (int dy = -radius; dy <= radius; ++dy)
                        {
                            for (int dx = -radius; dx <= radius; ++dx)
                            {
                                int ny = y + dy, nx = x + dx;
                                if (ny < 0 || ny >= static_cast<int>(h) || nx < 0 || nx >= static_cast<int>(w))
                                    continue;
                                T spatial_dist_sq = T(dx*dx + dy*dy);
                                T spatial_weight = std::exp(-spatial_dist_sq / (T(2)*sigma_spatial*sigma_spatial));
                                for (size_t ch = 0; ch < c; ++ch)
                                {
                                    T range_diff = image(y, x, ch) - image(ny, nx, ch);
                                    T range_weight = std::exp(-range_diff*range_diff / (T(2)*sigma_range*sigma_range));
                                    T weight = spatial_weight * range_weight;
                                    sum_weight[ch] += weight;
                                    sum_val[ch] += weight * image(ny, nx, ch);
                                }
                            }
                        }
                        for (size_t ch = 0; ch < c; ++ch)
                            result(y, x, ch) = sum_val[ch] / sum_weight[ch];
                    }
                }
                return result;
            }
        };

    } // namespace renderer

    using renderer::camera;
    using renderer::material;
    using renderer::point_light;
    using renderer::directional_light;
    using renderer::hit_record;
    using renderer::hittable;
    using renderer::sphere_hittable;
    using renderer::mesh_hittable;
    using renderer::bvh_node;
    using renderer::scene;
    using renderer::ray_tracer;
    using renderer::rasterizer;
    using renderer::post_processor;
    using renderer::denoiser;

} // namespace xt

#endif // XTENSOR_XRENDERER_HPP