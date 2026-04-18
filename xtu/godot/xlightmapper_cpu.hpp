// include/xtu/godot/xlightmapper_cpu.hpp
// xtensor-unified - CPU Lightmapper for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XLIGHTMAPPER_CPU_HPP
#define XTU_GODOT_XLIGHTMAPPER_CPU_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xrenderingserver.hpp"
#include "xtu/godot/xlightmapper.hpp"
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/graphics/xtransform.hpp"
#include "xtu/graphics/xintersection.hpp"
#include "xtu/parallel/xparallel.hpp"

#ifdef XTU_USE_EMBREE
#include <embree4/rtcore.h>
#endif

XTU_NAMESPACE_BEGIN
namespace godot {
namespace lightmapper {

// #############################################################################
// Forward declarations
// #############################################################################
class LightmapperCPU;
class RaytracerBVH;
class PathTracer;
class IrradianceCache;

// #############################################################################
// Sampling pattern types
// #############################################################################
enum class SamplingPattern : uint8_t {
    PATTERN_RANDOM = 0,
    PATTERN_STRATIFIED = 1,
    PATTERN_SOBOL = 2,
    PATTERN_HALTON = 3
};

// #############################################################################
// Ray tracing backend
// #############################################################################
enum class RaytracerBackend : uint8_t {
    BACKEND_INTERNAL = 0,
    BACKEND_EMBREE = 1
};

// #############################################################################
// IrradianceCache - Caches indirect lighting samples
// #############################################################################
class IrradianceCache {
private:
    struct CacheEntry {
        vec3f position;
        vec3f normal;
        vec3f irradiance;
        float radius;
        int samples = 0;
    };

    std::vector<CacheEntry> m_entries;
    float m_spacing = 1.0f;
    float m_max_distance = 10.0f;
    mutable std::mutex m_mutex;

public:
    void set_spacing(float spacing) { m_spacing = spacing; }
    void set_max_distance(float distance) { m_max_distance = distance; }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_entries.clear();
    }

    bool lookup(const vec3f& position, const vec3f& normal, vec3f& out_irradiance) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        vec3f weighted_sum(0);
        float total_weight = 0.0f;

        for (const auto& entry : m_entries) {
            float dist = (entry.position - position).length();
            if (dist > entry.radius) continue;

            float normal_weight = std::max(0.0f, dot(entry.normal, normal));
            if (normal_weight < 0.1f) continue;

            float weight = (1.0f - dist / entry.radius) * normal_weight;
            weighted_sum += entry.irradiance * weight;
            total_weight += weight;
        }

        if (total_weight > 0.0f) {
            out_irradiance = weighted_sum / total_weight;
            return true;
        }
        return false;
    }

    void insert(const vec3f& position, const vec3f& normal, const vec3f& irradiance, float radius) {
        std::lock_guard<std::mutex> lock(m_mutex);
        CacheEntry entry;
        entry.position = position;
        entry.normal = normal;
        entry.irradiance = irradiance;
        entry.radius = radius;
        m_entries.push_back(entry);
    }
};

// #############################################################################
// RaytracerBVH - Internal BVH ray tracer
// #############################################################################
class RaytracerBVH {
private:
    struct Triangle {
        vec3f v0, v1, v2;
        vec3f normal;
        int material_id = 0;
    };

    struct BVHNode {
        aabb bounds;
        int left = -1;
        int right = -1;
        int first_tri = 0;
        int tri_count = 0;
    };

    std::vector<Triangle> m_triangles;
    std::vector<BVHNode> m_nodes;
    int m_root_idx = -1;

public:
    void build(const std::vector<vec3f>& vertices, const std::vector<int>& indices,
               const std::vector<mat4f>& transforms) {
        m_triangles.clear();
        m_nodes.clear();

        // Collect all triangles
        for (size_t i = 0; i < indices.size(); i += 3) {
            Triangle tri;
            tri.v0 = vertices[indices[i]];
            tri.v1 = vertices[indices[i + 1]];
            tri.v2 = vertices[indices[i + 2]];
            tri.normal = normalize(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
            m_triangles.push_back(tri);
        }

        // Build BVH
        if (!m_triangles.empty()) {
            m_root_idx = build_node(0, static_cast<int>(m_triangles.size()));
        }
    }

    bool intersect(const ray& r, float& out_t, vec3f& out_normal, int& out_tri_idx) const {
        if (m_root_idx < 0) return false;
        return intersect_node(m_root_idx, r, out_t, out_normal, out_tri_idx);
    }

    bool occluded(const ray& r, float max_dist = std::numeric_limits<float>::max()) const {
        if (m_root_idx < 0) return false;
        return occluded_node(m_root_idx, r, max_dist);
    }

private:
    int build_node(int start, int end) {
        int idx = static_cast<int>(m_nodes.size());
        m_nodes.emplace_back();

        BVHNode& node = m_nodes.back();
        node.first_tri = start;
        node.tri_count = end - start;

        // Compute bounds
        for (int i = start; i < end; ++i) {
            node.bounds.extend(m_triangles[i].v0);
            node.bounds.extend(m_triangles[i].v1);
            node.bounds.extend(m_triangles[i].v2);
        }

        if (node.tri_count <= 4) {
            return idx;
        }

        // Find split axis
        vec3f extents = node.bounds.max - node.bounds.min;
        int axis = 0;
        if (extents.y() > extents.x()) axis = 1;
        if (extents.z() > extents[axis]) axis = 2;

        // Sort triangles by centroid along axis
        float split = (node.bounds.min[axis] + node.bounds.max[axis]) * 0.5f;
        int mid = start;
        for (int i = start; i < end; ++i) {
            vec3f centroid = (m_triangles[i].v0 + m_triangles[i].v1 + m_triangles[i].v2) / 3.0f;
            if (centroid[axis] < split) {
                std::swap(m_triangles[i], m_triangles[mid]);
                ++mid;
            }
        }

        if (mid == start || mid == end) {
            mid = (start + end) / 2;
        }

        node.left = build_node(start, mid);
        node.right = build_node(mid, end);
        return idx;
    }

    bool intersect_node(int node_idx, const ray& r, float& out_t, vec3f& out_normal, int& out_tri_idx) const {
        const BVHNode& node = m_nodes[node_idx];
        if (!node.bounds.intersect(r)) return false;

        if (node.left < 0) {
            // Leaf node
            bool hit = false;
            float closest = out_t;
            for (int i = node.first_tri; i < node.first_tri + node.tri_count; ++i) {
                float t;
                vec3f normal;
                if (intersect_triangle(r, m_triangles[i], t, normal)) {
                    if (t < closest) {
                        closest = t;
                        out_normal = normal;
                        out_tri_idx = i;
                        hit = true;
                    }
                }
            }
            if (hit) out_t = closest;
            return hit;
        }

        bool hit_left = intersect_node(node.left, r, out_t, out_normal, out_tri_idx);
        bool hit_right = intersect_node(node.right, r, out_t, out_normal, out_tri_idx);
        return hit_left || hit_right;
    }

    bool occluded_node(int node_idx, const ray& r, float max_dist) const {
        const BVHNode& node = m_nodes[node_idx];
        float t;
        if (!node.bounds.intersect(r, t, t)) return false;
        if (t > max_dist) return false;

        if (node.left < 0) {
            for (int i = node.first_tri; i < node.first_tri + node.tri_count; ++i) {
                float tri_t;
                vec3f normal;
                if (intersect_triangle(r, m_triangles[i], tri_t, normal) && tri_t < max_dist) {
                    return true;
                }
            }
            return false;
        }

        return occluded_node(node.left, r, max_dist) || occluded_node(node.right, r, max_dist);
    }

    bool intersect_triangle(const ray& r, const Triangle& tri, float& t, vec3f& normal) const {
        vec3f e1 = tri.v1 - tri.v0;
        vec3f e2 = tri.v2 - tri.v0;
        vec3f h = cross(r.direction, e2);
        float a = dot(e1, h);

        if (std::abs(a) < 1e-6f) return false;

        float f = 1.0f / a;
        vec3f s = r.origin - tri.v0;
        float u = f * dot(s, h);

        if (u < 0.0f || u > 1.0f) return false;

        vec3f q = cross(s, e1);
        float v = f * dot(r.direction, q);

        if (v < 0.0f || u + v > 1.0f) return false;

        t = f * dot(e2, q);
        if (t < r.t_min || t > r.t_max) return false;

        normal = tri.normal;
        return true;
    }
};

// #############################################################################
// PathTracer - Monte Carlo path tracing integrator
// #############################################################################
class PathTracer {
private:
    RaytracerBVH m_bvh;
    IrradianceCache m_irradiance_cache;
    std::mt19937 m_rng;
    int m_max_bounces = 4;
    int m_samples_per_pixel = 64;
    SamplingPattern m_pattern = SamplingPattern::PATTERN_SOBOL;
    bool m_use_irradiance_cache = true;
    float m_indirect_clamp = 10.0f;

public:
    PathTracer() : m_rng(std::random_device{}()) {}

    void set_max_bounces(int bounces) { m_max_bounces = bounces; }
    void set_samples_per_pixel(int spp) { m_samples_per_pixel = spp; }
    void set_sampling_pattern(SamplingPattern pattern) { m_pattern = pattern; }
    void set_use_irradiance_cache(bool use) { m_use_irradiance_cache = use; }

    void build_scene(const std::vector<vec3f>& vertices, const std::vector<int>& indices,
                     const std::vector<mat4f>& transforms) {
        m_bvh.build(vertices, indices, transforms);
        m_irradiance_cache.clear();
    }

    vec3f trace_pixel(const vec3f& position, const vec3f& normal, const vec3f& albedo,
                      const std::vector<LightmapBaker::LightData>& lights) {
        vec3f accum(0);

        for (int s = 0; s < m_samples_per_pixel; ++s) {
            vec3f sample_dir = sample_hemisphere(normal, s);
            ray r(position + normal * 0.01f, sample_dir);

            vec3f radiance = trace_path(r, albedo, lights, 0);
            accum += radiance;
        }

        return accum / static_cast<float>(m_samples_per_pixel);
    }

private:
    vec3f sample_hemisphere(const vec3f& normal, int sample_idx) {
        float u1 = halton(sample_idx, 2);
        float u2 = halton(sample_idx, 3);

        float phi = 2.0f * M_PI * u1;
        float cos_theta = std::sqrt(1.0f - u2);
        float sin_theta = std::sqrt(u2);

        vec3f tangent, bitangent;
        create_basis(normal, tangent, bitangent);

        return tangent * (std::cos(phi) * sin_theta) +
               bitangent * (std::sin(phi) * sin_theta) +
               normal * cos_theta;
    }

    float halton(int index, int base) {
        float result = 0.0f;
        float f = 1.0f;
        int i = index;
        while (i > 0) {
            f /= base;
            result += f * (i % base);
            i /= base;
        }
        return result;
    }

    void create_basis(const vec3f& n, vec3f& t, vec3f& b) {
        if (std::abs(n.x()) > 0.9f) {
            t = vec3f(0, 1, 0);
        } else {
            t = vec3f(1, 0, 0);
        }
        t = normalize(t - n * dot(n, t));
        b = cross(n, t);
    }

    vec3f trace_path(const ray& r, const vec3f& albedo, const std::vector<LightmapBaker::LightData>& lights, int depth) {
        if (depth >= m_max_bounces) return vec3f(0);

        float t;
        vec3f normal;
        int tri_idx;
        if (!m_bvh.intersect(r, t, normal, tri_idx)) {
            // Hit sky - sample lights
            vec3f sky(0);
            for (const auto& light : lights) {
                sky += light.color * light.energy;
            }
            return sky * 0.1f;
        }

        vec3f hit_pos = r.at(t);
        vec3f direct = sample_direct_lighting(hit_pos, normal, albedo, lights);

        if (depth == 0 && m_use_irradiance_cache) {
            vec3f cached;
            if (m_irradiance_cache.lookup(hit_pos, normal, cached)) {
                return direct + cached;
            }
        }

        // Sample indirect bounce
        vec3f indirect(0);
        if (depth < m_max_bounces - 1) {
            vec3f bounce_dir = sample_hemisphere(normal, depth * m_samples_per_pixel);
            ray bounce_ray(hit_pos + normal * 0.01f, bounce_dir);
            float pdf = dot(normal, bounce_dir) / M_PI;
            indirect = trace_path(bounce_ray, albedo, lights, depth + 1) * albedo / M_PI;
            indirect = vec3f(std::min(indirect.x(), m_indirect_clamp),
                             std::min(indirect.y(), m_indirect_clamp),
                             std::min(indirect.z(), m_indirect_clamp));
        }

        if (depth == 0 && m_use_irradiance_cache) {
            m_irradiance_cache.insert(hit_pos, normal, indirect, 2.0f);
        }

        return direct + indirect;
    }

    vec3f sample_direct_lighting(const vec3f& pos, const vec3f& normal, const vec3f& albedo,
                                  const std::vector<LightmapBaker::LightData>& lights) {
        vec3f direct(0);

        for (const auto& light : lights) {
            vec3f light_dir;
            float dist;
            float attenuation = 1.0f;

            if (light.type == 0) { // Directional
                light_dir = -normalize(light.direction);
                dist = std::numeric_limits<float>::max();
            } else { // Point
                light_dir = light.position - pos;
                dist = light_dir.length();
                light_dir /= dist;
                attenuation = 1.0f / (1.0f + 0.1f * dist + 0.01f * dist * dist);
                if (dist > light.range) continue;
            }

            float ndl = std::max(0.0f, dot(normal, light_dir));
            if (ndl <= 0.0f) continue;

            // Shadow ray
            ray shadow_ray(pos + normal * 0.01f, light_dir);
            if (!m_bvh.occluded(shadow_ray, dist)) {
                direct += light.color * light.energy * albedo * ndl * attenuation;
            }
        }

        return direct;
    }
};

// #############################################################################
// LightmapperCPU - Main CPU lightmapper
// #############################################################################
class LightmapperCPU : public LightmapBaker {
    XTU_GODOT_REGISTER_CLASS(LightmapperCPU, LightmapBaker)

private:
    RaytracerBackend m_backend = RaytracerBackend::BACKEND_INTERNAL;
    PathTracer m_path_tracer;
    int m_thread_count = 0;
    bool m_use_adaptive_sampling = true;
    float m_adaptive_threshold = 0.01f;
    int m_min_samples = 16;
    int m_max_samples = 256;

#ifdef XTU_USE_EMBREE
    RTCDevice m_embree_device = nullptr;
    RTCScene m_embree_scene = nullptr;
#endif

public:
    static StringName get_class_static() { return StringName("LightmapperCPU"); }

    LightmapperCPU() {
        m_thread_count = std::thread::hardware_concurrency();
    }

    ~LightmapperCPU() {
#ifdef XTU_USE_EMBREE
        if (m_embree_scene) rtcReleaseScene(m_embree_scene);
        if (m_embree_device) rtcReleaseDevice(m_embree_device);
#endif
    }

    void set_backend(RaytracerBackend backend) { m_backend = backend; }
    RaytracerBackend get_backend() const { return m_backend; }

    void set_thread_count(int count) { m_thread_count = std::max(1, count); }
    int get_thread_count() const { return m_thread_count; }

    void set_use_adaptive_sampling(bool use) { m_use_adaptive_sampling = use; }
    bool get_use_adaptive_sampling() const { return m_use_adaptive_sampling; }

    void set_max_bounces(int bounces) { m_path_tracer.set_max_bounces(bounces); }
    int get_max_bounces() const { return m_path_tracer.get_max_bounces(); }

    void set_samples_per_pixel(int spp) { m_path_tracer.set_samples_per_pixel(spp); }
    int get_samples_per_pixel() const { return m_path_tracer.get_samples_per_pixel(); }

    void set_use_irradiance_cache(bool use) { m_path_tracer.set_use_irradiance_cache(use); }
    bool get_use_irradiance_cache() const { return m_path_tracer.get_use_irradiance_cache(); }

    void bake_implementation() override {
        if (m_meshes.empty()) return;

        // Build acceleration structure
        if (m_backend == RaytracerBackend::BACKEND_EMBREE) {
            build_embree_scene();
        } else {
            build_internal_bvh();
        }

        size_t total_pixels = 0;
        for (const auto& mesh : m_meshes) {
            vec2i lm_size = compute_lightmap_size(mesh);
            total_pixels += lm_size.x() * lm_size.y();
        }

        std::atomic<size_t> baked_pixels{0};

        parallel::parallel_for(0, m_meshes.size(), [&](size_t mesh_idx) {
            const auto& mesh = m_meshes[mesh_idx];
            vec2i lm_size = compute_lightmap_size(mesh);
            std::vector<vec4f> lightmap(lm_size.x() * lm_size.y());

            parallel::parallel_for(0, lm_size.y(), [&](int y) {
                for (int x = 0; x < lm_size.x(); ++x) {
                    vec2f uv((x + 0.5f) / lm_size.x(), (y + 0.5f) / lm_size.y());
                    vec3f world_pos = get_world_position_from_uv(mesh, uv);
                    vec3f normal = get_world_normal_from_uv(mesh, uv);
                    vec3f albedo = get_albedo_from_uv(mesh, uv);

                    vec3f color = m_path_tracer.trace_pixel(world_pos, normal, albedo, m_lights);
                    lightmap[y * lm_size.x() + x] = vec4f(color.x(), color.y(), color.z(), 1.0f);

                    ++baked_pixels;
                    m_progress = static_cast<float>(baked_pixels) / static_cast<float>(total_pixels);
                }
            });

            // Store lightmap result
            std::lock_guard<std::mutex> lock(m_mutex);
            m_lightmap_results[mesh_idx] = lightmap;
        });

        if (m_settings.denoise) {
            apply_denoiser();
        }
    }

private:
    std::unordered_map<size_t, std::vector<vec4f>> m_lightmap_results;

    void build_internal_bvh() {
        std::vector<vec3f> all_vertices;
        std::vector<int> all_indices;
        std::vector<mat4f> transforms;
        int idx_offset = 0;

        for (const auto& mesh : m_meshes) {
            // Extract mesh data
            transforms.push_back(mesh.transform);
            idx_offset += static_cast<int>(all_vertices.size());
        }

        m_path_tracer.build_scene(all_vertices, all_indices, transforms);
    }

#ifdef XTU_USE_EMBREE
    void build_embree_scene() {
        if (m_embree_device) rtcReleaseDevice(m_embree_device);
        m_embree_device = rtcNewDevice(nullptr);
        m_embree_scene = rtcNewScene(m_embree_device);

        for (const auto& mesh : m_meshes) {
            RTCGeometry geom = rtcNewGeometry(m_embree_device, RTC_GEOMETRY_TYPE_TRIANGLE);
            // Upload mesh data to Embree
            rtcCommitGeometry(geom);
            rtcAttachGeometry(m_embree_scene, geom);
            rtcReleaseGeometry(geom);
        }

        rtcCommitScene(m_embree_scene);
    }
#endif

    vec2i compute_lightmap_size(const MeshInstance& mesh) const {
        float area = 1.0f;
        int size = static_cast<int>(std::sqrt(area) * 16.0f);
        return vec2i(std::clamp(size, 16, 1024), std::clamp(size, 16, 1024));
    }

    vec3f get_world_position_from_uv(const MeshInstance& mesh, const vec2f& uv) const {
        return vec3f(0);
    }

    vec3f get_world_normal_from_uv(const MeshInstance& mesh, const vec2f& uv) const {
        return vec3f(0, 1, 0);
    }

    vec3f get_albedo_from_uv(const MeshInstance& mesh, const vec2f& uv) const {
        return vec3f(0.8f);
    }

    void apply_denoiser() {
        // Apply bilateral filter or OIDN
    }
};

} // namespace lightmapper

// Bring into main namespace
using lightmapper::LightmapperCPU;
using lightmapper::RaytracerBackend;
using lightmapper::SamplingPattern;
using lightmapper::IrradianceCache;
using lightmapper::PathTracer;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XLIGHTMAPPER_CPU_HPP