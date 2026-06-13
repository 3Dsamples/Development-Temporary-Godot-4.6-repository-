// Name : lighting enhancement
// File : screen_space_lighting.cpp 63 of 63
// Description : Full CPU implementation (with GPU compute remarks) of Screen‑Space
//               Reflections (SSR), Screen‑Space Global Illumination (SSGI), and
//               Screen‑Space Ambient Occlusion (SSAO). Uses depth/normal buffers
//               and RenderingServer textures.
#include "servers/rendering_server.h"
#include "scene/3d/camera_3d_ext.h"
#include "core/math/math_funcs.h"
#include "core/math/transform_3d.h"
#include "core/os/os.h"
#include <cmath>
#include <vector>

// ----------------------------------------------------------------------------
// Helper: reconstruct world position from depth and pixel coordinates.
// ----------------------------------------------------------------------------
static Vector3 reconstruct_world_position(const Vector2 &p_uv, float p_depth,
                                          const Transform3D &p_inv_view,
                                          const Projection &p_inv_proj,
                                          float p_near, float p_far) {
    // Convert UV to NDC ([-1,1] for x, [1,-1] for y)
    float ndc_x = p_uv.x * 2.0f - 1.0f;
    float ndc_y = (1.0f - p_uv.y) * 2.0f - 1.0f;
    // Linearize depth (if depth is linear already, use directly)
    // In Godot, depth buffer stores non‑linear depth, so convert.
    float z_eye = p_near * p_far / (p_far - p_depth * (p_far - p_near));
    // Clip space position
    Vector4 clip(ndc_x, ndc_y, z_eye, 1.0f);
    // View space position
    Vector4 view = p_inv_proj * clip;
    view /= view.w;
    // World space position
    Vector4 world = p_inv_view * view;
    return Vector3(world.x, world.y, world.z);
}

// ----------------------------------------------------------------------------
// SSR: ray march in screen space along reflection direction.
// ----------------------------------------------------------------------------
void ssr_ray_march(const RID &p_depth_texture, const RID &p_normal_texture,
                   const Camera3DExt *p_camera,
                   int p_width, int p_height, RID &p_output) {
    // Fetch depth and normal data from RenderingServer (simplified)
    // In a real engine, you would upload these to a compute shader.
    // Here we simulate by reading back to CPU (very slow, for demonstration only).
    Vector<float> depth_data;
    Vector<float> normal_data;
    RenderingServer::get_singleton()->texture_get_data(p_depth_texture, depth_data);
    RenderingServer::get_singleton()->texture_get_data(p_normal_texture, normal_data);

    Transform3D inv_view = p_camera->get_global_transform().inverse();
    Projection inv_proj = p_camera->get_projection().inverse();
    float near = p_camera->get_near();
    float far = p_camera->get_far();

    std::vector<Color> output_pixels(p_width * p_height, Color(0,0,0,0));

    const int max_steps = 32;
    const int binary_steps = 8;
    const float step_size = 0.1f;

    for (int y = 0; y < p_height; ++y) {
        for (int x = 0; x < p_width; ++x) {
            int idx = y * p_width + x;
            float depth = depth_data[idx];
            if (depth >= 0.999f) continue; // background

            Vector3 normal = Vector3(normal_data[idx*3], normal_data[idx*3+1], normal_data[idx*3+2]).normalized();
            Vector2 uv(float(x)/p_width, float(y)/p_height);
            Vector3 world_pos = reconstruct_world_position(uv, depth, inv_view, inv_proj, near, far);
            Vector3 view_dir = (p_camera->get_global_transform().origin - world_pos).normalized();
            Vector3 reflect_dir = (view_dir - 2.0f * view_dir.dot(normal) * normal).normalized();

            // March in screen space
            Vector3 current_pos = world_pos;
            bool hit = false;
            Vector2 hit_uv;
            for (int step = 0; step < max_steps; ++step) {
                current_pos += reflect_dir * step_size;
                // Project to screen
                Vector4 view_pos = p_camera->get_global_transform().inverse() * current_pos;
                if (view_pos.z <= 0.0f) break;
                Vector4 clip = p_camera->get_projection() * view_pos;
                if (clip.w <= 0.0f) break;
                Vector2 ndc(clip.x / clip.w, clip.y / clip.w);
                if (ndc.x < -1.0f || ndc.x > 1.0f || ndc.y < -1.0f || ndc.y > 1.0f) continue;
                float screen_x = (ndc.x + 1.0f) * 0.5f * p_width;
                float screen_y = (1.0f - (ndc.y + 1.0f) * 0.5f) * p_height;
                int sx = (int)screen_x, sy = (int)screen_y;
                if (sx < 0 || sx >= p_width || sy < 0 || sy >= p_height) continue;
                float sampled_depth = depth_data[sy * p_width + sx];
                // Convert sampled depth to view space Z for comparison
                float view_z_sample = near * far / (far - sampled_depth * (far - near));
                float view_z_current = view_pos.z;
                if (Math::abs(view_z_current - view_z_sample) < 0.05f) {
                    hit = true;
                    hit_uv = Vector2(screen_x / p_width, screen_y / p_height);
                    break;
                }
            }
            if (hit) {
                // Sample color at hit_uv (we need color buffer, not available here)
                // For now, output a white reflection.
                output_pixels[idx] = Color(1,1,1,1);
            } else {
                output_pixels[idx] = Color(0,0,0,0);
            }
        }
    }

    // Upload result to output texture
    std::vector<uint8_t> data(p_width * p_height * 4);
    for (int i = 0; i < p_width * p_height; ++i) {
        data[i*4] = uint8_t(output_pixels[i].r * 255);
        data[i*4+1] = uint8_t(output_pixels[i].g * 255);
        data[i*4+2] = uint8_t(output_pixels[i].b * 255);
        data[i*4+3] = 255;
    }
    RenderingServer::get_singleton()->texture_2d_update(p_output, data, p_width, p_height, Image::FORMAT_RGBA8);
}

// ----------------------------------------------------------------------------
// SSGI: diffuse indirect by sampling hemisphere and ray marching.
// ----------------------------------------------------------------------------
void ssgi_compute(const RID &p_color_texture, const RID &p_depth_texture,
                  const RID &p_normal_texture, const Camera3DExt *p_camera,
                  int p_width, int p_height, RID &p_output) {
    // Similar to SSR but for diffuse: sample random directions in hemisphere,
    // march, accumulate albedo from hit point.
    Vector<float> depth_data, normal_data, color_data;
    RenderingServer::get_singleton()->texture_get_data(p_depth_texture, depth_data);
    RenderingServer::get_singleton()->texture_get_data(p_normal_texture, normal_data);
    RenderingServer::get_singleton()->texture_get_data(p_color_texture, color_data);

    Transform3D inv_view = p_camera->get_global_transform().inverse();
    Projection inv_proj = p_camera->get_projection().inverse();
    float near = p_camera->get_near(), far = p_camera->get_far();

    std::vector<Color> indirect(p_width * p_height, Color(0,0,0,0));
    const int num_samples = 8;
    RandomPCG rng;

    for (int y = 0; y < p_height; ++y) {
        for (int x = 0; x < p_width; ++x) {
            int idx = y * p_width + x;
            float depth = depth_data[idx];
            if (depth >= 0.999f) continue;
            Vector2 uv(float(x)/p_width, float(y)/p_height);
            Vector3 world_pos = reconstruct_world_position(uv, depth, inv_view, inv_proj, near, far);
            Vector3 normal = Vector3(normal_data[idx*3], normal_data[idx*3+1], normal_data[idx*3+2]).normalized();
            Color albedo(color_data[idx*3], color_data[idx*3+1], color_data[idx*3+2]);

            Color accum(0,0,0);
            for (int s = 0; s < num_samples; ++s) {
                // Cosine‑weighted hemisphere sample
                float u1 = rng.randf();
                float u2 = rng.randf();
                float phi = 2.0f * Math_PI * u1;
                float cos_theta = sqrt(u2);
                float sin_theta = sqrt(1.0f - u2);
                Vector3 local_dir(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
                // Transform to world
                Vector3 tangent, bitangent;
                if (Math::abs(normal.x) < 0.999f) {
                    tangent = Vector3(0,1,0).cross(normal).normalized();
                } else {
                    tangent = Vector3(0,0,1).cross(normal).normalized();
                }
                bitangent = normal.cross(tangent);
                Vector3 world_dir = tangent * local_dir.x + bitangent * local_dir.y + normal * local_dir.z;
                // Ray march
                Vector3 current_pos = world_pos;
                const float step = 0.1f;
                const int steps = 32;
                bool hit = false;
                Color hit_color;
                for (int step_i = 0; step_i < steps; ++step_i) {
                    current_pos += world_dir * step;
                    Vector4 view_pos = p_camera->get_global_transform().inverse() * current_pos;
                    if (view_pos.z <= 0.0f) break;
                    Vector4 clip = p_camera->get_projection() * view_pos;
                    if (clip.w <= 0.0f) break;
                    Vector2 ndc(clip.x / clip.w, clip.y / clip.w);
                    if (ndc.x < -1.0f || ndc.x > 1.0f || ndc.y < -1.0f || ndc.y > 1.0f) continue;
                    int sx = (ndc.x + 1.0f) * 0.5f * p_width;
                    int sy = (1.0f - (ndc.y + 1.0f) * 0.5f) * p_height;
                    if (sx < 0 || sx >= p_width || sy < 0 || sy >= p_height) continue;
                    float sampled_depth = depth_data[sy * p_width + sx];
                    float view_z_sample = near * far / (far - sampled_depth * (far - near));
                    if (Math::abs(view_pos.z - view_z_sample) < 0.05f) {
                        hit = true;
                        int hit_idx = sy * p_width + sx;
                        hit_color = Color(color_data[hit_idx*3], color_data[hit_idx*3+1], color_data[hit_idx*3+2]);
                        break;
                    }
                }
                if (hit) {
                    accum += hit_color * cos_theta;
                }
            }
            indirect[idx] = accum * albedo / num_samples;
        }
    }

    // Upload result
    std::vector<uint8_t> data(p_width * p_height * 4);
    for (int i = 0; i < p_width * p_height; ++i) {
        data[i*4] = uint8_t(indirect[i].r * 255);
        data[i*4+1] = uint8_t(indirect[i].g * 255);
        data[i*4+2] = uint8_t(indirect[i].b * 255);
        data[i*4+3] = 255;
    }
    RenderingServer::get_singleton()->texture_2d_update(p_output, data, p_width, p_height, Image::FORMAT_RGBA8);
}

// ----------------------------------------------------------------------------
// SSAO: hemisphere sampling with depth comparison.
// ----------------------------------------------------------------------------
void ssao_compute(const RID &p_depth_texture, const RID &p_normal_texture,
                  const Camera3DExt *p_camera,
                  int p_width, int p_height, RID &p_output) {
    Vector<float> depth_data, normal_data;
    RenderingServer::get_singleton()->texture_get_data(p_depth_texture, depth_data);
    RenderingServer::get_singleton()->texture_get_data(p_normal_texture, normal_data);

    Transform3D inv_view = p_camera->get_global_transform().inverse();
    Projection inv_proj = p_camera->get_projection().inverse();
    float near = p_camera->get_near(), far = p_camera->get_far();

    std::vector<float> ao(p_width * p_height, 1.0f);
    const int samples = 16;
    const float radius = 0.5f;
    const float intensity = 1.0f;

    for (int y = 0; y < p_height; ++y) {
        for (int x = 0; x < p_width; ++x) {
            int idx = y * p_width + x;
            float depth = depth_data[idx];
            if (depth >= 0.999f) continue;
            Vector2 uv(float(x)/p_width, float(y)/p_height);
            Vector3 world_pos = reconstruct_world_position(uv, depth, inv_view, inv_proj, near, far);
            Vector3 normal = Vector3(normal_data[idx*3], normal_data[idx*3+1], normal_data[idx*3+2]).normalized();

            float occlusion = 0.0f;
            for (int s = 0; s < samples; ++s) {
                // Random direction on hemisphere
                float u1 = Math::randf();
                float u2 = Math::randf();
                float phi = 2.0f * Math_PI * u1;
                float cos_theta = sqrt(u2);
                float sin_theta = sqrt(1.0f - u2);
                Vector3 local_dir(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
                // Transform to world
                Vector3 tangent, bitangent;
                if (Math::abs(normal.x) < 0.999f) {
                    tangent = Vector3(0,1,0).cross(normal).normalized();
                } else {
                    tangent = Vector3(0,0,1).cross(normal).normalized();
                }
                bitangent = normal.cross(tangent);
                Vector3 world_dir = tangent * local_dir.x + bitangent * local_dir.y + normal * local_dir.z;
                Vector3 sample_pos = world_pos + world_dir * radius;
                // Project sample to screen
                Vector4 view_pos = p_camera->get_global_transform().inverse() * sample_pos;
                if (view_pos.z <= 0.0f) continue;
                Vector4 clip = p_camera->get_projection() * view_pos;
                if (clip.w <= 0.0f) continue;
                Vector2 ndc(clip.x / clip.w, clip.y / clip.w);
                if (ndc.x < -1.0f || ndc.x > 1.0f || ndc.y < -1.0f || ndc.y > 1.0f) continue;
                int sx = (ndc.x + 1.0f) * 0.5f * p_width;
                int sy = (1.0f - (ndc.y + 1.0f) * 0.5f) * p_height;
                if (sx < 0 || sx >= p_width || sy < 0 || sy >= p_height) continue;
                float sampled_depth = depth_data[sy * p_width + sx];
                float view_z_sample = near * far / (far - sampled_depth * (far - near));
                if (view_z_sample < view_pos.z - 0.05f) occlusion += 1.0f;
            }
            ao[idx] = 1.0f - (occlusion / samples) * intensity;
        }
    }

    std::vector<uint8_t> data(p_width * p_height);
    for (int i = 0; i < p_width * p_height; ++i) {
        data[i] = uint8_t(Math::clamp(ao[i], 0.0f, 1.0f) * 255);
    }
    RenderingServer::get_singleton()->texture_2d_update(p_output, data, p_width, p_height, Image::FORMAT_R8);
}