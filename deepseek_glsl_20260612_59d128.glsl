// Name : lighting enhancement updated
// File : lighting_compute.glsl
// Description : High-performance GLSL compute shader for real-time lighting techniques:
//               SSR, SSGI, SSAO, CSM, PCSS, VSM, and full temporal denoising.
//               Includes temporal_denoise.glsl for motion-vector-guided accumulation.
// ----------------------------------------------------------------------------
// Include the temporal denoising module
#include "temporal_denoise.glsl"

// ----------------------------------------------------------------------------
// Constants
// ----------------------------------------------------------------------------
#define PI 3.14159265358979323846
#define MAX_SSR_STEPS 32
#define MAX_SSGI_SAMPLES 8
#define MAX_SSAO_SAMPLES 32
#define TILE_SIZE 16

// ----------------------------------------------------------------------------
// Uniforms
// ----------------------------------------------------------------------------
layout(set = 0, binding = 0) uniform sampler2D color_texture;
layout(set = 0, binding = 1) uniform sampler2D depth_texture;
layout(set = 0, binding = 2) uniform sampler2D normal_texture;
layout(set = 0, binding = 3) uniform sampler2D motion_vectors_texture;
layout(set = 0, binding = 4) uniform sampler2D shadow_map_texture;      // for CSM/PCSS
layout(set = 0, binding = 5) uniform sampler2D shadow_map2_texture;     // for VSM (depth^2)
layout(set = 0, binding = 6) uniform samplerCube environment_cubemap;   // for reflections
// History textures (from temporal_denoise.glsl expectations)
layout(set = 0, binding = 7) uniform sampler2D history_color_texture;
layout(set = 0, binding = 8) uniform sampler2D history_depth_texture;
layout(set = 0, binding = 9) uniform sampler2D history_variance_texture; // for variance‑guided accumulation
layout(set = 0, binding = 10) uniform sampler2D history_motion_texture;

layout(set = 0, binding = 11) uniform CameraUniforms {
    mat4 inv_proj;
    mat4 inv_view;
    mat4 prev_view_proj;
    vec4 camera_position;   // xyz, w = 1
    vec2 focal_length;      // fx, fy
    vec2 principal_point;   // cx, cy
    vec2 screen_size;       // width, height
    float near_plane;
    float far_plane;
    float time;
} camera;

layout(set = 0, binding = 12) uniform LightingParams {
    int enable_ssr;
    int enable_ssgi;
    int enable_ssao;
    int enable_shadows;
    int shadow_technique;   // 0=PCF,1=PCSS,2=VSM,3=CSM
    float ssr_step_size;
    int ssr_max_steps;
    float ssao_radius;
    float ssao_intensity;
    int ssgi_num_samples;
    float ssgi_radius;
    float pcss_light_size;
    float vsm_exponent;
    float temporal_blend;
    int enable_denoiser;
} params;

layout(set = 0, binding = 13) uniform TemporalParams {
    float feedback_factor;
    float variance_clip;
    float color_sigma;
    float depth_sigma;
    float normal_sigma;
    float motion_blend;
    int max_history;
    int use_variance_clamping;
} temporal;

layout(set = 0, binding = 14) writeonly uniform image2D output_image;

// ----------------------------------------------------------------------------
// Helper: reconstruct world position from depth and UV
// ----------------------------------------------------------------------------
vec3 reconstruct_world_position(vec2 uv, float depth) {
    vec4 clip = vec4(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    vec4 view = camera.inv_proj * clip;
    view /= view.w;
    vec4 world = camera.inv_view * view;
    return world.xyz;
}

// ----------------------------------------------------------------------------
// Helper: screen‑space ray march
// ----------------------------------------------------------------------------
bool screen_space_ray_march(vec3 origin, vec3 direction, out vec2 hit_uv, out float hit_depth, int max_steps, float step_size) {
    vec3 current = origin;
    for (int i = 0; i < max_steps; ++i) {
        current += direction * step_size;
        vec4 view_pos = camera.inv_view * vec4(current, 1.0);
        vec4 clip = camera.inv_proj * view_pos;
        if (clip.z <= 0.0) continue;
        vec3 ndc = clip.xyz / clip.w;
        if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) continue;
        vec2 uv = ndc.xy * 0.5 + 0.5;
        float sampled_depth = texture(depth_texture, uv).r;
        float view_z = camera.near_plane * camera.far_plane / (camera.far_plane - sampled_depth * (camera.far_plane - camera.near_plane));
        if (abs(view_pos.z - view_z) < 0.05) {
            hit_uv = uv;
            hit_depth = sampled_depth;
            return true;
        }
    }
    return false;
}

// ----------------------------------------------------------------------------
// SSAO compute (single pixel)
// ----------------------------------------------------------------------------
float compute_ssao(vec2 uv, float depth, vec3 normal) {
    float occlusion = 0.0;
    vec3 world_pos = reconstruct_world_position(uv, depth);
    for (int i = 0; i < MAX_SSAO_SAMPLES; ++i) {
        // random hemisphere direction (cosine-weighted)
        float u1 = fract(sin(float(i) * 12.9898 + camera.time) * 43758.5453);
        float u2 = fract(cos(float(i) * 78.233 + camera.time) * 43758.5453);
        float phi = 2.0 * PI * u1;
        float cos_theta = sqrt(u2);
        float sin_theta = sqrt(1.0 - u2);
        vec3 local_dir = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
        // tangent basis from normal
        vec3 tangent = (abs(normal.x) < 0.999) ? normalize(cross(vec3(0,1,0), normal)) : normalize(cross(vec3(0,0,1), normal));
        vec3 bitangent = cross(normal, tangent);
        vec3 world_dir = tangent * local_dir.x + bitangent * local_dir.y + normal * local_dir.z;
        vec3 sample_pos = world_pos + world_dir * params.ssao_radius;
        // project to screen
        vec4 view_pos = camera.inv_view * vec4(sample_pos, 1.0);
        vec4 clip = camera.inv_proj * view_pos;
        vec2 sample_uv = clip.xy / clip.w * 0.5 + 0.5;
        if (sample_uv.x >= 0.0 && sample_uv.x <= 1.0 && sample_uv.y >= 0.0 && sample_uv.y <= 1.0) {
            float sample_depth = texture(depth_texture, sample_uv).r;
            float view_z_sample = camera.near_plane * camera.far_plane / (camera.far_plane - sample_depth * (camera.far_plane - camera.near_plane));
            if (view_z_sample < view_pos.z - 0.05) occlusion += 1.0;
        }
    }
    return 1.0 - (occlusion / float(MAX_SSAO_SAMPLES)) * params.ssao_intensity;
}

// ----------------------------------------------------------------------------
// SSR compute (single pixel)
// ----------------------------------------------------------------------------
vec3 compute_ssr(vec2 uv, float depth, vec3 normal, vec3 view_dir) {
    vec3 world_pos = reconstruct_world_position(uv, depth);
    vec3 reflect_dir = normalize(reflect(view_dir, normal));
    vec2 hit_uv;
    float hit_depth;
    if (screen_space_ray_march(world_pos, reflect_dir, hit_uv, hit_depth, params.ssr_max_steps, params.ssr_step_size)) {
        return texture(color_texture, hit_uv).rgb;
    } else {
        // fallback: sample environment cubemap
        return texture(environment_cubemap, reflect_dir).rgb;
    }
}

// ----------------------------------------------------------------------------
// SSGI compute (single pixel)
// ----------------------------------------------------------------------------
vec3 compute_ssgi(vec2 uv, float depth, vec3 normal, vec3 view_dir) {
    vec3 world_pos = reconstruct_world_position(uv, depth);
    vec3 albedo = texture(color_texture, uv).rgb;
    vec3 indirect = vec3(0.0);
    int samples = params.ssgi_num_samples;
    for (int i = 0; i < samples; ++i) {
        // random cosine‑weighted hemisphere direction
        float u1 = fract(sin(float(i) * 12.9898 + camera.time + 1.0) * 43758.5453);
        float u2 = fract(cos(float(i) * 78.233 + camera.time + 1.0) * 43758.5453);
        float phi = 2.0 * PI * u1;
        float cos_theta = sqrt(u2);
        float sin_theta = sqrt(1.0 - u2);
        vec3 local_dir = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
        vec3 tangent = (abs(normal.x) < 0.999) ? normalize(cross(vec3(0,1,0), normal)) : normalize(cross(vec3(0,0,1), normal));
        vec3 bitangent = cross(normal, tangent);
        vec3 world_dir = tangent * local_dir.x + bitangent * local_dir.y + normal * local_dir.z;
        vec3 sample_pos = world_pos + world_dir * params.ssgi_radius;
        vec2 hit_uv;
        float hit_depth;
        if (screen_space_ray_march(sample_pos, world_dir, hit_uv, hit_depth, 32, 0.1)) {
            indirect += texture(color_texture, hit_uv).rgb * cos_theta;
        }
    }
    if (samples > 0) indirect /= float(samples);
    return albedo * indirect;
}

// ----------------------------------------------------------------------------
// CSM shadow evaluation (using a shadow atlas with cascades)
// ----------------------------------------------------------------------------
float sample_csm(vec3 world_pos) {
    // In production, you would fetch the cascade index based on distance.
    // For simplicity, we sample the first cascade shadow map.
    vec3 shadow_uv = world_pos; // dummy; proper conversion needed.
    return texture(shadow_map_texture, shadow_uv.xy).r;
}

// ----------------------------------------------------------------------------
// PCSS soft shadow
// ----------------------------------------------------------------------------
float pcss_shadow(vec3 uv_depth, float receiver_depth) {
    // 1. Blocker search (simplified)
    float blocker_depth = 0.0;
    int blocker_count = 0;
    float light_size = params.pcss_light_size;
    const int search_radius = 8;
    vec2 texel_size = 1.0 / textureSize(shadow_map_texture, 0);
    for (int y = -search_radius; y <= search_radius; ++y) {
        for (int x = -search_radius; x <= search_radius; ++x) {
            vec2 offset = vec2(x, y) * texel_size;
            float d = texture(shadow_map_texture, uv_depth.xy + offset).r;
            if (d < receiver_depth - 0.01) {
                blocker_depth += d;
                blocker_count++;
            }
        }
    }
    if (blocker_count == 0) return 1.0;
    float avg_blocker = blocker_depth / float(blocker_count);
    float penumbra = light_size * (receiver_depth - avg_blocker) / avg_blocker;
    penumbra = clamp(penumbra, 0.5, 5.0);
    int kernel_radius = int(penumbra) + 1;
    float shadow = 0.0;
    int samples = 0;
    for (int dy = -kernel_radius; dy <= kernel_radius; ++dy) {
        for (int dx = -kernel_radius; dx <= kernel_radius; ++dx) {
            vec2 offset = vec2(dx, dy) * texel_size;
            float d = texture(shadow_map_texture, uv_depth.xy + offset).r;
            if (receiver_depth <= d + 0.005) shadow += 1.0;
            samples++;
        }
    }
    return shadow / float(samples);
}

// ----------------------------------------------------------------------------
// VSM shadow evaluation
// ----------------------------------------------------------------------------
float vsm_shadow(vec3 uv_depth, float receiver_depth) {
    vec2 moments = texture(shadow_map_texture, uv_depth.xy).rg;
    float depth2 = texture(shadow_map2_texture, uv_depth.xy).r;
    float variance = max(depth2 - moments.x * moments.x, 1e-4);
    float diff = receiver_depth - moments.x;
    if (diff <= 0.0) return 1.0;
    return variance / (variance + diff * diff);
}

// ----------------------------------------------------------------------------
// Compute local variance (simplified, could be replaced with sample variance)
// ----------------------------------------------------------------------------
float estimate_local_variance(vec2 uv) {
    // For demonstration, use a small 3x3 window on luminance.
    const int r = 1;
    float mean = 0.0;
    float mean_sq = 0.0;
    int cnt = 0;
    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            vec2 off = vec2(dx, dy) / camera.screen_size;
            float lum = dot(texture(color_texture, uv + off).rgb, vec3(0.2126, 0.7152, 0.0722));
            mean += lum;
            mean_sq += lum * lum;
            cnt++;
        }
    }
    mean /= float(cnt);
    mean_sq /= float(cnt);
    return max(mean_sq - mean * mean, 0.0);
}

// ----------------------------------------------------------------------------
// Main compute shader
// ----------------------------------------------------------------------------
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= int(camera.screen_size.x) || pixel.y >= int(camera.screen_size.y)) return;

    vec2 uv = (vec2(pixel) + 0.5) / camera.screen_size;
    float depth = texture(depth_texture, uv).r;
    if (depth >= 0.999) {
        imageStore(output_image, pixel, vec4(0.0));
        return;
    }

    vec3 normal = texture(normal_texture, uv).rgb;
    normal = normalize(normal * 2.0 - 1.0); // decode from [0,1]
    vec3 world_pos = reconstruct_world_position(uv, depth);
    vec3 view_dir = normalize(camera.camera_position.xyz - world_pos);

    // Start with base color (direct lighting already applied)
    vec3 color = texture(color_texture, uv).rgb;

    // SSAO
    if (params.enable_ssao != 0) {
        float ao = compute_ssao(uv, depth, normal);
        color *= ao;
    }

    // SSR
    if (params.enable_ssr != 0) {
        vec3 reflection = compute_ssr(uv, depth, normal, view_dir);
        color += reflection * 0.3;
    }

    // SSGI
    if (params.enable_ssgi != 0) {
        vec3 indirect = compute_ssgi(uv, depth, normal, view_dir);
        color += indirect * 0.5;
    }

    // Shadows
    if (params.enable_shadows != 0) {
        float shadow = 1.0;
        if (params.shadow_technique == 0) { // PCF (simple)
            vec2 texel = 1.0 / textureSize(shadow_map_texture, 0);
            float sum = 0.0;
            for (int y = -1; y <= 1; ++y) {
                for (int x = -1; x <= 1; ++x) {
                    vec2 offset = vec2(x, y) * texel;
                    sum += texture(shadow_map_texture, uv + offset).r;
                }
            }
            shadow = sum / 9.0;
        } else if (params.shadow_technique == 1) { // PCSS
            vec3 shadow_uv = vec3(uv, depth);
            shadow = pcss_shadow(shadow_uv, depth);
        } else if (params.shadow_technique == 2) { // VSM
            vec3 shadow_uv = vec3(uv, depth);
            shadow = vsm_shadow(shadow_uv, depth);
        } else if (params.shadow_technique == 3) { // CSM
            shadow = sample_csm(world_pos);
        }
        color *= shadow;
    }

    // Temporal denoising
    if (params.enable_denoiser != 0) {
        vec2 motion = texture(motion_vectors_texture, uv).xy;
        // Estimate variance (for ray tracing this would come from sample variance)
        float variance = estimate_local_variance(uv);
        // Call the denoising function from included header
        color = denoise_pixel(uv, color, depth, normal, motion, variance);
    }

    imageStore(output_image, pixel, vec4(color, 1.0));
}