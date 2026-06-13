// Name : lighting enhancement updated
// File : lighting_pass.cpp 79 of 63
// Description : Implementation of host-side lighting controller for lighting_compute.glsl.
//               Updates uniform buffers (camera, lighting, temporal), binds textures,
//               dispatches compute shader, and manages history textures.
#include "lighting_pass.h"
#include "core/math/math_funcs.h"
#include "core/math/transform_3d.h"
#include "core/math/projection.h"
#include "core/math/vector3.h"
#include "servers/rendering_server.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/world_3d.h"
#include "core/os/os.h"
#include <cmath>
#include <cstring>

// ----------------------------------------------------------------------------
// Helper: convert Godot camera matrices to GLSL column-major layout
// ----------------------------------------------------------------------------
static void fill_camera_uniforms(const Camera3D *camera, float width, float height,
                                 float *out_inv_proj, float *out_inv_view,
                                 float *out_prev_view_proj, float *out_cam_pos,
                                 float *out_focal, float *out_principal,
                                 float *out_near_far, float *out_time) {
    // Inverse projection (column-major)
    Projection proj = camera->get_projection();
    Projection inv_proj = proj.inverse();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            out_inv_proj[j * 4 + i] = inv_proj[i][j];

    // Inverse view (world → camera) – column-major
    Transform3D view = camera->get_global_transform().inverse();
    Transform3D inv_view = view.inverse(); // camera → world
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            out_inv_view[j * 4 + i] = inv_view[i][j];

    // Previous view-projection (simulate identity for first frame; would be stored in practice)
    memset(out_prev_view_proj, 0, 16 * sizeof(float));
    out_prev_view_proj[0] = out_prev_view_proj[5] = out_prev_view_proj[10] = out_prev_view_proj[15] = 1.0f;

    // Camera position
    Vector3 pos = camera->get_global_transform().origin;
    out_cam_pos[0] = pos.x; out_cam_pos[1] = pos.y; out_cam_pos[2] = pos.z; out_cam_pos[3] = 1.0f;

    // Focal length from vertical FOV (pixels)
    float fov_y = camera->get_fov(); // degrees
    float rad_fov = Math::deg_to_rad(fov_y);
    float focal_pixels = (height * 0.5f) / tanf(rad_fov * 0.5f);
    out_focal[0] = focal_pixels;
    out_focal[1] = focal_pixels;
    out_principal[0] = width * 0.5f;
    out_principal[1] = height * 0.5f;

    // Near and far planes
    out_near_far[0] = camera->get_near();
    out_near_far[1] = camera->get_far();

    // Time
    out_time[0] = (float)OS::get_singleton()->get_ticks_msec() / 1000.0f;
}

// ----------------------------------------------------------------------------
// Constructor / Destructor
// ----------------------------------------------------------------------------
LightingPass::LightingPass() {
    RenderingServer *rs = RenderingServer::get_singleton();

    // Create compute shader (in production, you would compile the GLSL source)
    m_compute_shader = rs->shader_create();
    // For demo, assume shader is already loaded.

    // Create uniform buffers with sizes matching GLSL layout
    // CameraUniforms: 4 matrices (16 floats each) + 4 floats + 2+2+2+1 = 16*4 + 4+2+2+2+1 = 75 floats = 300 bytes
    size_t cam_size = 16 * 4 * sizeof(float) + 4 * sizeof(float) + 2 * sizeof(float) + 2 * sizeof(float) + 2 * sizeof(float) + 1 * sizeof(float);
    m_uniform_buffer_camera = rs->uniform_buffer_create(cam_size);
    // LightingParams: ints (5+1) + floats (8) + int (1) = ~ 64 bytes
    size_t light_size = 6 * sizeof(int) + 8 * sizeof(float) + 1 * sizeof(int);
    m_uniform_buffer_lighting = rs->uniform_buffer_create(light_size);
    // TemporalParams: 6 floats + 2 ints = 32 bytes
    size_t temp_size = 6 * sizeof(float) + 2 * sizeof(int);
    m_uniform_buffer_temporal = rs->uniform_buffer_create(temp_size);

    // Create temporary output texture (will be resized later)
    m_temp_output = rs->texture_2d_create();
    // Initialize with dummy size
    rs->texture_2d_initialize(m_temp_output, 1920, 1080, Image::FORMAT_RGBA8);

    m_history.width = m_history.height = 0;
}

LightingPass::~LightingPass() {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (m_compute_shader.is_valid()) rs->free(m_compute_shader);
    if (m_uniform_buffer_camera.is_valid()) rs->free(m_uniform_buffer_camera);
    if (m_uniform_buffer_lighting.is_valid()) rs->free(m_uniform_buffer_lighting);
    if (m_uniform_buffer_temporal.is_valid()) rs->free(m_uniform_buffer_temporal);
    if (m_temp_output.is_valid()) rs->free(m_temp_output);
    if (m_history.color_texture.is_valid()) rs->free(m_history.color_texture);
    if (m_history.depth_texture.is_valid()) rs->free(m_history.depth_texture);
    if (m_history.variance_texture.is_valid()) rs->free(m_history.variance_texture);
    if (m_history.motion_texture.is_valid()) rs->free(m_history.motion_texture);
}

// ----------------------------------------------------------------------------
// Public setters
// ----------------------------------------------------------------------------
void LightingPass::set_params(const LightingPassParams &p_params) { m_params = p_params; }
void LightingPass::set_camera(const Camera3D *p_camera) { m_camera = p_camera; }
void LightingPass::set_world(World3D *p_world) { m_world = p_world; }
void LightingPass::set_shadow_map(const RID &p_shadow_map) { m_shadow_map = p_shadow_map; }
void LightingPass::set_shadow_map2(const RID &p_shadow_map2) { m_shadow_map2 = p_shadow_map2; }
void LightingPass::set_environment_cubemap(const RID &p_cubemap) { m_env_cubemap = p_cubemap; }

// ----------------------------------------------------------------------------
// Private uniform updaters (full math)
// ----------------------------------------------------------------------------
void LightingPass::_update_camera_uniforms() {
    if (!m_camera) return;
    int w = m_history.width > 0 ? m_history.width : 1920;
    int h = m_history.height > 0 ? m_history.height : 1080;

    // Layout must exactly match the GLSL struct CameraUniforms
    struct {
        float inv_proj[16];
        float inv_view[16];
        float prev_view_proj[16];
        float cam_pos[4];
        float focal[2];
        float principal[2];
        float near_far[2];
        float time;
    } cam_data;
    memset(&cam_data, 0, sizeof(cam_data));

    fill_camera_uniforms(m_camera, (float)w, (float)h,
                         cam_data.inv_proj, cam_data.inv_view,
                         cam_data.prev_view_proj, cam_data.cam_pos,
                         cam_data.focal, cam_data.principal,
                         cam_data.near_far, &cam_data.time);

    RenderingServer::get_singleton()->uniform_buffer_update(m_uniform_buffer_camera, 0, sizeof(cam_data), &cam_data);
}

void LightingPass::_update_lighting_uniforms() {
    // GLSL LightingParams layout:
    // int enable_ssr, enable_ssgi, enable_ssao, enable_shadows, shadow_technique;
    // float ssr_step_size; int ssr_max_steps;
    // float ssao_radius, ssao_intensity; int ssgi_num_samples;
    // float ssgi_radius, pcss_light_size, vsm_exponent, temporal_blend; int enable_denoiser;
    struct {
        int enable_ssr, enable_ssgi, enable_ssao, enable_shadows, shadow_technique;
        float ssr_step_size;
        int ssr_max_steps;
        float ssao_radius, ssao_intensity;
        int ssgi_num_samples;
        float ssgi_radius, pcss_light_size, vsm_exponent, temporal_blend;
        int enable_denoiser;
        int _padding[3];
    } data;
    memset(&data, 0, sizeof(data));
    data.enable_ssr = m_params.enable_ssr ? 1 : 0;
    data.enable_ssgi = m_params.enable_ssgi ? 1 : 0;
    data.enable_ssao = m_params.enable_ssao ? 1 : 0;
    data.enable_shadows = m_params.enable_shadows ? 1 : 0;
    data.shadow_technique = m_params.shadow_technique;
    data.ssr_step_size = m_params.ssr_step_size;
    data.ssr_max_steps = m_params.ssr_max_steps;
    data.ssao_radius = m_params.ssao_radius;
    data.ssao_intensity = m_params.ssao_intensity;
    data.ssgi_num_samples = m_params.ssgi_num_samples;
    data.ssgi_radius = m_params.ssgi_radius;
    data.pcss_light_size = m_params.pcss_light_size;
    data.vsm_exponent = m_params.vsm_exponent;
    data.temporal_blend = m_params.temporal_blend;
    data.enable_denoiser = m_params.enable_denoiser ? 1 : 0;

    RenderingServer::get_singleton()->uniform_buffer_update(m_uniform_buffer_lighting, 0, sizeof(data), &data);
}

void LightingPass::_update_temporal_uniforms() {
    // GLSL TemporalParams: 6 floats + 2 ints
    struct {
        float feedback_factor, variance_clip, color_sigma, depth_sigma, normal_sigma, motion_blend;
        int max_history, use_variance_clamping;
        float _padding[2];
    } data;
    memset(&data, 0, sizeof(data));
    data.feedback_factor = 0.95f;
    data.variance_clip = 1.5f;
    data.color_sigma = 0.1f;
    data.depth_sigma = 0.05f;
    data.normal_sigma = 0.2f;
    data.motion_blend = 0.9f;
    data.max_history = 2;
    data.use_variance_clamping = 1;

    RenderingServer::get_singleton()->uniform_buffer_update(m_uniform_buffer_temporal, 0, sizeof(data), &data);
}

void LightingPass::_create_history_textures(int width, int height) {
    RenderingServer *rs = RenderingServer::get_singleton();
    auto ensure_tex = [&](RID &rid, Image::Format fmt) {
        if (rid.is_valid()) rs->free(rid);
        rid = rs->texture_2d_create();
        rs->texture_2d_initialize(rid, width, height, fmt);
    };
    ensure_tex(m_history.color_texture, Image::FORMAT_RGBA8);
    ensure_tex(m_history.depth_texture, Image::FORMAT_R32F);
    ensure_tex(m_history.variance_texture, Image::FORMAT_R32F);
    ensure_tex(m_history.motion_texture, Image::FORMAT_RG32F);
    m_history.width = width;
    m_history.height = height;
}

void LightingPass::_bind_textures(const RID &p_color, const RID &p_depth, const RID &p_normal,
                                  const RID &p_motion) {
    RenderingServer *rs = RenderingServer::get_singleton();
    // Binding indices match GLSL layout
    rs->material_set_texture(m_compute_shader, 0, p_color);
    rs->material_set_texture(m_compute_shader, 1, p_depth);
    rs->material_set_texture(m_compute_shader, 2, p_normal);
    rs->material_set_texture(m_compute_shader, 3, p_motion);
    rs->material_set_texture(m_compute_shader, 4, m_shadow_map);
    rs->material_set_texture(m_compute_shader, 5, m_shadow_map2);
    rs->material_set_texture(m_compute_shader, 6, m_env_cubemap);
    rs->material_set_texture(m_compute_shader, 7, m_history.color_texture);
    rs->material_set_texture(m_compute_shader, 8, m_history.depth_texture);
    rs->material_set_texture(m_compute_shader, 9, m_history.variance_texture);
    rs->material_set_texture(m_compute_shader, 10, m_history.motion_texture);

    rs->material_set_uniform_buffer(m_compute_shader, 11, m_uniform_buffer_camera);
    rs->material_set_uniform_buffer(m_compute_shader, 12, m_uniform_buffer_lighting);
    rs->material_set_uniform_buffer(m_compute_shader, 13, m_uniform_buffer_temporal);

    rs->shader_set_image(m_compute_shader, 14, m_temp_output);
}

void LightingPass::_dispatch(int width, int height) {
    int groups_x = (width + 7) / 8;
    int groups_y = (height + 7) / 8;
    RenderingServer::get_singleton()->compute_shader_dispatch(m_compute_shader, groups_x, groups_y, 1);
}

// ----------------------------------------------------------------------------
// Main execution
// ----------------------------------------------------------------------------
void LightingPass::execute(const RID &p_color_in, const RID &p_depth, const RID &p_normal,
                           const RID &p_motion, RID &out_color) {
    if (!m_camera) return;
    RenderingServer *rs = RenderingServer::get_singleton();
    int width = rs->texture_get_width(p_depth);
    int height = rs->texture_get_height(p_depth);
    if (width <= 0 || height <= 0) return;

    if (m_history.width != width || m_history.height != height) {
        _create_history_textures(width, height);
    }

    _update_camera_uniforms();
    _update_lighting_uniforms();
    _update_temporal_uniforms();
    _bind_textures(p_color_in, p_depth, p_normal, p_motion);
    _dispatch(width, height);

    // After dispatch, copy the output to history color and depth for next frame
    rs->texture_2d_copy(m_temp_output, m_history.color_texture);
    rs->texture_2d_copy(p_depth, m_history.depth_texture);
    rs->texture_2d_copy(p_motion, m_history.motion_texture);
    // Variance texture would be written by the shader, but we assume shader writes it.
    // For now, we leave variance texture as is.

    out_color = m_temp_output;
}