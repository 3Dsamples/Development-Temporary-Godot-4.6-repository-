// Name : lighting enhancement updated
// File : lighting_pass.h 78 of 63
// Description : Host-side controller for lighting_compute.glsl shader.
//               Manages uniforms, textures, history buffers, and dispatch.
#pragma once

#include "core/math/transform_3d.h"
#include "core/math/projection.h"
#include "servers/rendering_server.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/world_3d.h"

// ----------------------------------------------------------------------------
// Parameters for the lighting pass (matches GLSL struct LightingParams)
// ----------------------------------------------------------------------------
struct LightingPassParams {
    bool enable_ssr = true;
    bool enable_ssgi = true;
    bool enable_ssao = true;
    bool enable_shadows = true;
    int shadow_technique = 1;      // 0=PCF,1=PCSS,2=VSM,3=CSM
    float ssr_step_size = 0.1f;
    int ssr_max_steps = 32;
    float ssao_radius = 0.5f;
    float ssao_intensity = 1.0f;
    int ssgi_num_samples = 8;
    float ssgi_radius = 1.0f;
    float pcss_light_size = 1.0f;
    float vsm_exponent = 40.0f;
    float temporal_blend = 0.95f;
    bool enable_denoiser = true;
};

// ----------------------------------------------------------------------------
// Temporal history buffers (matches GLSL history textures)
// ----------------------------------------------------------------------------
struct TemporalHistory {
    RID color_texture;     // binding 7
    RID depth_texture;     // binding 8
    RID variance_texture;  // binding 9
    RID motion_texture;    // binding 10
    int width = 0, height = 0;
};

// ----------------------------------------------------------------------------
// Main lighting pass controller
// ----------------------------------------------------------------------------
class LightingPass {
public:
    LightingPass();
    ~LightingPass();

    void set_params(const LightingPassParams &p_params);
    void set_camera(const Camera3D *p_camera);
    void set_world(World3D *p_world);
    void set_shadow_map(const RID &p_shadow_map);
    void set_shadow_map2(const RID &p_shadow_map2); // for VSM
    void set_environment_cubemap(const RID &p_cubemap);

    // Execute the lighting pass on the given G‑buffer textures
    // Output: out_color (RID of generated texture)
    void execute(const RID &p_color_in, const RID &p_depth, const RID &p_normal,
                 const RID &p_motion, RID &out_color);

    // Get history textures for next frame (call after execute)
    const TemporalHistory &get_history() const { return m_history; }

private:
    LightingPassParams m_params;
    const Camera3D *m_camera = nullptr;
    World3D *m_world = nullptr;
    RID m_shadow_map;
    RID m_shadow_map2;
    RID m_env_cubemap;

    // Compute shader resources
    RID m_compute_shader;
    RID m_uniform_buffer_camera;
    RID m_uniform_buffer_lighting;
    RID m_uniform_buffer_temporal;
    RID m_temp_output;          // output of the compute shader (binding 14)

    // History buffers
    TemporalHistory m_history;

    void _update_camera_uniforms();
    void _update_lighting_uniforms();
    void _update_temporal_uniforms();
    void _create_history_textures(int width, int height);
    void _bind_textures(const RID &p_color, const RID &p_depth, const RID &p_normal,
                        const RID &p_motion);
    void _dispatch(int width, int height);
};