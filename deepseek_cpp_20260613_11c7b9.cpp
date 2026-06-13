// Name : lighting enhancement updated
// File : lighting_manager.cpp 87 of 63
// Description : Implementation of LightingManager – integrates all lighting techniques
//               with Godot’s scene, manages shadow atlas, and calls the compute shader.
#include "lighting_manager.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/world_3d.h"
#include "servers/rendering_server.h"
#include "core/math/transform_3d.h"
#include "core/math/projection.h"
#include <cmath>

// ----------------------------------------------------------------------------
// Constructor / Destructor
// ----------------------------------------------------------------------------
LightingManager::LightingManager() {
    m_params = LightingPassParams(); // use defaults
    m_shadow_atlas = RID();
}

LightingManager::~LightingManager() {
    if (m_shadow_atlas.is_valid()) {
        RenderingServer::get_singleton()->free(m_shadow_atlas);
    }
}

// ----------------------------------------------------------------------------
// Configuration setters
// ----------------------------------------------------------------------------
void LightingManager::set_ssr_enabled(bool p_enabled) { m_params.enable_ssr = p_enabled; }
bool LightingManager::is_ssr_enabled() const { return m_params.enable_ssr; }
void LightingManager::set_ssgi_enabled(bool p_enabled) { m_params.enable_ssgi = p_enabled; }
bool LightingManager::is_ssgi_enabled() const { return m_params.enable_ssgi; }
void LightingManager::set_ssao_enabled(bool p_enabled) { m_params.enable_ssao = p_enabled; }
bool LightingManager::is_ssao_enabled() const { return m_params.enable_ssao; }
void LightingManager::set_shadows_enabled(bool p_enabled) { m_params.enable_shadows = p_enabled; }
bool LightingManager::are_shadows_enabled() const { return m_params.enable_shadows; }
void LightingManager::set_shadow_quality(int p_quality) {
    if (p_quality == 0) {
        m_params.shadow_technique = 0; // PCF low
        m_params.pcss_light_size = 1.0f;
    } else if (p_quality == 1) {
        m_params.shadow_technique = 1; // PCSS medium
        m_params.pcss_light_size = 1.5f;
    } else {
        m_params.shadow_technique = 2; // VSM high
        m_params.vsm_exponent = 60.0f;
    }
}
void LightingManager::set_temporal_denoising(bool p_enabled) { m_params.enable_denoiser = p_enabled; }
bool LightingManager::is_temporal_denoising_enabled() const { return m_params.enable_denoiser; }

void LightingManager::set_environment(const Ref<WorldEnvironment> &p_env) {
    m_environment = p_env;
    if (p_env.is_valid() && p_env->get_sky().is_valid()) {
        RID sky_rid = p_env->get_sky()->get_rid();
        // In a real engine, you would extract the environment cubemap from the sky.
        // For simulation, we assume the environment cubemap is set elsewhere.
    }
}

void LightingManager::set_shadow_map_size(int p_size) {
    // Recreate shadow atlas with new size
    if (m_shadow_atlas.is_valid()) {
        RenderingServer::get_singleton()->free(m_shadow_atlas);
        m_shadow_atlas = RID();
    }
    _ensure_shadow_atlas(p_size, p_size);
    m_lighting_pass.set_shadow_map(m_shadow_atlas);
}

TextureManager &LightingManager::get_texture_manager() {
    return m_texture_manager;
}

// ----------------------------------------------------------------------------
// Notification: when the node enters the scene tree, connect to the main viewport.
// ----------------------------------------------------------------------------
void LightingManager::_notification(int p_what) {
    if (p_what == NOTIFICATION_ENTER_TREE) {
        // Get the main viewport size
        Viewport *viewport = get_viewport();
        if (viewport) {
            m_viewport_width = viewport->get_visible_rect().size.width;
            m_viewport_height = viewport->get_visible_rect().size.height;
            _ensure_shadow_atlas(2048, 2048); // default size
            m_initialized = true;
        }
    } else if (p_what == NOTIFICATION_EXIT_TREE) {
        m_initialized = false;
    }
}

// ----------------------------------------------------------------------------
// Ensure the shadow atlas texture exists and is up‑to‑date.
// ----------------------------------------------------------------------------
void LightingManager::_ensure_shadow_atlas(int p_width, int p_height) {
    if (m_shadow_atlas.is_valid()) {
        int w = RenderingServer::get_singleton()->texture_get_width(m_shadow_atlas);
        int h = RenderingServer::get_singleton()->texture_get_height(m_shadow_atlas);
        if (w == p_width && h == p_height) return;
        RenderingServer::get_singleton()->free(m_shadow_atlas);
        m_shadow_atlas = RID();
    }
    m_shadow_atlas = RenderingServer::get_singleton()->texture_2d_create();
    RenderingServer::get_singleton()->texture_2d_initialize(m_shadow_atlas, p_width, p_height, Image::FORMAT_DEPTH);
    m_lighting_pass.set_shadow_map(m_shadow_atlas);
}

// ----------------------------------------------------------------------------
// Per‑frame update: camera, motion vectors, G‑buffer, dispatch compute shader.
// ----------------------------------------------------------------------------
void LightingManager::update(double p_delta) {
    if (!m_initialized) return;

    Viewport *viewport = get_viewport();
    if (!viewport) return;
    Camera3D *camera = viewport->get_camera_3d();
    if (!camera) return;

    int w = viewport->get_visible_rect().size.width;
    int h = viewport->get_visible_rect().size.height;
    if (w != m_viewport_width || h != m_viewport_height) {
        m_viewport_width = w;
        m_viewport_height = h;
        // Recreate shadow atlas if needed
        _ensure_shadow_atlas(2048, 2048);
    }

    // Update lighting pass parameters
    m_lighting_pass.set_params(m_params);
    m_lighting_pass.set_camera(camera);
    // In a real engine, you would obtain the environment cubemap from the sky.
    // For now, we pass an empty RID; the shader will fallback to a default black cubemap.
    m_lighting_pass.set_environment_cubemap(RID());

    // Retrieve G‑buffer textures from the viewport's rendering server (simplified)
    // In practice, these would be the color, depth, normal, and motion textures from the previous frame.
    // We assume they are already rendered and available as RIDs.
    // For simulation, we create dummy RIDs (not recommended for real use).
    RID color_tex = RenderingServer::get_singleton()->viewport_get_texture(viewport->get_viewport_rid());
    RID depth_tex = RID();   // would be from a separate render pass
    RID normal_tex = RID();
    RID motion_tex = RID();

    // Execute the lighting pass compute shader
    RID final_color;
    m_lighting_pass.execute(color_tex, depth_tex, normal_tex, motion_tex, final_color);

    // Set the final color as the viewport's texture (or composite later)
    // In a real engine, you would blend with the existing framebuffer.
    // Here we just output to the viewport (simplified).
    RenderingServer::get_singleton()->viewport_set_texture(viewport->get_viewport_rid(), final_color);
}

// ----------------------------------------------------------------------------
// Helper: update camera matrices and motion vectors (called internally by lighting_pass)
// ----------------------------------------------------------------------------
void LightingManager::_update_camera_and_buffers(Camera3D *p_camera) {
    // This method would be called inside m_lighting_pass.execute, but we keep it here for completeness.
    // The actual uniform updates are handled by LightingPass::_update_camera_uniforms.
}