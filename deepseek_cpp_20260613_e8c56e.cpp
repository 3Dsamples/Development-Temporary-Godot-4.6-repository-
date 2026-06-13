// Name : lighting enhancement updated
// File : lighting_manager.h 86 of 63
// Description : High-level manager that integrates GPU lighting techniques (SSR, SSGI, SSAO,
//               shadows, temporal denoising, variant textures) with Godot’s scene system.
//               Uses the compute shaders from lighting_compute.glsl and variant_texture.glsl.
#pragma once

#include "scene/3d/node_3d.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/world_environment.h"
#include "servers/rendering_server.h"
#include "lighting_pass.h"          // LightingPass, LightingPassParams
#include "texture_manager.h"        // TextureManager, VariantTextureParams

// ----------------------------------------------------------------------------
// Main lighting manager – attaches to a world and controls all real‑time GI effects
// ----------------------------------------------------------------------------
class LightingManager : public Node3D {
    GDCLASS(LightingManager, Node3D);

public:
    LightingManager();
    ~LightingManager();

    // ------------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------------
    void set_ssr_enabled(bool p_enabled);
    bool is_ssr_enabled() const;
    void set_ssgi_enabled(bool p_enabled);
    bool is_ssgi_enabled() const;
    void set_ssao_enabled(bool p_enabled);
    bool is_ssao_enabled() const;
    void set_shadows_enabled(bool p_enabled);
    bool are_shadows_enabled() const;
    void set_shadow_quality(int p_quality); // 0=low,1=medium,2=high
    void set_temporal_denoising(bool p_enabled);
    bool is_temporal_denoising_enabled() const;

    // ------------------------------------------------------------------------
    // Environment and resources
    // ------------------------------------------------------------------------
    void set_environment(const Ref<WorldEnvironment> &p_env);
    void set_shadow_map_size(int p_size); // per‑cascade resolution

    // ------------------------------------------------------------------------
    // Per‑frame update (must be called each frame)
    // ------------------------------------------------------------------------
    void update(double p_delta);

    // ------------------------------------------------------------------------
    // Access the internal texture manager (to customize missing texture behaviour)
    // ------------------------------------------------------------------------
    TextureManager &get_texture_manager();

protected:
    void _notification(int p_what);

private:
    LightingPass m_lighting_pass;
    TextureManager m_texture_manager;
    LightingPassParams m_params;
    Ref<WorldEnvironment> m_environment;
    RID m_shadow_atlas;          // combined CSM texture
    bool m_initialized = false;
    int m_viewport_width = 0;
    int m_viewport_height = 0;

    void _ensure_shadow_atlas(int p_width, int p_height);
    void _update_camera_and_buffers(Camera3D *p_camera);
};