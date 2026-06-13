// Name : lighting enhancement
// File : scene/3d/light_3d_ext.h 3 of 60
// Description : Extended light node with advanced shadow parameters, IBL integration,
//               dynamic GI modes, and real‑time rendering server updates.
#pragma once

#include "scene/3d/light_3d.h"
#include "servers/rendering_server.h"

class Light3DExt : public Light3D {
    GDCLASS(Light3DExt, Light3D);

public:
    Light3DExt();
    ~Light3DExt();

    // ------------------------------------------------------------------------
    // Light parameters (override from Light3D)
    // ------------------------------------------------------------------------
    void set_color(const Color &p_color);
    void set_param(int p_param, float p_value); // param: energy, range, attenuation, etc.
    void set_shadow_enabled(bool p_enabled);
    void set_shadow_bias(float p_bias);
    void set_shadow_normal_bias(float p_normal_bias);

    // ------------------------------------------------------------------------
    // Real‑time dynamic resolution for shadows (performance)
    // ------------------------------------------------------------------------
    void set_shadow_map_resolution(int p_resolution);
    int get_shadow_map_resolution() const;

    // ------------------------------------------------------------------------
    // Global illumination contribution
    // ------------------------------------------------------------------------
    void set_gi_mode(int p_mode) override;
    void set_gi_contribution(float p_amount) override;

    // ------------------------------------------------------------------------
    // Reflection probe influence (for IBL)
    // ------------------------------------------------------------------------
    void set_reflection_probe_intensity(float p_intensity);
    float get_reflection_probe_intensity() const;

    // ------------------------------------------------------------------------
    // Rendering server sync (push all parameters to RenderingServer)
    // ------------------------------------------------------------------------
    void sync_light_params();
    void sync_shadow_params();

    // ------------------------------------------------------------------------
    // Light culling integration (used by RenderingServer)
    // ------------------------------------------------------------------------
    void set_cull_mask(uint32_t p_mask);
    uint32_t get_cull_mask() const;

private:
    struct Impl;
    Impl *pimpl;
};