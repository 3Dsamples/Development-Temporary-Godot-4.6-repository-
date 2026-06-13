// world_environment.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// WorldEnvironment – global environment settings (background, ambient light,
// fog, tonemap, glow, etc.). When added to the scene, it overrides the
// default environment of the current world. Only one instance is active
// at a time (per world). This node does not cast shadows, but affects the
// lighting of the entire scene.
// ============================================================================

enum class BackgroundMode : uint8_t {
    SKY,
    COLOR,
    CLEAR_COLOR,
    CANVAS
};

enum class SkyMode : uint8_t {
    PROCEDURAL,
    PANORAMA,
    CUBEMAP,
    NONE
};

struct EnvironmentSky {
    SkyMode mode = SkyMode::NONE;
    int64_t texture_rid = -1;           // cubemap / panorama texture
    int64_t material_rid = -1;          // custom shader material
    // Procedural sky parameters (simplified)
    float sun_angle = 0.0f;
    float sun_intensity = 1.0f;
    float rayleigh_coefficient = 5.0f;
    float mie_coefficient = 2.0f;
    float mie_directional_g = 0.8f;
};

struct AmbientLight {
    bool enabled = true;
    float color[3] = {0.2f, 0.2f, 0.2f};       // constant color
    float energy = 1.0f;
    bool use_sky_contribution = true;           // mix with sky
    float sky_contribution = 0.5f;
    // Spherical harmonics (optional)
    std::vector<float> sh_coefficients;         // 27 floats (RGB SH9)
};

struct Fog {
    bool enabled = false;
    float density = 0.01f;                     // exponential fog
    float height = 0.0f;
    float height_falloff = 0.0f;
    float start_distance = 10.0f;
    float end_distance = 1000.0f;
    float max_radius = 1000.0f;
    float fog_sky_affect = 0.8f;
    float fog_density_texture = 0.0f;          // density texture contribution
};

struct Tonemap {
    bool enabled = true;
    int mode = 0;                              // 0=ACES,1=Filmic,2=Reinhard,3=Linear
    float exposure = 1.0f;
    float white = 1.0f;
};

struct Glow {
    bool enabled = false;
    float intensity = 0.8f;
    float strength = 0.8f;
    float blend_mode = 0;                      // 0=additive,1=screen,2=soft light
    float hdr_threshold = 0.0f;
    float hdr_scale = 1.0f;
    float bloom = 0.0f;
    std::vector<float> levels;                 // per mip level weights
};

struct Adjustment {
    bool enabled = true;
    float brightness = 0.0f;
    float contrast = 1.0f;
    float saturation = 1.0f;
    float color_correction = 0.0f;
    int64_t color_correction_texture = -1;     // 3D LUT
};

class WorldEnvironment : public Node3D {
public:
    WorldEnvironment();
    ~WorldEnvironment();

    // ------------------------------------------------------------------------
    // Background
    // ------------------------------------------------------------------------
    void set_background_mode(BackgroundMode mode);
    BackgroundMode get_background_mode() const;
    void set_background_color(const float* rgb);
    void get_background_color(float* out_rgb) const;

    // ------------------------------------------------------------------------
    // Sky
    // ------------------------------------------------------------------------
    void set_sky(const EnvironmentSky& sky);
    const EnvironmentSky& get_sky() const;

    // ------------------------------------------------------------------------
    // Ambient light
    // ------------------------------------------------------------------------
    void set_ambient_light(const AmbientLight& ambient);
    const AmbientLight& get_ambient_light() const;

    // ------------------------------------------------------------------------
    // Fog
    // ------------------------------------------------------------------------
    void set_fog(const Fog& fog);
    const Fog& get_fog() const;

    // ------------------------------------------------------------------------
    // Tonemap
    // ------------------------------------------------------------------------
    void set_tonemap(const Tonemap& tonemap);
    const Tonemap& get_tonemap() const;

    // ------------------------------------------------------------------------
    // Glow (bloom)
    // ------------------------------------------------------------------------
    void set_glow(const Glow& glow);
    const Glow& get_glow() const;

    // ------------------------------------------------------------------------
    // Color adjustment
    // ------------------------------------------------------------------------
    void set_adjustment(const Adjustment& adj);
    const Adjustment& get_adjustment() const;

    // ------------------------------------------------------------------------
    // Override all settings from another environment (for blending)
    // ------------------------------------------------------------------------
    void override_from(const WorldEnvironment& other, float blend_factor);

    // ------------------------------------------------------------------------
    // Apply to rendering server (call after any change)
    // ------------------------------------------------------------------------
    void apply_environment();

    // ------------------------------------------------------------------------
    // Node overrides
    // ------------------------------------------------------------------------
    void ready() override;
    void process(double delta) override;
    void synchronize_render_server(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting