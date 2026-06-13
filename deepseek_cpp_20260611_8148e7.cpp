// world_environment.cpp
#include "world_environment.h"
#include <cstring>
#include <cmath>

namespace lighting {

// ============================================================================
// WorldEnvironment implementation
// ============================================================================
struct WorldEnvironment::Impl {
    BackgroundMode background_mode = BackgroundMode::SKY;
    float background_color[3] = {0.0f, 0.0f, 0.0f};

    EnvironmentSky sky;
    AmbientLight ambient;
    Fog fog;
    Tonemap tonemap;
    Glow glow;
    Adjustment adjustment;

    // Flag to re‑apply to RenderingServer
    bool dirty = true;

    // In a real engine, we would have a handle to the environment resource
    int64_t environment_rid = -1;
    int64_t sky_rid = -1;

    ~Impl() {
        // Free rendering server resources
    }
};

WorldEnvironment::WorldEnvironment() : pimpl(std::make_unique<Impl>()) {
    // Set default values
    pimpl->ambient.color[0] = 0.2f; pimpl->ambient.color[1] = 0.2f; pimpl->ambient.color[2] = 0.2f;
    pimpl->ambient.energy = 1.0f;
    pimpl->ambient.use_sky_contribution = true;
    pimpl->ambient.sky_contribution = 0.5f;
    pimpl->tonemap.mode = 0;   // ACES
    pimpl->tonemap.exposure = 1.0f;
    pimpl->tonemap.white = 1.0f;
    pimpl->adjustment.brightness = 0.0f;
    pimpl->adjustment.contrast = 1.0f;
    pimpl->adjustment.saturation = 1.0f;
}
WorldEnvironment::~WorldEnvironment() = default;

void WorldEnvironment::set_background_mode(BackgroundMode mode) {
    pimpl->background_mode = mode;
    pimpl->dirty = true;
}
BackgroundMode WorldEnvironment::get_background_mode() const { return pimpl->background_mode; }
void WorldEnvironment::set_background_color(const float* rgb) {
    memcpy(pimpl->background_color, rgb, 3*sizeof(float));
    pimpl->dirty = true;
}
void WorldEnvironment::get_background_color(float* out_rgb) const {
    memcpy(out_rgb, pimpl->background_color, 3*sizeof(float));
}

void WorldEnvironment::set_sky(const EnvironmentSky& sky) {
    pimpl->sky = sky;
    pimpl->dirty = true;
}
const EnvironmentSky& WorldEnvironment::get_sky() const { return pimpl->sky; }

void WorldEnvironment::set_ambient_light(const AmbientLight& ambient) {
    pimpl->ambient = ambient;
    pimpl->dirty = true;
}
const AmbientLight& WorldEnvironment::get_ambient_light() const { return pimpl->ambient; }

void WorldEnvironment::set_fog(const Fog& fog) {
    pimpl->fog = fog;
    pimpl->dirty = true;
}
const Fog& WorldEnvironment::get_fog() const { return pimpl->fog; }

void WorldEnvironment::set_tonemap(const Tonemap& tonemap) {
    pimpl->tonemap = tonemap;
    pimpl->dirty = true;
}
const Tonemap& WorldEnvironment::get_tonemap() const { return pimpl->tonemap; }

void WorldEnvironment::set_glow(const Glow& glow) {
    pimpl->glow = glow;
    pimpl->dirty = true;
}
const Glow& WorldEnvironment::get_glow() const { return pimpl->glow; }

void WorldEnvironment::set_adjustment(const Adjustment& adj) {
    pimpl->adjustment = adj;
    pimpl->dirty = true;
}
const Adjustment& WorldEnvironment::get_adjustment() const { return pimpl->adjustment; }

void WorldEnvironment::override_from(const WorldEnvironment& other, float blend_factor) {
    // Linear blend of each parameter (simplified)
    blend_factor = std::clamp(blend_factor, 0.0f, 1.0f);
    float inv = 1.0f - blend_factor;

    // Background color
    for (int i=0;i<3;++i) {
        pimpl->background_color[i] = pimpl->background_color[i] * inv + other.pimpl->background_color[i] * blend_factor;
    }
    // Ambient color and energy
    for (int i=0;i<3;++i) {
        pimpl->ambient.color[i] = pimpl->ambient.color[i] * inv + other.pimpl->ambient.color[i] * blend_factor;
    }
    pimpl->ambient.energy = pimpl->ambient.energy * inv + other.pimpl->ambient.energy * blend_factor;
    pimpl->ambient.sky_contribution = pimpl->ambient.sky_contribution * inv + other.pimpl->ambient.sky_contribution * blend_factor;
    // Fog density
    pimpl->fog.density = pimpl->fog.density * inv + other.pimpl->fog.density * blend_factor;
    // Tonemap exposure
    pimpl->tonemap.exposure = pimpl->tonemap.exposure * inv + other.pimpl->tonemap.exposure * blend_factor;
    pimpl->tonemap.white = pimpl->tonemap.white * inv + other.pimpl->tonemap.white * blend_factor;
    // Glow intensity
    pimpl->glow.intensity = pimpl->glow.intensity * inv + other.pimpl->glow.intensity * blend_factor;
    // Adjustment
    pimpl->adjustment.brightness = pimpl->adjustment.brightness * inv + other.pimpl->adjustment.brightness * blend_factor;
    pimpl->adjustment.contrast = pimpl->adjustment.contrast * inv + other.pimpl->adjustment.contrast * blend_factor;
    pimpl->adjustment.saturation = pimpl->adjustment.saturation * inv + other.pimpl->adjustment.saturation * blend_factor;

    pimpl->dirty = true;
}

void WorldEnvironment::apply_environment() {
    if (!pimpl->dirty) return;

    // In a real engine, we would create or update the environment resource.
    if (pimpl->environment_rid == -1) {
        // pimpl->environment_rid = RenderingServer::environment_create();
    }
    // Set background mode and color
    // RenderingServer::environment_set_background(pimpl->environment_rid, pimpl->background_mode, pimpl->background_color);
    // Set sky if present
    if (pimpl->sky.mode != SkyMode::NONE) {
        if (pimpl->sky_rid == -1) {
            // pimpl->sky_rid = RenderingServer::sky_create();
        }
        // RenderingServer::sky_set_texture(pimpl->sky_rid, pimpl->sky.texture_rid, pimpl->sky.mode);
        // RenderingServer::environment_set_sky(pimpl->environment_rid, pimpl->sky_rid);
    }
    // Set ambient light
    // RenderingServer::environment_set_ambient_light(pimpl->environment_rid, pimpl->ambient.color, pimpl->ambient.energy, pimpl->ambient.use_sky_contribution, pimpl->ambient.sky_contribution);
    // Set fog
    // RenderingServer::environment_set_fog(pimpl->environment_rid, pimpl->fog.enabled, pimpl->fog.density, pimpl->fog.height, pimpl->fog.height_falloff, pimpl->fog.start_distance, pimpl->fog.end_distance, pimpl->fog.max_radius, pimpl->fog.fog_sky_affect);
    // Set tonemap
    // RenderingServer::environment_set_tonemap(pimpl->environment_rid, pimpl->tonemap.mode, pimpl->tonemap.exposure, pimpl->tonemap.white);
    // Set glow
    // RenderingServer::environment_set_glow(pimpl->environment_rid, pimpl->glow.enabled, pimpl->glow.intensity, pimpl->glow.strength, pimpl->glow.blend_mode, pimpl->glow.hdr_threshold, pimpl->glow.hdr_scale, pimpl->glow.bloom, pimpl->glow.levels);
    // Set adjustment
    // RenderingServer::environment_set_adjustment(pimpl->environment_rid, pimpl->adjustment.enabled, pimpl->adjustment.brightness, pimpl->adjustment.contrast, pimpl->adjustment.saturation, pimpl->adjustment.color_correction_texture);
    // Finally, assign this environment to the current world (SceneTree)
    // RenderingServer::viewport_set_environment(viewport, pimpl->environment_rid);

    pimpl->dirty = false;
}

void WorldEnvironment::ready() {
    Node3D::ready();
    apply_environment();
}

void WorldEnvironment::process(double delta) {
    Node3D::process(delta);
    // Re‑apply if any parameter changes (dirty flag set)
    if (pimpl->dirty) {
        apply_environment();
    }
}

void WorldEnvironment::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    // Nothing else; environment settings are applied globally via RenderingServer.
}

} // namespace lighting