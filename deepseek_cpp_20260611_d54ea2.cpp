// lightmap_gi.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <unordered_map>

namespace lighting {

// ============================================================================
// LightmapGI – bakes global illumination into texture maps.
// Supports direct + indirect lighting, emissive surfaces, multiple bounces,
// and quality settings. Lightmaps are applied to static geometry.
// Integrates with the rendering server and provides real‑time GI probes.
// ============================================================================

class LightmapGI : public Node3D {
public:
    LightmapGI();
    ~LightmapGI();

    // ------------------------------------------------------------------------
    // Baking control
    // ------------------------------------------------------------------------
    void bake(bool use_denoiser = true);
    void clear();
    bool is_baked() const;

    // ------------------------------------------------------------------------
    // Quality and performance
    // ------------------------------------------------------------------------
    void set_quality(int level);   // 0 = low, 1 = medium, 2 = high, 3 = ultra
    int get_quality() const;
    void set_bounces(int bounces);
    int get_bounces() const;
    void set_texel_per_unit(float texels);
    float get_texel_per_unit() const;
    void set_max_texture_size(int size);
    int get_max_texture_size() const;
    void set_use_denoiser(bool use);
    bool get_use_denoiser() const;

    // ------------------------------------------------------------------------
    // Lightmap data access (for rendering server)
    // ------------------------------------------------------------------------
    int get_lightmap_count() const;
    int64_t get_lightmap_texture(int index) const;
    void get_lightmap_uv_scale(int index, float& scale_x, float& scale_y) const;

    // ------------------------------------------------------------------------
    // Probe support (lightmap probes for dynamic objects)
    // ------------------------------------------------------------------------
    void set_probe_bake(bool enabled);
    bool is_probe_bake_enabled() const;
    int get_probe_count() const;
    void get_probe_position(int index, double* out_pos) const;
    void get_probe_sh_coeffs(int index, float* out_coeffs) const; // 3x9 floats

    // ------------------------------------------------------------------------
    // Lighting & environment override (for baking)
    // ------------------------------------------------------------------------
    void set_env_ambient(const float* color, float intensity);
    void get_env_ambient(float* out_color, float& out_intensity) const;
    void set_env_directional(const float* color, float intensity, const double* direction);
    void get_env_directional(float* out_color, float& out_intensity, double* out_direction) const;
    void set_env_sky_rotation(const double* euler_rad);
    void get_env_sky_rotation(double* out_euler_rad) const;

    // ------------------------------------------------------------------------
    // Node overrides
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting