// lightmap_gi.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// LightmapGI – bakes static global illumination into a texture atlas.
// Supports dynamic lighting updates (realtime GI with lightmap blending),
// directional lightmaps, and integrated with reflection probes for specular.
// High performance using GPU‑based baking (compute shaders) and runtime
// streaming of lightmap textures.
// ============================================================================

class LightmapGI : public Node3D {
public:
    LightmapGI();
    ~LightmapGI();

    // ------------------------------------------------------------------------
    // Baking control
    // ------------------------------------------------------------------------
    void set_bake_quality(int quality);   // 0 = low, 1 = medium, 2 = high, 3 = ultra
    int get_bake_quality() const;
    void set_bake_denoiser(bool enable);
    bool is_bake_denoiser_enabled() const;
    void set_bake_max_bounces(int bounces);
    int get_bake_max_bounces() const;
    void set_bake_light_resolution(int resolution); // texels per unit (e.g., 4)
    int get_bake_light_resolution() const;

    // ------------------------------------------------------------------------
    // Lightmap texture management
    // ------------------------------------------------------------------------
    void set_atlas_size(int size);       // power of two, e.g., 4096
    int get_atlas_size() const;
    void set_use_compression(bool enable);
    bool is_compression_enabled() const;
    void set_texture_format(int format); // 0 = RGB8, 1 = RGBA8, 2 = BC6H, 3 = BC7
    int get_texture_format() const;

    // ------------------------------------------------------------------------
    // Baking (non‑realtime, called from editor)
    // ------------------------------------------------------------------------
    void bake();                         // bakes current scene to lightmap
    bool is_baking() const;
    float get_bake_progress() const;
    void cancel_bake();
    void clear_lightmaps();              // remove all baked data

    // ------------------------------------------------------------------------
    // Runtime GI settings (applies to baked lightmaps)
    // ------------------------------------------------------------------------
    void set_gi_mode(int mode);          // 0 = disabled, 1 = static lightmap, 2 = dynamic (realtime blend)
    int get_gi_mode() const;
    void set_energy(float multiplier);
    float get_energy() const;
    void set_bias(float bias);
    float get_bias() const;
    void set_normal_bias(float bias);
    float get_normal_bias() const;
    void set_interior(bool interior);
    bool is_interior() const;

    // ------------------------------------------------------------------------
    // Directional lightmap (baked dominant direction)
    // ------------------------------------------------------------------------
    void set_directional_enabled(bool enable);
    bool is_directional_enabled() const;
    void set_directional_half_lobe(float angle);
    float get_directional_half_lobe() const;

    // ------------------------------------------------------------------------
    // Lightmap probes for indirect specular (optional)
    // ------------------------------------------------------------------------
    void set_probe_capture_enabled(bool enable);
    bool is_probe_capture_enabled() const;

    // ------------------------------------------------------------------------
    // Performance: dynamic level‑of‑detail for lightmap mipmaps
    // ------------------------------------------------------------------------
    void set_use_mipmaps(bool enable);
    bool are_mipmaps_enabled() const;

    // ------------------------------------------------------------------------
    // Node overrides (sync with render server)
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;
    void process(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting