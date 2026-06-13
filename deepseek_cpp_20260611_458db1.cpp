// lightmap_gi.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <functional>

namespace lighting {

// ============================================================================
// LightmapGI – bakes global illumination (indirect lighting, shadows) into
// lightmap textures. Supports real‑time updates when lights or static geometry
// change (progressive refinement). Integrates with the scene for dynamic GI.
// ============================================================================

enum class LightmapQuality : uint8_t {
    LOW = 32,       // texels per unit
    MEDIUM = 64,
    HIGH = 128,
    ULTRA = 256
};

struct LightmapData {
    int width = 0, height = 0;
    std::vector<float> data_r;   // red channel (linear)
    std::vector<float> data_g;
    std::vector<float> data_b;
    int atlas_index = -1;
};

class LightmapGI : public Node3D {
public:
    LightmapGI();
    ~LightmapGI();

    // ------------------------------------------------------------------------
    // Baking control
    // ------------------------------------------------------------------------
    void set_quality(LightmapQuality quality);
    LightmapQuality get_quality() const;
    void set_bounce_count(int bounces);
    int get_bounce_count() const;
    void set_texel_per_unit(int texels);   // overrides quality
    int get_texel_per_unit() const;
    void set_bake_shadows(bool shadows);
    bool get_bake_shadows() const;
    void set_bake_emissive(bool emissive);
    bool get_bake_emissive() const;

    // ------------------------------------------------------------------------
    // Baking process (asynchronous)
    // ------------------------------------------------------------------------
    void bake();                                 // starts baking (async)
    bool is_baking() const;
    void cancel_bake();
    float get_bake_progress() const;             // 0..1
    void set_bake_completed_callback(std::function<void()> callback);

    // ------------------------------------------------------------------------
    // Lightmap textures access (for rendering server)
    // ------------------------------------------------------------------------
    int get_lightmap_atlas_count() const;
    int64_t get_lightmap_texture_id(int atlas_index) const; // RenderingServer texture RID
    void get_lightmap_uv_scale(int atlas_index, double* out_scale) const; // UV scale for meshes

    // ------------------------------------------------------------------------
    // Dynamic GI (light probes from baked data)
    // ------------------------------------------------------------------------
    void set_probe_update_frequency(float fps);
    float get_probe_update_frequency() const;
    void set_use_light_probes(bool use);
    bool is_using_light_probes() const;
    int get_probe_grid_resolution() const;

    // ------------------------------------------------------------------------
    // Lighting contribution (emissive from lightmaps to real‑time GI)
    // ------------------------------------------------------------------------
    void set_gi_contribution(float amount);
    float get_gi_contribution() const;

    // ------------------------------------------------------------------------
    // Rendering server synchronization
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;
    void process(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting