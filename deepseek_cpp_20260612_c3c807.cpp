// Name : lighting enhancement
// File : scene/3d/lightmap_gi_ext.h 21 of 60
// Description : Extended lightmap GI node with baking controls, quality settings,
//               lightmap texture management, and real‑time RenderingServer sync.
#pragma once

#include "scene/3d/lightmap_gi.h"
#include "servers/rendering_server.h"

class LightmapGIExt : public LightmapGI {
    GDCLASS(LightmapGIExt, LightmapGI);

public:
    LightmapGIExt();
    ~LightmapGIExt();

    // ------------------------------------------------------------------------
    // Baking parameters
    // ------------------------------------------------------------------------
    void set_quality(int p_quality);            // 0 = low, 1 = medium, 2 = high, 3 = ultra
    int get_quality() const;
    void set_bounce_count(int p_bounces);
    int get_bounce_count() const;
    void set_texel_per_unit(float p_texels);
    float get_texel_per_unit() const;
    void set_bake_shadows(bool p_enabled);
    bool get_bake_shadows() const;
    void set_bake_emissive(bool p_enabled);
    bool get_bake_emissive() const;

    // ------------------------------------------------------------------------
    // Lightmap atlas management
    // ------------------------------------------------------------------------
    RID get_lightmap_atlas_texture() const;
    int get_lightmap_atlas_size() const;
    void set_atlas_resolution(int p_resolution);
    int get_atlas_resolution() const;

    // ------------------------------------------------------------------------
    // Real‑time updates (for dynamic lighting)
    // ------------------------------------------------------------------------
    void set_dynamic_update(bool p_enabled);
    bool is_dynamic_update() const;
    void set_update_frequency(float p_fps);
    float get_update_frequency() const;
    void request_update();                     // force re‑bake (async)

    // ------------------------------------------------------------------------
    // Baking process (asynchronous)
    // ------------------------------------------------------------------------
    void bake();                               // starts baking in background
    bool is_baking() const;
    float get_bake_progress() const;
    void cancel_bake();

    // ------------------------------------------------------------------------
    // Rendering server synchronization
    // ------------------------------------------------------------------------
    void sync_lightmap();

private:
    struct Impl;
    Impl *pimpl;
};