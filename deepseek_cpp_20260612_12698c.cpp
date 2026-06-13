// Name : lighting enhancement
// File : scene/3d/lightmap_gi_ext.h 21 of 60
// Description : Extended lightmap GI node with baking controls, texel density,
//               dynamic lightmap updates, and RenderingServer synchronization.
#pragma once

#include "scene/3d/lightmap_gi.h"
#include "servers/rendering_server.h"

class LightmapGIExt : public LightmapGI {
    GDCLASS(LightmapGIExt, LightmapGI);

public:
    LightmapGIExt();
    ~LightmapGIExt();

    // ------------------------------------------------------------------------
    // Baking parameters (quality, resolution, bounces)
    // ------------------------------------------------------------------------
    void set_texel_per_unit(int p_texels);
    int get_texel_per_unit() const;
    void set_bounce_count(int p_bounces);
    int get_bounce_count() const;
    void set_bake_shadows(bool p_shadows);
    bool get_bake_shadows() const;
    void set_bake_emissive(bool p_emissive);
    bool get_bake_emissive() const;

    // ------------------------------------------------------------------------
    // Lightmap data (texture atlas and UV scale)
    // ------------------------------------------------------------------------
    RID get_lightmap_atlas_texture() const;
    void get_uv_scale(float &r_scale_u, float &r_scale_v) const;

    // ------------------------------------------------------------------------
    // Bake control (asynchronous)
    // ------------------------------------------------------------------------
    void bake_async();
    void cancel_bake();
    bool is_baking() const;
    float get_bake_progress() const;

    // ------------------------------------------------------------------------
    // Real-time updates (for dynamic GI)
    // ------------------------------------------------------------------------
    void set_gi_mode(int p_mode);
    int get_gi_mode() const;
    void set_gi_contribution(float p_amount);
    float get_gi_contribution() const;

    // ------------------------------------------------------------------------
    // Rendering server synchronization
    // ------------------------------------------------------------------------
    void sync_lightmap();

private:
    struct Impl;
    Impl *pimpl;
};