// Name : lighting enhancement
// File : scene/3d/visual_instance_3d_ext.h 7 of 60
// Description : Extended visual instance with LOD, bounding box culling,
//               shadow casting, GI modes, and full rendering server synchronization.
#pragma once

#include "scene/3d/visual_instance_3d.h"
#include "servers/rendering_server.h"

class VisualInstance3DExt : public VisualInstance3D {
    GDCLASS(VisualInstance3DExt, VisualInstance3D);

public:
    VisualInstance3DExt();
    ~VisualInstance3DExt();

    // ------------------------------------------------------------------------
    // Base instance configuration (override from VisualInstance3D)
    // ------------------------------------------------------------------------
    void set_base(const RID &p_base) override;
    void set_transform(const Transform3D &p_transform) override;
    void set_visible(bool p_visible) override;
    void set_layer_mask(uint32_t p_layer_mask) override;

    // ------------------------------------------------------------------------
    // Shadow and GI settings
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool p_cast) override;
    bool get_cast_shadow() const override;
    void set_gi_mode(int p_mode) override;
    int get_gi_mode() const override;
    void set_gi_contribution(float p_amount) override;
    float get_gi_contribution() const override;
    void set_emissive(const Color &p_color, float p_intensity) override;
    Color get_emissive() const override;
    float get_emissive_intensity() const override;

    // ------------------------------------------------------------------------
    // Level of Detail (LOD) and distance culling
    // ------------------------------------------------------------------------
    void set_lod_bias(float p_bias);
    float get_lod_bias() const;
    void set_visibility_range(float p_min, float p_max, float p_fade_margin = 0.0f);
    void get_visibility_range(float &p_min, float &p_max, float &p_fade_margin) const;

    // ------------------------------------------------------------------------
    // Rendering server synchronization (push all parameters)
    // ------------------------------------------------------------------------
    void sync_instance();
    void sync_instance_transform();
    void sync_instance_visibility();
    void sync_instance_lighting();
    void sync_instance_lod();

    // ------------------------------------------------------------------------
    // Get render instance ID (for use by RenderingServer)
    // ------------------------------------------------------------------------
    RID get_instance_rid() const;

private:
    struct Impl;
    Impl *pimpl;
};