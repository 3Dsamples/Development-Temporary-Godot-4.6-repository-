// Name : lighting enhancement
// File : scene/3d/geometry_instance_3d_ext.h 9 of 60
// Description : Extended geometry instance with material overrides, shadow settings,
//               LOD, and full rendering server synchronization for meshes.
#pragma once

#include "scene/3d/geometry_instance_3d.h"
#include "servers/rendering_server.h"

class GeometryInstance3DExt : public GeometryInstance3D {
    GDCLASS(GeometryInstance3DExt, GeometryInstance3D);

public:
    GeometryInstance3DExt();
    ~GeometryInstance3DExt();

    // ------------------------------------------------------------------------
    // Material overrides (per surface and overall)
    // ------------------------------------------------------------------------
    void set_material_override(const RID &p_material) override;
    RID get_material_override() const override;
    void set_surface_material(int p_surface, const RID &p_material);
    RID get_surface_material(int p_surface) const;
    void clear_surface_material(int p_surface);

    // ------------------------------------------------------------------------
    // Shadow and GI settings (per geometry)
    // ------------------------------------------------------------------------
    void set_cast_shadow(int p_cast_shadow) override; // SHADOW_CASTING_SETTING values
    int get_cast_shadow() const override;
    void set_receive_shadow(bool p_receive);
    bool get_receive_shadow() const;

    // ------------------------------------------------------------------------
    // Distance fade (visibility fade)
    // ------------------------------------------------------------------------
    void set_distance_fade_enabled(bool p_enabled);
    bool is_distance_fade_enabled() const;
    void set_distance_fade_range(float p_min, float p_max, float p_length = 0.0f);
    void get_distance_fade_range(float &p_min, float &p_max, float &p_length) const;

    // ------------------------------------------------------------------------
    // Level of Detail (LOD) for meshes
    // ------------------------------------------------------------------------
    void set_lod_distance(float p_distance);
    float get_lod_distance() const;

    // ------------------------------------------------------------------------
    // Rendering server synchronization (push all parameters)
    // ------------------------------------------------------------------------
    void sync_geometry();
    void sync_materials();
    void sync_shadow_params();
    void sync_fade();

private:
    struct Impl;
    Impl *pimpl;
};