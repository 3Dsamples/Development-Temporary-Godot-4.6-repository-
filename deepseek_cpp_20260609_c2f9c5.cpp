// geometry_instance_3d.h
#pragma once

#include "visual_instance_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// GeometryInstance3D – base for all visible 3D geometry
// Adds material slots, shadow casting settings, and visibility ranges.
// ============================================================================

class GeometryInstance3D : public VisualInstance3D {
public:
    GeometryInstance3D();
    ~GeometryInstance3D();

    // ------------------------------------------------------------------------
    // Material system (multiple materials per surface)
    // ------------------------------------------------------------------------
    void set_material(int surface_idx, const char* material_path);
    void clear_material(int surface_idx);
    void clear_all_materials();
    bool has_material(int surface_idx) const;
    int get_material_count() const;

    // ------------------------------------------------------------------------
    // Material override (replaces all surface materials)
    // ------------------------------------------------------------------------
    void set_material_override(const char* material_path) override;
    void clear_material_override() override;

    // ------------------------------------------------------------------------
    // Shadow casting and receiving
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast);
    bool get_cast_shadow() const;
    void set_receive_shadow(bool receive);
    bool get_receive_shadow() const;
    void set_double_sided_shadow(bool double_sided);
    bool is_double_sided_shadow() const;

    // ------------------------------------------------------------------------
    // Visibility range (distance-based fade)
    // ------------------------------------------------------------------------
    void set_visibility_range(float begin, float end, float fade_margin = 0.0f);
    void get_visibility_range(float& begin, float& end, float& fade_margin) const;
    void set_visibility_range_fade_mode(int mode); // 0=disabled, 1=opacity, 2=scale
    int get_visibility_range_fade_mode() const;

    // ------------------------------------------------------------------------
    // LOD overrides (per‑instance)
    // ------------------------------------------------------------------------
    void set_lod_bias(float bias);
    float get_lod_bias() const;
    void set_lod_thresholds(const float* thresholds, int count); // replaces default

    // ------------------------------------------------------------------------
    // GI / lightmap settings
    // ------------------------------------------------------------------------
    void set_lightmap_scale(float scale);
    float get_lightmap_scale() const;
    void set_lightmap_index(int index);
    int get_lightmap_index() const;

    // ------------------------------------------------------------------------
    // Per‑object culling toggle
    // ------------------------------------------------------------------------
    void set_ignore_frustum_culling(bool ignore);
    bool get_ignore_frustum_culling() const;

    // ------------------------------------------------------------------------
    // Render server synchronization
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;

protected:
    // Called when materials change
    virtual void _materials_updated();

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting