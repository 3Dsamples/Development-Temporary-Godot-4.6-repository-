// reflection_probe_3d.h
#pragma once

#include "visual_instance_3d.h"
#include <memory>
#include <array>

namespace lighting {

// ============================================================================
// ReflectionProbe3D – captures a dynamic cubemap for local reflections
// Supports real‑time updates, blend distance, and GI influence.
// ============================================================================

enum class ReflectionProbeUpdateMode : uint8_t {
    ALWAYS,          // every frame
    ONCE,            // capture once then static
    ON_MOVE          // recapture when probe moves
};

class ReflectionProbe3D : public VisualInstance3D {
public:
    ReflectionProbe3D();
    ~ReflectionProbe3D();

    // ------------------------------------------------------------------------
    // Capture settings
    // ------------------------------------------------------------------------
    void set_update_mode(ReflectionProbeUpdateMode mode);
    ReflectionProbeUpdateMode get_update_mode() const;
    void set_resolution(int resolution);        // cubemap face size
    int get_resolution() const;
    void set_update_rate(float hz);             // updates per second (for ALWAYS)
    float get_update_rate() const;

    // ------------------------------------------------------------------------
    // Influence volume (blend distance, intensity)
    // ------------------------------------------------------------------------
    void set_extents(const Vector3& extents);   // box size
    Vector3 get_extents() const;
    void set_origin_offset(const Vector3& offset);
    Vector3 get_origin_offset() const;
    void set_intensity(float intensity);
    float get_intensity() const;
    void set_blend_distance(float distance);    // linear blend near edges
    float get_blend_distance() const;

    // ------------------------------------------------------------------------
    // Culling mask (which layers to capture)
    // ------------------------------------------------------------------------
    void set_cull_mask(uint32_t mask);
    uint32_t get_cull_mask() const;

    // ------------------------------------------------------------------------
    // Global illumination contribution (influence on light probes)
    // ------------------------------------------------------------------------
    void set_gi_importance(float importance);  // 0 = none, 1 = full
    float get_gi_importance() const;

    // ------------------------------------------------------------------------
    // Shadow casting from captured cubemap (affects reflections on objects)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast);
    bool get_cast_shadow() const;
    void set_ambient_contribution(float amount); // environment light from probe

    // ------------------------------------------------------------------------
    // Real‑time capture (call each frame if needed)
    // ------------------------------------------------------------------------
    void update_cubemap();          // force capture now
    bool is_cubemap_ready() const;
    int64_t get_cubemap_texture_rid() const;

    // ------------------------------------------------------------------------
    // Rendering server synchronization
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;

    // ------------------------------------------------------------------------
    // For dynamic reflection of moving objects (character reflection)
    // ------------------------------------------------------------------------
    void set_include_dynamic_objects(bool include);
    bool get_include_dynamic_objects() const;

protected:
    void _transform_changed() override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting