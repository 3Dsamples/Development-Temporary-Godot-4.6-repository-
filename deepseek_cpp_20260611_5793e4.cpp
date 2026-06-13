// reflection_probe.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <functional>

namespace lighting {

// ============================================================================
// ReflectionProbe – captures a cubemap of the environment for reflections.
// Supports both static (baked) and dynamic (real‑time) updates, box/blend
// distances, interior/ambient mode, and global illumination contribution.
// ============================================================================

enum class ReflectionUpdateMode : uint8_t {
    STATIC,         // capture once (baked)
    DYNAMIC,        // update every frame (expensive)
    ONCE_THEN_FREE  // capture once then stop (for dynamic but periodic)
};

enum class ReflectionProbeShape : uint8_t {
    BOX,
    SPHERE
};

class ReflectionProbe : public Node3D {
public:
    ReflectionProbe();
    ~ReflectionProbe();

    // ------------------------------------------------------------------------
    // Capture control
    // ------------------------------------------------------------------------
    void set_update_mode(ReflectionUpdateMode mode);
    ReflectionUpdateMode get_update_mode() const;
    void set_update_frequency(float frames_per_second); // for periodic dynamic update
    float get_update_frequency() const;
    void capture();          // force immediate capture (synchronous)
    void capture_async();    // background capture
    bool is_capturing() const;

    // ------------------------------------------------------------------------
    // Resolution and quality
    // ------------------------------------------------------------------------
    void set_resolution(int resolution);  // e.g., 128, 256, 512
    int get_resolution() const;
    void set_roughness_filtering(bool enable);
    bool is_roughness_filtering_enabled() const;

    // ------------------------------------------------------------------------
    // Influence shape and extents
    // ------------------------------------------------------------------------
    void set_shape(ReflectionProbeShape shape);
    ReflectionProbeShape get_shape() const;
    void set_extents(const double* extents); // half extents for box, radius for sphere
    void get_extents(double* out_extents) const;

    // ------------------------------------------------------------------------
    // Blend and distance falloff
    // ------------------------------------------------------------------------
    void set_blend_distance(double distance);
    double get_blend_distance() const;
    void set_intensity(float intensity);
    float get_intensity() const;
    void set_ambient_color(const float* rgb);
    void get_ambient_color(float* out_rgb) const;
    void set_ambient_mode(bool interior); // interior = use probe as ambient source
    bool get_ambient_mode() const;

    // ------------------------------------------------------------------------
    // Culling mask (which objects are reflected)
    // ------------------------------------------------------------------------
    void set_reflection_mask(uint32_t mask);
    uint32_t get_reflection_mask() const;

    // ------------------------------------------------------------------------
    // Lighting & GI (probe can contribute to GI as an emissive volume)
    // ------------------------------------------------------------------------
    void set_gi_contribution(float amount);
    float get_gi_contribution() const;
    void set_emissive(const float* color, float intensity);
    void get_emissive(float* out_color, float& out_intensity) const;
    void set_cast_shadow(bool cast);    // for debug visualization
    bool get_cast_shadow() const;

    // ------------------------------------------------------------------------
    // Debug visualization
    // ------------------------------------------------------------------------
    void set_debug_visible(bool visible);
    bool is_debug_visible() const;
    void set_debug_color(const float* rgb);
    void get_debug_color(float* out_rgb) const;

    // ------------------------------------------------------------------------
    // Access captured cubemap texture (for rendering server)
    // ------------------------------------------------------------------------
    int64_t get_cubemap_texture_rid() const; // RenderingServer texture ID

    // ------------------------------------------------------------------------
    // Node overrides
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;
    void process(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting