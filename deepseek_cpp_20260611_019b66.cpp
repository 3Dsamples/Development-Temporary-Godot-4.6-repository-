// lightmap_probe.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// LightmapProbe – captures incident lighting at a point and stores it as
// spherical harmonics coefficients (or cubemap) for use in lightmapped GI.
// Probes can be placed in static scenes to provide indirect lighting for
// dynamic objects (character lighting, material sampling). Supports real‑time
// updates when lights or static geometry change (re‑baking required).
// ============================================================================

enum class ProbeCaptureMode : uint8_t {
    SH9,          // 9 spherical harmonics coefficients (3 bands)
    SH4,          // 4 coefficients (2 bands, faster)
    CUBEMAP_32,   // 32x32 cubemap
    CUBEMAP_128
};

struct LightmapProbeData {
    double position[3];
    ProbeCaptureMode mode;
    std::vector<float> coefficients; // SH coefficients (27 floats for RGB SH9)
    // For cubemap mode, we could store compressed textures.
};

class LightmapProbe : public Node3D {
public:
    LightmapProbe();
    ~LightmapProbe();

    // ------------------------------------------------------------------------
    // Capture control
    // ------------------------------------------------------------------------
    void set_capture_mode(ProbeCaptureMode mode);
    ProbeCaptureMode get_capture_mode() const;
    void set_capture_extents(const double* extents); // sphere or box size for influence
    void get_capture_extents(double* out_extents) const;
    void set_interior(bool interior); // if true, only affects objects inside extents
    bool is_interior() const;

    // ------------------------------------------------------------------------
    // Capture process (synchronous or asynchronous)
    // ------------------------------------------------------------------------
    void capture();                     // immediate capture (may block)
    void capture_async();               // starts capture in background
    bool is_capturing() const;
    void cancel_capture();

    // ------------------------------------------------------------------------
    // Access captured data
    // ------------------------------------------------------------------------
    const LightmapProbeData* get_captured_data() const;
    void get_irradiance(const double* direction, float* out_color) const; // sample SH

    // ------------------------------------------------------------------------
    // Influence on dynamic objects (GI blending)
    // ------------------------------------------------------------------------
    void set_influence_radius(double radius);
    double get_influence_radius() const;
    void set_influence_weight(float weight);
    float get_influence_weight() const;
    void set_baked_gi_contribution(float amount);
    float get_baked_gi_contribution() const;

    // ------------------------------------------------------------------------
    // Rendering server sync (upload probe data to GPU)
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;
    void process(double delta) override;

    // ------------------------------------------------------------------------
    // Gizmo / debug visualization
    // ------------------------------------------------------------------------
    void set_debug_visible(bool visible);
    bool is_debug_visible() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting