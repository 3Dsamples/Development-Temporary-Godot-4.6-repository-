// lightmap_probe.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// LightmapProbe – captures indirect lighting at a single point in space
// and stores spherical harmonics (SH). Used for dynamic objects that move
// through a baked scene. Probes can be placed manually or automatically
// during lightmap baking.
// ============================================================================

class LightmapProbe : public Node3D {
public:
    LightmapProbe();
    ~LightmapProbe();

    // ------------------------------------------------------------------------
    // SH data access
    // ------------------------------------------------------------------------
    void set_sh_coeffs(const float* coeffs); // 3x9 floats (RGB SH)
    void get_sh_coeffs(float* out_coeffs) const;

    // ------------------------------------------------------------------------
    // Influence radius (for blending with other probes)
    // ------------------------------------------------------------------------
    void set_radius(double radius);
    double get_radius() const;

    // ------------------------------------------------------------------------
    // Baked data (set after baking)
    // ------------------------------------------------------------------------
    void set_baked_data(const float* coeffs, double radius);
    bool is_baked() const;

    // ------------------------------------------------------------------------
    // Lighting contributions to environment (probes can emit light)
    // ------------------------------------------------------------------------
    void set_emissive(const float* color, float intensity);
    void get_emissive(float* out_color, float& out_intensity) const;

    // ------------------------------------------------------------------------
    // Node overrides
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting