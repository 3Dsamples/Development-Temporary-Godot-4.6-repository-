// lightmap_probe.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// LightmapProbe – captures lighting information at a specific position
// to be used with LightmapGI. Stores incident radiance as spherical harmonics
// (SH) and optionally a cubemap for specular reflections.
// Probes are placed in the scene and baked together with lightmaps.
// ============================================================================

class LightmapProbe : public Node3D {
public:
    LightmapProbe();
    ~LightmapProbe();

    // ------------------------------------------------------------------------
    // Probe capture parameters
    // ------------------------------------------------------------------------
    void set_capture_size(int resolution); // cubemap face resolution (for specular)
    int get_capture_size() const;
    void set_capture_extents(const double* extents); // world space bounds (for influence)
    void get_capture_extents(double* out_extents) const;

    // ------------------------------------------------------------------------
    // Baked data (set after baking)
    // ------------------------------------------------------------------------
    void set_sh_coefficients(const float* coeffs_27); // 3x9 = 27 floats (RGB SH)
    void get_sh_coefficients(float* out_coeffs_27) const;
    void set_specular_cubemap(int64_t texture_rid);
    int64_t get_specular_cubemap() const;

    // ------------------------------------------------------------------------
    // Interpolation blending
    // ------------------------------------------------------------------------
    void set_interior(bool interior);
    bool is_interior() const;
    void set_energy(float multiplier);
    float get_energy() const;

    // ------------------------------------------------------------------------
    // Rendering server sync (baking results)
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;
    void process(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting