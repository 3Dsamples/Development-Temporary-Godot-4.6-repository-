// Name : lighting enhancement
// File : scene/3d/lightmap_probe_ext.h 23 of 60
// Description : Extended lightmap probe node with spherical harmonics capture,
//               influence radius, interpolation weights, and RenderingServer sync.
#pragma once

#include "scene/3d/lightmap_probe.h"
#include "servers/rendering_server.h"

class LightmapProbeExt : public LightmapProbe {
    GDCLASS(LightmapProbeExt, LightmapProbe);

public:
    LightmapProbeExt();
    ~LightmapProbeExt();

    // ------------------------------------------------------------------------
    // Capture control
    // ------------------------------------------------------------------------
    void set_capture_mode(int p_mode);          // 0 = SH9, 1 = SH4, 2 = cubemap_32, 3 = cubemap_128
    int get_capture_mode() const;
    void capture();                             // immediate capture (synchronous)
    void capture_async();                       // background capture
    bool is_capturing() const;
    void cancel_capture();

    // ------------------------------------------------------------------------
    // Influence extents and weight
    // ------------------------------------------------------------------------
    void set_influence_radius(float p_radius);
    float get_influence_radius() const;
    void set_influence_weight(float p_weight);
    float get_influence_weight() const;
    void set_interior(bool p_interior);
    bool is_interior() const;
    void set_interior_center(const Vector3 &p_center);
    Vector3 get_interior_center() const;

    // ------------------------------------------------------------------------
    // Captured data (spherical harmonics coefficients)
    // ------------------------------------------------------------------------
    void get_sh_coefficients(Vector<float> &r_coeffs) const; // 27 floats (RGB SH9)
    void set_sh_coefficients(const Vector<float> &p_coeffs); // for manual override

    // ------------------------------------------------------------------------
    // Global illumination contribution
    // ------------------------------------------------------------------------
    void set_gi_contribution(float p_amount);
    float get_gi_contribution() const;

    // ------------------------------------------------------------------------
    // Rendering server synchronization
    // ------------------------------------------------------------------------
    void sync_probe();

private:
    struct Impl;
    Impl *pimpl;
};