// Name : lighting enhancement
// File : scene/3d/reflection_probe_ext.h 17 of 60
// Description : Extended reflection probe with box/sphere influence, real‑time cubemap
//               capture, blend distances, intensity, and full RenderingServer sync.
#pragma once

#include "scene/3d/reflection_probe.h"
#include "servers/rendering_server.h"

class ReflectionProbeExt : public ReflectionProbe {
    GDCLASS(ReflectionProbeExt, ReflectionProbe);

public:
    ReflectionProbeExt();
    ~ReflectionProbeExt();

    // ------------------------------------------------------------------------
    // Influence shape and extents
    // ------------------------------------------------------------------------
    void set_shape(int p_shape);                // 0 = box, 1 = sphere
    int get_shape() const;
    void set_extents(const Vector3 &p_extents); // half extents for box, radius for sphere
    Vector3 get_extents() const;

    // ------------------------------------------------------------------------
    // Blend distances and intensity
    // ------------------------------------------------------------------------
    void set_intensity(float p_intensity);
    float get_intensity() const;
    void set_blend_distance(float p_distance);
    float get_blend_distance() const;
    void set_ambient_color(const Color &p_color);
    Color get_ambient_color() const;
    void set_ambient_mode(bool p_interior);     // interior = probe acts as ambient source
    bool get_ambient_mode() const;

    // ------------------------------------------------------------------------
    // Update mode (static / dynamic / real‑time)
    // ------------------------------------------------------------------------
    void set_update_mode(int p_mode);           // 0 = static, 1 = dynamic, 2 = always
    int get_update_mode() const;
    void set_update_frequency(float p_fps);     // how often to recapture (Hz)
    float get_update_frequency() const;

    // ------------------------------------------------------------------------
    // Resolution and quality
    // ------------------------------------------------------------------------
    void set_resolution(int p_resolution);
    int get_resolution() const;
    void set_roughness_filtering(bool p_enable);
    bool get_roughness_filtering() const;

    // ------------------------------------------------------------------------
    // Culling mask (which objects are reflected)
    // ------------------------------------------------------------------------
    void set_cull_mask(uint32_t p_mask);
    uint32_t get_cull_mask() const;

    // ------------------------------------------------------------------------
    // Rendering server synchronization
    // ------------------------------------------------------------------------
    void sync_probe();
    void capture_cubemap();                    // force immediate capture (async)
    RID get_probe_rid() const;

private:
    struct Impl;
    Impl *pimpl;
};