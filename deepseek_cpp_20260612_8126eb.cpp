// Name : lighting enhancement
// File : scene/3d/area_3d_ext.h 37 of 60
// Description : Extended area node with gravity/damping overrides, overlap detection,
//               priority, emissive lighting, and full RenderingServer sync for GI.
#pragma once

#include "scene/3d/area_3d.h"
#include "servers/rendering_server.h"

class Area3DExt : public Area3D {
    GDCLASS(Area3DExt, Area3D);

public:
    Area3DExt();
    ~Area3DExt();

    // ------------------------------------------------------------------------
    // Gravity override
    // ------------------------------------------------------------------------
    void set_gravity_enabled(bool p_enabled);
    bool is_gravity_enabled() const;
    void set_gravity(const Vector3 &p_gravity);
    Vector3 get_gravity() const;
    void set_gravity_point(bool p_point);
    bool is_gravity_point() const;
    void set_gravity_point_center(const Vector3 &p_center);
    Vector3 get_gravity_point_center() const;
    void set_gravity_distance_scale(float p_scale);
    float get_gravity_distance_scale() const;

    // ------------------------------------------------------------------------
    // Damping overrides (linear / angular)
    // ------------------------------------------------------------------------
    void set_linear_damp_enabled(bool p_enabled);
    bool is_linear_damp_enabled() const;
    void set_linear_damp(float p_damp);
    float get_linear_damp() const;
    void set_angular_damp_enabled(bool p_enabled);
    bool is_angular_damp_enabled() const;
    void set_angular_damp(float p_damp);
    float get_angular_damp() const;
    void set_damp_priority(int p_priority);
    int get_damp_priority() const;

    // ------------------------------------------------------------------------
    // Overlap detection and signals
    // ------------------------------------------------------------------------
    using BodyCallback = Callable;
    void set_body_entered_callback(const BodyCallback &p_callback);
    void set_body_exited_callback(const BodyCallback &p_callback);
    void set_area_entered_callback(const BodyCallback &p_callback);
    void set_area_exited_callback(const BodyCallback &p_callback);

    // ------------------------------------------------------------------------
    // Priority (for overlapping areas)
    // ------------------------------------------------------------------------
    void set_priority(int p_priority);
    int get_priority() const;

    // ------------------------------------------------------------------------
    // Global illumination (area can emit light or affect GI intensity)
    // ------------------------------------------------------------------------
    void set_gi_mode(int p_mode) override;      // 0=off,1=static,2=dynamic
    int get_gi_mode() const override;
    void set_gi_contribution(float p_amount);
    float get_gi_contribution() const;
    void set_emissive_gi(const Color &p_color, float p_intensity);
    Color get_emissive_gi_color() const;
    float get_emissive_gi_intensity() const;

    // ------------------------------------------------------------------------
    // Rendering server synchronization (update area parameters)
    // ------------------------------------------------------------------------
    void sync_area();

private:
    struct Impl;
    Impl *pimpl;
};