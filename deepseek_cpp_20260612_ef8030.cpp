// Name : lighting enhancement
// File : scene/3d/vehicle_wheel_3d_ext.h 57 of 60
// Description : Extended vehicle wheel node with suspension, friction, roll influence,
//               rotation angle, steering angle, and full RenderingServer sync.
#pragma once

#include "scene/3d/vehicle_wheel_3d.h"
#include "servers/rendering_server.h"

class VehicleWheel3DExt : public VehicleWheel3D {
    GDCLASS(VehicleWheel3DExt, VehicleWheel3D);

public:
    VehicleWheel3DExt();
    ~VehicleWheel3DExt();

    // ------------------------------------------------------------------------
    // Wheel geometry and suspension
    // ------------------------------------------------------------------------
    void set_radius(float p_radius);
    float get_radius() const;
    void set_width(float p_width);
    float get_width() const;
    void set_suspension_rest_length(float p_length);
    float get_suspension_rest_length() const;
    void set_suspension_travel(float p_travel);
    float get_suspension_travel() const;

    // ------------------------------------------------------------------------
    // Suspension physics (used by VehicleBody3D)
    // ------------------------------------------------------------------------
    void set_suspension_stiffness(float p_stiffness);
    float get_suspension_stiffness() const;
    void set_suspension_damping(float p_damping);
    float get_suspension_damping() const;
    void set_friction(float p_friction);
    float get_friction() const;
    void set_roll_influence(float p_influence);
    float get_roll_influence() const;

    // ------------------------------------------------------------------------
    // Runtime state (updated by VehicleBody3D)
    // ------------------------------------------------------------------------
    void set_compression(float p_compression);   // 0 = extended, 1 = compressed
    float get_compression() const;
    void set_rotation_angle(float p_radians);
    float get_rotation_angle() const;
    void set_steering_angle(float p_radians);
    float get_steering_angle() const;

    // ------------------------------------------------------------------------
    // Steering constraint (for visual wheel alignment)
    // ------------------------------------------------------------------------
    void set_use_as_steering(bool p_steering);
    bool is_steering_wheel() const;

    // ------------------------------------------------------------------------
    // Lighting & shadows (visual wheel)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool p_cast) override;
    void set_receive_shadow(bool p_receive) override;
    void set_gi_mode(int p_mode) override;
    void set_gi_contribution(float p_amount) override;
    void set_emissive(const Color &p_color, float p_intensity) override;
    Color get_emissive() const override;
    float get_emissive_intensity() const override;

    // ------------------------------------------------------------------------
    // Rendering server synchronization (update wheel transform and mesh)
    // ------------------------------------------------------------------------
    void sync_wheel();

private:
    struct Impl;
    Impl *pimpl;
};