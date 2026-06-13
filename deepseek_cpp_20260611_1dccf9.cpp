// vehicle_wheel_3d.h
#pragma once

#include "geometry_instance_3d.h"
#include <cstdint>
#include <memory>

namespace lighting {

// ============================================================================
// VehicleWheel3D – a wheel node for use with VehicleBody3D.
// Handles visual wheel mesh, suspension travel, rotation, and shadow casting.
// ============================================================================

class VehicleWheel3D : public GeometryInstance3D {
public:
    VehicleWheel3D();
    ~VehicleWheel3D();

    // ------------------------------------------------------------------------
    // Wheel geometry (visual)
    // ------------------------------------------------------------------------
    void set_radius(float radius);
    float get_radius() const;
    void set_width(float width);
    float get_width() const;
    void set_suspension_rest_length(float length);
    float get_suspension_rest_length() const;
    void set_suspension_travel(float travel);
    float get_suspension_travel() const;

    // ------------------------------------------------------------------------
    // Suspension physics (used by VehicleBody3D)
    // ------------------------------------------------------------------------
    void set_suspension_stiffness(float stiffness);
    float get_suspension_stiffness() const;
    void set_suspension_damping(float damping);
    float get_suspension_damping() const;
    void set_friction(float friction);
    float get_friction() const;
    void set_roll_influence(float influence);
    float get_roll_influence() const;

    // ------------------------------------------------------------------------
    // Runtime state (updated by VehicleBody3D)
    // ------------------------------------------------------------------------
    void set_compression(float compression);   // current suspension compression (0=full extension, 1=full compression)
    float get_compression() const;
    void set_rotation_angle(float radians);
    float get_rotation_angle() const;
    void set_steering_angle(float radians);
    float get_steering_angle() const;

    // ------------------------------------------------------------------------
    // Steering constraint (for visual wheel alignment)
    // ------------------------------------------------------------------------
    void set_use_as_steering(bool steering);
    bool is_steering_wheel() const;

    // ------------------------------------------------------------------------
    // Lighting & shadows (visual wheel)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;
    void set_gi_contribution(float amount) override;
    void set_emissive(const float* color, float intensity) override;
    void get_emissive(float* out_color, float& out_intensity) const override;

    // ------------------------------------------------------------------------
    // Force update (call after changing geometry parameters)
    // ------------------------------------------------------------------------
    void update_wheel_mesh();

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