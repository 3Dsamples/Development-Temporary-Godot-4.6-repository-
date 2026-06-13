// vehicle_body_3d.h
#pragma once

#include "rigid_body_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// VehicleWheel3D – attached to VehicleBody3D for suspension and traction
// ============================================================================

class VehicleWheel3D : public Node3D {
public:
    VehicleWheel3D();
    ~VehicleWheel3D();

    void set_radius(float radius);
    float get_radius() const;
    void set_width(float width);
    float get_width() const;
    void set_suspension_rest_length(float length);
    float get_suspension_rest_length() const;
    void set_suspension_travel(float travel);
    float get_suspension_travel() const;
    void set_suspension_stiffness(float stiffness);
    float get_suspension_stiffness() const;
    void set_suspension_damping(float damping);
    float get_suspension_damping() const;
    void set_friction(float friction);
    float get_friction() const;
    void set_roll_influence(float influence);
    float get_roll_influence() const;

    // Lighting & shadows (wheels can cast shadows)
    void set_cast_shadow(bool cast);
    bool get_cast_shadow() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

// ============================================================================
// VehicleBody3D – rigid body with wheel constraints, engine, steering
// Supports dynamic lighting, shadows, and GI.
// ============================================================================

class VehicleBody3D : public RigidBody3D {
public:
    VehicleBody3D();
    ~VehicleBody3D();

    // ------------------------------------------------------------------------
    // Wheel management
    // ------------------------------------------------------------------------
    void add_wheel(VehicleWheel3D* wheel, const Transform3D& local_transform);
    void remove_wheel(VehicleWheel3D* wheel);
    int get_wheel_count() const;
    VehicleWheel3D* get_wheel(int index) const;

    // ------------------------------------------------------------------------
    // Engine & driving
    // ------------------------------------------------------------------------
    void set_engine_force(float force);
    float get_engine_force() const;
    void set_brake(float brake);
    float get_brake() const;
    void set_steering(float angle); // in radians
    float get_steering() const;
    void set_max_engine_force(float max_force);
    float get_max_engine_force() const;
    void set_max_brake(float max_brake);
    float get_max_brake() const;
    void set_max_steering_angle(float angle);
    float get_max_steering_angle() const;

    // ------------------------------------------------------------------------
    // Transmission
    // ------------------------------------------------------------------------
    void set_use_abs(bool enabled);
    bool is_abs_enabled() const;
    void set_use_tcs(bool enabled);
    bool is_tcs_enabled() const;

    // ------------------------------------------------------------------------
    // Per‑wheel control (if needed)
    // ------------------------------------------------------------------------
    void set_wheel_engine_force(int wheel_index, float force);
    void set_wheel_brake(int wheel_index, float brake);
    void set_wheel_steering(int wheel_index, float angle);
    float get_wheel_rpm(int wheel_index) const;

    // ------------------------------------------------------------------------
    // Lighting integration – vehicles move and should cast dynamic shadows
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;
    void set_gi_contribution(float amount) override;

    // ------------------------------------------------------------------------
    // Physics override (to apply engine forces)
    // ------------------------------------------------------------------------
    void update_physics(double delta_time) override;

    // ------------------------------------------------------------------------
    // Render sync (for wheel transforms and shadows)
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting