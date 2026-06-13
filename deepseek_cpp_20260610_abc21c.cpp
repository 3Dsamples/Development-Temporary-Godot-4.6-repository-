// vehicle_body_3d.h
#pragma once

#include "rigid_body_3d.h"
#include <memory>
#include <vector>
#include <cstdint>

namespace lighting {

class VehicleWheel3D;

// ============================================================================
// VehicleBody3D – raycast vehicle simulation (suspension, steering, engine)
// Supports high‑performance physics, dynamic shadows, and GI for the whole vehicle.
// ============================================================================

class VehicleBody3D : public RigidBody3D {
public:
    VehicleBody3D();
    ~VehicleBody3D();

    // ------------------------------------------------------------------------
    // Wheel management
    // ------------------------------------------------------------------------
    void add_wheel(VehicleWheel3D* wheel);
    void remove_wheel(VehicleWheel3D* wheel);
    void clear_wheels();
    int get_wheel_count() const;
    VehicleWheel3D* get_wheel(int index) const;

    // ------------------------------------------------------------------------
    // Engine and drivetrain
    // ------------------------------------------------------------------------
    void set_engine_force(float force);          // force in Newtons
    float get_engine_force() const;
    void set_brake(float brake);                // braking force (0..1)
    float get_brake() const;
    void set_steering(float steering);          // steering angle (radians)
    float get_steering() const;

    void set_engine_power(float power);         // max engine power (Watts)
    float get_engine_power() const;
    void set_engine_power_curve(float* curve_points, int count); // torque/rpm curve
    void set_engine_max_rpm(float rpm);
    float get_engine_max_rpm() const;
    void set_engine_min_rpm(float rpm);
    float get_engine_min_rpm() const;

    // ------------------------------------------------------------------------
    // Gearbox (automatic)
    // ------------------------------------------------------------------------
    void set_gearbox_auto(bool enable);
    bool is_gearbox_auto() const;
    void set_forward_gears(int count);
    int get_forward_gears() const;
    void set_reverse_gears(int count);
    int get_reverse_gears() const;
    void set_gear(int gear);
    int get_gear() const;
    void shift_gear_up();
    void shift_gear_down();

    // ------------------------------------------------------------------------
    // Clutch and differential
    // ------------------------------------------------------------------------
    void set_clutch_strength(float strength);
    float get_clutch_strength() const;
    void set_differential_ratio(float ratio);
    float get_differential_ratio() const;

    // ------------------------------------------------------------------------
    // Mass distribution (center of gravity offset)
    // ------------------------------------------------------------------------
    void set_center_of_gravity(const Vector3& offset);
    Vector3 get_center_of_gravity() const;

    // ------------------------------------------------------------------------
    // Lighting integration (whole vehicle casts shadows, GI from headlights)
    // ------------------------------------------------------------------------
    void set_headlight_enabled(bool enabled);
    bool is_headlight_enabled() const;
    void set_headlight_intensity(float intensity);
    float get_headlight_intensity() const;
    void set_headlight_color(const Vector3& color);
    Vector3 get_headlight_color() const;
    void set_brake_light_intensity(float intensity);
    float get_brake_light_intensity() const;
    void set_reverse_light_intensity(float intensity);
    float get_reverse_light_intensity() const;

    // ------------------------------------------------------------------------
    // Performance (tire smoke, skid marks – visual effects)
    // ------------------------------------------------------------------------
    void set_tire_smoke_enabled(bool enable);
    bool is_tire_smoke_enabled() const;

    // ------------------------------------------------------------------------
    // Physics update (called by engine)
    // ------------------------------------------------------------------------
    void update_physics(double delta_time) override;

    // ------------------------------------------------------------------------
    // Render server sync (lights, shadows)
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

// ============================================================================
// VehicleWheel3D – wheel node for VehicleBody
// ============================================================================

class VehicleWheel3D : public Node3D {
public:
    VehicleWheel3D();
    ~VehicleWheel3D();

    void set_use_as_steering(bool enable);
    bool is_use_as_steering() const;
    void set_use_as_traction(bool enable);
    bool is_use_as_traction() const;
    void set_use_as_brake(bool enable);
    bool is_use_as_brake() const;

    void set_suspension_rest_length(float length);
    float get_suspension_rest_length() const;
    void set_suspension_stiffness(float stiffness);
    float get_suspension_stiffness() const;
    void set_suspension_damping(float damping);
    float get_suspension_damping() const;
    void set_suspension_max_force(float force);
    float get_suspension_max_force() const;

    void set_wheel_radius(float radius);
    float get_wheel_radius() const;
    void set_wheel_forward_offset(float offset);
    float get_wheel_forward_offset() const;
    void set_wheel_side_offset(float offset);
    float get_wheel_side_offset() const;

    void set_steering_angle(float angle);
    float get_steering_angle() const;

    void set_roll_influence(float influence);
    float get_roll_influence() const;

    // Visual rotation and position (for wheel animation)
    void set_visual_rotation(float radians);
    void set_visual_offset(const Vector3& offset);

    // Lighting integration: wheel may have emissive brake glow
    void set_emissive_intensity_when_braking(float intensity);
    float get_emissive_intensity_when_braking() const;

private:
    friend class VehicleBody3D;
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting