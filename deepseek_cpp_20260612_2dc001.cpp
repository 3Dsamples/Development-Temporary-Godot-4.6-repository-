// Name : lighting enhancement
// File : scene/3d/physics_body_3d_ext.h 31 of 60
// Description : Extended physics body with mass, velocity, gravity, shadows, GI,
//               emissive, and physics interpolation for real‑time lighting.
#pragma once

#include "scene/3d/physics_body_3d.h"
#include "servers/rendering_server.h"

class PhysicsBody3DExt : public PhysicsBody3D {
    GDCLASS(PhysicsBody3DExt, PhysicsBody3D);

public:
    PhysicsBody3DExt();
    ~PhysicsBody3DExt();

    // ------------------------------------------------------------------------
    // Physics state (mass, velocity, damping)
    // ------------------------------------------------------------------------
    void set_mass(float p_mass);
    float get_mass() const;
    void set_linear_velocity(const Vector3 &p_velocity);
    Vector3 get_linear_velocity() const;
    void set_angular_velocity(const Vector3 &p_velocity);
    Vector3 get_angular_velocity() const;
    void set_gravity_scale(float p_scale);
    float get_gravity_scale() const;
    void set_linear_damp(float p_damp);
    float get_linear_damp() const;
    void set_angular_damp(float p_damp);
    float get_angular_damp() const;

    // ------------------------------------------------------------------------
    // Forces and impulses (apply in world space)
    // ------------------------------------------------------------------------
    void apply_force(const Vector3 &p_force, const Vector3 &p_position = Vector3());
    void apply_impulse(const Vector3 &p_impulse, const Vector3 &p_position = Vector3());
    void apply_torque(const Vector3 &p_torque);
    void clear_forces();

    // ------------------------------------------------------------------------
    // Sleeping (performance)
    // ------------------------------------------------------------------------
    void set_sleeping(bool p_sleep);
    bool is_sleeping() const;
    void set_can_sleep(bool p_can_sleep);
    bool can_sleep() const;

    // ------------------------------------------------------------------------
    // Continuous collision detection (CCD)
    // ------------------------------------------------------------------------
    void set_ccd_enabled(bool p_enabled);
    bool is_ccd_enabled() const;
    void set_ccd_threshold(float p_threshold);
    float get_ccd_threshold() const;

    // ------------------------------------------------------------------------
    // Lighting & shadows (body can cast shadows and affect GI)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool p_cast) override;
    bool get_cast_shadow() const override;
    void set_gi_mode(int p_mode) override;
    int get_gi_mode() const override;
    void set_gi_contribution(float p_amount) override;
    float get_gi_contribution() const override;
    void set_emissive(const Color &p_color, float p_intensity) override;
    Color get_emissive() const override;
    float get_emissive_intensity() const override;

    // ------------------------------------------------------------------------
    // Physics interpolation (smooth motion for render)
    // ------------------------------------------------------------------------
    void set_interpolate(bool p_enabled);
    bool is_interpolated() const;
    void set_physics_fraction(double p_fraction);
    void apply_interpolated_transform();

    // ------------------------------------------------------------------------
    // Rendering server synchronization (push transform and lighting params)
    // ------------------------------------------------------------------------
    void sync_physics_body();

private:
    struct Impl;
    Impl *pimpl;
};