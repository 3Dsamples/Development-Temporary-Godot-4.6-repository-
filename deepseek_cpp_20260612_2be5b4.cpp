// Name : lighting enhancement
// File : scene/3d/rigid_body_3d_ext.h 33 of 60
// Description : Extended rigid body with mass, inertia tensor, freeze axes,
//               continuous collision detection (CCD), and full RenderingServer sync for dynamic lighting.
#pragma once

#include "scene/3d/physics_body_3d_ext.h"
#include "servers/rendering_server.h"

class RigidBody3DExt : public PhysicsBody3DExt {
    GDCLASS(RigidBody3DExt, PhysicsBody3DExt);

public:
    RigidBody3DExt();
    ~RigidBody3DExt();

    // ------------------------------------------------------------------------
    // Mass and inertia tensor
    // ------------------------------------------------------------------------
    void set_mass(float p_mass) override;
    float get_mass() const override;
    void set_inertia(const Vector3 &p_inertia); // diagonal inertia tensor (Ixx, Iyy, Izz)
    Vector3 get_inertia() const;

    // ------------------------------------------------------------------------
    // Freeze (lock movement or rotation per axis)
    // ------------------------------------------------------------------------
    void set_freeze_enabled(bool p_enabled);
    bool is_freeze_enabled() const;
    void set_freeze_linear(bool p_freeze_x, bool p_freeze_y, bool p_freeze_z);
    void get_freeze_linear(bool &r_freeze_x, bool &r_freeze_y, bool &r_freeze_z) const;
    void set_freeze_angular(bool p_freeze_x, bool p_freeze_y, bool p_freeze_z);
    void get_freeze_angular(bool &r_freeze_x, bool &r_freeze_y, bool &r_freeze_z) const;

    // ------------------------------------------------------------------------
    // Continuous collision detection (CCD)
    // ------------------------------------------------------------------------
    void set_ccd_enabled(bool p_enabled) override;
    bool is_ccd_enabled() const override;
    void set_ccd_threshold(float p_threshold) override;
    float get_ccd_threshold() const override;

    // ------------------------------------------------------------------------
    // Max velocities (clamping)
    // ------------------------------------------------------------------------
    void set_max_linear_velocity(float p_max);
    float get_max_linear_velocity() const;
    void set_max_angular_velocity(float p_max);
    float get_max_angular_velocity() const;

    // ------------------------------------------------------------------------
    // Lighting & shadows (overrides from base)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool p_cast) override;
    void set_gi_mode(int p_mode) override;
    void set_gi_contribution(float p_amount) override;
    void set_emissive(const Color &p_color, float p_intensity) override;

    // ------------------------------------------------------------------------
    // Rendering server synchronization
    // ------------------------------------------------------------------------
    void sync_rigid_body();

private:
    struct Impl;
    Impl *pimpl;
};