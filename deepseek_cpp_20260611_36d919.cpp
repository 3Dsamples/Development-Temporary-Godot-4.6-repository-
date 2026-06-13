// rigid_body_3d.h
#pragma once

#include "physics_body_3d.h"
#include <cstdint>
#include <memory>

namespace lighting {

// ============================================================================
// RigidBody3D – dynamic rigid body with forces, gravity, collisions.
// Supports mass, inertia, damping, continuous collision detection (CCD),
// sleeping, freezing axes, and locking linear/angular motion.
// Integrated with full lighting: shadows, GI, emissive (via base).
// ============================================================================

enum class BodyFreezeMode : uint8_t {
    FREEZE_NONE,
    FREEZE_STATIC,
    FREEZE_KINEMATIC
};

class RigidBody3D : public PhysicsBody3D {
public:
    RigidBody3D();
    ~RigidBody3D();

    // ------------------------------------------------------------------------
    // Mass & inertia
    // ------------------------------------------------------------------------
    void set_mass(float mass);
    float get_mass() const;
    void set_inertia(const float* inertia_tensor); // 3x3 row‑major
    void get_inertia(float* out_inertia) const;

    // ------------------------------------------------------------------------
    // Gravity and damping
    // ------------------------------------------------------------------------
    void set_gravity_scale(float scale);
    float get_gravity_scale() const;
    void set_linear_damping(float damping);
    float get_linear_damping() const;
    void set_angular_damping(float damping);
    float get_angular_damping() const;

    // ------------------------------------------------------------------------
    // Linear / angular velocity control
    // ------------------------------------------------------------------------
    void set_linear_velocity(const double* velocity);
    void get_linear_velocity(double* out_velocity) const;
    void set_angular_velocity(const double* velocity);
    void get_angular_velocity(double* out_velocity) const;

    // ------------------------------------------------------------------------
    // Forces and impulses
    // ------------------------------------------------------------------------
    void apply_force(const double* force, const double* at_position = nullptr);
    void apply_impulse(const double* impulse, const double* at_position = nullptr);
    void apply_torque(const double* torque);
    void apply_torque_impulse(const double* torque_impulse);
    void clear_forces();

    // ------------------------------------------------------------------------
    // Sleeping (optimization)
    // ------------------------------------------------------------------------
    void set_sleeping(bool sleeping);
    bool is_sleeping() const;
    void set_can_sleep(bool can_sleep);
    bool can_sleep() const;

    // ------------------------------------------------------------------------
    // Freezing (stop movement completely)
    // ------------------------------------------------------------------------
    void set_freeze_enabled(bool freeze);
    bool is_freeze_enabled() const;
    void set_freeze_mode(BodyFreezeMode mode);
    BodyFreezeMode get_freeze_mode() const;

    // ------------------------------------------------------------------------
    // Axis locking (lock linear/angular movement per axis)
    // ------------------------------------------------------------------------
    void set_linear_lock_axes(bool lock_x, bool lock_y, bool lock_z);
    void set_angular_lock_axes(bool lock_x, bool lock_y, bool lock_z);
    void get_linear_lock_axes(bool& lock_x, bool& lock_y, bool& lock_z) const;
    void get_angular_lock_axes(bool& lock_x, bool& lock_y, bool& lock_z) const;

    // ------------------------------------------------------------------------
    // Continuous collision detection (CCD)
    // ------------------------------------------------------------------------
    void set_ccd_enabled(bool enabled);
    bool is_ccd_enabled() const;
    void set_ccd_motion_threshold(float threshold);
    float get_ccd_motion_threshold() const;

    // ------------------------------------------------------------------------
    // Maximum velocities (clamping)
    // ------------------------------------------------------------------------
    void set_max_linear_velocity(float max_vel);
    float get_max_linear_velocity() const;
    void set_max_angular_velocity(float max_vel);
    float get_max_angular_velocity() const;

    // ------------------------------------------------------------------------
    // Lighting & GI overrides (same as base, but we ensure they are exposed)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;
    void set_gi_contribution(float amount) override;
    void set_emissive(const float* color, float intensity) override;
    void get_emissive(float* out_color, float& out_intensity) const override;

    // ------------------------------------------------------------------------
    // Physics server synchronization
    // ------------------------------------------------------------------------
    void update_physics(double delta_time) override;
    void synchronize_render_server(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting