// joint_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>

namespace lighting {

// ============================================================================
// Joint3D – connects two physics bodies with constraints
// Supports: hinge, prismatic, ball, generic 6-DOF, slider, cone twist.
// Integrated with shadows and GI via the connected bodies.
// ============================================================================

enum class JointType : uint8_t {
    HINGE,          // rotation around one axis
    PRISMATIC,      // linear translation along one axis
    BALL,           // ball socket (spherical)
    GENERIC_6_DOF,  // six degrees of freedom with limits
    SLIDER,         // translation + rotation
    CONE_TWIST      // ragdoll cone twist
};

class Joint3D : public Node3D {
public:
    Joint3D();
    ~Joint3D();

    // ------------------------------------------------------------------------
    // Body references (node IDs or RIDs)
    // ------------------------------------------------------------------------
    void set_body_a(int64_t body_rid, const Transform3D& local_a);
    void set_body_b(int64_t body_rid, const Transform3D& local_b);
    int64_t get_body_a_rid() const;
    int64_t get_body_b_rid() const;

    // ------------------------------------------------------------------------
    // Joint type and parameters
    // ------------------------------------------------------------------------
    void set_joint_type(JointType type);
    JointType get_joint_type() const;

    // Hinge / Prismatic / Slider limits
    void set_limit_lower(float lower);
    float get_limit_lower() const;
    void set_limit_upper(float upper);
    float get_limit_upper() const;
    void set_limit_restitution(float restitution);
    float get_limit_restitution() const;
    void set_limit_damping(float damping);
    float get_limit_damping() const;
    void set_limit_softness(float softness);
    float get_limit_softness() const;

    // Motor (for actuated joints)
    void set_motor_enabled(bool enabled);
    bool is_motor_enabled() const;
    void set_motor_target_velocity(float velocity);
    float get_motor_target_velocity() const;
    void set_motor_max_force(float force);
    float get_motor_max_force() const;

    // Spring (return to neutral)
    void set_spring_enabled(bool enabled);
    bool is_spring_enabled() const;
    void set_spring_stiffness(float stiffness);
    float get_spring_stiffness() const;
    void set_spring_damping(float damping);
    float get_spring_damping() const;

    // ------------------------------------------------------------------------
    // 6-DOF specific: angular and linear limits per axis
    // ------------------------------------------------------------------------
    void set_angular_limit_x(float lower, float upper);
    void set_angular_limit_y(float lower, float upper);
    void set_angular_limit_z(float lower, float upper);
    void set_linear_limit_x(float lower, float upper);
    void set_linear_limit_y(float lower, float upper);
    void set_linear_limit_z(float lower, float upper);

    // ------------------------------------------------------------------------
    // Breakable joints
    // ------------------------------------------------------------------------
    void set_break_force(float force);
    float get_break_force() const;
    void set_break_torque(float torque);
    float get_break_torque() const;

    // ------------------------------------------------------------------------
    // Lighting considerations (joints can cast shadow via bodies, no direct light)
    // ------------------------------------------------------------------------
    void set_collision_affects_gi(bool affects);
    bool get_collision_affects_gi() const;

    // ------------------------------------------------------------------------
    // Physics server synchronization
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting