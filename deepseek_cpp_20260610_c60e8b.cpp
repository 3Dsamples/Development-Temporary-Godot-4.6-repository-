// joint_3d.cpp
#include "joint_3d.h"
#include <cstring>
#include <algorithm>
#include <cmath>

namespace lighting {

struct Joint3D::Impl {
    // Bodies
    int64_t body_a_rid = -1;
    int64_t body_b_rid = -1;
    Transform3D local_a;
    Transform3D local_b;

    JointType type = JointType::HINGE;

    // Common limits
    float limit_lower = -1e10f;
    float limit_upper = 1e10f;
    float limit_restitution = 0.5f;
    float limit_damping = 1.0f;
    float limit_softness = 0.0f;

    // Motor
    bool motor_enabled = false;
    float motor_target_velocity = 0.0f;
    float motor_max_force = 1e8f;

    // Spring
    bool spring_enabled = false;
    float spring_stiffness = 0.0f;
    float spring_damping = 1.0f;

    // 6-DOF per‑axis limits (angular in radians, linear in meters)
    struct AxisLimits { float lower, upper; };
    AxisLimits angular_limit[3] = {{-1e10f,1e10f}, {-1e10f,1e10f}, {-1e10f,1e10f}};
    AxisLimits linear_limit[3] = {{-1e10f,1e10f}, {-1e10f,1e10f}, {-1e10f,1e10f}};

    // Breakable
    float break_force = 1e8f;
    float break_torque = 1e8f;

    // GI
    bool collision_affects_gi = true;

    // Physics server RID
    int64_t joint_rid = -1;
};

Joint3D::Joint3D() : pimpl(std::make_unique<Impl>()) {}
Joint3D::~Joint3D() = default;

void Joint3D::set_body_a(int64_t body_rid, const Transform3D& local_a) {
    pimpl->body_a_rid = body_rid;
    pimpl->local_a = local_a;
}
void Joint3D::set_body_b(int64_t body_rid, const Transform3D& local_b) {
    pimpl->body_b_rid = body_rid;
    pimpl->local_b = local_b;
}
int64_t Joint3D::get_body_a_rid() const { return pimpl->body_a_rid; }
int64_t Joint3D::get_body_b_rid() const { return pimpl->body_b_rid; }

void Joint3D::set_joint_type(JointType type) { pimpl->type = type; }
JointType Joint3D::get_joint_type() const { return pimpl->type; }

void Joint3D::set_limit_lower(float lower) { pimpl->limit_lower = lower; }
float Joint3D::get_limit_lower() const { return pimpl->limit_lower; }
void Joint3D::set_limit_upper(float upper) { pimpl->limit_upper = upper; }
float Joint3D::get_limit_upper() const { return pimpl->limit_upper; }
void Joint3D::set_limit_restitution(float restitution) { pimpl->limit_restitution = restitution; }
float Joint3D::get_limit_restitution() const { return pimpl->limit_restitution; }
void Joint3D::set_limit_damping(float damping) { pimpl->limit_damping = damping; }
float Joint3D::get_limit_damping() const { return pimpl->limit_damping; }
void Joint3D::set_limit_softness(float softness) { pimpl->limit_softness = softness; }
float Joint3D::get_limit_softness() const { return pimpl->limit_softness; }

void Joint3D::set_motor_enabled(bool enabled) { pimpl->motor_enabled = enabled; }
bool Joint3D::is_motor_enabled() const { return pimpl->motor_enabled; }
void Joint3D::set_motor_target_velocity(float velocity) { pimpl->motor_target_velocity = velocity; }
float Joint3D::get_motor_target_velocity() const { return pimpl->motor_target_velocity; }
void Joint3D::set_motor_max_force(float force) { pimpl->motor_max_force = force; }
float Joint3D::get_motor_max_force() const { return pimpl->motor_max_force; }

void Joint3D::set_spring_enabled(bool enabled) { pimpl->spring_enabled = enabled; }
bool Joint3D::is_spring_enabled() const { return pimpl->spring_enabled; }
void Joint3D::set_spring_stiffness(float stiffness) { pimpl->spring_stiffness = stiffness; }
float Joint3D::get_spring_stiffness() const { return pimpl->spring_stiffness; }
void Joint3D::set_spring_damping(float damping) { pimpl->spring_damping = damping; }
float Joint3D::get_spring_damping() const { return pimpl->spring_damping; }

void Joint3D::set_angular_limit_x(float lower, float upper) {
    pimpl->angular_limit[0] = {lower, upper};
}
void Joint3D::set_angular_limit_y(float lower, float upper) {
    pimpl->angular_limit[1] = {lower, upper};
}
void Joint3D::set_angular_limit_z(float lower, float upper) {
    pimpl->angular_limit[2] = {lower, upper};
}
void Joint3D::set_linear_limit_x(float lower, float upper) {
    pimpl->linear_limit[0] = {lower, upper};
}
void Joint3D::set_linear_limit_y(float lower, float upper) {
    pimpl->linear_limit[1] = {lower, upper};
}
void Joint3D::set_linear_limit_z(float lower, float upper) {
    pimpl->linear_limit[2] = {lower, upper};
}

void Joint3D::set_break_force(float force) { pimpl->break_force = force; }
float Joint3D::get_break_force() const { return pimpl->break_force; }
void Joint3D::set_break_torque(float torque) { pimpl->break_torque = torque; }
float Joint3D::get_break_torque() const { return pimpl->break_torque; }

void Joint3D::set_collision_affects_gi(bool affects) { pimpl->collision_affects_gi = affects; }
bool Joint3D::get_collision_affects_gi() const { return pimpl->collision_affects_gi; }

void Joint3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    // If joint_rid is valid, send updated parameters to physics server.
    // For GI, joints themselves don't cast shadows but their connected bodies do.
    // We might update the global illumination influence of the constraints.
}

} // namespace lighting