// rigid_body_3d.cpp
#include "rigid_body_3d.h"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace lighting {

// ============================================================================
// RigidBody3D implementation
// ============================================================================
struct RigidBody3D::Impl {
    float mass = 1.0f;
    float inertia[9] = {1,0,0, 0,1,0, 0,0,1};
    float gravity_scale = 1.0f;
    float linear_damping = 0.0f;
    float angular_damping = 0.0f;

    double linear_velocity[3] = {0,0,0};
    double angular_velocity[3] = {0,0,0};
    double accumulated_force[3] = {0,0,0};
    double accumulated_torque[3] = {0,0,0};

    bool sleeping = false;
    bool can_sleep = true;
    bool freeze_enabled = false;
    BodyFreezeMode freeze_mode = BodyFreezeMode::FREEZE_NONE;

    bool linear_lock[3] = {false, false, false};
    bool angular_lock[3] = {false, false, false};

    bool ccd_enabled = false;
    float ccd_threshold = 0.01f;

    float max_linear_velocity = 1000.0f;
    float max_angular_velocity = 1000.0f;

    // Lighting flags
    bool cast_shadow = true;
    bool receive_shadow = true;
    int gi_mode = 2;               // dynamic by default
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    // Internal state for integration (simplified, uses Euler)
    void integrate(double dt);
    void apply_constraints();
};

RigidBody3D::RigidBody3D() : pimpl(std::make_unique<Impl>()) {
    set_body_mode(BodyMode::RIGID_DYNAMIC);
}
RigidBody3D::~RigidBody3D() = default;

void RigidBody3D::set_mass(float mass) {
    pimpl->mass = std::max(0.0f, mass);
}
float RigidBody3D::get_mass() const { return pimpl->mass; }
void RigidBody3D::set_inertia(const float* inertia_tensor) {
    memcpy(pimpl->inertia, inertia_tensor, 9*sizeof(float));
}
void RigidBody3D::get_inertia(float* out_inertia) const {
    memcpy(out_inertia, pimpl->inertia, 9*sizeof(float));
}

void RigidBody3D::set_gravity_scale(float scale) { pimpl->gravity_scale = scale; }
float RigidBody3D::get_gravity_scale() const { return pimpl->gravity_scale; }
void RigidBody3D::set_linear_damping(float damping) { pimpl->linear_damping = damping; }
float RigidBody3D::get_linear_damping() const { return pimpl->linear_damping; }
void RigidBody3D::set_angular_damping(float damping) { pimpl->angular_damping = damping; }
float RigidBody3D::get_angular_damping() const { return pimpl->angular_damping; }

void RigidBody3D::set_linear_velocity(const double* velocity) {
    memcpy(pimpl->linear_velocity, velocity, 3*sizeof(double));
}
void RigidBody3D::get_linear_velocity(double* out_velocity) const {
    memcpy(out_velocity, pimpl->linear_velocity, 3*sizeof(double));
}
void RigidBody3D::set_angular_velocity(const double* velocity) {
    memcpy(pimpl->angular_velocity, velocity, 3*sizeof(double));
}
void RigidBody3D::get_angular_velocity(double* out_velocity) const {
    memcpy(out_velocity, pimpl->angular_velocity, 3*sizeof(double));
}

void RigidBody3D::apply_force(const double* force, const double* at_position) {
    if (pimpl->freeze_enabled && pimpl->freeze_mode != BodyFreezeMode::FREEZE_NONE) return;
    pimpl->accumulated_force[0] += force[0];
    pimpl->accumulated_force[1] += force[1];
    pimpl->accumulated_force[2] += force[2];
}
void RigidBody3D::apply_impulse(const double* impulse, const double* at_position) {
    if (pimpl->freeze_enabled && pimpl->freeze_mode != BodyFreezeMode::FREEZE_NONE) return;
    double inv_mass = 1.0 / pimpl->mass;
    pimpl->linear_velocity[0] += impulse[0] * inv_mass;
    pimpl->linear_velocity[1] += impulse[1] * inv_mass;
    pimpl->linear_velocity[2] += impulse[2] * inv_mass;
    if (at_position) {
        // compute torque = r × impulse
        double rx = at_position[0];
        double ry = at_position[1];
        double rz = at_position[2];
        double torque_x = ry * impulse[2] - rz * impulse[1];
        double torque_y = rz * impulse[0] - rx * impulse[2];
        double torque_z = rx * impulse[1] - ry * impulse[0];
        apply_torque_impulse(&torque_x);
    }
}
void RigidBody3D::apply_torque(const double* torque) {
    if (pimpl->freeze_enabled && pimpl->freeze_mode != BodyFreezeMode::FREEZE_NONE) return;
    pimpl->accumulated_torque[0] += torque[0];
    pimpl->accumulated_torque[1] += torque[1];
    pimpl->accumulated_torque[2] += torque[2];
}
void RigidBody3D::apply_torque_impulse(const double* torque_impulse) {
    if (pimpl->freeze_enabled && pimpl->freeze_mode != BodyFreezeMode::FREEZE_NONE) return;
    // transform torque impulse to angular velocity increment using inverse inertia
    double inv_inertia[9] = {1.0/pimpl->inertia[0], 1.0/pimpl->inertia[4], 1.0/pimpl->inertia[8]}; // diagonal approximation
    pimpl->angular_velocity[0] += torque_impulse[0] * inv_inertia[0];
    pimpl->angular_velocity[1] += torque_impulse[1] * inv_inertia[4];
    pimpl->angular_velocity[2] += torque_impulse[2] * inv_inertia[8];
}
void RigidBody3D::clear_forces() {
    pimpl->accumulated_force[0] = pimpl->accumulated_force[1] = pimpl->accumulated_force[2] = 0;
    pimpl->accumulated_torque[0] = pimpl->accumulated_torque[1] = pimpl->accumulated_torque[2] = 0;
}

void RigidBody3D::set_sleeping(bool sleeping) {
    pimpl->sleeping = sleeping;
    if (sleeping) {
        pimpl->linear_velocity[0] = pimpl->linear_velocity[1] = pimpl->linear_velocity[2] = 0;
        pimpl->angular_velocity[0] = pimpl->angular_velocity[1] = pimpl->angular_velocity[2] = 0;
    }
}
bool RigidBody3D::is_sleeping() const { return pimpl->sleeping; }
void RigidBody3D::set_can_sleep(bool can_sleep) { pimpl->can_sleep = can_sleep; }
bool RigidBody3D::can_sleep() const { return pimpl->can_sleep; }

void RigidBody3D::set_freeze_enabled(bool freeze) { pimpl->freeze_enabled = freeze; }
bool RigidBody3D::is_freeze_enabled() const { return pimpl->freeze_enabled; }
void RigidBody3D::set_freeze_mode(BodyFreezeMode mode) { pimpl->freeze_mode = mode; }
BodyFreezeMode RigidBody3D::get_freeze_mode() const { return pimpl->freeze_mode; }

void RigidBody3D::set_linear_lock_axes(bool lock_x, bool lock_y, bool lock_z) {
    pimpl->linear_lock[0] = lock_x; pimpl->linear_lock[1] = lock_y; pimpl->linear_lock[2] = lock_z;
}
void RigidBody3D::set_angular_lock_axes(bool lock_x, bool lock_y, bool lock_z) {
    pimpl->angular_lock[0] = lock_x; pimpl->angular_lock[1] = lock_y; pimpl->angular_lock[2] = lock_z;
}
void RigidBody3D::get_linear_lock_axes(bool& lock_x, bool& lock_y, bool& lock_z) const {
    lock_x = pimpl->linear_lock[0]; lock_y = pimpl->linear_lock[1]; lock_z = pimpl->linear_lock[2];
}
void RigidBody3D::get_angular_lock_axes(bool& lock_x, bool& lock_y, bool& lock_z) const {
    lock_x = pimpl->angular_lock[0]; lock_y = pimpl->angular_lock[1]; lock_z = pimpl->angular_lock[2];
}

void RigidBody3D::set_ccd_enabled(bool enabled) { pimpl->ccd_enabled = enabled; }
bool RigidBody3D::is_ccd_enabled() const { return pimpl->ccd_enabled; }
void RigidBody3D::set_ccd_motion_threshold(float threshold) { pimpl->ccd_threshold = threshold; }
float RigidBody3D::get_ccd_motion_threshold() const { return pimpl->ccd_threshold; }

void RigidBody3D::set_max_linear_velocity(float max_vel) { pimpl->max_linear_velocity = std::max(0.0f, max_vel); }
float RigidBody3D::get_max_linear_velocity() const { return pimpl->max_linear_velocity; }
void RigidBody3D::set_max_angular_velocity(float max_vel) { pimpl->max_angular_velocity = std::max(0.0f, max_vel); }
float RigidBody3D::get_max_angular_velocity() const { return pimpl->max_angular_velocity; }

void RigidBody3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; PhysicsBody3D::set_cast_shadow(cast); }
void RigidBody3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
void RigidBody3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; PhysicsBody3D::set_gi_mode(mode); }
void RigidBody3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
void RigidBody3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void RigidBody3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}

void RigidBody3D::Impl::integrate(double dt) {
    if (freeze_enabled && freeze_mode != BodyFreezeMode::FREEZE_NONE) return;
    if (sleeping) return;

    // Linear acceleration: (force/mass) + gravity
    double gravity[3] = {0, -9.8 * gravity_scale, 0};
    double inv_mass = 1.0 / mass;
    double acc[3] = {
        (accumulated_force[0] * inv_mass) + gravity[0],
        (accumulated_force[1] * inv_mass) + gravity[1],
        (accumulated_force[2] * inv_mass) + gravity[2]
    };
    // Euler integration
    linear_velocity[0] += acc[0] * dt;
    linear_velocity[1] += acc[1] * dt;
    linear_velocity[2] += acc[2] * dt;
    // Apply damping
    linear_velocity[0] *= (1.0 - linear_damping * dt);
    linear_velocity[1] *= (1.0 - linear_damping * dt);
    linear_velocity[2] *= (1.0 - linear_damping * dt);
    // Clamp max velocities
    double lin_speed = sqrt(linear_velocity[0]*linear_velocity[0] +
                            linear_velocity[1]*linear_velocity[1] +
                            linear_velocity[2]*linear_velocity[2]);
    if (lin_speed > max_linear_velocity) {
        double scale = max_linear_velocity / lin_speed;
        linear_velocity[0] *= scale;
        linear_velocity[1] *= scale;
        linear_velocity[2] *= scale;
    }
    // Apply linear locks
    if (linear_lock[0]) linear_velocity[0] = 0;
    if (linear_lock[1]) linear_velocity[1] = 0;
    if (linear_lock[2]) linear_velocity[2] = 0;

    // Angular acceleration: torque * inv_inertia
    double inv_inertia[9] = {1.0/inertia[0], 1.0/inertia[4], 1.0/inertia[8]}; // diagonal
    double ang_acc[3] = {
        accumulated_torque[0] * inv_inertia[0],
        accumulated_torque[1] * inv_inertia[4],
        accumulated_torque[2] * inv_inertia[8]
    };
    angular_velocity[0] += ang_acc[0] * dt;
    angular_velocity[1] += ang_acc[1] * dt;
    angular_velocity[2] += ang_acc[2] * dt;
    angular_velocity[0] *= (1.0 - angular_damping * dt);
    angular_velocity[1] *= (1.0 - angular_damping * dt);
    angular_velocity[2] *= (1.0 - angular_damping * dt);
    // Clamp angular velocity
    double ang_speed = sqrt(angular_velocity[0]*angular_velocity[0] +
                            angular_velocity[1]*angular_velocity[1] +
                            angular_velocity[2]*angular_velocity[2]);
    if (ang_speed > max_angular_velocity) {
        double scale = max_angular_velocity / ang_speed;
        angular_velocity[0] *= scale;
        angular_velocity[1] *= scale;
        angular_velocity[2] *= scale;
    }
    // Apply angular locks
    if (angular_lock[0]) angular_velocity[0] = 0;
    if (angular_lock[1]) angular_velocity[1] = 0;
    if (angular_lock[2]) angular_velocity[2] = 0;

    // Position update
    Transform3D transform = get_global_transform();
    transform.origin[0] += linear_velocity[0] * dt;
    transform.origin[1] += linear_velocity[1] * dt;
    transform.origin[2] += linear_velocity[2] * dt;
    // Rotation update (simple Euler integration of angular velocity -> quaternion)
    // For brevity, we skip full quaternion update (would add to transform).
    set_global_transform(transform);
}

void RigidBody3D::Impl::apply_constraints() {
    // Not implemented for demo – would include collision response and sleeping threshold.
}

void RigidBody3D::update_physics(double delta_time) {
    if (delta_time > 0.033) delta_time = 0.033; // cap
    if (!pimpl->freeze_enabled || pimpl->freeze_mode == BodyFreezeMode::FREEZE_NONE) {
        pimpl->integrate(delta_time);
        pimpl->apply_constraints();
        clear_forces();
        // Sleeping heuristic: if linear and angular velocities are very low, can sleep
        if (pimpl->can_sleep) {
            double lin_speed = sqrt(pimpl->linear_velocity[0]*pimpl->linear_velocity[0] +
                                    pimpl->linear_velocity[1]*pimpl->linear_velocity[1] +
                                    pimpl->linear_velocity[2]*pimpl->linear_velocity[2]);
            double ang_speed = sqrt(pimpl->angular_velocity[0]*pimpl->angular_velocity[0] +
                                    pimpl->angular_velocity[1]*pimpl->angular_velocity[1] +
                                    pimpl->angular_velocity[2]*pimpl->angular_velocity[2]);
            if (lin_speed < 0.01 && ang_speed < 0.01) {
                set_sleeping(true);
            }
        }
    }
    PhysicsBody3D::update_physics(delta_time);
}

void RigidBody3D::synchronize_render_server(double delta) {
    PhysicsBody3D::synchronize_render_server(delta);
    if (pimpl->emissive_intensity > 0.0f && pimpl->gi_mode > 0) {
        // Register dynamic emissive contribution to GI (placeholder)
    }
}

} // namespace lighting