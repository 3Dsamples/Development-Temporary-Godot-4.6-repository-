// physics_body_3d.cpp
#include "physics_body_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace lighting {

struct PhysicsBody3D::Impl {
    BodyMode mode = BodyMode::STATIC;
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
    bool ccd_enabled = false;
    float ccd_threshold = 0.01f;

    float gi_contribution = 1.0f;
    bool cast_gi_shadow = true;

    // Internal physics state (simplified)
    double world_position[3] = {0,0,0};
    double world_rotation[9] = {1,0,0, 0,1,0, 0,0,1};
};

PhysicsBody3D::PhysicsBody3D() : pimpl(std::make_unique<Impl>()) {}
PhysicsBody3D::~PhysicsBody3D() = default;

void PhysicsBody3D::set_body_mode(BodyMode mode) {
    pimpl->mode = mode;
    // Notify physics server
}
BodyMode PhysicsBody3D::get_body_mode() const { return pimpl->mode; }

void PhysicsBody3D::set_mass(float mass) { pimpl->mass = mass; }
float PhysicsBody3D::get_mass() const { return pimpl->mass; }
void PhysicsBody3D::set_inertia(const float* inertia_tensor) {
    memcpy(pimpl->inertia, inertia_tensor, 9*sizeof(float));
}
void PhysicsBody3D::get_inertia(float* out_inertia) const {
    memcpy(out_inertia, pimpl->inertia, 9*sizeof(float));
}
void PhysicsBody3D::set_gravity_scale(float scale) { pimpl->gravity_scale = scale; }
float PhysicsBody3D::get_gravity_scale() const { return pimpl->gravity_scale; }
void PhysicsBody3D::set_linear_damping(float damping) { pimpl->linear_damping = damping; }
float PhysicsBody3D::get_linear_damping() const { return pimpl->linear_damping; }
void PhysicsBody3D::set_angular_damping(float damping) { pimpl->angular_damping = damping; }
float PhysicsBody3D::get_angular_damping() const { return pimpl->angular_damping; }

void PhysicsBody3D::set_linear_velocity(const double* velocity) {
    memcpy(pimpl->linear_velocity, velocity, 3*sizeof(double));
}
void PhysicsBody3D::get_linear_velocity(double* out_velocity) const {
    memcpy(out_velocity, pimpl->linear_velocity, 3*sizeof(double));
}
void PhysicsBody3D::set_angular_velocity(const double* velocity) {
    memcpy(pimpl->angular_velocity, velocity, 3*sizeof(double));
}
void PhysicsBody3D::get_angular_velocity(double* out_velocity) const {
    memcpy(out_velocity, pimpl->angular_velocity, 3*sizeof(double));
}

void PhysicsBody3D::apply_force(const double* force, const double* at_position) {
    if (pimpl->mode != BodyMode::RIGID_DYNAMIC) return;
    pimpl->accumulated_force[0] += force[0];
    pimpl->accumulated_force[1] += force[1];
    pimpl->accumulated_force[2] += force[2];
}
void PhysicsBody3D::apply_impulse(const double* impulse, const double* at_position) {
    if (pimpl->mode != BodyMode::RIGID_DYNAMIC) return;
    double inv_mass = 1.0 / pimpl->mass;
    pimpl->linear_velocity[0] += impulse[0] * inv_mass;
    pimpl->linear_velocity[1] += impulse[1] * inv_mass;
    pimpl->linear_velocity[2] += impulse[2] * inv_mass;
}
void PhysicsBody3D::apply_torque(const double* torque) {
    if (pimpl->mode != BodyMode::RIGID_DYNAMIC) return;
    pimpl->accumulated_torque[0] += torque[0];
    pimpl->accumulated_torque[1] += torque[1];
    pimpl->accumulated_torque[2] += torque[2];
}
void PhysicsBody3D::clear_forces() {
    pimpl->accumulated_force[0] = pimpl->accumulated_force[1] = pimpl->accumulated_force[2] = 0;
    pimpl->accumulated_torque[0] = pimpl->accumulated_torque[1] = pimpl->accumulated_torque[2] = 0;
}

void PhysicsBody3D::set_sleeping(bool sleeping) { pimpl->sleeping = sleeping; }
bool PhysicsBody3D::is_sleeping() const { return pimpl->sleeping; }
void PhysicsBody3D::set_can_sleep(bool can_sleep) { pimpl->can_sleep = can_sleep; }
bool PhysicsBody3D::can_sleep() const { return pimpl->can_sleep; }

void PhysicsBody3D::set_ccd_enabled(bool enabled) { pimpl->ccd_enabled = enabled; }
bool PhysicsBody3D::is_ccd_enabled() const { return pimpl->ccd_enabled; }
void PhysicsBody3D::set_ccd_motion_threshold(float threshold) { pimpl->ccd_threshold = threshold; }
float PhysicsBody3D::get_ccd_motion_threshold() const { return pimpl->ccd_threshold; }

void PhysicsBody3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
float PhysicsBody3D::get_gi_contribution() const { return pimpl->gi_contribution; }
void PhysicsBody3D::set_cast_gi_shadow(bool cast) { pimpl->cast_gi_shadow = cast; }
bool PhysicsBody3D::get_cast_gi_shadow() const { return pimpl->cast_gi_shadow; }

void PhysicsBody3D::_integrate_forces(double delta_time) {
    if (pimpl->mode != BodyMode::RIGID_DYNAMIC || pimpl->sleeping) return;
    double inv_mass = 1.0 / pimpl->mass;
    // Linear acceleration: F/m + gravity
    double gravity[3] = {0, -9.8 * pimpl->gravity_scale, 0};
    double acc[3] = {
        pimpl->accumulated_force[0] * inv_mass + gravity[0],
        pimpl->accumulated_force[1] * inv_mass + gravity[1],
        pimpl->accumulated_force[2] * inv_mass + gravity[2]
    };
    // Euler integration
    pimpl->linear_velocity[0] += acc[0] * delta_time;
    pimpl->linear_velocity[1] += acc[1] * delta_time;
    pimpl->linear_velocity[2] += acc[2] * delta_time;
    pimpl->linear_velocity[0] *= (1.0 - pimpl->linear_damping * delta_time);
    pimpl->linear_velocity[1] *= (1.0 - pimpl->linear_damping * delta_time);
    pimpl->linear_velocity[2] *= (1.0 - pimpl->linear_damping * delta_time);
    // Position update
    pimpl->world_position[0] += pimpl->linear_velocity[0] * delta_time;
    pimpl->world_position[1] += pimpl->linear_velocity[1] * delta_time;
    pimpl->world_position[2] += pimpl->linear_velocity[2] * delta_time;
    // Rotation (simple Euler for angular velocity)
    pimpl->angular_velocity[0] *= (1.0 - pimpl->angular_damping * delta_time);
    pimpl->angular_velocity[1] *= (1.0 - pimpl->angular_damping * delta_time);
    pimpl->angular_velocity[2] *= (1.0 - pimpl->angular_damping * delta_time);
    // Update transform (simplified – would integrate angular velocity to quaternion)
    Transform3D transform = get_global_transform();
    transform.origin[0] = pimpl->world_position[0];
    transform.origin[1] = pimpl->world_position[1];
    transform.origin[2] = pimpl->world_position[2];
    set_global_transform(transform);
    clear_forces();
    // Sleeping heuristic (if velocity very low)
    if (pimpl->can_sleep) {
        double speed_sq = pimpl->linear_velocity[0]*pimpl->linear_velocity[0] +
                          pimpl->linear_velocity[1]*pimpl->linear_velocity[1] +
                          pimpl->linear_velocity[2]*pimpl->linear_velocity[2];
        if (speed_sq < 1e-6) pimpl->sleeping = true;
    }
}

void PhysicsBody3D::update_physics(double delta_time) {
    if (delta_time > 0.033) delta_time = 0.033; // cap
    _integrate_forces(delta_time);
}

void PhysicsBody3D::synchronize_render_server(double delta) {
    CollisionObject3D::synchronize_render_server(delta);
    // Update visual instance transform
    _update_render_instance_transform();
}

// ============================================================================
// RigidBody3D
// ============================================================================
struct RigidBody3D::Impl {
    bool frozen = false;
    int freeze_mode = 0;
};
RigidBody3D::RigidBody3D() : pimpl(std::make_unique<Impl>()) {}
RigidBody3D::~RigidBody3D() = default;
void RigidBody3D::set_freeze(bool freeze) { pimpl->frozen = freeze; }
bool RigidBody3D::is_frozen() const { return pimpl->frozen; }
void RigidBody3D::set_freeze_mode(int mode) { pimpl->freeze_mode = mode; }

// ============================================================================
// CharacterBody3D
// ============================================================================
struct CharacterBody3D::Impl {
    double velocity[3] = {0,0,0};
    double up_direction[3] = {0,1,0};
    bool floor_stop_on_slope = true;
    float floor_max_angle = 1.396f; // 80 degrees in radians
    float wall_min_angle = 0.0f;
    bool on_floor = false;
    bool on_wall = false;
    bool on_ceiling = false;
    std::vector<std::pair<double[3], double[3]>> slide_collisions; // position, normal
};
CharacterBody3D::CharacterBody3D() : pimpl(std::make_unique<Impl>()) {}
CharacterBody3D::~CharacterBody3D() = default;
void CharacterBody3D::set_velocity(const double* velocity) { memcpy(pimpl->velocity, velocity, 3*sizeof(double)); }
const double* CharacterBody3D::get_velocity() const { return pimpl->velocity; }
void CharacterBody3D::set_up_direction(const double* up) { memcpy(pimpl->up_direction, up, 3*sizeof(double)); }
void CharacterBody3D::set_floor_stop_on_slope(bool enabled) { pimpl->floor_stop_on_slope = enabled; }
void CharacterBody3D::set_floor_max_angle(float radians) { pimpl->floor_max_angle = radians; }
void CharacterBody3D::set_wall_min_angle(float radians) { pimpl->wall_min_angle = radians; }
void CharacterBody3D::move_and_slide() {
    // Simplified movement with sliding (not full implementation)
    double new_pos[3];
    new_pos[0] = pimpl->velocity[0] * 0.016; // assuming delta
    new_pos[1] = pimpl->velocity[1] * 0.016;
    new_pos[2] = pimpl->velocity[2] * 0.016;
    // Collision detection and response (placeholder)
    pimpl->on_floor = (pimpl->velocity[1] <= 0.0);
    pimpl->on_wall = false;
    pimpl->on_ceiling = false;
    Transform3D trans = get_global_transform();
    trans.origin[0] += new_pos[0];
    trans.origin[1] += new_pos[1];
    trans.origin[2] += new_pos[2];
    set_global_transform(trans);
}
bool CharacterBody3D::is_on_floor() const { return pimpl->on_floor; }
bool CharacterBody3D::is_on_wall() const { return pimpl->on_wall; }
bool CharacterBody3D::is_on_ceiling() const { return pimpl->on_ceiling; }
int CharacterBody3D::get_slide_count() const { return (int)pimpl->slide_collisions.size(); }
void CharacterBody3D::get_last_slide_collision(int idx, double* out_position, double* out_normal) {
    if (idx < (int)pimpl->slide_collisions.size()) {
        memcpy(out_position, pimpl->slide_collisions[idx].first, 3*sizeof(double));
        memcpy(out_normal, pimpl->slide_collisions[idx].second, 3*sizeof(double));
    }
}

} // namespace lighting