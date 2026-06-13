// Name : lighting enhancement
// File : scene/3d/physics_body_3d_ext.cpp 32 of 60
// Description : Implementation of PhysicsBody3DExt with mass, velocity, forces,
//               damping, CCD, sleeping, physics interpolation, and full RenderingServer sync.
#include "physics_body_3d_ext.h"
#include "servers/rendering_server.h"
#include "core/math/math_funcs.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "scene/3d/node_3d_ext.h" // for transform sync
#include <cmath>

struct PhysicsBody3DExt::Impl {
    float mass = 1.0f;
    float inv_mass = 1.0f;
    Vector3 linear_velocity;
    Vector3 angular_velocity;
    float gravity_scale = 1.0f;
    float linear_damp = 0.0f;
    float angular_damp = 0.0f;
    Vector3 accumulated_force;
    Vector3 accumulated_torque;
    bool sleeping = false;
    bool can_sleep = true;
    bool ccd_enabled = false;
    float ccd_threshold = 0.01f;

    bool cast_shadow = true;
    int gi_mode = 2;              // dynamic
    float gi_contribution = 1.0f;
    Color emissive_color;
    float emissive_intensity = 0.0f;

    bool interpolate = true;
    double physics_fraction = 0.0;
    Transform3D previous_physics_transform;
    Transform3D current_physics_transform;

    bool body_dirty = true;

    Impl() {
        previous_physics_transform.set_identity();
        current_physics_transform.set_identity();
    }

    void update_mass() {
        inv_mass = (mass > 0.0f) ? 1.0f / mass : 0.0f;
    }

    void apply_force(const Vector3 &p_force, const Vector3 &p_position) {
        if (sleeping) return;
        accumulated_force += p_force;
        if (p_position != Vector3()) {
            // Torque = r × F
            Vector3 r = p_position - current_physics_transform.origin;
            accumulated_torque += r.cross(p_force);
        }
    }

    void apply_impulse(const Vector3 &p_impulse, const Vector3 &p_position) {
        if (sleeping) return;
        linear_velocity += p_impulse * inv_mass;
        if (p_position != Vector3()) {
            Vector3 r = p_position - current_physics_transform.origin;
            Vector3 torque_impulse = r.cross(p_impulse);
            // Convert torque impulse to angular velocity delta: Δω = I⁻¹ · τ (diagonal inertia assumed)
            // For simplicity, use a constant inertia of 1.0 per axis.
            angular_velocity += torque_impulse;
        }
        if (can_sleep) sleeping = false;
    }

    void apply_torque(const Vector3 &p_torque) {
        if (sleeping) return;
        accumulated_torque += p_torque;
    }

    void integrate(double p_dt) {
        if (sleeping || p_dt <= 0.0) return;

        // Limit timestep to avoid explosion
        p_dt = MIN(p_dt, 0.033);

        // Linear acceleration: a = F/m + g
        Vector3 gravity(0, -9.8f * gravity_scale, 0);
        Vector3 acceleration = accumulated_force * inv_mass + gravity;
        linear_velocity += acceleration * p_dt;
        linear_velocity *= (1.0f - linear_damp * p_dt);
        // CCD: check if velocity exceeds threshold, would need swept tests (simplified)
        if (ccd_enabled && linear_velocity.length() > ccd_threshold) {
            // In full engine, perform CCD collision detection here
        }

        // Angular acceleration: α = τ / I (I = 1)
        Vector3 ang_acc = accumulated_torque;
        angular_velocity += ang_acc * p_dt;
        angular_velocity *= (1.0f - angular_damp * p_dt);

        // Update transform (Euler)
        Vector3 new_position = current_physics_transform.origin + linear_velocity * p_dt;
        // Rotation: integrate angular velocity into quaternion (simplified: Euler angles)
        // For simplicity, we update basis using Rodrigues' rotation formula.
        // In production, use quaternion integration.
        float angle = angular_velocity.length() * p_dt;
        if (angle > 0.0f) {
            Vector3 axis = angular_velocity.normalized();
            Transform3D rot;
            rot.rotate_basis(axis, angle);
            current_physics_transform = rot * current_physics_transform;
        }
        current_physics_transform.origin = new_position;

        // Clear forces for next step
        accumulated_force = Vector3();
        accumulated_torque = Vector3();

        // Sleeping heuristic (if very low velocity)
        if (can_sleep && linear_velocity.length_squared() < 0.01f && angular_velocity.length_squared() < 0.01f) {
            sleeping = true;
        }
    }

    void sync_render_server(Transform3D &p_current, Transform3D &p_previous, double p_frac) {
        if (!interpolate) {
            // Directly set transform (no interpolation)
            set_global_transform(current_physics_transform);
        } else {
            // Interpolate between previous and current transforms
            Transform3D interp = previous_physics_transform.interpolate_with(current_physics_transform, p_frac);
            set_global_transform(interp);
        }
        // Also sync lighting flags via Node3DExt (assuming our node is Node3DExt)
        // The base node will handle rendering server instance updates via its own sync.
    }
};

PhysicsBody3DExt::PhysicsBody3DExt() {
    pimpl = new Impl();
}

PhysicsBody3DExt::~PhysicsBody3DExt() {
    delete pimpl;
}

void PhysicsBody3DExt::set_mass(float p_mass) {
    pimpl->mass = p_mass;
    pimpl->update_mass();
}
float PhysicsBody3DExt::get_mass() const { return pimpl->mass; }

void PhysicsBody3DExt::set_linear_velocity(const Vector3 &p_velocity) {
    pimpl->linear_velocity = p_velocity;
}
Vector3 PhysicsBody3DExt::get_linear_velocity() const { return pimpl->linear_velocity; }

void PhysicsBody3DExt::set_angular_velocity(const Vector3 &p_velocity) {
    pimpl->angular_velocity = p_velocity;
}
Vector3 PhysicsBody3DExt::get_angular_velocity() const { return pimpl->angular_velocity; }

void PhysicsBody3DExt::set_gravity_scale(float p_scale) { pimpl->gravity_scale = p_scale; }
float PhysicsBody3DExt::get_gravity_scale() const { return pimpl->gravity_scale; }

void PhysicsBody3DExt::set_linear_damp(float p_damp) { pimpl->linear_damp = p_damp; }
float PhysicsBody3DExt::get_linear_damp() const { return pimpl->linear_damp; }

void PhysicsBody3DExt::set_angular_damp(float p_damp) { pimpl->angular_damp = p_damp; }
float PhysicsBody3DExt::get_angular_damp() const { return pimpl->angular_damp; }

void PhysicsBody3DExt::apply_force(const Vector3 &p_force, const Vector3 &p_position) {
    pimpl->apply_force(p_force, p_position);
}
void PhysicsBody3DExt::apply_impulse(const Vector3 &p_impulse, const Vector3 &p_position) {
    pimpl->apply_impulse(p_impulse, p_position);
}
void PhysicsBody3DExt::apply_torque(const Vector3 &p_torque) {
    pimpl->apply_torque(p_torque);
}
void PhysicsBody3DExt::clear_forces() {
    pimpl->accumulated_force = Vector3();
    pimpl->accumulated_torque = Vector3();
}

void PhysicsBody3DExt::set_sleeping(bool p_sleep) { pimpl->sleeping = p_sleep; }
bool PhysicsBody3DExt::is_sleeping() const { return pimpl->sleeping; }
void PhysicsBody3DExt::set_can_sleep(bool p_can_sleep) { pimpl->can_sleep = p_can_sleep; }
bool PhysicsBody3DExt::can_sleep() const { return pimpl->can_sleep; }

void PhysicsBody3DExt::set_ccd_enabled(bool p_enabled) { pimpl->ccd_enabled = p_enabled; }
bool PhysicsBody3DExt::is_ccd_enabled() const { return pimpl->ccd_enabled; }
void PhysicsBody3DExt::set_ccd_threshold(float p_threshold) { pimpl->ccd_threshold = p_threshold; }
float PhysicsBody3DExt::get_ccd_threshold() const { return pimpl->ccd_threshold; }

void PhysicsBody3DExt::set_cast_shadow(bool p_cast) { pimpl->cast_shadow = p_cast; sync_physics_body(); }
bool PhysicsBody3DExt::get_cast_shadow() const { return pimpl->cast_shadow; }
void PhysicsBody3DExt::set_gi_mode(int p_mode) { pimpl->gi_mode = p_mode; sync_physics_body(); }
int PhysicsBody3DExt::get_gi_mode() const { return pimpl->gi_mode; }
void PhysicsBody3DExt::set_gi_contribution(float p_amount) { pimpl->gi_contribution = p_amount; sync_physics_body(); }
float PhysicsBody3DExt::get_gi_contribution() const { return pimpl->gi_contribution; }
void PhysicsBody3DExt::set_emissive(const Color &p_color, float p_intensity) {
    pimpl->emissive_color = p_color;
    pimpl->emissive_intensity = p_intensity;
    sync_physics_body();
}
Color PhysicsBody3DExt::get_emissive() const { return pimpl->emissive_color; }
float PhysicsBody3DExt::get_emissive_intensity() const { return pimpl->emissive_intensity; }

void PhysicsBody3DExt::set_interpolate(bool p_enabled) { pimpl->interpolate = p_enabled; }
bool PhysicsBody3DExt::is_interpolated() const { return pimpl->interpolate; }

void PhysicsBody3DExt::set_physics_fraction(double p_fraction) {
    pimpl->physics_fraction = p_fraction;
}
void PhysicsBody3DExt::apply_interpolated_transform() {
    pimpl->sync_render_server(pimpl->current_physics_transform, pimpl->previous_physics_transform, pimpl->physics_fraction);
}

void PhysicsBody3DExt::sync_physics_body() {
    // Store previous physics transform before integrating new state
    pimpl->previous_physics_transform = pimpl->current_physics_transform;
    // Integration will be called by physics server, not here.
    // Update rendering server flags via base node (assume derived from Node3DExt)
    if (Node3DExt *node = cast_to<Node3DExt>(this)) {
        node->set_cast_shadow(pimpl->cast_shadow);
        node->set_gi_mode(pimpl->gi_mode);
        node->set_gi_contribution(pimpl->gi_contribution);
        node->set_emissive(pimpl->emissive_color, pimpl->emissive_intensity);
        node->sync_render_server_lighting_params();
    }
}