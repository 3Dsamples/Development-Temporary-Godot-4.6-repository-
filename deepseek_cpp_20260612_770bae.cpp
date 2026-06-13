// Name : lighting enhancement
// File : scene/3d/rigid_body_3d_ext.cpp 34 of 60
// Description : Implementation of RigidBody3DExt with inertia tensor, freeze axes,
//               max velocity clamping, CCD, and RenderingServer lighting sync.
#include "rigid_body_3d_ext.h"
#include "servers/rendering_server.h"
#include "core/math/math_funcs.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include <cmath>

struct RigidBody3DExt::Impl {
    Vector3 inertia = Vector3(1.0f, 1.0f, 1.0f);
    Vector3 inv_inertia = Vector3(1.0f, 1.0f, 1.0f);
    bool freeze_enabled = false;
    bool freeze_linear[3] = {false, false, false};
    bool freeze_angular[3] = {false, false, false};
    float max_linear_velocity = 1000.0f;
    float max_angular_velocity = 1000.0f;

    void update_inertia() {
        // Assuming mass is accessible via base, but we store inertia directly.
        // For simplicity, set inv_inertia = 1 / inertia (if non-zero)
        inv_inertia.x = (inertia.x > 0.0f) ? 1.0f / inertia.x : 0.0f;
        inv_inertia.y = (inertia.y > 0.0f) ? 1.0f / inertia.y : 0.0f;
        inv_inertia.z = (inertia.z > 0.0f) ? 1.0f / inertia.z : 0.0f;
    }

    void apply_freeze_and_clamp(Vector3 &lin_vel, Vector3 &ang_vel) const {
        if (freeze_enabled) {
            if (freeze_linear[0]) lin_vel.x = 0.0f;
            if (freeze_linear[1]) lin_vel.y = 0.0f;
            if (freeze_linear[2]) lin_vel.z = 0.0f;
            if (freeze_angular[0]) ang_vel.x = 0.0f;
            if (freeze_angular[1]) ang_vel.y = 0.0f;
            if (freeze_angular[2]) ang_vel.z = 0.0f;
        }
        // Clamp linear velocity
        float lin_speed = lin_vel.length();
        if (lin_speed > max_linear_velocity && max_linear_velocity > 0.0f) {
            lin_vel *= max_linear_velocity / lin_speed;
        }
        // Clamp angular velocity
        float ang_speed = ang_vel.length();
        if (ang_speed > max_angular_velocity && max_angular_velocity > 0.0f) {
            ang_vel *= max_angular_velocity / ang_speed;
        }
    }
};

RigidBody3DExt::RigidBody3DExt() {
    pimpl = new Impl();
    pimpl->update_inertia();
}

RigidBody3DExt::~RigidBody3DExt() {
    delete pimpl;
}

void RigidBody3DExt::set_mass(float p_mass) {
    PhysicsBody3DExt::set_mass(p_mass);
    // Inertia might depend on mass; but we keep explicit inertia.
    pimpl->update_inertia();
}
float RigidBody3DExt::get_mass() const { return PhysicsBody3DExt::get_mass(); }

void RigidBody3DExt::set_inertia(const Vector3 &p_inertia) {
    pimpl->inertia = p_inertia;
    pimpl->update_inertia();
}
Vector3 RigidBody3DExt::get_inertia() const { return pimpl->inertia; }

void RigidBody3DExt::set_freeze_enabled(bool p_enabled) {
    pimpl->freeze_enabled = p_enabled;
}
bool RigidBody3DExt::is_freeze_enabled() const { return pimpl->freeze_enabled; }

void RigidBody3DExt::set_freeze_linear(bool p_freeze_x, bool p_freeze_y, bool p_freeze_z) {
    pimpl->freeze_linear[0] = p_freeze_x;
    pimpl->freeze_linear[1] = p_freeze_y;
    pimpl->freeze_linear[2] = p_freeze_z;
}
void RigidBody3DExt::get_freeze_linear(bool &r_freeze_x, bool &r_freeze_y, bool &r_freeze_z) const {
    r_freeze_x = pimpl->freeze_linear[0];
    r_freeze_y = pimpl->freeze_linear[1];
    r_freeze_z = pimpl->freeze_linear[2];
}
void RigidBody3DExt::set_freeze_angular(bool p_freeze_x, bool p_freeze_y, bool p_freeze_z) {
    pimpl->freeze_angular[0] = p_freeze_x;
    pimpl->freeze_angular[1] = p_freeze_y;
    pimpl->freeze_angular[2] = p_freeze_z;
}
void RigidBody3DExt::get_freeze_angular(bool &r_freeze_x, bool &r_freeze_y, bool &r_freeze_z) const {
    r_freeze_x = pimpl->freeze_angular[0];
    r_freeze_y = pimpl->freeze_angular[1];
    r_freeze_z = pimpl->freeze_angular[2];
}

void RigidBody3DExt::set_ccd_enabled(bool p_enabled) {
    PhysicsBody3DExt::set_ccd_enabled(p_enabled);
}
bool RigidBody3DExt::is_ccd_enabled() const { return PhysicsBody3DExt::is_ccd_enabled(); }
void RigidBody3DExt::set_ccd_threshold(float p_threshold) {
    PhysicsBody3DExt::set_ccd_threshold(p_threshold);
}
float RigidBody3DExt::get_ccd_threshold() const { return PhysicsBody3DExt::get_ccd_threshold(); }

void RigidBody3DExt::set_max_linear_velocity(float p_max) {
    pimpl->max_linear_velocity = p_max;
}
float RigidBody3DExt::get_max_linear_velocity() const { return pimpl->max_linear_velocity; }
void RigidBody3DExt::set_max_angular_velocity(float p_max) {
    pimpl->max_angular_velocity = p_max;
}
float RigidBody3DExt::get_max_angular_velocity() const { return pimpl->max_angular_velocity; }

void RigidBody3DExt::set_cast_shadow(bool p_cast) {
    PhysicsBody3DExt::set_cast_shadow(p_cast);
}
void RigidBody3DExt::set_gi_mode(int p_mode) {
    PhysicsBody3DExt::set_gi_mode(p_mode);
}
void RigidBody3DExt::set_gi_contribution(float p_amount) {
    PhysicsBody3DExt::set_gi_contribution(p_amount);
}
void RigidBody3DExt::set_emissive(const Color &p_color, float p_intensity) {
    PhysicsBody3DExt::set_emissive(p_color, p_intensity);
}

void RigidBody3DExt::sync_rigid_body() {
    PhysicsBody3DExt::sync_physics_body();
    // Additional sync if needed (inertia, freeze) – but rendering server does not care.
}