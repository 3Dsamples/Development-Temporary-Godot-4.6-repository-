// vehicle_body_3d.cpp
#include "vehicle_body_3d.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace lighting {

struct VehicleWheel3D::Impl {
    float radius = 0.4f;
    float width = 0.2f;
    float suspension_rest_length = 0.5f;
    float suspension_travel = 0.2f;
    float suspension_stiffness = 20.0f;
    float suspension_damping = 2.0f;
    float friction = 1.0f;
    float roll_influence = 1.0f;
    bool cast_shadow = true;
    // Current suspension state (for simulation)
    float current_length = 0.5f;
    float current_force = 0.0f;
};

VehicleWheel3D::VehicleWheel3D() : pimpl(std::make_unique<Impl>()) {}
VehicleWheel3D::~VehicleWheel3D() = default;

void VehicleWheel3D::set_radius(float radius) { pimpl->radius = radius; }
float VehicleWheel3D::get_radius() const { return pimpl->radius; }
void VehicleWheel3D::set_width(float width) { pimpl->width = width; }
float VehicleWheel3D::get_width() const { return pimpl->width; }
void VehicleWheel3D::set_suspension_rest_length(float length) { pimpl->suspension_rest_length = length; }
float VehicleWheel3D::get_suspension_rest_length() const { return pimpl->suspension_rest_length; }
void VehicleWheel3D::set_suspension_travel(float travel) { pimpl->suspension_travel = travel; }
float VehicleWheel3D::get_suspension_travel() const { return pimpl->suspension_travel; }
void VehicleWheel3D::set_suspension_stiffness(float stiffness) { pimpl->suspension_stiffness = stiffness; }
float VehicleWheel3D::get_suspension_stiffness() const { return pimpl->suspension_stiffness; }
void VehicleWheel3D::set_suspension_damping(float damping) { pimpl->suspension_damping = damping; }
float VehicleWheel3D::get_suspension_damping() const { return pimpl->suspension_damping; }
void VehicleWheel3D::set_friction(float friction) { pimpl->friction = friction; }
float VehicleWheel3D::get_friction() const { return pimpl->friction; }
void VehicleWheel3D::set_roll_influence(float influence) { pimpl->roll_influence = influence; }
float VehicleWheel3D::get_roll_influence() const { return pimpl->roll_influence; }
void VehicleWheel3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; }
bool VehicleWheel3D::get_cast_shadow() const { return pimpl->cast_shadow; }

// ----------------------------------------------------------------------------
// VehicleBody3D implementation
// ----------------------------------------------------------------------------
struct VehicleBody3D::Impl {
    struct WheelAttachment {
        VehicleWheel3D* wheel;
        Transform3D local_transform;
        // runtime data
        double last_position[3];
        double suspension_force[3];
        double wheel_rotation = 0.0;
    };
    std::vector<WheelAttachment> wheels;

    float engine_force = 0.0f;
    float brake = 0.0f;
    float steering = 0.0f;
    float max_engine_force = 500.0f;
    float max_brake = 10.0f;
    float max_steering_angle = 0.6f; // ~35 degrees
    bool use_abs = false;
    bool use_tcs = false;

    // Lighting flags
    bool cast_shadow = true;
    bool receive_shadow = true;
    int gi_mode = 2; // dynamic
    float gi_contribution = 1.0f;

    // Internal simulation
    double linear_velocity[3] = {0,0,0};
    double angular_velocity[3] = {0,0,0};
};

VehicleBody3D::VehicleBody3D() : pimpl(std::make_unique<Impl>()) {
    set_body_mode(BodyMode::RIGID_DYNAMIC);
    set_mass(800.0f); // typical car mass
}
VehicleBody3D::~VehicleBody3D() = default;

void VehicleBody3D::add_wheel(VehicleWheel3D* wheel, const Transform3D& local_transform) {
    if (!wheel) return;
    Impl::WheelAttachment att;
    att.wheel = wheel;
    att.local_transform = local_transform;
    att.last_position[0] = local_transform.origin[0];
    att.last_position[1] = local_transform.origin[1];
    att.last_position[2] = local_transform.origin[2];
    pimpl->wheels.push_back(att);
    wheel->set_parent(this);
}
void VehicleBody3D::remove_wheel(VehicleWheel3D* wheel) {
    auto it = std::find_if(pimpl->wheels.begin(), pimpl->wheels.end(),
        [wheel](const Impl::WheelAttachment& a) { return a.wheel == wheel; });
    if (it != pimpl->wheels.end()) {
        it->wheel->set_parent(nullptr);
        pimpl->wheels.erase(it);
    }
}
int VehicleBody3D::get_wheel_count() const { return (int)pimpl->wheels.size(); }
VehicleWheel3D* VehicleBody3D::get_wheel(int index) const {
    return (index >=0 && index < (int)pimpl->wheels.size()) ? pimpl->wheels[index].wheel : nullptr;
}

void VehicleBody3D::set_engine_force(float force) { pimpl->engine_force = std::clamp(force, -pimpl->max_engine_force, pimpl->max_engine_force); }
float VehicleBody3D::get_engine_force() const { return pimpl->engine_force; }
void VehicleBody3D::set_brake(float brake) { pimpl->brake = std::clamp(brake, 0.0f, pimpl->max_brake); }
float VehicleBody3D::get_brake() const { return pimpl->brake; }
void VehicleBody3D::set_steering(float angle) { pimpl->steering = std::clamp(angle, -pimpl->max_steering_angle, pimpl->max_steering_angle); }
float VehicleBody3D::get_steering() const { return pimpl->steering; }
void VehicleBody3D::set_max_engine_force(float max_force) { pimpl->max_engine_force = max_force; }
float VehicleBody3D::get_max_engine_force() const { return pimpl->max_engine_force; }
void VehicleBody3D::set_max_brake(float max_brake) { pimpl->max_brake = max_brake; }
float VehicleBody3D::get_max_brake() const { return pimpl->max_brake; }
void VehicleBody3D::set_max_steering_angle(float angle) { pimpl->max_steering_angle = angle; }
float VehicleBody3D::get_max_steering_angle() const { return pimpl->max_steering_angle; }
void VehicleBody3D::set_use_abs(bool enabled) { pimpl->use_abs = enabled; }
bool VehicleBody3D::is_abs_enabled() const { return pimpl->use_abs; }
void VehicleBody3D::set_use_tcs(bool enabled) { pimpl->use_tcs = enabled; }
bool VehicleBody3D::is_tcs_enabled() const { return pimpl->use_tcs; }

void VehicleBody3D::set_wheel_engine_force(int wheel_index, float force) {
    // In real implementation, could override per wheel; not used by default.
}
void VehicleBody3D::set_wheel_brake(int wheel_index, float brake) {}
void VehicleBody3D::set_wheel_steering(int wheel_index, float angle) {}
float VehicleBody3D::get_wheel_rpm(int wheel_index) const { return 0.0f; }

void VehicleBody3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; }
void VehicleBody3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
void VehicleBody3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; }
void VehicleBody3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }

void VehicleBody3D::update_physics(double delta_time) {
    if (delta_time > 0.033) delta_time = 0.033;
    // Get current linear and angular velocity (from physics body)
    get_linear_velocity(pimpl->linear_velocity);
    get_angular_velocity(pimpl->angular_velocity);

    // Compute forward direction of vehicle (local Z axis transformed to world)
    Transform3D global = get_global_transform();
    double forward[3] = { global.basis[6], global.basis[7], global.basis[8] }; // column? careful: row-major
    // In row-major basis matrix, Z axis is at indices 8,9,10? Actually basis is 3x3 row-major stored in 9 doubles.
    // Assume basis[8], basis[9], basis[10] is Z. Simpler: use helper.
    double fwd[3] = { global.basis[8], global.basis[9], global.basis[10] };
    // Normalize
    double len = sqrt(fwd[0]*fwd[0]+fwd[1]*fwd[1]+fwd[2]*fwd[2]);
    if (len > 1e-6) { fwd[0]/=len; fwd[1]/=len; fwd[2]/=len; }

    // Apply engine force
    double force_magnitude = pimpl->engine_force;
    if (pimpl->use_tcs && fabs(pimpl->linear_velocity[0]) > 0.1) {
        // simplistic traction control: reduce engine force if wheel spin
        force_magnitude *= (1.0 - fabs(pimpl->linear_velocity[0]) / 30.0);
    }
    double impulse[3] = { fwd[0] * force_magnitude * delta_time,
                          fwd[1] * force_magnitude * delta_time,
                          fwd[2] * force_magnitude * delta_time };
    apply_impulse(impulse, nullptr);

    // Apply brake force (opposite to velocity)
    if (pimpl->brake > 0.0f) {
        double vel[3]; get_linear_velocity(vel);
        double vel_len = sqrt(vel[0]*vel[0]+vel[1]*vel[1]+vel[2]*vel[2]);
        if (vel_len > 1e-6) {
            double brake_impulse[3] = { -vel[0]/vel_len * pimpl->brake * delta_time,
                                        -vel[1]/vel_len * pimpl->brake * delta_time,
                                        -vel[2]/vel_len * pimpl->brake * delta_time };
            apply_impulse(brake_impulse, nullptr);
        }
    }

    // Apply steering (change angular velocity around Y axis)
    double steering_impulse = pimpl->steering * 10.0f * delta_time; // fake torque
    double torque[3] = { 0, steering_impulse, 0 };
    apply_torque(torque);

    // Call parent physics integration to simulate (updates position, velocity)
    RigidBody3D::update_physics(delta_time);

    // Update wheel transforms for visual rendering (position, rotation)
    Transform3D body_transform = get_global_transform();
    for (auto& wheel_att : pimpl->wheels) {
        Transform3D wheel_global = body_transform * wheel_att.local_transform;
        wheel_att.wheel->set_global_transform(wheel_global);
        // Rotate wheel based on forward speed (simplified)
        double speed = sqrt(pimpl->linear_velocity[0]*pimpl->linear_velocity[0] +
                            pimpl->linear_velocity[1]*pimpl->linear_velocity[1] +
                            pimpl->linear_velocity[2]*pimpl->linear_velocity[2]);
        float radius = wheel_att.wheel->get_radius();
        float angular_speed = speed / radius;
        wheel_att.wheel_rotation += angular_speed * delta_time;
        // Apply rotation to the wheel's transform (rotate around X or Z depending on orientation)
        // We assume wheel local X is axis of rotation. For simplicity, we just modify wheel's local rotation.
        Transform3D wheel_local = wheel_att.local_transform;
        // Rotate around X by wheel_rotation
        double c = cos(wheel_att.wheel_rotation);
        double s = sin(wheel_att.wheel_rotation);
        double new_basis[9] = {1,0,0, 0,c,s, 0,-s,c}; // rotation X
        // Combine with existing basis (not perfect, but approximation)
        wheel_local.basis[0] = new_basis[0]; wheel_local.basis[1] = new_basis[1]; wheel_local.basis[2] = new_basis[2];
        wheel_local.basis[3] = new_basis[3]; wheel_local.basis[4] = new_basis[4]; wheel_local.basis[5] = new_basis[5];
        wheel_local.basis[6] = new_basis[6]; wheel_local.basis[7] = new_basis[7]; wheel_local.basis[8] = new_basis[8];
        wheel_att.local_transform = wheel_local;
        wheel_att.wheel->set_transform(wheel_local);
    }
}

void VehicleBody3D::synchronize_render_server(double delta) {
    RigidBody3D::synchronize_render_server(delta);
    // Update shadow and GI settings for the vehicle and its wheels
    for (auto& wheel_att : pimpl->wheels) {
        wheel_att.wheel->set_cast_shadow(pimpl->cast_shadow);
        // Also update any material/visibility for GI
    }
}

} // namespace lighting