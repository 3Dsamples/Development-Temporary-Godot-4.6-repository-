// File 332: modules/wicked/src/bodies/wicked_body.cpp
// Implementation of WickedBody – mass, inertia, velocity, damping, forces,
// integration, sleep, CCD flags, and Godot bindings.

#include "wicked_body.h"
#include "../collision/wicked_shape.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

namespace wicked {

void WickedBody::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_type", "type"), &WickedBody::set_type);
    ClassDB::bind_method(D_METHOD("get_type"), &WickedBody::get_type);
    ClassDB::bind_method(D_METHOD("set_transform", "xform"), &WickedBody::set_transform);
    ClassDB::bind_method(D_METHOD("get_transform"), &WickedBody::get_transform);
    ClassDB::bind_method(D_METHOD("set_linear_velocity", "vel"), &WickedBody::set_linear_velocity);
    ClassDB::bind_method(D_METHOD("get_linear_velocity"), &WickedBody::get_linear_velocity);
    ClassDB::bind_method(D_METHOD("set_angular_velocity", "vel"), &WickedBody::set_angular_velocity);
    ClassDB::bind_method(D_METHOD("get_angular_velocity"), &WickedBody::get_angular_velocity);
    ClassDB::bind_method(D_METHOD("set_mass", "mass"), &WickedBody::set_mass);
    ClassDB::bind_method(D_METHOD("get_mass"), &WickedBody::get_mass);
    ClassDB::bind_method(D_METHOD("set_inertia", "inertia"), &WickedBody::set_inertia);
    ClassDB::bind_method(D_METHOD("get_inertia_local"), &WickedBody::get_inertia_local);
    ClassDB::bind_method(D_METHOD("set_linear_damping", "damping"), &WickedBody::set_linear_damping);
    ClassDB::bind_method(D_METHOD("get_linear_damping"), &WickedBody::get_linear_damping);
    ClassDB::bind_method(D_METHOD("set_angular_damping", "damping"), &WickedBody::set_angular_damping);
    ClassDB::bind_method(D_METHOD("get_angular_damping"), &WickedBody::get_angular_damping);
    ClassDB::bind_method(D_METHOD("apply_force", "force", "world_point"), &WickedBody::apply_force, DEFVAL(vec3()));
    ClassDB::bind_method(D_METHOD("apply_impulse", "impulse", "world_point"), &WickedBody::apply_impulse, DEFVAL(vec3()));
    ClassDB::bind_method(D_METHOD("apply_torque_impulse", "torque"), &WickedBody::apply_torque_impulse);
    ClassDB::bind_method(D_METHOD("set_gravity_enabled", "enabled"), &WickedBody::set_gravity_enabled);
    ClassDB::bind_method(D_METHOD("is_gravity_enabled"), &WickedBody::is_gravity_enabled);
    ClassDB::bind_method(D_METHOD("set_collision_shape", "shape"), &WickedBody::set_collision_shape);
    ClassDB::bind_method(D_METHOD("get_collision_shape"), &WickedBody::get_collision_shape);
    ClassDB::bind_method(D_METHOD("set_material_id", "id"), &WickedBody::set_material_id);
    ClassDB::bind_method(D_METHOD("get_material_id"), &WickedBody::get_material_id);
    ClassDB::bind_method(D_METHOD("set_ccd_enabled", "enabled"), &WickedBody::set_ccd_enabled);
    ClassDB::bind_method(D_METHOD("is_ccd_enabled"), &WickedBody::is_ccd_enabled);
    ClassDB::bind_method(D_METHOD("set_ccd_motion_threshold", "threshold"), &WickedBody::set_ccd_motion_threshold);
    ClassDB::bind_method(D_METHOD("get_ccd_motion_threshold"), &WickedBody::get_ccd_motion_threshold);
    ClassDB::bind_method(D_METHOD("set_ccd_swept_sphere_radius", "radius"), &WickedBody::set_ccd_swept_sphere_radius);
    ClassDB::bind_method(D_METHOD("get_ccd_swept_sphere_radius"), &WickedBody::get_ccd_swept_sphere_radius);
    ClassDB::bind_method(D_METHOD("set_deactivation_enabled", "enabled"), &WickedBody::set_deactivation_enabled);
    ClassDB::bind_method(D_METHOD("is_deactivation_enabled"), &WickedBody::is_deactivation_enabled);
    ClassDB::bind_method(D_METHOD("activate", "force"), &WickedBody::activate, DEFVAL(true));
    ClassDB::bind_method(D_METHOD("deactivate"), &WickedBody::deactivate);

    ADD_PROPERTY(PropertyInfo(Variant::INT, "type"), "set_type", "get_type");
    ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "transform"), "set_transform", "get_transform");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mass"), "set_mass", "get_mass");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "linear_damping"), "set_linear_damping", "get_linear_damping");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_damping"), "set_angular_damping", "get_angular_damping");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gravity_enabled"), "set_gravity_enabled", "is_gravity_enabled");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "collision_shape", PROPERTY_HINT_RESOURCE_TYPE, "WickedShape"), "set_collision_shape", "get_collision_shape");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "material_id"), "set_material_id", "get_material_id");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ccd_enabled"), "set_ccd_enabled", "is_ccd_enabled");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ccd_motion_threshold"), "set_ccd_motion_threshold", "get_ccd_motion_threshold");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ccd_swept_sphere_radius"), "set_ccd_swept_sphere_radius", "get_ccd_swept_sphere_radius");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "deactivation_enabled"), "set_deactivation_enabled", "is_deactivation_enabled");
}

WickedBody::WickedBody() :
    body_type(BodyType::DYNAMIC),
    mass(1.0),
    inverse_mass(1.0),
    linear_damping(0.0),
    angular_damping(0.0),
    mat_id(0),
    activation_state(ActivationState::ACTIVE_TAG),
    sleep_counter(0),
    gravity_enabled(true),
    deactivation_enabled(true),
    ccd_enabled(false),
    ccd_motion_threshold(0.0),
    ccd_swept_sphere_radius(0.2) {
    inertia_local.set_identity();
    inverse_inertia_local.set_identity();
    inverse_inertia_world.set_identity();
}

WickedBody::~WickedBody() {}

void WickedBody::set_type(BodyType p_type) {
    body_type = p_type;
    if (body_type != BodyType::DYNAMIC) {
        inverse_mass = 0.0;
        inverse_inertia_local = Basis();
        inverse_inertia_world = Basis();
        linear_velocity = vec3();
        angular_velocity = vec3();
    } else {
        set_mass(mass);
        set_inertia(inertia_local);
    }
}

void WickedBody::set_transform(const mat4 &p_xform) { transform = p_xform; update_inverse_inertia(); }
void WickedBody::set_linear_velocity(const vec3 &p_vel) { linear_velocity = p_vel; }
void WickedBody::set_angular_velocity(const vec3 &p_vel) { angular_velocity = p_vel; }

void WickedBody::set_mass(real_t p_mass) {
    mass = MAX(p_mass, 0.0);
    inverse_mass = (mass > 0.0) ? 1.0 / mass : 0.0;
}

void WickedBody::set_inertia(const mat3 &p_inertia) {
    inertia_local = p_inertia;
    inverse_inertia_local = inertia_local.inverse();
    update_inverse_inertia();
}

void WickedBody::update_inverse_inertia() {
    inverse_inertia_world = transform.basis * inverse_inertia_local * transform.basis.transposed();
}

void WickedBody::set_linear_damping(real_t p_damp) { linear_damping = CLAMP(p_damp, 0.0, 1.0); }
real_t WickedBody::get_linear_damping() const { return linear_damping; }
void WickedBody::set_angular_damping(real_t p_damp) { angular_damping = CLAMP(p_damp, 0.0, 1.0); }
real_t WickedBody::get_angular_damping() const { return angular_damping; }

void WickedBody::apply_force(const vec3 &p_force, const vec3 &p_world_point) {
    force_accum += p_force;
    vec3 r = p_world_point - transform.origin;
    torque_accum += r.cross(p_force);
}

void WickedBody::apply_impulse(const vec3 &p_impulse, const vec3 &p_world_point) {
    if (inverse_mass > 0.0) {
        linear_velocity += p_impulse * inverse_mass;
    }
    vec3 r = p_world_point - transform.origin;
    angular_velocity += inverse_inertia_world.xform(r.cross(p_impulse));
}

void WickedBody::apply_torque_impulse(const vec3 &p_torque) {
    angular_velocity += inverse_inertia_world.xform(p_torque);
}

void WickedBody::clear_forces() { force_accum = vec3(); torque_accum = vec3(); }

void WickedBody::integrate_velocity(real_t p_dt) {
    if (body_type != BodyType::DYNAMIC || inverse_mass <= 0.0) return;
    linear_velocity += force_accum * (inverse_mass * p_dt);
    angular_velocity += inverse_inertia_world.xform(torque_accum * p_dt);
    // Apply damping (v = v * (1 - damping * dt))
    linear_velocity *= (1.0 - linear_damping * p_dt);
    angular_velocity *= (1.0 - angular_damping * p_dt);
    clear_forces();
}

void WickedBody::integrate_position(real_t p_dt) {
    if (body_type != BodyType::DYNAMIC) return;
    transform.origin += linear_velocity * p_dt;
    real_t angle = angular_velocity.length();
    if (angle > CMP_EPSILON) {
        vec3 axis = angular_velocity / angle;
        real_t half_angle = angle * p_dt * 0.5;
        quat omega_quat(axis, half_angle);
        quat q_current(transform.basis);
        q_current = omega_quat * q_current;
        q_current.normalize();
        transform.basis = q_current.get_basis().orthonormalized();
    }
    update_inverse_inertia();
}

void WickedBody::set_activation_state(ActivationState p_state) { activation_state = p_state; }
ActivationState WickedBody::get_activation_state() const { return activation_state; }

void WickedBody::activate(bool p_force) {
    if (p_force || activation_state == ActivationState::ISLAND_SLEEPING || activation_state == ActivationState::WANTS_DEACTIVATION) {
        activation_state = ActivationState::ACTIVE_TAG;
        sleep_counter = 0;
    }
}

void WickedBody::deactivate() {
    activation_state = ActivationState::ISLAND_SLEEPING;
    linear_velocity = vec3();
    angular_velocity = vec3();
}

void WickedBody::set_collision_shape(const Ref<WickedShape> &p_shape) {
    collision_shape = p_shape;
    if (collision_shape.is_valid()) {
        cached_aabb = collision_shape->get_local_aabb();
    }
}

void WickedBody::set_material_id(material_id p_id) { mat_id = p_id; }
material_id WickedBody::get_material_id() const { return mat_id; }

void WickedBody::set_ccd_enabled(bool p_enabled) { ccd_enabled = p_enabled; }
bool WickedBody::is_ccd_enabled() const { return ccd_enabled; }

void WickedBody::set_ccd_motion_threshold(real_t p_threshold) { ccd_motion_threshold = MAX(p_threshold, 0.0); }
real_t WickedBody::get_ccd_motion_threshold() const { return ccd_motion_threshold; }

void WickedBody::set_ccd_swept_sphere_radius(real_t p_radius) { ccd_swept_sphere_radius = MAX(p_radius, 0.0); }
real_t WickedBody::get_ccd_swept_sphere_radius() const { return ccd_swept_sphere_radius; }

} // namespace wicked