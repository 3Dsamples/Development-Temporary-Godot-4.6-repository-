// File 178: modules/newton/src/bodies/newton_body.cpp
// NewtonBody implementation – mass properties, transform, velocity,
// force accumulation, impulse application, and integration.

#include "newton_body.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "core/typedefs.h"

namespace newton {

void NewtonBody::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_type", "type"), &NewtonBody::set_type);
	ClassDB::bind_method(D_METHOD("get_type"), &NewtonBody::get_type);
	ClassDB::bind_method(D_METHOD("set_transform", "xform"), &NewtonBody::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &NewtonBody::get_transform);
	ClassDB::bind_method(D_METHOD("set_linear_velocity", "vel"), &NewtonBody::set_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_linear_velocity"), &NewtonBody::get_linear_velocity);
	ClassDB::bind_method(D_METHOD("set_angular_velocity", "vel"), &NewtonBody::set_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_angular_velocity"), &NewtonBody::get_angular_velocity);
	ClassDB::bind_method(D_METHOD("set_mass", "mass"), &NewtonBody::set_mass);
	ClassDB::bind_method(D_METHOD("get_mass"), &NewtonBody::get_mass);
	ClassDB::bind_method(D_METHOD("set_inertia", "inertia"), &NewtonBody::set_inertia);
	ClassDB::bind_method(D_METHOD("get_inertia_local"), &NewtonBody::get_inertia_local);
	ClassDB::bind_method(D_METHOD("apply_force", "force", "world_point"), &NewtonBody::apply_force, DEFVAL(vec3()));
	ClassDB::bind_method(D_METHOD("apply_impulse", "impulse", "world_point"), &NewtonBody::apply_impulse, DEFVAL(vec3()));
	ClassDB::bind_method(D_METHOD("set_gravity_enabled", "enabled"), &NewtonBody::set_gravity_enabled);
	ClassDB::bind_method(D_METHOD("is_gravity_enabled"), &NewtonBody::is_gravity_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "type"), "set_type", "get_type");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "transform"), "set_transform", "get_transform");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mass"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gravity_enabled"), "set_gravity_enabled", "is_gravity_enabled");
}

NewtonBody::NewtonBody() :
	body_type(BodyType::DYNAMIC),
	mass(1.0),
	inverse_mass(1.0) {
	inertia_local.set_identity();
	inverse_inertia_local.set_identity();
	inverse_inertia_world.set_identity();
}

NewtonBody::~NewtonBody() {}

void NewtonBody::set_type(BodyType p_type) {
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

void NewtonBody::set_transform(const mat4 &p_xform) {
	transform = p_xform;
	update_inverse_inertia();
}

void NewtonBody::set_linear_velocity(const vec3 &p_vel) { linear_velocity = p_vel; }
void NewtonBody::set_angular_velocity(const vec3 &p_vel) { angular_velocity = p_vel; }

void NewtonBody::set_mass(real_t p_mass) {
	mass = MAX(p_mass, 0.0);
	inverse_mass = (mass > 0.0) ? 1.0 / mass : 0.0;
}

void NewtonBody::set_inertia(const mat3 &p_inertia) {
	inertia_local = p_inertia;
	// Invert the 3x3 matrix via Godot Basis inverse
	inverse_inertia_local = inertia_local.inverse();
	update_inverse_inertia();
}

void NewtonBody::update_inverse_inertia() {
	// I_world^{-1} = R * I_local^{-1} * R^T
	inverse_inertia_world = transform.basis * inverse_inertia_local * transform.basis.transposed();
}

void NewtonBody::apply_force(const vec3 &p_force, const vec3 &p_world_point) {
	force_accum += p_force;
	vec3 r = p_world_point - transform.origin;
	torque_accum += r.cross(p_force);
}

void NewtonBody::apply_impulse(const vec3 &p_impulse, const vec3 &p_world_point) {
	if (inverse_mass > 0.0) {
		linear_velocity += p_impulse * inverse_mass;
	}
	vec3 r = p_world_point - transform.origin;
	angular_velocity += inverse_inertia_world.xform(r.cross(p_impulse));
}

void NewtonBody::clear_forces() {
	force_accum = vec3();
	torque_accum = vec3();
}

void NewtonBody::integrate_velocity(real_t p_dt) {
	if (body_type != BodyType::DYNAMIC || inverse_mass <= 0.0) return;
	// v += (F/m) * dt
	linear_velocity += force_accum * (inverse_mass * p_dt);
	// ω += I⁻¹ * τ * dt
	angular_velocity += inverse_inertia_world.xform(torque_accum * p_dt);
	clear_forces();
}

void NewtonBody::integrate_position(real_t p_dt) {
	if (body_type != BodyType::DYNAMIC) return;
	// x += v * dt
	transform.origin += linear_velocity * p_dt;
	// integrate rotation: q_new = q_old + 0.5 * ω * q_old * dt
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

void NewtonBody::set_active(bool p_active) { active = p_active; }

void NewtonBody::set_gravity_enabled(bool p_enabled) { gravity_enabled = p_enabled; }
bool NewtonBody::is_gravity_enabled() const { return gravity_enabled; }

} // namespace newton