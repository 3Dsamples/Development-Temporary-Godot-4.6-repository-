// File 272: modules/vienna/src/bodies/vienna_body.cpp
// ViennaBody implementation – transform, mass, inertia, damping, forces,
// integration, and Godot bindings.

#include "vienna_body.h"
#include "../collision/vienna_shape.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

namespace vienna {

void ViennaBody::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_type", "type"), &ViennaBody::set_type);
	ClassDB::bind_method(D_METHOD("get_type"), &ViennaBody::get_type);
	ClassDB::bind_method(D_METHOD("set_transform", "xform"), &ViennaBody::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &ViennaBody::get_transform);
	ClassDB::bind_method(D_METHOD("set_linear_velocity", "vel"), &ViennaBody::set_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_linear_velocity"), &ViennaBody::get_linear_velocity);
	ClassDB::bind_method(D_METHOD("set_angular_velocity", "vel"), &ViennaBody::set_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_angular_velocity"), &ViennaBody::get_angular_velocity);
	ClassDB::bind_method(D_METHOD("set_mass", "mass"), &ViennaBody::set_mass);
	ClassDB::bind_method(D_METHOD("get_mass"), &ViennaBody::get_mass);
	ClassDB::bind_method(D_METHOD("set_inertia", "inertia"), &ViennaBody::set_inertia);
	ClassDB::bind_method(D_METHOD("get_inertia_local"), &ViennaBody::get_inertia_local);
	ClassDB::bind_method(D_METHOD("set_linear_damping", "damping"), &ViennaBody::set_linear_damping);
	ClassDB::bind_method(D_METHOD("get_linear_damping"), &ViennaBody::get_linear_damping);
	ClassDB::bind_method(D_METHOD("set_angular_damping", "damping"), &ViennaBody::set_angular_damping);
	ClassDB::bind_method(D_METHOD("get_angular_damping"), &ViennaBody::get_angular_damping);
	ClassDB::bind_method(D_METHOD("apply_force", "force", "world_point"), &ViennaBody::apply_force, DEFVAL(vec3()));
	ClassDB::bind_method(D_METHOD("apply_impulse", "impulse", "world_point"), &ViennaBody::apply_impulse, DEFVAL(vec3()));
	ClassDB::bind_method(D_METHOD("set_gravity_enabled", "enabled"), &ViennaBody::set_gravity_enabled);
	ClassDB::bind_method(D_METHOD("is_gravity_enabled"), &ViennaBody::is_gravity_enabled);
	ClassDB::bind_method(D_METHOD("set_collision_shape", "shape"), &ViennaBody::set_collision_shape);
	ClassDB::bind_method(D_METHOD("get_collision_shape"), &ViennaBody::get_collision_shape);
	ClassDB::bind_method(D_METHOD("set_material_id", "id"), &ViennaBody::set_material_id);
	ClassDB::bind_method(D_METHOD("get_material_id"), &ViennaBody::get_material_id);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "type"), "set_type", "get_type");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "transform"), "set_transform", "get_transform");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mass"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "linear_damping"), "set_linear_damping", "get_linear_damping");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_damping"), "set_angular_damping", "get_angular_damping");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gravity_enabled"), "set_gravity_enabled", "is_gravity_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "collision_shape", PROPERTY_HINT_RESOURCE_TYPE, "ViennaShape"), "set_collision_shape", "get_collision_shape");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "material_id"), "set_material_id", "get_material_id");
}

ViennaBody::ViennaBody() :
	body_type(BodyType::DYNAMIC),
	mass(1.0),
	inverse_mass(1.0),
	linear_damping(0.0),
	angular_damping(0.0),
	mat_id(0),
	active(true),
	gravity_enabled(true),
	sleep_counter(0) {
	inertia_local.set_identity();
	inverse_inertia_local.set_identity();
	inverse_inertia_world.set_identity();
}

ViennaBody::~ViennaBody() {}

void ViennaBody::set_type(BodyType p_type) {
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

void ViennaBody::set_transform(const mat4 &p_xform) { transform = p_xform; update_inverse_inertia(); }
void ViennaBody::set_linear_velocity(const vec3 &p_vel) { linear_velocity = p_vel; }
void ViennaBody::set_angular_velocity(const vec3 &p_vel) { angular_velocity = p_vel; }

void ViennaBody::set_mass(real_t p_mass) { mass = MAX(p_mass, 0.0); inverse_mass = (mass > 0.0) ? 1.0 / mass : 0.0; }
void ViennaBody::set_inertia(const mat3 &p_inertia) { inertia_local = p_inertia; inverse_inertia_local = inertia_local.inverse(); update_inverse_inertia(); }
void ViennaBody::update_inverse_inertia() { inverse_inertia_world = transform.basis * inverse_inertia_local * transform.basis.transposed(); }

void ViennaBody::set_linear_damping(real_t p_damp) { linear_damping = CLAMP(p_damp, 0.0, 1.0); }
real_t ViennaBody::get_linear_damping() const { return linear_damping; }
void ViennaBody::set_angular_damping(real_t p_damp) { angular_damping = CLAMP(p_damp, 0.0, 1.0); }
real_t ViennaBody::get_angular_damping() const { return angular_damping; }

void ViennaBody::apply_force(const vec3 &p_force, const vec3 &p_world_point) {
	force_accum += p_force;
	vec3 r = p_world_point - transform.origin;
	torque_accum += r.cross(p_force);
}

void ViennaBody::apply_impulse(const vec3 &p_impulse, const vec3 &p_world_point) {
	if (inverse_mass > 0.0) linear_velocity += p_impulse * inverse_mass;
	vec3 r = p_world_point - transform.origin;
	angular_velocity += inverse_inertia_world.xform(r.cross(p_impulse));
}

void ViennaBody::clear_forces() { force_accum = vec3(); torque_accum = vec3(); }

void ViennaBody::integrate_velocity(real_t p_dt) {
	if (body_type != BodyType::DYNAMIC || inverse_mass <= 0.0) return;
	linear_velocity += force_accum * (inverse_mass * p_dt);
	angular_velocity += inverse_inertia_world.xform(torque_accum * p_dt);
	linear_velocity *= (1.0 - linear_damping * p_dt);
	angular_velocity *= (1.0 - angular_damping * p_dt);
	clear_forces();
}

void ViennaBody::integrate_position(real_t p_dt) {
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

void ViennaBody::set_active(bool p_active) { active = p_active; }

void ViennaBody::set_collision_shape(const Ref<ViennaShape> &p_shape) {
	collision_shape = p_shape;
	if (collision_shape.is_valid()) cached_aabb = collision_shape->get_local_aabb();
}

void ViennaBody::set_material_id(material_id p_id) { mat_id = p_id; }
material_id ViennaBody::get_material_id() const { return mat_id; }

} // namespace vienna