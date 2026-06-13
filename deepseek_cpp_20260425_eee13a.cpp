// File 56: modules/genesis/src/entities/base_entity.h
// Base class for all physics entities in Genesis (rigid, FEM, MPM, tool).
// Inherits from Godot Resource for easy serialisation and editor integration.

#ifndef GENESIS_ENTITIES_BASE_ENTITY_H
#define GENESIS_ENTITIES_BASE_ENTITY_H

#include "core/io/resource.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/templates/local_vector.h"
#include "../materials/material_base.h"
#include "../core/genesis_types.h"
#include "../options/options_system.h"

namespace genesis {

class GenesisMaterial;

class BaseEntity : public Resource {
	GDCLASS(BaseEntity, Resource);

public:
	BaseEntity() :
		entity_uid(0),
		solver_type(SolverType::RIGID),
		active(true),
		enable_gravity(true),
		linear_velocity(Vector3()),
		angular_velocity(Vector3()),
		force_accum(Vector3()),
		torque_accum(Vector3()),
		material(nullptr) {}

	// --- Identity ---
	void set_entity_uid(entity_id_t p_uid) { entity_uid = p_uid; }
	entity_id_t get_entity_uid() const { return entity_uid; }

	void set_solver_type(SolverType p_type) { solver_type = p_type; }
	SolverType get_solver_type() const { return solver_type; }

	void set_active(bool p_active) { active = p_active; }
	bool is_active() const { return active; }

	// --- Transform ---
	void set_transform(const Transform3D &p_transform) { transform = p_transform; }
	Transform3D get_transform() const { return transform; }

	void set_position(const Vector3 &p_pos) { transform.origin = p_pos; }
	Vector3 get_position() const { return transform.origin; }

	void set_rotation(const Basis &p_basis) { transform.basis = p_basis; }
	Basis get_rotation() const { return transform.basis; }

	// --- Velocities ---
	void set_linear_velocity(const Vector3 &p_vel) { linear_velocity = p_vel; }
	Vector3 get_linear_velocity() const { return linear_velocity; }

	void set_angular_velocity(const Vector3 &p_vel) { angular_velocity = p_vel; }
	Vector3 get_angular_velocity() const { return angular_velocity; }

	// --- Physics parameters ---
	void set_enable_gravity(bool p_val) { enable_gravity = p_val; }
	bool is_gravity_enabled() const { return enable_gravity; }

	void set_material(const Ref<GenesisMaterial> &p_mat) { material = p_mat; }
	Ref<GenesisMaterial> get_material() const { return material; }

	// --- Force accumulation ---
	void apply_force(const Vector3 &p_force, const Vector3 &p_at_world_pos = Vector3()) {
		force_accum += p_force;
		if (p_at_world_pos != Vector3() && p_at_world_pos != transform.origin) {
			Vector3 r = p_at_world_pos - transform.origin;
			torque_accum += r.cross(p_force);
		}
	}

	void apply_impulse(const Vector3 &p_impulse, const Vector3 &p_at_world_pos = Vector3()) {
		if (material.is_null() || material->get_density() <= 0) return;
		// For rigid bodies, this changes velocity directly. For deformables, it adds per-vertex
		// impulses (to be handled by derived classes).
		// Base implementation for simple rigid.
		real_t mass = get_mass();
		if (mass > 0) {
			linear_velocity += p_impulse / mass;
			if (p_at_world_pos != Vector3() && p_at_world_pos != transform.origin) {
				Vector3 r = p_at_world_pos - transform.origin;
				// Inertia handling: assume isotropic inertia for base
				real_t inv_inertia = 1.0 / get_inertia_scalar();
				angular_velocity += r.cross(p_impulse) * inv_inertia;
			}
		}
	}

	void clear_forces() {
		force_accum = Vector3();
		torque_accum = Vector3();
	}

	// --- Mass / inertia (overridden by derived classes) ---
	virtual real_t get_mass() const {
		if (material.is_valid()) return material->get_density() * get_volume();
		return 0.0;
	}
	virtual real_t get_volume() const { return 1.0; } // placeholder
	virtual real_t get_inertia_scalar() const {
		real_t mass = get_mass();
		return mass > 0 ? mass * 0.4 : 0.0; // sphere approx
	}

	// --- AABB (virtual, overridden by shape-specific entities) ---
	virtual AABB get_aabb() const {
		// Default: a small box around position
		return AABB(transform.origin - Vector3(0.1, 0.1, 0.1), Vector3(0.2, 0.2, 0.2));
	}

	// --- Step (called by solver) ---
	virtual void integrate_velocity(real_t dt) {
		if (!active || material.is_null()) return;
		real_t mass = get_mass();
		if (mass <= 0) return;
		linear_velocity += force_accum * (dt / mass);
		// simplified angular: use scalar inertia
		real_t inertia = get_inertia_scalar();
		if (inertia > 0) {
			angular_velocity += torque_accum * (dt / inertia);
		}
		clear_forces();
	}

	virtual void integrate_position(real_t dt) {
		if (!active) return;
		transform.origin += linear_velocity * dt;
		// integrate rotation from angular velocity: q = q + 0.5 * omega * q * dt
		// using quaternion multiplication
		Quaternion q(transform.basis);
		real_t angle = angular_velocity.length();
		if (angle > CMP_EPSILON) {
			Vector3 axis = angular_velocity / angle;
			real_t half_angle = angle * dt * 0.5;
			Quaternion omega_quat(axis, half_angle);
			q = omega_quat * q;
			q.normalize();
		}
		transform.basis = q.get_basis();
	}

	// --- Virtual methods for derived entity specific setup ---
	virtual void init_from_options(const genesis::options::Options &opts) {
		set_position(opts.get_vector3("pos", get_position()));
		set_rotation(Basis(opts.get_vector3("rot", Vector3())));
		set_linear_velocity(opts.get_vector3("vel", linear_velocity));
		set_angular_velocity(opts.get_vector3("angvel", angular_velocity));
		set_enable_gravity(opts.get_bool("gravity", true));
	}

	// --- Serialisation helpers for Resource ---
	virtual void _get_property_list(List<PropertyInfo> *p_list) const override {
		Resource::_get_property_list(p_list);
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_entity_uid", "uid"), &BaseEntity::set_entity_uid);
		ClassDB::bind_method(D_METHOD("get_entity_uid"), &BaseEntity::get_entity_uid);
		ClassDB::bind_method(D_METHOD("set_solver_type", "type"), &BaseEntity::set_solver_type);
		ClassDB::bind_method(D_METHOD("get_solver_type"), &BaseEntity::get_solver_type);
		ClassDB::bind_method(D_METHOD("set_active", "active"), &BaseEntity::set_active);
		ClassDB::bind_method(D_METHOD("is_active"), &BaseEntity::is_active);
		ClassDB::bind_method(D_METHOD("set_transform", "transform"), &BaseEntity::set_transform);
		ClassDB::bind_method(D_METHOD("get_transform"), &BaseEntity::get_transform);
		ClassDB::bind_method(D_METHOD("set_linear_velocity", "vel"), &BaseEntity::set_linear_velocity);
		ClassDB::bind_method(D_METHOD("get_linear_velocity"), &BaseEntity::get_linear_velocity);
		ClassDB::bind_method(D_METHOD("set_angular_velocity", "vel"), &BaseEntity::set_angular_velocity);
		ClassDB::bind_method(D_METHOD("get_angular_velocity"), &BaseEntity::get_angular_velocity);
		ClassDB::bind_method(D_METHOD("set_enable_gravity", "enable"), &BaseEntity::set_enable_gravity);
		ClassDB::bind_method(D_METHOD("is_gravity_enabled"), &BaseEntity::is_gravity_enabled);
		ClassDB::bind_method(D_METHOD("set_material", "material"), &BaseEntity::set_material);
		ClassDB::bind_method(D_METHOD("get_material"), &BaseEntity::get_material);
		ClassDB::bind_method(D_METHOD("apply_force", "force", "pos"), &BaseEntity::apply_force, DEFVAL(Vector3()));
		ClassDB::bind_method(D_METHOD("apply_impulse", "impulse", "pos"), &BaseEntity::apply_impulse, DEFVAL(Vector3()));
		ClassDB::bind_method(D_METHOD("get_mass"), &BaseEntity::get_mass);

		ADD_PROPERTY(PropertyInfo(Variant::INT, "entity_uid"), "set_entity_uid", "get_entity_uid");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "solver_type", PROPERTY_HINT_ENUM, "Rigid,FEM,MPM,PBD,SPH,SF,Tool"), "set_solver_type", "get_solver_type");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "active"), "set_active", "is_active");
		ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "transform"), "set_transform", "get_transform");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gravity"), "set_enable_gravity", "is_gravity_enabled");
		ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "GenesisMaterial"), "set_material", "get_material");
	}

private:
	entity_id_t entity_uid;
	SolverType solver_type;
	bool active;
	bool enable_gravity;
	Transform3D transform;
	Vector3 linear_velocity;
	Vector3 angular_velocity;
	Vector3 force_accum;
	Vector3 torque_accum;
	Ref<GenesisMaterial> material;
};

} // namespace genesis

#endif // GENESIS_ENTITIES_BASE_ENTITY_H