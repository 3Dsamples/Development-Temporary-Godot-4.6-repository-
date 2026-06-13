// File 57: modules/genesis/src/entities/rigid_entity.h
// Rigid entity: a convex-shaped dynamic/static body driven by the rigid solver.

#ifndef GENESIS_ENTITIES_RIGID_ENTITY_H
#define GENESIS_ENTITIES_RIGID_ENTITY_H

#include "base_entity.h"
#include "../core/genesis_types.h"
#include "../core/genesis_constants.h"
#include "../materials/material_base.h"
#include "../collision/collider.h"           // forward declared? We'll include full later; we can forward declare.
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"

namespace genesis {

// Forward declare collider to avoid circular includes (will be resolved later).
class Collider;

class RigidEntity : public BaseEntity {
	GDCLASS(RigidEntity, BaseEntity);

public:
	RigidEntity() :
		geometry_type(GeometryType::SPHERE),
		radius(0.5),
		half_extents(0.5, 0.5, 0.5),
		height(1.0),
		mass(1.0),
		inertia(Basis()),
		inverse_mass(1.0),
		inverse_inertia(Basis()),
		collider(nullptr) {
		solver_type = SolverType::RIGID;
	}

	~RigidEntity() {
		if (collider) {
			memdelete(collider);
			collider = nullptr;
		}
	}

	// --- Geometry ---
	void set_geometry_type(GeometryType p_type) { geometry_type = p_type; }
	GeometryType get_geometry_type() const { return geometry_type; }

	// Sphere
	void set_radius(real_t p_r) { radius = MAX(p_r, 0.0); }
	real_t get_radius() const { return radius; }

	// Box
	void set_half_extents(const Vector3 &p_ext) { half_extents = p_ext.abs(); }
	Vector3 get_half_extents() const { return half_extents; }

	// Capsule / Cylinder
	void set_height(real_t p_h) { height = MAX(p_h, 0.0); }
	real_t get_height() const { return height; }

	// Mass override (if > 0, overrides density-based calculation)
	void set_override_mass(real_t p_mass) { mass = p_mass; }
	real_t get_override_mass() const { return mass; }

	// --- Inertia (computed automatically from geometry and material) ---
	void update_mass_properties() {
		real_t density = material.is_valid() ? material->get_density() : 1000.0;
		if (mass > 0) {
			// Override mass present
		} else {
			mass = compute_volume() * density;
		}
		inverse_mass = mass > 0 ? 1.0 / mass : 0.0;
		inertia = compute_inertia_tensor();
		inverse_inertia = inertia.inverse(); // Basis inverse
	}

	virtual real_t get_mass() const override {
		return mass;
	}
	virtual real_t get_volume() const override {
		return compute_volume();
	}

	// --- Collider proxy ---
	Collider *get_collider() {
		if (!collider) {
			collider = memnew(Collider);
		}
		return collider;
	}

	// --- AABB ---
	virtual AABB get_aabb() const override {
		switch (geometry_type) {
			case GeometryType::SPHERE:
				return AABB(transform.origin - Vector3(radius, radius, radius),
				            Vector3(radius * 2, radius * 2, radius * 2));
			case GeometryType::BOX:
				return AABB(transform.origin - half_extents, half_extents * 2);
			case GeometryType::CAPSULE:
			case GeometryType::CYLINDER: {
				real_t r = radius;
				real_t hh = height * 0.5;
				return AABB(transform.origin - Vector3(r, hh, r), Vector3(r * 2, height, r * 2));
			}
			default:
				return AABB(transform.origin - Vector3(0.2, 0.2, 0.2), Vector3(0.4, 0.4, 0.4));
		}
	}

	// --- Override integration for rigid bodies (includes rotation update) ---
	virtual void integrate_velocity(real_t dt) override {
		if (!active || mass <= 0.0) return;
		linear_velocity += force_accum * (dt * inverse_mass);
		angular_velocity += inverse_inertia.xform(torque_accum * dt);
		clear_forces();
	}

	virtual void integrate_position(real_t dt) override {
		if (!active) return;
		transform.origin += linear_velocity * dt;
		real_t angle = angular_velocity.length();
		if (angle > CMP_EPSILON) {
			Vector3 axis = angular_velocity / angle;
			Quaternion rot(axis, angle * dt);
			transform.basis = rot * transform.basis;
			transform.basis.orthonormalize();
		}
		// update world inverse inertia
		inverse_inertia_world = transform.basis * inverse_inertia * transform.basis.transposed();
	}

	virtual void init_from_options(const genesis::options::Options &opts) override {
		BaseEntity::init_from_options(opts);
		geometry_type = GeometryType(opts.get_int("shape", 0));
		radius = opts.get_real("radius", 0.5);
		if (opts.has("half_extents")) {
			half_extents = opts.get_vector3("half_extents", Vector3(0.5, 0.5, 0.5));
		}
		height = opts.get_real("height", 1.0);
		mass = opts.get_real("mass", 0.0); // 0 means auto
		update_mass_properties();
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_geometry_type", "type"), &RigidEntity::set_geometry_type);
		ClassDB::bind_method(D_METHOD("get_geometry_type"), &RigidEntity::get_geometry_type);
		ClassDB::bind_method(D_METHOD("set_radius", "radius"), &RigidEntity::set_radius);
		ClassDB::bind_method(D_METHOD("get_radius"), &RigidEntity::get_radius);
		ClassDB::bind_method(D_METHOD("set_half_extents", "extents"), &RigidEntity::set_half_extents);
		ClassDB::bind_method(D_METHOD("get_half_extents"), &RigidEntity::get_half_extents);
		ClassDB::bind_method(D_METHOD("set_height", "height"), &RigidEntity::set_height);
		ClassDB::bind_method(D_METHOD("get_height"), &RigidEntity::get_height);
		ClassDB::bind_method(D_METHOD("set_override_mass", "mass"), &RigidEntity::set_override_mass);
		ClassDB::bind_method(D_METHOD("get_override_mass"), &RigidEntity::get_override_mass);
		ClassDB::bind_method(D_METHOD("update_mass_properties"), &RigidEntity::update_mass_properties);

		ADD_PROPERTY(PropertyInfo(Variant::INT, "geometry_type", PROPERTY_HINT_ENUM, "Sphere,Box,Capsule,Cylinder,ConvexMesh,HeightField,TriMesh,SDF,PointCloud"), "set_geometry_type", "get_geometry_type");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0,100,0.01"), "set_radius", "get_radius");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "half_extents"), "set_half_extents", "get_half_extents");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_RANGE, "0,100,0.01"), "set_height", "get_height");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "override_mass", PROPERTY_HINT_RANGE, "0,100000,0.1"), "set_override_mass", "get_override_mass");
	}

private:
	real_t compute_volume() const {
		switch (geometry_type) {
			case GeometryType::SPHERE: return (4.0 / 3.0) * Math_PI * radius * radius * radius;
			case GeometryType::BOX: return half_extents.x * half_extents.y * half_extents.z * 8;
			case GeometryType::CAPSULE:
				// cylinder + 2 hemispheres = cylinder + sphere volume
				return Math_PI * radius * radius * (height - 2 * radius) + (4.0 / 3.0) * Math_PI * radius * radius * radius;
			case GeometryType::CYLINDER:
				return Math_PI * radius * radius * height;
			default: return 1.0;
		}
	}

	Basis compute_inertia_tensor() const {
		if (mass <= 0) return Basis();
		switch (geometry_type) {
			case GeometryType::SPHERE: {
				real_t I = 0.4 * mass * radius * radius;
				return Basis().scaled(Vector3(I, I, I));
			}
			case GeometryType::BOX: {
				real_t x = half_extents.x, y = half_extents.y, z = half_extents.z;
				real_t Ix = (1.0 / 12.0) * mass * (y * y + z * z);
				real_t Iy = (1.0 / 12.0) * mass * (x * x + z * z);
				real_t Iz = (1.0 / 12.0) * mass * (x * x + y * y);
				return Basis().scaled(Vector3(Ix, Iy, Iz));
			}
			case GeometryType::CAPSULE:
			case GeometryType::CYLINDER: {
				real_t h = height, r = radius;
				real_t Ixy = (1.0 / 12.0) * mass * (3 * r * r + h * h);
				real_t Iz = 0.5 * mass * r * r;
				return Basis().scaled(Vector3(Ixy, Ixy, Iz));
			}
			default: {
				real_t I = 0.4 * mass;
				return Basis().scaled(Vector3(I, I, I));
			}
		}
	}

	GeometryType geometry_type;
	real_t radius;
	Vector3 half_extents;
	real_t height;
	real_t mass;
	Basis inertia, inverse_inertia;
	real_t inverse_mass;
	Basis inverse_inertia_world;
	Collider *collider;
};

} // namespace genesis

#endif // GENESIS_ENTITIES_RIGID_ENTITY_H