// File 93: modules/genesis/src/nodes/genesis_rigid_body_3d.h
// GenesisRigidBody3D – a Node3D that wraps a RigidEntity, connecting it to
// the GenesisWorld simulation. Provides a drop‑in replacement for Godot's
// RigidDynamicBody3D with additional material assignment and solver options.

#ifndef GENESIS_NODES_RIGID_BODY_3D_H
#define GENESIS_NODES_RIGID_BODY_3D_H

#include "scene/3d/node_3d.h"
#include "scene/main/scene_tree.h"
#include "scene/resources/mesh.h"

#include "../entities/rigid_entity.h"
#include "../materials/material_base.h"
#include "../core/genesis_types.h"

// Forward declare GenesisWorld (node that drives the physics)
namespace genesis {
class GenesisWorld;
}

namespace genesis {

class GenesisRigidBody3D : public Node3D {
	GDCLASS(GenesisRigidBody3D, Node3D);

public:
	enum Mode {
		MODE_DYNAMIC,
		MODE_STATIC,
		MODE_KINEMATIC
	};

	GenesisRigidBody3D() :
		mode(MODE_DYNAMIC),
		entity_uid(0),
		world(nullptr),
		_collider_type(GeometryType::SPHERE),
		_collider_radius(0.5),
		_collider_half_extents(0.5, 0.5, 0.5),
		_collider_height(1.0),
		_collider_mesh_path("") {
		set_process(false);
		set_physics_process(false);
	}

	// --- Mode ---
	void set_mode(Mode p_mode) {
		mode = p_mode;
		if (rigid_entity.is_valid()) {
			switch (mode) {
				case MODE_DYNAMIC: rigid_entity->set_type(RigidEntity::DYNAMIC); break;
				case MODE_STATIC: rigid_entity->set_type(RigidEntity::STATIC); break;
				case MODE_KINEMATIC: rigid_entity->set_type(RigidEntity::KINEMATIC); break;
			}
		}
	}
	Mode get_mode() const { return mode; }

	// --- Material ---
	void set_material(const Ref<GenesisMaterial> &p_mat) {
		_material = p_mat;
		if (rigid_entity.is_valid()) rigid_entity->set_material(p_mat);
	}
	Ref<GenesisMaterial> get_material() const { return _material; }

	// --- Collision shape ---
	void set_collider_type(GeometryType p_type) {
		_collider_type = p_type;
		update_collider();
	}
	GeometryType get_collider_type() const { return _collider_type; }

	void set_collider_radius(real_t p_r) { _collider_radius = p_r; update_collider(); }
	real_t get_collider_radius() const { return _collider_radius; }

	void set_collider_half_extents(const Vector3 &p_he) { _collider_half_extents = p_he; update_collider(); }
	Vector3 get_collider_half_extents() const { return _collider_half_extents; }

	void set_collider_height(real_t p_h) { _collider_height = p_h; update_collider(); }
	real_t get_collider_height() const { return _collider_height; }

	void set_collider_mesh_path(const String &p_path) { _collider_mesh_path = p_path; update_collider(); }
	String get_collider_mesh_path() const { return _collider_mesh_path; }

	// --- Access RigidEntity (for advanced users) ---
	Ref<RigidEntity> get_rigid_entity() { return rigid_entity; }

	// --- Godot lifecycle ---
	void _notification(int p_what) {
		if (p_what == NOTIFICATION_READY) {
			_initialize();
		}
		if (p_what == NOTIFICATION_PROCESS) {
			_sync_to_scene();
		}
	}

	// --- Scripting API ---
	void apply_impulse(const Vector3 &impulse, const Vector3 &world_point = Vector3()) {
		if (rigid_entity.is_valid()) rigid_entity->apply_impulse(impulse, world_point);
	}
	void apply_central_impulse(const Vector3 &impulse) {
		if (rigid_entity.is_valid()) rigid_entity->apply_impulse(impulse, rigid_entity->get_position());
	}
	void apply_force(const Vector3 &force, const Vector3 &world_point = Vector3()) {
		if (rigid_entity.is_valid()) rigid_entity->apply_force(force, world_point);
	}
	void apply_central_force(const Vector3 &force) {
		if (rigid_entity.is_valid()) rigid_entity->apply_force(force, rigid_entity->get_position());
	}

	Vector3 get_linear_velocity() const {
		return rigid_entity.is_valid() ? rigid_entity->get_linear_velocity() : Vector3();
	}
	Vector3 get_angular_velocity() const {
		return rigid_entity.is_valid() ? rigid_entity->get_angular_velocity() : Vector3();
	}

	void set_linear_velocity(const Vector3 &p_vel) {
		if (rigid_entity.is_valid()) rigid_entity->set_linear_velocity(p_vel);
	}
	void set_angular_velocity(const Vector3 &p_vel) {
		if (rigid_entity.is_valid()) rigid_entity->set_angular_velocity(p_vel);
	}

private:
	void _initialize() {
		// Find the GenesisWorld node (ancestor or sibling) to register this entity
		if (!_find_world()) {
			ERR_PRINT("GenesisRigidBody3D: no GenesisWorld found in the scene tree.");
			return;
		}

		// Create the RigidEntity
		rigid_entity.instantiate();
		rigid_entity->set_entity_uid(_generate_uid());
		entity_uid = rigid_entity->get_entity_uid();
		rigid_entity->set_transform(get_global_transform());
		if (_material.is_valid()) rigid_entity->set_material(_material);
		set_mode(mode); // apply dynamic/static/kinematic
		update_collider();

		// Register with world
		world->add_entity(rigid_entity);

		// Start processing to sync Godot transform from physics
		set_process(true);
	}

	bool _find_world() {
		// Search ancestors
		Node *parent = get_parent();
		while (parent) {
			world = Object::cast_to<GenesisWorld>(parent);
			if (world) return true;
			parent = parent->get_parent();
		}
		// Search siblings (child of world?)
		return false; // not found, user must assign manually
	}

	uint64_t _generate_uid() {
		return (uint64_t(get_instance_id()) << 32) | (uint64_t(Math::rand()));
	}

	void update_collider() {
		if (rigid_entity.is_null()) return;
		rigid_entity->set_geometry_type(_collider_type);
		rigid_entity->set_radius(_collider_radius);
		rigid_entity->set_half_extents(_collider_half_extents);
		rigid_entity->set_height(_collider_height);
		// If convex mesh path is set, load it (not implemented fully)
		if (!_collider_mesh_path.is_empty() && _collider_type == GeometryType::CONVEX_MESH) {
			// Load mesh and set vertices
		}
		rigid_entity->update_mass_properties();
	}

	void _sync_to_scene() {
		if (rigid_entity.is_null()) return;
		// Move the Godot node to match physics entity transform
		Transform3D phys_xform = rigid_entity->get_transform();
		set_global_transform(phys_xform);
	}

	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_mode", "mode"), &GenesisRigidBody3D::set_mode);
		ClassDB::bind_method(D_METHOD("get_mode"), &GenesisRigidBody3D::get_mode);
		ClassDB::bind_method(D_METHOD("set_material", "material"), &GenesisRigidBody3D::set_material);
		ClassDB::bind_method(D_METHOD("get_material"), &GenesisRigidBody3D::get_material);

		ClassDB::bind_method(D_METHOD("set_collider_type", "type"), &GenesisRigidBody3D::set_collider_type);
		ClassDB::bind_method(D_METHOD("get_collider_type"), &GenesisRigidBody3D::get_collider_type);
		ClassDB::bind_method(D_METHOD("set_collider_radius", "radius"), &GenesisRigidBody3D::set_collider_radius);
		ClassDB::bind_method(D_METHOD("get_collider_radius"), &GenesisRigidBody3D::get_collider_radius);
		ClassDB::bind_method(D_METHOD("set_collider_half_extents", "extents"), &GenesisRigidBody3D::set_collider_half_extents);
		ClassDB::bind_method(D_METHOD("get_collider_half_extents"), &GenesisRigidBody3D::get_collider_half_extents);
		ClassDB::bind_method(D_METHOD("set_collider_height", "height"), &GenesisRigidBody3D::set_collider_height);
		ClassDB::bind_method(D_METHOD("get_collider_height"), &GenesisRigidBody3D::get_collider_height);
		ClassDB::bind_method(D_METHOD("set_collider_mesh_path", "path"), &GenesisRigidBody3D::set_collider_mesh_path);
		ClassDB::bind_method(D_METHOD("get_collider_mesh_path"), &GenesisRigidBody3D::get_collider_mesh_path);

		ClassDB::bind_method(D_METHOD("apply_impulse", "impulse", "world_point"), &GenesisRigidBody3D::apply_impulse, DEFVAL(Vector3()));
		ClassDB::bind_method(D_METHOD("apply_central_impulse", "impulse"), &GenesisRigidBody3D::apply_central_impulse);
		ClassDB::bind_method(D_METHOD("apply_force", "force", "world_point"), &GenesisRigidBody3D::apply_force, DEFVAL(Vector3()));
		ClassDB::bind_method(D_METHOD("apply_central_force", "force"), &GenesisRigidBody3D::apply_central_force);
		ClassDB::bind_method(D_METHOD("get_linear_velocity"), &GenesisRigidBody3D::get_linear_velocity);
		ClassDB::bind_method(D_METHOD("get_angular_velocity"), &GenesisRigidBody3D::get_angular_velocity);
		ClassDB::bind_method(D_METHOD("set_linear_velocity", "velocity"), &GenesisRigidBody3D::set_linear_velocity);
		ClassDB::bind_method(D_METHOD("set_angular_velocity", "velocity"), &GenesisRigidBody3D::set_angular_velocity);

		BIND_ENUM_CONSTANT(MODE_DYNAMIC);
		BIND_ENUM_CONSTANT(MODE_STATIC);
		BIND_ENUM_CONSTANT(MODE_KINEMATIC);

		ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Dynamic,Static,Kinematic"), "set_mode", "get_mode");
		ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "GenesisMaterial"), "set_material", "get_material");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "collider_type", PROPERTY_HINT_ENUM, "Sphere,Box,Capsule,Cylinder,ConvexMesh"), "set_collider_type", "get_collider_type");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.001,100,0.01"), "set_collider_radius", "get_collider_radius");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "half_extents"), "set_collider_half_extents", "get_collider_half_extents");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_RANGE, "0.001,100,0.01"), "set_collider_height", "get_collider_height");
		ADD_PROPERTY(PropertyInfo(Variant::STRING, "collider_mesh_path", PROPERTY_HINT_FILE, "*.obj,*.mesh"), "set_collider_mesh_path", "get_collider_mesh_path");
	}

	Ref<RigidEntity> rigid_entity;
	Mode mode;
	entity_id_t entity_uid;
	GenesisWorld *world; // cached pointer

	// Collider configuration
	GeometryType _collider_type;
	real_t _collider_radius;
	Vector3 _collider_half_extents;
	real_t _collider_height;
	String _collider_mesh_path;

	Ref<GenesisMaterial> _material;
};

} // namespace genesis

#endif // GENESIS_NODES_RIGID_BODY_3D_H