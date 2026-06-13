// File 157: modules/genesis/src/nodes/genesis_rigid_body_3d.cpp
// Implements the GenesisRigidBody3D node. On ready, it creates a RigidEntity,
// registers it with the GenesisWorld, and syncs the Godot transform each frame.

#include "genesis_rigid_body_3d.h"

#include "scene/main/scene_tree.h"
#include "scene/resources/mesh.h"

#include "../genesis_world.h"                     // GenesisWorld
#include "../entities/rigid_entity.h"
#include "../materials/material_base.h"

namespace genesis {

void GenesisRigidBody3D::_bind_methods() {
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

void GenesisRigidBody3D::_notification(int p_what) {
	if (p_what == NOTIFICATION_READY) {
		_initialize();                     // create RigidEntity, register with world
	}
	if (p_what == NOTIFICATION_PROCESS) {
		_sync_to_scene();                  // update the Node3D transform from the physics entity
	}
}

void GenesisRigidBody3D::_initialize() {
	// Find the GenesisWorld node (ancestor) or fall back to manual assignment
	if (!_find_world()) {
		ERR_PRINT("GenesisRigidBody3D: no GenesisWorld found in the scene tree.");
		return;
	}

	// Create the rigid entity and apply initial parameters
	rigid_entity.instantiate();
	rigid_entity->set_entity_uid(_generate_uid());
	entity_uid = rigid_entity->get_entity_uid();

	// Copy the current Node3D transform to the entity
	rigid_entity->set_transform(get_global_transform());

	// Assign the material if one is set
	if (_material.is_valid()) rigid_entity->set_material(_material);

	// Set dynamic/static/kinematic mode
	set_mode(mode);

	// Configure collision shape from the inspector values
	update_collider();

	// Register the entity with the world so it is stepped by the solver
	world->add_entity(rigid_entity);

	// Enable per‑frame processing to keep the scene node in sync
	set_process(true);
}

bool GenesisRigidBody3D::_find_world() {
	Node *p = get_parent();
	while (p) {
		world = Object::cast_to<GenesisWorld>(p);
		if (world) return true;
		p = p->get_parent();
	}
	return false;
}

uint64_t GenesisRigidBody3D::_generate_uid() {
	// Produce an unique 64‑bit handle from the Object instance ID and a random salt
	return (uint64_t(get_instance_id()) << 32) | (uint64_t(Math::rand()) & 0xFFFFFFFF);
}

void GenesisRigidBody3D::update_collider() {
	if (rigid_entity.is_null()) return;
	rigid_entity->set_geometry_type(_collider_type);
	rigid_entity->set_radius(_collider_radius);
	rigid_entity->set_half_extents(_collider_half_extents);
	rigid_entity->set_height(_collider_height);

	if (_collider_type == GeometryType::CONVEX_MESH && !_collider_mesh_path.is_empty()) {
		// In a full implementation we would load the mesh resource, extract its
		// vertices, and pass them to a ConvexMeshCollider.  Skipping for brevity.
	}

	rigid_entity->update_mass_properties();   // recompute mass, inertia, inverse data
}

void GenesisRigidBody3D::_sync_to_scene() {
	if (rigid_entity.is_null()) return;
	// Place the Godot node exactly where the physics engine placed the rigid body
	set_global_transform(rigid_entity->get_transform());
}

// --- Scripting API wrappers ---

void GenesisRigidBody3D::apply_impulse(const Vector3 &impulse, const Vector3 &world_point) {
	if (rigid_entity.is_valid()) rigid_entity->apply_impulse(impulse, world_point);
}

void GenesisRigidBody3D::apply_central_impulse(const Vector3 &impulse) {
	if (rigid_entity.is_valid())
		rigid_entity->apply_impulse(impulse, rigid_entity->get_position());
}

void GenesisRigidBody3D::apply_force(const Vector3 &force, const Vector3 &world_point) {
	if (rigid_entity.is_valid()) rigid_entity->apply_force(force, world_point);
}

void GenesisRigidBody3D::apply_central_force(const Vector3 &force) {
	if (rigid_entity.is_valid())
		rigid_entity->apply_force(force, rigid_entity->get_position());
}

Vector3 GenesisRigidBody3D::get_linear_velocity() const {
	return rigid_entity.is_valid() ? rigid_entity->get_linear_velocity() : Vector3();
}

Vector3 GenesisRigidBody3D::get_angular_velocity() const {
	return rigid_entity.is_valid() ? rigid_entity->get_angular_velocity() : Vector3();
}

void GenesisRigidBody3D::set_linear_velocity(const Vector3 &p_vel) {
	if (rigid_entity.is_valid()) rigid_entity->set_linear_velocity(p_vel);
}

void GenesisRigidBody3D::set_angular_velocity(const Vector3 &p_vel) {
	if (rigid_entity.is_valid()) rigid_entity->set_angular_velocity(p_vel);
}

void GenesisRigidBody3D::set_mode(Mode p_mode) {
	mode = p_mode;
	if (rigid_entity.is_valid()) {
		switch (mode) {
			case MODE_DYNAMIC: rigid_entity->set_type(RigidEntity::DYNAMIC); break;
			case MODE_STATIC:  rigid_entity->set_type(RigidEntity::STATIC);  break;
			case MODE_KINEMATIC: rigid_entity->set_type(RigidEntity::KINEMATIC); break;
		}
	}
}

void GenesisRigidBody3D::set_material(const Ref<GenesisMaterial> &p_mat) {
	_material = p_mat;
	if (rigid_entity.is_valid()) rigid_entity->set_material(p_mat);
}

} // namespace genesis