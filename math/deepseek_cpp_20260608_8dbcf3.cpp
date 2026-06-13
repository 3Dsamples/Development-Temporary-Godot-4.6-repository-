// File 241: modules/newton/src/nodes/newton_rigid_body_3d.cpp
// NewtonRigidBody3D implementation – a Node3D that drives a NewtonBody
// in the scene, syncing transforms and handling forces, impulses, collisions.

#include "newton_rigid_body_3d.h"

#include "scene/main/scene_tree.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "../world/newton_world.h"
#include "../bodies/newton_body.h"
#include "../collision/newton_collision.h"

namespace newton {

void NewtonRigidBody3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &NewtonRigidBody3D::set_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &NewtonRigidBody3D::get_mode);
	ClassDB::bind_method(D_METHOD("set_mass", "mass"), &NewtonRigidBody3D::set_mass);
	ClassDB::bind_method(D_METHOD("get_mass"), &NewtonRigidBody3D::get_mass);
	ClassDB::bind_method(D_METHOD("set_collision_shape", "shape"), &NewtonRigidBody3D::set_collision_shape);
	ClassDB::bind_method(D_METHOD("get_collision_shape"), &NewtonRigidBody3D::get_collision_shape);
	ClassDB::bind_method(D_METHOD("set_linear_velocity", "vel"), &NewtonRigidBody3D::set_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_linear_velocity"), &NewtonRigidBody3D::get_linear_velocity);
	ClassDB::bind_method(D_METHOD("set_angular_velocity", "vel"), &NewtonRigidBody3D::set_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_angular_velocity"), &NewtonRigidBody3D::get_angular_velocity);
	ClassDB::bind_method(D_METHOD("apply_impulse", "impulse", "world_point"), &NewtonRigidBody3D::apply_impulse, DEFVAL(vec3()));
	ClassDB::bind_method(D_METHOD("apply_force", "force", "world_point"), &NewtonRigidBody3D::apply_force, DEFVAL(vec3()));
	ClassDB::bind_method(D_METHOD("set_gravity_enabled", "enabled"), &NewtonRigidBody3D::set_gravity_enabled);
	ClassDB::bind_method(D_METHOD("is_gravity_enabled"), &NewtonRigidBody3D::is_gravity_enabled);
	ClassDB::bind_method(D_METHOD("set_ccd_enabled", "enabled"), &NewtonRigidBody3D::set_ccd_enabled);
	ClassDB::bind_method(D_METHOD("is_ccd_enabled"), &NewtonRigidBody3D::is_ccd_enabled);

	BIND_ENUM_CONSTANT(MODE_DYNAMIC);
	BIND_ENUM_CONSTANT(MODE_STATIC);
	BIND_ENUM_CONSTANT(MODE_KINEMATIC);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Dynamic,Static,Kinematic"), "set_mode", "get_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mass"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "collision_shape", PROPERTY_HINT_RESOURCE_TYPE, "NewtonCollision"), "set_collision_shape", "get_collision_shape");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gravity_enabled"), "set_gravity_enabled", "is_gravity_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ccd_enabled"), "set_ccd_enabled", "is_ccd_enabled");
}

NewtonRigidBody3D::NewtonRigidBody3D() :
	mode(MODE_DYNAMIC),
	world(nullptr),
	body_id(0),
	world_found(false) {
	set_process(true);
	newton_body.instantiate();
}

void NewtonRigidBody3D::_notification(int p_what) {
	if (p_what == NOTIFICATION_READY) {
		_find_world();                              // locate NewtonWorld in ancestors
		if (world) {
			// Configure body type
			switch (mode) {
				case MODE_DYNAMIC: newton_body->set_type(BodyType::DYNAMIC); break;
				case MODE_STATIC:  newton_body->set_type(BodyType::STATIC); break;
				case MODE_KINEMATIC: newton_body->set_type(BodyType::KINEMATIC); break;
			}
			// Set initial transform from the node
			newton_body->set_transform(get_global_transform());
			// Attach collision shape if set
			if (collision_shape.is_valid()) {
				newton_body->set_collision_shape(collision_shape);
				// Compute and set inertia from shape and mass
				if (mode == MODE_DYNAMIC) {
					real_t m = newton_body->get_mass();
					if (m > 0.0) {
						mat3 inertia = collision_shape->compute_inertia(m);
						newton_body->set_inertia(inertia);
					}
				}
			}
			// Register with world
			body_id = world->create_body(newton_body);
			world_found = true;
		}
	}
	if (p_what == NOTIFICATION_PROCESS && world_found) {
		_sync_from_newton();                         // update Node3D transform from physics
	}
}

void NewtonRigidBody3D::_find_world() {
	Node *p = get_parent();
	while (p) {
		// Look for a node that has a NewtonWorld attached (user could add a custom node).
		// Alternatively, we can rely on a static singleton.
		NewtonWorld *w = Object::cast_to<NewtonWorld>(p->get("newton_world"));
		if (!w) {
			// try to get via method get_newton_world
			if (p->has_method("get_newton_world")) {
				Variant ret = p->call("get_newton_world");
				w = Object::cast_to<NewtonWorld>(ret);
			}
		}
		if (w) {
			world = w;
			return;
		}
		p = p->get_parent();
	}
	// Fallback: get a global singleton if set via ProjectSettings (not implemented).
}

void NewtonRigidBody3D::_sync_from_newton() {
	if (!newton_body.is_valid()) return;
	// Update the Godot node transform from the Newton body
	set_global_transform(newton_body->get_transform());
}

void NewtonRigidBody3D::_sync_to_newton() {
	if (!newton_body.is_valid()) return;
	// Push the node's transform to Newton (for kinematic or manual override)
	Transform3D xform = get_global_transform();
	newton_body->set_transform(xform);
}

void NewtonRigidBody3D::set_mode(Mode p_mode) {
	mode = p_mode;
	if (newton_body.is_valid()) {
		switch (mode) {
			case MODE_DYNAMIC: newton_body->set_type(BodyType::DYNAMIC); break;
			case MODE_STATIC:  newton_body->set_type(BodyType::STATIC); break;
			case MODE_KINEMATIC: newton_body->set_type(BodyType::KINEMATIC); break;
		}
	}
}

NewtonRigidBody3D::Mode NewtonRigidBody3D::get_mode() const { return mode; }

void NewtonRigidBody3D::set_mass(real_t p_mass) {
	if (newton_body.is_valid()) {
		newton_body->set_mass(p_mass);
		// Recompute inertia if shape exists
		if (collision_shape.is_valid()) {
			newton_body->set_inertia(collision_shape->compute_inertia(p_mass));
		}
	}
}

real_t NewtonRigidBody3D::get_mass() const {
	return newton_body.is_valid() ? newton_body->get_mass() : 0.0;
}

void NewtonRigidBody3D::set_collision_shape(Ref<NewtonCollision> p_shape) {
	collision_shape = p_shape;
	if (newton_body.is_valid() && collision_shape.is_valid()) {
		newton_body->set_collision_shape(collision_shape);
		// Update AABB
		newton_body->set_collision_aabb(collision_shape->get_local_aabb());
		// If dynamic, update inertia
		if (mode == MODE_DYNAMIC) {
			real_t m = newton_body->get_mass();
			if (m > 0.0) newton_body->set_inertia(collision_shape->compute_inertia(m));
		}
	}
}

Ref<NewtonCollision> NewtonRigidBody3D::get_collision_shape() const { return collision_shape; }

void NewtonRigidBody3D::set_linear_velocity(const vec3 &p_vel) {
	if (newton_body.is_valid()) newton_body->set_linear_velocity(p_vel);
}

vec3 NewtonRigidBody3D::get_linear_velocity() const {
	return newton_body.is_valid() ? newton_body->get_linear_velocity() : vec3();
}

void NewtonRigidBody3D::set_angular_velocity(const vec3 &p_vel) {
	if (newton_body.is_valid()) newton_body->set_angular_velocity(p_vel);
}

vec3 NewtonRigidBody3D::get_angular_velocity() const {
	return newton_body.is_valid() ? newton_body->get_angular_velocity() : vec3();
}

void NewtonRigidBody3D::apply_impulse(const vec3 &p_impulse, const vec3 &p_world_point) {
	if (newton_body.is_valid()) {
		vec3 point = p_world_point;
		if (point == vec3()) point = newton_body->get_position();
		newton_body->apply_impulse(p_impulse, point);
	}
}

void NewtonRigidBody3D::apply_force(const vec3 &p_force, const vec3 &p_world_point) {
	if (newton_body.is_valid()) {
		vec3 point = p_world_point;
		if (point == vec3()) point = newton_body->get_position();
		newton_body->apply_force(p_force, point);
	}
}

void NewtonRigidBody3D::set_gravity_enabled(bool p_enabled) {
	if (newton_body.is_valid()) newton_body->set_gravity_enabled(p_enabled);
}

bool NewtonRigidBody3D::is_gravity_enabled() const {
	return newton_body.is_valid() ? newton_body->is_gravity_enabled() : true;
}

void NewtonRigidBody3D::set_ccd_enabled(bool p_enabled) {
	if (newton_body.is_valid()) newton_body->set_ccd_enabled(p_enabled);
}

bool NewtonRigidBody3D::is_ccd_enabled() const {
	return newton_body.is_valid() ? newton_body->is_ccd_enabled() : false;
}

} // namespace newton