// File 310: modules/vienna/src/nodes/vienna_rigid_body_3d.cpp
// Implementation of ViennaRigidBody3D – syncs with ViennaWorld, applies forces,
// manages mode, mass, shape, damping, gravity, sleep, and CCD.  All operations
// delegate to the underlying ViennaBody.

#include "vienna_rigid_body_3d.h"
#include "../world/vienna_world.h"
#include "../bodies/vienna_body.h"
#include "../collision/vienna_shape.h"
#include "../materials/vienna_material.h"
#include "../nodes/vienna_world_node_3d.h"       // to locate the world

#include "core/object/class_db.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/typedefs.h"

namespace vienna {

void ViennaRigidBody3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &ViennaRigidBody3D::set_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &ViennaRigidBody3D::get_mode);
	ClassDB::bind_method(D_METHOD("set_mass", "mass"), &ViennaRigidBody3D::set_mass);
	ClassDB::bind_method(D_METHOD("get_mass"), &ViennaRigidBody3D::get_mass);
	ClassDB::bind_method(D_METHOD("set_collision_shape", "shape"), &ViennaRigidBody3D::set_collision_shape);
	ClassDB::bind_method(D_METHOD("get_collision_shape"), &ViennaRigidBody3D::get_collision_shape);
	ClassDB::bind_method(D_METHOD("set_linear_velocity", "velocity"), &ViennaRigidBody3D::set_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_linear_velocity"), &ViennaRigidBody3D::get_linear_velocity);
	ClassDB::bind_method(D_METHOD("set_angular_velocity", "velocity"), &ViennaRigidBody3D::set_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_angular_velocity"), &ViennaRigidBody3D::get_angular_velocity);
	ClassDB::bind_method(D_METHOD("apply_force", "force", "world_point"), &ViennaRigidBody3D::apply_force, DEFVAL(vec3()));
	ClassDB::bind_method(D_METHOD("apply_central_force", "force"), &ViennaRigidBody3D::apply_central_force);
	ClassDB::bind_method(D_METHOD("apply_impulse", "impulse", "world_point"), &ViennaRigidBody3D::apply_impulse, DEFVAL(vec3()));
	ClassDB::bind_method(D_METHOD("apply_central_impulse", "impulse"), &ViennaRigidBody3D::apply_central_impulse);
	ClassDB::bind_method(D_METHOD("set_linear_damping", "damping"), &ViennaRigidBody3D::set_linear_damping);
	ClassDB::bind_method(D_METHOD("get_linear_damping"), &ViennaRigidBody3D::get_linear_damping);
	ClassDB::bind_method(D_METHOD("set_angular_damping", "damping"), &ViennaRigidBody3D::set_angular_damping);
	ClassDB::bind_method(D_METHOD("get_angular_damping"), &ViennaRigidBody3D::get_angular_damping);
	ClassDB::bind_method(D_METHOD("set_gravity_enabled", "enabled"), &ViennaRigidBody3D::set_gravity_enabled);
	ClassDB::bind_method(D_METHOD("is_gravity_enabled"), &ViennaRigidBody3D::is_gravity_enabled);
	ClassDB::bind_method(D_METHOD("set_sleeping", "sleeping"), &ViennaRigidBody3D::set_sleeping);
	ClassDB::bind_method(D_METHOD("is_sleeping"), &ViennaRigidBody3D::is_sleeping);
	ClassDB::bind_method(D_METHOD("set_material", "material"), &ViennaRigidBody3D::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &ViennaRigidBody3D::get_material);
	ClassDB::bind_method(D_METHOD("set_ccd_enabled", "enabled"), &ViennaRigidBody3D::set_ccd_enabled);
	ClassDB::bind_method(D_METHOD("is_ccd_enabled"), &ViennaRigidBody3D::is_ccd_enabled);

	BIND_ENUM_CONSTANT(MODE_DYNAMIC);
	BIND_ENUM_CONSTANT(MODE_STATIC);
	BIND_ENUM_CONSTANT(MODE_KINEMATIC);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Dynamic,Static,Kinematic"), "set_mode", "get_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mass"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "collision_shape", PROPERTY_HINT_RESOURCE_TYPE, "ViennaShape"), "set_collision_shape", "get_collision_shape");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "linear_damping"), "set_linear_damping", "get_linear_damping");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_damping"), "set_angular_damping", "get_angular_damping");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gravity_enabled"), "set_gravity_enabled", "is_gravity_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sleeping"), "set_sleeping", "is_sleeping");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "ViennaMaterial"), "set_material", "get_material");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ccd_enabled"), "set_ccd_enabled", "is_ccd_enabled");
}

ViennaRigidBody3D::ViennaRigidBody3D() :
	mode(MODE_DYNAMIC),
	world(nullptr),
	body_id(0),
	world_found(false),
	force_accum(vec3()),
	torque_accum(vec3()),
	linear_damping(0.0),
	angular_damping(0.0),
	ccd_enabled(false),
	gravity_enabled(true),
	is_asleep(false) {
	vienna_body.instantiate();
	set_process(true);
}

void ViennaRigidBody3D::_notification(int p_what) {
	if (p_what == NOTIFICATION_READY) {
		_find_world();                               // locate ViennaWorldNode3D ancestor
		if (world) {
			// Configure body type
			switch (mode) {
				case MODE_DYNAMIC:   vienna_body->set_type(BodyType::DYNAMIC);   break;
				case MODE_STATIC:    vienna_body->set_type(BodyType::STATIC);    break;
				case MODE_KINEMATIC: vienna_body->set_type(BodyType::KINEMATIC); break;
			}
			vienna_body->set_mass(get_mass());
			vienna_body->set_linear_damping(linear_damping);
			vienna_body->set_angular_damping(angular_damping);
			vienna_body->set_gravity_enabled(gravity_enabled);
			vienna_body->set_ccd_enabled(ccd_enabled);
			vienna_body->set_active(!is_asleep);
			if (material.is_valid()) {
				material_id mat_id = world->create_material(material);
				vienna_body->set_material_id(mat_id);
			}
			// Set initial transform from the node
			vienna_body->set_transform(get_global_transform());
			// Apply collision shape if set
			if (collision_shape.is_valid()) {
				vienna_body->set_collision_shape(collision_shape);
				vienna_body->set_collision_aabb(collision_shape->get_local_aabb());
				if (mode == MODE_DYNAMIC && vienna_body->get_mass() > 0.0) {
					vienna_body->set_inertia(collision_shape->compute_inertia(vienna_body->get_mass()));
				}
			}
			body_id = world->create_body(vienna_body);
			world_found = true;
		}
	}
	if (p_what == NOTIFICATION_PHYSICS_PROCESS) {
		// Accumulated forces are applied before the physics step
		_sync_to_physics();
	}
	if (p_what == NOTIFICATION_PROCESS) {
		if (world_found) {
			_sync_from_physics();                         // update the Node3D transform from ViennaBody
		}
	}
}

void ViennaRigidBody3D::_find_world() {
	Node *p = get_parent();
	while (p) {
		// Look for a ViennaWorldNode3D companion (or any node providing get_vienna_world)
		if (p->has_method("get_vienna_world")) {
			Variant ret = p->call("get_vienna_world");
			world = Object::cast_to<ViennaWorld>(ret);
			if (world) return;
		}
		// Alternatively, check if the parent itself is a ViennaWorldNode3D
		ViennaWorldNode3D *wn = Object::cast_to<ViennaWorldNode3D>(p);
		if (wn) {
			world = wn->get_vienna_world();
			return;
		}
		p = p->get_parent();
	}
	// If not found, the body will not participate in physics; print warning.
	WARN_PRINT("ViennaRigidBody3D: No ViennaWorldNode3D found in the scene tree.");
}

void ViennaRigidBody3D::_sync_from_physics() {
	if (!vienna_body.is_valid()) return;
	// Update the Godot node’s transform from the physics body
	set_global_transform(vienna_body->get_transform());
}

void ViennaRigidBody3D::_sync_to_physics() {
	if (!vienna_body.is_valid()) return;
	// If kinematic, push the node’s transform to the physics body
	if (mode == MODE_KINEMATIC) {
		vienna_body->set_transform(get_global_transform());
	}
	// Apply accumulated forces before the next physics step
	if (mode == MODE_DYNAMIC && vienna_body->is_active()) {
		vienna_body->apply_force(force_accum, get_global_position());
		vienna_body->clear_forces(); // clear after applying? No, we accumulate externally and send per frame.
		// Actually each frame we apply and reset.
		force_accum = vec3();
		torque_accum = vec3();
	}
}

void ViennaRigidBody3D::set_mode(Mode p_mode) {
	mode = p_mode;
	if (vienna_body.is_valid()) {
		switch (mode) {
			case MODE_DYNAMIC:   vienna_body->set_type(BodyType::DYNAMIC);   break;
			case MODE_STATIC:    vienna_body->set_type(BodyType::STATIC);    break;
			case MODE_KINEMATIC: vienna_body->set_type(BodyType::KINEMATIC); break;
		}
	}
}

ViennaRigidBody3D::Mode ViennaRigidBody3D::get_mode() const { return mode; }

void ViennaRigidBody3D::set_mass(real_t p_mass) {
	if (vienna_body.is_valid()) {
		vienna_body->set_mass(p_mass);
		// Recompute inertia if shape exists
		if (collision_shape.is_valid()) {
			vienna_body->set_inertia(collision_shape->compute_inertia(p_mass));
		}
	}
}

real_t ViennaRigidBody3D::get_mass() const {
	return vienna_body.is_valid() ? vienna_body->get_mass() : 0.0;
}

void ViennaRigidBody3D::set_collision_shape(const Ref<ViennaShape> &p_shape) {
	collision_shape = p_shape;
	if (vienna_body.is_valid() && collision_shape.is_valid()) {
		vienna_body->set_collision_shape(collision_shape);
		vienna_body->set_collision_aabb(collision_shape->get_local_aabb());
		if (mode == MODE_DYNAMIC && vienna_body->get_mass() > 0.0) {
			vienna_body->set_inertia(collision_shape->compute_inertia(vienna_body->get_mass()));
		}
	}
}

Ref<ViennaShape> ViennaRigidBody3D::get_collision_shape() const { return collision_shape; }

void ViennaRigidBody3D::set_linear_velocity(const vec3 &p_vel) {
	if (vienna_body.is_valid()) vienna_body->set_linear_velocity(p_vel);
}

vec3 ViennaRigidBody3D::get_linear_velocity() const {
	return vienna_body.is_valid() ? vienna_body->get_linear_velocity() : vec3();
}

void ViennaRigidBody3D::set_angular_velocity(const vec3 &p_vel) {
	if (vienna_body.is_valid()) vienna_body->set_angular_velocity(p_vel);
}

vec3 ViennaRigidBody3D::get_angular_velocity() const {
	return vienna_body.is_valid() ? vienna_body->get_angular_velocity() : vec3();
}

void ViennaRigidBody3D::apply_force(const vec3 &p_force, const vec3 &p_world_point) {
	if (world_found && vienna_body.is_valid() && vienna_body->is_active()) {
		vienna_body->apply_force(p_force, p_world_point);
	} else {
		// accumulate for later when world is ready
		force_accum += p_force;
		vec3 r = p_world_point - get_global_position();
		torque_accum += r.cross(p_force);
	}
}

void ViennaRigidBody3D::apply_central_force(const vec3 &p_force) {
	apply_force(p_force, get_global_position());
}

void ViennaRigidBody3D::apply_impulse(const vec3 &p_impulse, const vec3 &p_world_point) {
	if (world_found && vienna_body.is_valid() && vienna_body->is_active()) {
		vienna_body->apply_impulse(p_impulse, p_world_point);
	}
}

void ViennaRigidBody3D::apply_central_impulse(const vec3 &p_impulse) {
	apply_impulse(p_impulse, get_global_position());
}

void ViennaRigidBody3D::set_linear_damping(real_t p_damping) {
	linear_damping = p_damping;
	if (vienna_body.is_valid()) vienna_body->set_linear_damping(linear_damping);
}

real_t ViennaRigidBody3D::get_linear_damping() const { return linear_damping; }

void ViennaRigidBody3D::set_angular_damping(real_t p_damping) {
	angular_damping = p_damping;
	if (vienna_body.is_valid()) vienna_body->set_angular_damping(angular_damping);
}

real_t ViennaRigidBody3D::get_angular_damping() const { return angular_damping; }

void ViennaRigidBody3D::set_gravity_enabled(bool p_enabled) {
	gravity_enabled = p_enabled;
	if (vienna_body.is_valid()) vienna_body->set_gravity_enabled(p_enabled);
}

bool ViennaRigidBody3D::is_gravity_enabled() const { return gravity_enabled; }

void ViennaRigidBody3D::set_sleeping(bool p_sleeping) {
	is_asleep = p_sleeping;
	if (vienna_body.is_valid()) vienna_body->set_active(!p_sleeping);
}

bool ViennaRigidBody3D::is_sleeping() const { return is_asleep; }

void ViennaRigidBody3D::set_material(const Ref<ViennaMaterial> &p_material) {
	material = p_material;
	if (world_found && material.is_valid() && vienna_body.is_valid()) {
		material_id mat_id = world->create_material(material);
		vienna_body->set_material_id(mat_id);
	}
}

Ref<ViennaMaterial> ViennaRigidBody3D::get_material() const { return material; }

void ViennaRigidBody3D::set_ccd_enabled(bool p_enabled) {
	ccd_enabled = p_enabled;
	if (vienna_body.is_valid()) vienna_body->set_ccd_enabled(p_enabled);
}

bool ViennaRigidBody3D::is_ccd_enabled() const { return ccd_enabled; }

} // namespace vienna