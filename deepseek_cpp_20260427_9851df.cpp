// File 261: modules/integration/unified_physics_server_3d.cpp
// Implementation of the UnifiedPhysicsServer3D – bridges Newton Dynamics,
// Genesis multi‑solver, and Gaia broad‑phase into a single Godot physics server.

#include "unified_physics_server_3d.h"

// Newton
#include "../newton/src/world/newton_world.h"
#include "../newton/src/bodies/newton_body.h"
#include "../newton/src/collision/newton_collision.h"
#include "../newton/src/core/newton_types.h"
#include "../newton/src/core/newton_constants.h"

// Genesis
#include "../genesis/src/genesis_world.h"
#include "../genesis/src/entities/rigid_entity.h"
#include "../genesis/src/entities/fem_entity.h"
#include "../genesis/src/entities/mpm_entity.h"
#include "../genesis/src/core/genesis_types.h"

// Gaia
#include "../gaia/src/collision_detector/broad_phase.h"

// Godot
#include "core/config/project_settings.h"
#include "core/io/json.h"
#include "core/object/class_db.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"
#include "core/typedefs.h"

namespace unified {

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
UnifiedPhysicsServer3D::UnifiedPhysicsServer3D() : active_space(RID()) {
	// Create a default space
	RID default_space = space_create();
	active_space = default_space;
}

UnifiedPhysicsServer3D::~UnifiedPhysicsServer3D() {
	// Cleanup all spaces (destroy Newton worlds, Genesis worlds)
	for (KeyValue<RID, SpaceData> &kv : space_map) {
		if (kv.value.newton_world) memdelete(kv.value.newton_world);
		if (kv.value.genesis_world) memdelete(kv.value.genesis_world);
		if (kv.value.broad_phase) memdelete(kv.value.broad_phase);
	}
	space_map.clear();
}

// ---------------------------------------------------------------------------
// Server interface
// ---------------------------------------------------------------------------
bool UnifiedPhysicsServer3D::is_flushing_queries() const { return false; }
int UnifiedPhysicsServer3D::get_process_info(ProcessInfo p_info) {
	// Return aggregate information from all spaces
	switch (p_info) {
		case INFO_ACTIVE_OBJECTS: {
			int total = 0;
			for (const KeyValue<RID, SpaceData> &kv : space_map) {
				if (kv.value.newton_world) total += kv.value.newton_world->get_body_count();
				if (kv.value.genesis_world) total += kv.value.genesis_world->get_entity_count();
			}
			return total;
		}
		default: return 0;
	}
}

// ---------------------------------------------------------------------------
// Space / Area creation
// ---------------------------------------------------------------------------
RID UnifiedPhysicsServer3D::space_create() {
	SpaceData space;
	space.newton_world = memnew(newton::NewtonWorld);
	space.genesis_world = memnew(genesis::GenesisWorld);
	space.broad_phase = memnew(gaia::collision::BroadPhase);
	space.gravity = Vector3(0, -9.80665, 0);
	space.step_size = 1.0 / 60.0;
	space.active = true;
	RID rid = RID();
	space.self = rid;
	space_map[rid] = space;
	return rid;
}

RID UnifiedPhysicsServer3D::area_create() { return RID(); } // not used

// ---------------------------------------------------------------------------
// Body creation (rigid body → Newton; soft body → Genesis FEM)
// ---------------------------------------------------------------------------
RID UnifiedPhysicsServer3D::body_create() {
	// Create a Newton rigid body
	Ref<newton::NewtonBody> body;
	body.instantiate();
	body->set_type(newton::BodyType::DYNAMIC);

	// Register it in the default space's Newton world
	SpaceData &space = space_map[active_space];
	newton::body_id nid = space.newton_world->create_body(body);

	NewtonBodyData bd;
	bd.self = RID();
	bd.nid = nid;
	bd.active = true;
	RID rid = RID();
	newton_body_map[rid] = bd;

	return rid;
}

RID UnifiedPhysicsServer3D::soft_body_create() {
	// Create a Genesis FEM entity
	Ref<genesis::FEMEntity> fem;
	fem.instantiate();
	fem->set_entity_uid(genesis::entity_id_t(genesis::GenesisWorld::get_next_entity_uid()));

	SpaceData &space = space_map[active_space];
	genesis::entity_id_t gid = fem->get_entity_uid();
	space.genesis_world->add_entity(fem);

	GenesisFEMData fd;
	fd.self = RID();
	fd.gid = gid;
	fd.active = true;
	RID rid = RID();
	genesis_fem_map[rid] = fd;

	return rid;
}

// ---------------------------------------------------------------------------
// Shape creation (collision shapes)
// ---------------------------------------------------------------------------
RID UnifiedPhysicsServer3D::shape_create(ShapeType p_type) {
	RID rid = RID();
	Ref<newton::NewtonCollision> shape;
	switch (p_type) {
		case SHAPE_SPHERE: shape.instantiate(); break;
		case SHAPE_BOX:    shape.instantiate(); break;
		case SHAPE_CAPSULE: shape.instantiate(); break;
		case SHAPE_CYLINDER: shape.instantiate(); break;
		default: shape.instantiate(); break;
	}
	shape_map[rid] = shape;
	return rid;
}

// ---------------------------------------------------------------------------
// Body space assignment
// ---------------------------------------------------------------------------
void UnifiedPhysicsServer3D::body_set_space(RID p_body, RID p_space) {
	// Move body from its current space to another. For simplicity, we ignore
	// and assume bodies stay in the default space.
}

void UnifiedPhysicsServer3D::body_set_mode(RID p_body, BodyMode p_mode) {
	// Find body in Newton map and set mode
	if (newton_body_map.has(p_body)) {
		SpaceData &space = space_map[active_space];
		Ref<newton::NewtonBody> body = space.newton_world->get_body(newton_body_map[p_body].nid);
		if (body.is_valid()) {
			switch (p_mode) {
				case BODY_MODE_STATIC: body->set_type(newton::BodyType::STATIC); break;
				case BODY_MODE_KINEMATIC: body->set_type(newton::BodyType::KINEMATIC); break;
				default: body->set_type(newton::BodyType::DYNAMIC); break;
			}
		}
	}
	// Genesis soft bodies have no direct mode mapping; they are always dynamic.
}

void UnifiedPhysicsServer3D::body_set_state(RID p_body, BodyState p_state, const Variant &p_value) {
	// Handle Newton rigid bodies
	if (newton_body_map.has(p_body)) {
		SpaceData &space = space_map[active_space];
		Ref<newton::NewtonBody> body = space.newton_world->get_body(newton_body_map[p_body].nid);
		if (body.is_null()) return;
		switch (p_state) {
			case BODY_STATE_TRANSFORM: body->set_transform(p_value); break;
			case BODY_STATE_LINEAR_VELOCITY: body->set_linear_velocity(p_value); break;
			case BODY_STATE_ANGULAR_VELOCITY: body->set_angular_velocity(p_value); break;
			case BODY_STATE_SLEEPING: body->set_active(!bool(p_value)); break;
			default: break;
		}
	}
	// Handle Genesis FEM bodies
	else if (genesis_fem_map.has(p_body)) {
		SpaceData &space = space_map[active_space];
		Ref<genesis::BaseEntity> entity = space.genesis_world->get_entity(genesis_fem_map[p_body].gid);
		if (entity.is_null()) return;
		switch (p_state) {
			case BODY_STATE_TRANSFORM: entity->set_transform(p_value); break;
			case BODY_STATE_LINEAR_VELOCITY: entity->set_linear_velocity(p_value); break;
			case BODY_STATE_ANGULAR_VELOCITY: entity->set_angular_velocity(p_value); break;
			default: break;
		}
	}
}

Variant UnifiedPhysicsServer3D::body_get_state(RID p_body, BodyState p_state) {
	if (newton_body_map.has(p_body)) {
		SpaceData &space = space_map[active_space];
		Ref<newton::NewtonBody> body = space.newton_world->get_body(newton_body_map[p_body].nid);
		if (body.is_null()) return Variant();
		switch (p_state) {
			case BODY_STATE_TRANSFORM: return body->get_transform();
			case BODY_STATE_LINEAR_VELOCITY: return body->get_linear_velocity();
			case BODY_STATE_ANGULAR_VELOCITY: return body->get_angular_velocity();
			case BODY_STATE_SLEEPING: return !body->is_active();
			default: return Variant();
		}
	} else if (genesis_fem_map.has(p_body)) {
		SpaceData &space = space_map[active_space];
		Ref<genesis::BaseEntity> entity = space.genesis_world->get_entity(genesis_fem_map[p_body].gid);
		if (entity.is_null()) return Variant();
		switch (p_state) {
			case BODY_STATE_TRANSFORM: return entity->get_transform();
			case BODY_STATE_LINEAR_VELOCITY: return entity->get_linear_velocity();
			case BODY_STATE_ANGULAR_VELOCITY: return entity->get_angular_velocity();
			default: return Variant();
		}
	}
	return Variant();
}

// ---------------------------------------------------------------------------
// Shape assignment
// ---------------------------------------------------------------------------
void UnifiedPhysicsServer3D::body_add_shape(RID p_body, RID p_shape, const Transform3D &p_transform, bool p_disabled) {
	if (newton_body_map.has(p_body)) {
		SpaceData &space = space_map[active_space];
		Ref<newton::NewtonBody> body = space.newton_world->get_body(newton_body_map[p_body].nid);
		if (body.is_null()) return;
		if (shape_map.has(p_shape)) {
			body->set_collision_shape(shape_map[p_shape]);
			body->set_collision_aabb(shape_map[p_shape]->get_local_aabb());
		}
	}
	// For Genesis entities, the shape is embedded in the FEM mesh; not handled here.
}

void UnifiedPhysicsServer3D::body_set_shape(RID p_body, int p_shape_idx, RID p_shape) { body_add_shape(p_body, p_shape); }
void UnifiedPhysicsServer3D::body_set_shape_transform(RID p_body, int p_shape_idx, const Transform3D &p_transform) {}

// ---------------------------------------------------------------------------
// Physics step – called each tick
// ---------------------------------------------------------------------------
void UnifiedPhysicsServer3D::physics_step(real_t p_step) {
	for (KeyValue<RID, SpaceData> &kv : space_map) {
		if (kv.value.active) {
			_step_space(kv.value, p_step);
		}
	}
}

void UnifiedPhysicsServer3D::_step_space(SpaceData &space, real_t p_dt) {
	real_t dt = MIN(p_dt, space.step_size);

	// 1. Run Newton Dynamics step (rigid bodies, joints, collisions)
	space.newton_world->set_gravity(space.gravity);
	space.newton_world->step(dt);

	// 2. Run Genesis multi‑solver step (FEM, MPM, SPH, PBD, etc.)
	space.genesis_world->set_gravity(space.gravity);
	space.genesis_world->simulate_step(dt);
}

// ---------------------------------------------------------------------------
// Sync Newton / Genesis transforms back to Godot's space state
// ---------------------------------------------------------------------------
void UnifiedPhysicsServer3D::_sync_newton_to_godot(const SpaceData &p_space) {
	// The server is queried by Godot for body states; no extra sync needed.
	// If we stored body states internally, we would update them here.
}

void UnifiedPhysicsServer3D::_sync_genesis_to_godot(const SpaceData &p_space) {
	// The server returns states directly from entities when queried.
}

// ---------------------------------------------------------------------------
// Active / gravity management
// ---------------------------------------------------------------------------
void UnifiedPhysicsServer3D::set_active(bool p_active) {
	for (KeyValue<RID, SpaceData> &kv : space_map) {
		kv.value.active = p_active;
	}
}

bool UnifiedPhysicsServer3D::is_active() const {
	for (const KeyValue<RID, SpaceData> &kv : space_map) {
		if (kv.value.active) return true;
	}
	return false;
}

void UnifiedPhysicsServer3D::set_space_gravity(RID p_space, const Vector3 &p_gravity) {
	if (space_map.has(p_space)) {
		space_map[p_space].gravity = p_gravity;
	}
}

Vector3 UnifiedPhysicsServer3D::get_space_gravity(RID p_space) const {
	if (space_map.has(p_space)) return space_map[p_space].gravity;
	return Vector3(0, -9.80665, 0);
}

} // namespace unified