// File 160: modules/genesis/src/genesis_physics_server.cpp
// Implementation of the GenesisPhysicsServer3D – the custom PhysicsServer3DExtension
// that replaces Godot's default physics with the Gaia + Genesis multi‑solver engine.
// Provides body/space/shape management and calls the Genesis solvers each physics tick.

#include "genesis_physics_server.h"

// Gaia broad‑phase
#include "../../gaia/src/collision_detector/broad_phase.h"
#include "../../gaia/src/collision_detector/narrow_phase.h"
#include "../../gaia/src/collision_detector/contact.h"

// Genesis entities and solvers
#include "entities/rigid_entity.h"
#include "entities/fem_entity.h"
#include "entities/mpm_entity.h"
#include "entities/tool_entity.h"
#include "solvers/rigid_solver.h"
#include "solvers/fem_solver.h"
#include "solvers/mpm_solver.h"
#include "solvers/pbd_solver.h"
#include "solvers/sph_solver.h"
#include "solvers/sf_solver.h"
#include "solvers/kinematic_solver.h"
#include "sensors/base_sensor.h"
#include "sensors/contact_force_sensor.h"

// Godot core
#include "core/os/os.h"
#include "core/variant/variant.h"

namespace genesis {

void GenesisPhysicsServer3D::_bind_methods() {
	// No additional bindings beyond the extension base for now.
}

// ---- space / area (not used) ----
RID GenesisPhysicsServer3D::space_create() { return RID(); }
RID GenesisPhysicsServer3D::area_create()  { return RID(); }

// ---- body creation ----
RID GenesisPhysicsServer3D::body_create() {
	Ref<RigidEntity> entity;
	entity.instantiate();
	entity->set_material(Ref<GenesisMaterial>(memnew(GenesisMaterial)));   // default material
	entity->set_entity_uid(_next_uid());
	RID rid = entity_owner.make_rid(entity);
	all_entities.push_back(entity);
	// Add to rigid solver automatically
	rigid_solver->add_entity(entity);
	return rid;
}

RID GenesisPhysicsServer3D::soft_body_create() {
	Ref<FEMEntity> fem;
	fem.instantiate();
	fem->set_entity_uid(_next_uid());
	RID rid = entity_owner.make_rid(fem);
	all_entities.push_back(fem);
	fem_solver->add_entity(fem);
	return rid;
}

// ---- shape (ignored, shapes are built via entity properties) ----
RID GenesisPhysicsServer3D::shape_create(ShapeType p_type) { return RID(); }

// ---- body state ----
void GenesisPhysicsServer3D::body_set_state(RID p_body, BodyState p_state, const Variant &p_value) {
	Ref<BaseEntity> entity = entity_owner.get_or_null(p_body);
	if (entity.is_null()) return;
	switch (p_state) {
		case BODY_STATE_TRANSFORM:
			entity->set_transform(p_value);
			break;
		case BODY_STATE_LINEAR_VELOCITY:
			entity->set_linear_velocity(p_value);
			break;
		case BODY_STATE_ANGULAR_VELOCITY:
			entity->set_angular_velocity(p_value);
			break;
		case BODY_STATE_SLEEPING: {
			bool sleep = p_value;
			entity->set_active(!sleep);
		} break;
		case BODY_STATE_CAN_SLEEP:
			// ignore for now
			break;
	}
}

Variant GenesisPhysicsServer3D::body_get_state(RID p_body, BodyState p_state) {
	Ref<BaseEntity> entity = entity_owner.get_or_null(p_body);
	if (entity.is_null()) return Variant();
	switch (p_state) {
		case BODY_STATE_TRANSFORM:          return entity->get_transform();
		case BODY_STATE_LINEAR_VELOCITY:    return entity->get_linear_velocity();
		case BODY_STATE_ANGULAR_VELOCITY:   return entity->get_angular_velocity();
		case BODY_STATE_SLEEPING:           return !entity->is_active();
		default: return Variant();
	}
}

// ---- shape attachment ----
void GenesisPhysicsServer3D::body_add_shape(RID p_body, RID p_shape, const Transform3D &p_transform, bool p_disabled) {}
void GenesisPhysicsServer3D::body_set_shape(RID p_body, int p_shape_idx, RID p_shape) {}
void GenesisPhysicsServer3D::body_set_shape_transform(RID p_body, int p_shape_idx, const Transform3D &p_transform) {}

// ---- main stepping ----
void GenesisPhysicsServer3D::physics_step(real_t p_step) {
	if (!active) return;

	// In a true server, sync all body transforms from the Godot scene tree
	// before stepping. Here we assume they were already set via body_set_state.

	real_t dt = MIN(p_step, fixed_step);
	time_accumulator += dt;

	while (time_accumulator >= fixed_step) {
		// --- Force accumulation & gravity ---
		for (Ref<BaseEntity> &ent : all_entities) {
			if (ent.is_null() || !ent->is_active()) continue;
			if (ent->is_gravity_enabled()) {
				ent->apply_force(gravity * ent->get_mass(), ent->get_position());
			}
		}

		// --- Kinematic solver (avatar / tool entities) ---
		kinematic_solver->set_dt(fixed_step);
		kinematic_solver->step();

		// --- Rigid body dynamics + collision ---
		rigid_solver->set_dt(fixed_step);
		rigid_solver->set_gravity(gravity);
		rigid_solver->step();

		// --- Deformable (FEM) ---
		fem_solver->set_dt(fixed_step);
		fem_solver->set_gravity(gravity);
		fem_solver->step();

		// --- Material Point Method ---
		mpm_solver->set_dt(fixed_step);
		mpm_solver->set_gravity(gravity);
		mpm_solver->step();

		// --- Smoothed Particle Hydrodynamics ---
		sph_solver->set_dt(fixed_step);
		sph_solver->set_gravity(gravity);
		sph_solver->step();

		// --- Position Based Dynamics (cloth, etc.) ---
		pbd_solver->set_gravity(gravity);
		pbd_solver->step();            // PBD solver internally uses its own dt

		// --- Stable Fluids ---
		sf_solver->set_gravity(gravity);
		sf_solver->step();

		// --- Update sensors ---
		for (Ref<BaseSensor> &sensor : sensors) {
			if (sensor.is_null() || !sensor->is_enabled()) continue;
			entity_id_t uid = sensor->get_entity_uid();
			Ref<BaseEntity> ent = get_entity_by_uid(uid);
			if (ent.is_null()) continue;
			sensor->step(fixed_step, ent);
		}

		time_accumulator -= fixed_step;
	}
}

void GenesisPhysicsServer3D::set_active(bool p_active) { active = p_active; }
bool GenesisPhysicsServer3D::is_active() const { return active; }

void GenesisPhysicsServer3D::load_options(const String &path) {
	world_options.load_from_json(path);
	// apply sub‑dicts to each solver
	rigid_solver->init_from_options(world_options.get_dict("rigid_solver"));
	fem_solver->init_from_options(world_options.get_dict("fem_solver"));
	mpm_solver->init_from_options(world_options.get_dict("mpm_solver"));
	sph_solver->init_from_options(world_options.get_dict("sph_solver"));
	pbd_solver->init_from_options(world_options.get_dict("pbd_solver"));
	sf_solver->init_from_options(world_options.get_dict("sf_solver"));
	kinematic_solver->init_from_options(world_options.get_dict("kinematic_solver"));
}

uint64_t GenesisPhysicsServer3D::_next_uid() {
	static uint64_t counter = 1;
	return counter++;
}

Ref<BaseEntity> GenesisPhysicsServer3D::get_entity_by_uid(entity_id_t p_uid) const {
	for (const Ref<BaseEntity> &e : all_entities) {
		if (e.is_valid() && e->get_entity_uid() == p_uid) return e;
	}
	return Ref<BaseEntity>();
}

} // namespace genesis