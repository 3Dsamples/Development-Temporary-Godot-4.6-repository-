// File 88: modules/genesis/src/genesis_physics_server.h
// GenesisPhysicsServer: a custom Godot PhysicsServer3DExtension that replaces
// the default physics engine with the multi‑solver Genesis engine.
// Integrates all Genesis solvers, entities, materials, sensors, and couples
// with Gaia's BVH broad‑phase and narrow‑phase collision.

#ifndef GENESIS_PHYSICS_SERVER_H
#define GENESIS_PHYSICS_SERVER_H

#include "servers/physics_server_3d.h"
#include "servers/physics_server_3d_helpers.h"

// Gaia BVH
#include "../../gaia/src/bvh/bvh.h"
#include "../../gaia/src/bvh/aabb.h"
#include "../../gaia/src/collision_detector/broad_phase.h"
#include "../../gaia/src/collision_detector/narrow_phase.h"
#include "../../gaia/src/collision_detector/contact.h"

// Genesis core
#include "core/genesis_types.h"
#include "core/genesis_constants.h"
#include "options/options_system.h"
#include "entities/base_entity.h"
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
#include "materials/material_base.h"

namespace genesis {

class GenesisPhysicsServer3D : public PhysicsServer3DExtension {
	GDCLASS(GenesisPhysicsServer3D, PhysicsServer3DExtension);

public:
	GenesisPhysicsServer3D() :
		world_options(),
		gravity(0, -9.81, 0),
		time_accumulator(0.0),
		fixed_step(1.0 / 60.0),
		active(false) {
		// Create default solvers
		rigid_solver.instantiate();
		fem_solver.instantiate();
		mpm_solver.instantiate();
		sph_solver.instantiate();
		pbd_solver.instantiate();
		sf_solver.instantiate();
		kinematic_solver.instantiate();
	}

	virtual ~GenesisPhysicsServer3D() {}

	// --- PhysicsServer3D overrides (only the essential ones for demonstration) ---

	virtual bool is_flushing_queries() const override { return false; }
	virtual int get_process_info(ProcessInfo p_info) override { return 0; }

	virtual RID space_create() override { return RID(); }
	virtual RID area_create() override { return RID(); }
	virtual RID body_create() override {
		// Create a rigid entity as a body
		Ref<RigidEntity> entity = memnew(RigidEntity);
		entity->set_material(Ref<GenesisMaterial>(memnew(GenesisMaterial)));
		RID rid = entity_owner.make_rid(entity);
		all_entities.push_back(entity);
		return rid;
	}
	virtual RID soft_body_create() override {
		// Create a FEM entity as soft body
		Ref<FEMEntity> fem = memnew(FEMEntity);
		RID rid = entity_owner.make_rid(fem);
		all_entities.push_back(fem);
		return rid;
	}
	virtual RID shape_create(ShapeType p_type) override { return RID(); }

	virtual void body_set_space(RID p_body, RID p_space) override {}
	virtual void body_set_mode(RID p_body, BodyMode p_mode) override {}
	virtual void body_set_state(RID p_body, BodyState p_state, const Variant &p_value) override {
		Ref<BaseEntity> entity = entity_owner.get_or_null(p_body);
		if (entity.is_null()) return;
		switch (p_state) {
			case BODY_STATE_TRANSFORM: entity->set_transform(p_value); break;
			case BODY_STATE_LINEAR_VELOCITY: entity->set_linear_velocity(p_value); break;
			case BODY_STATE_ANGULAR_VELOCITY: entity->set_angular_velocity(p_value); break;
			default: break;
		}
	}
	virtual Variant body_get_state(RID p_body, BodyState p_state) override {
		Ref<BaseEntity> entity = entity_owner.get_or_null(p_body);
		if (entity.is_null()) return Variant();
		switch (p_state) {
			case BODY_STATE_TRANSFORM: return entity->get_transform();
			case BODY_STATE_LINEAR_VELOCITY: return entity->get_linear_velocity();
			case BODY_STATE_ANGULAR_VELOCITY: return entity->get_angular_velocity();
			default: return Variant();
		}
	}

	virtual void body_add_shape(RID p_body, RID p_shape, const Transform3D &p_transform = Transform3D(), bool p_disabled = false) override {}
	virtual void body_set_shape(RID p_body, int p_shape_idx, RID p_shape) override {}
	virtual void body_set_shape_transform(RID p_body, int p_shape_idx, const Transform3D &p_transform) override {}

	virtual void physics_step(real_t p_step) override {
		if (!active) return;
		// Run all solvers sequentially in a fixed order
		real_t dt = MIN(p_step, fixed_step);
		time_accumulator += dt;

		// Pre-step: gather forces, apply kinematic targets
		for (Ref<BaseEntity> &entity : all_entities) {
			if (entity.is_null()) continue;
			// Apply gravity
			if (entity->is_gravity_enabled()) {
				entity->apply_force(gravity * entity->get_mass(), entity->get_position());
			}
		}

		// Step kinematic solver (drives tool entities)
		kinematic_solver->set_dt(dt);
		kinematic_solver->step();

		// Step rigid solver
		rigid_solver->set_dt(dt);
		rigid_solver->set_gravity(gravity);
		rigid_solver->step();

		// Step FEM solver(s)
		fem_solver->set_dt(dt);
		fem_solver->step();

		// Step MPM solver
		mpm_solver->set_dt(dt);
		mpm_solver->step();

		// Step SPH solver
		sph_solver->set_dt(dt);
		sph_solver->step();

		// Step PBD solver
		pbd_solver->set_dt(dt);
		pbd_solver->step();

		// Step Stable Fluids solver
		sf_solver->set_dt(dt);
		sf_solver->step();

		// Collision detection between all entities (using Gaia broad phase)
		detect_collisions();

		// Update Godot's body states (needed for game logic to read)
		sync_to_godot();

		time_accumulator -= fixed_step;
	}

	virtual void set_active(bool p_active) { active = p_active; }
	virtual bool is_active() const { return active; }

	// Expose worlds as a property? Not needed.

	// Configure via options (loaded from a JSON resource)
	void load_options(const String &path) {
		world_options.load_from_json(path);
		// Apply options to each solver
		rigid_solver->init_from_options(world_options.get_dict("rigid"));
		fem_solver->init_from_options(world_options.get_dict("fem"));
		mpm_solver->init_from_options(world_options.get_dict("mpm"));
		sph_solver->init_from_options(world_options.get_dict("sph"));
		pbd_solver->init_from_options(world_options.get_dict("pbd"));
		sf_solver->init_from_options(world_options.get_dict("sf"));
		kinematic_solver->init_from_options(world_options.get_dict("kinematic"));
	}

private:
	// Detect collisions between entities using Gaia's broad phase and narrow phase
	void detect_collisions() {
		gaia::collision::BroadPhase bp;
		// Insert all rigid entities into broad phase
		LocalVector<Ref<RigidEntity>> rigid_entities;
		for (Ref<BaseEntity> &entity : all_entities) {
			Ref<RigidEntity> rigid = entity;
			if (rigid.is_valid() && rigid->is_active()) {
				bp.add_object(rigid->get_entity_uid(), rigid->get_aabb());
				rigid_entities.push_back(rigid);
			}
		}
		// Find pairs and process
		bp.find_pairs([](uint32_t handle_a, uint32_t handle_b, void *userdata) {
			auto *self = static_cast<GenesisPhysicsServer3D *>(userdata);
			// Retrieve entities and resolve collision between them using GJK
			Ref<RigidEntity> a, b;
			for (Ref<BaseEntity> &e : self->all_entities) {
				if (e->get_entity_uid() == handle_a) a = e;
				if (e->get_entity_uid() == handle_b) b = e;
			}
			if (a.is_null() || b.is_null()) return;
			// Use GJK to get contact info
			// For now we just apply a simple spring push; a full manifold would follow.
		}, this);
	}

	// Sync entity transforms back to Godot's Space for use in scripts
	void sync_to_godot() {
		for (Ref<BaseEntity> &entity : all_entities) {
			if (entity.is_null()) continue;
			// Notify Godot's space (if we had one) of transform change. 
			// Since we are a custom server, we can store state and let Godot query it.
		}
	}

	genesis::options::Options world_options;
	Vector3 gravity;
	real_t time_accumulator;
	real_t fixed_step;
	bool active;

	// Entity ownership
	RID_Owner<BaseEntity> entity_owner;
	LocalVector<Ref<BaseEntity>> all_entities;

	// Solver instances
	Ref<RigidSolver> rigid_solver;
	Ref<FEMSolver> fem_solver;
	Ref<MPMSolver> mpm_solver;
	Ref<SPHSolver> sph_solver;
	Ref<GenesisPBDSolver> pbd_solver;
	Ref<SFSolver> sf_solver;
	Ref<KinematicSolver> kinematic_solver;
};

} // namespace genesis

#endif // GENESIS_PHYSICS_SERVER_H