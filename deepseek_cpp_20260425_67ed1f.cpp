// File 59: modules/genesis/src/solvers/base_solver.h
// Abstract base solver for Genesis physics solvers (rigid, FEM, MPM, PBD, SPH, SF).
// Each derived solver implements the simulation step for its entity type.

#ifndef GENESIS_SOLVERS_BASE_SOLVER_H
#define GENESIS_SOLVERS_BASE_SOLVER_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "../entities/base_entity.h"
#include "../options/options_system.h"
#include "../core/genesis_types.h"
#include "../core/genesis_constants.h"

namespace genesis {

/**
 * Base solver interface. Derived solvers (RigidSolver, FEMSolver, MPMSolver, etc.)
 * implement the step() method to advance their specific entities in time.
 *
 * The solver owns a set of entities and provides hooks for collision detection,
 * boundary conditions, and constraint enforcement.
 */
class BaseSolver : public RefCounted {
	GDCLASS(BaseSolver, RefCounted);

public:
	BaseSolver() :
		dt(DEFAULT_DT),
		gravity(0, -GRAVITY_EARTH, 0),
		time(0.0),
		sub_steps(1),
		iterations(DEFAULT_SOLVER_ITERATIONS),
		collision_enabled(true),
		world(nullptr) {}

	virtual ~BaseSolver() {}

	// --- Time step ---
	void set_dt(real_t p_dt) { dt = MAX(p_dt, 1e-6); }
	real_t get_dt() const { return dt; }

	void set_sub_steps(int p_sub) { sub_steps = MAX(p_sub, 1); }
	int get_sub_steps() const { return sub_steps; }

	void set_iterations(int p_iter) { iterations = MAX(p_iter, 1); }
	int get_iterations() const { return iterations; }

	void set_gravity(const Vector3 &p_g) { gravity = p_g; }
	Vector3 get_gravity() const { return gravity; }

	void set_collision_enabled(bool p_enabled) { collision_enabled = p_enabled; }
	bool is_collision_enabled() const { return collision_enabled; }

	real_t get_time() const { return time; }

	// --- Entity management ---
	void add_entity(Ref<BaseEntity> p_entity) {
		ERR_FAIL_COND(p_entity.is_null());
		entity_id_t uid = p_entity->get_entity_uid();
		entities[uid] = p_entity;
	}

	void remove_entity(entity_id_t p_uid) {
		entities.erase(p_uid);
	}

	Ref<BaseEntity> get_entity(entity_id_t p_uid) const {
		HashMap<entity_id_t, Ref<BaseEntity>>::ConstIterator it = entities.find(p_uid);
		if (it) return it->value;
		return Ref<BaseEntity>();
	}

	int get_entity_count() const { return entities.size(); }

	void clear_entities() { entities.clear(); }

	// --- Main step: called every physics frame ---
	virtual void step() {
		real_t sub_dt = dt / real_t(sub_steps);
		for (int substep = 0; substep < sub_steps; ++substep) {
			// 1. Pre-step (apply forces, boundary conditions)
			pre_step(sub_dt);
			// 2. Solve constraints / collisions
			if (collision_enabled) {
				detect_collisions(sub_dt);
			}
			// 3. Integrate equations of motion
			solve(sub_dt);
			// 4. Update positions / velocities
			post_step(sub_dt);
			time += sub_dt;
		}
	}

	// --- Virtual methods to be overridden ---
	virtual void pre_step(real_t p_sub_dt) {
		// Default: apply gravity to all entities
		for (KeyValue<entity_id_t, Ref<BaseEntity>> &kv : entities) {
			Ref<BaseEntity> e = kv.value;
			if (e.is_valid() && e->is_active() && e->is_gravity_enabled()) {
				e->apply_force(gravity * e->get_mass(), e->get_position());
			}
		}
	}

	virtual void detect_collisions(real_t p_sub_dt) {
		// Base: no collision detection. Override in rigid/MPM solvers.
		return;
	}

	virtual void solve(real_t p_sub_dt) = 0; // pure virtual

	virtual void post_step(real_t p_sub_dt) {
		// Default: integrate velocities and positions for all entities
		for (KeyValue<entity_id_t, Ref<BaseEntity>> &kv : entities) {
			Ref<BaseEntity> e = kv.value;
			if (e.is_valid() && e->is_active()) {
				e->integrate_velocity(p_sub_dt);
				e->integrate_position(p_sub_dt);
			}
		}
	}

	// --- Configuration ---
	virtual void init_from_options(const genesis::options::Options &opts) {
		dt = opts.get_real("dt", DEFAULT_DT);
		gravity = opts.get_vector3("gravity", Vector3(0, -GRAVITY_EARTH, 0));
		sub_steps = opts.get_int("sub_steps", 1);
		iterations = opts.get_int("iterations", DEFAULT_SOLVER_ITERATIONS);
		collision_enabled = opts.get_bool("collision", true);
	}

	// --- Optional: attach to a Gaia simulation world (for broadphase etc.) ---
	void set_world(void *p_world) { world = p_world; }
	void *get_world() const { return world; }

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_dt", "dt"), &BaseSolver::set_dt);
		ClassDB::bind_method(D_METHOD("get_dt"), &BaseSolver::get_dt);
		ClassDB::bind_method(D_METHOD("set_sub_steps", "sub_steps"), &BaseSolver::set_sub_steps);
		ClassDB::bind_method(D_METHOD("get_sub_steps"), &BaseSolver::get_sub_steps);
		ClassDB::bind_method(D_METHOD("set_iterations", "iterations"), &BaseSolver::set_iterations);
		ClassDB::bind_method(D_METHOD("get_iterations"), &BaseSolver::get_iterations);
		ClassDB::bind_method(D_METHOD("set_gravity", "gravity"), &BaseSolver::set_gravity);
		ClassDB::bind_method(D_METHOD("get_gravity"), &BaseSolver::get_gravity);
		ClassDB::bind_method(D_METHOD("set_collision_enabled", "enabled"), &BaseSolver::set_collision_enabled);
		ClassDB::bind_method(D_METHOD("is_collision_enabled"), &BaseSolver::is_collision_enabled);
		ClassDB::bind_method(D_METHOD("get_time"), &BaseSolver::get_time);
		ClassDB::bind_method(D_METHOD("add_entity", "entity"), &BaseSolver::add_entity);
		ClassDB::bind_method(D_METHOD("remove_entity", "uid"), &BaseSolver::remove_entity);
		ClassDB::bind_method(D_METHOD("get_entity", "uid"), &BaseSolver::get_entity);
		ClassDB::bind_method(D_METHOD("get_entity_count"), &BaseSolver::get_entity_count);
		ClassDB::bind_method(D_METHOD("clear_entities"), &BaseSolver::clear_entities);
		ClassDB::bind_method(D_METHOD("step"), &BaseSolver::step);

		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dt", PROPERTY_HINT_RANGE, "1e-6,1,1e-6"), "set_dt", "get_dt");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "sub_steps", PROPERTY_HINT_RANGE, "1,100,1"), "set_sub_steps", "get_sub_steps");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "iterations", PROPERTY_HINT_RANGE, "1,500,1"), "set_iterations", "get_iterations");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "gravity"), "set_gravity", "get_gravity");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collision_enabled"), "set_collision_enabled", "is_collision_enabled");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "time"), "", "get_time");
	}

	real_t dt;
	Vector3 gravity;
	real_t time;
	int sub_steps;
	int iterations;
	bool collision_enabled;
	void *world; // optional pointer to Gaia simulation world
	HashMap<entity_id_t, Ref<BaseEntity>> entities;
};

} // namespace genesis

#endif // GENESIS_SOLVERS_BASE_SOLVER_H