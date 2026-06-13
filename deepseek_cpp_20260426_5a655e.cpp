// File 90: modules/genesis/src/genesis_world.h
// GenesisWorld – the main Godot Node3D that owns and manages a Genesis
// multi-physics simulation. It creates solvers, entities, sensors, and
// drives stepping within _physics_process().

#ifndef GENESIS_WORLD_H
#define GENESIS_WORLD_H

#include "core/os/threaded_array_processor.h"
#include "scene/3d/node_3d.h"

#include "core/genesis_types.h"
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
#include "states/entity_state.h"
#include "states/solver_state.h"
#include "collision/collider.h"

namespace genesis {

class GenesisWorld : public Node3D {
	GDCLASS(GenesisWorld, Node3D);

public:
	GenesisWorld() :
		simulation_time(0.0),
		physics_dt(1.0 / 60.0),
		sub_steps(1),
		gravity(0, -9.81, 0),
		options_path("res://genesis_options.json") {
		set_process(true);
		set_physics_process(true);
	}

	// --- Configuration ---
	void set_physics_dt(real_t p_dt) { physics_dt = MAX(p_dt, 1e-6); }
	real_t get_physics_dt() const { return physics_dt; }

	void set_sub_steps(int p_sub) { sub_steps = MAX(p_sub, 1); }
	int get_sub_steps() const { return sub_steps; }

	void set_gravity(const Vector3 &p_g) { gravity = p_g; }
	Vector3 get_gravity() const { return gravity; }

	void set_options_path(const String &p_path) { options_path = p_path; }
	String get_options_path() const { return options_path; }

	// --- Entity management ---
	void add_entity(const Ref<BaseEntity> &p_entity) {
		ERR_FAIL_COND(p_entity.is_null());
		entity_id_t uid = p_entity->get_entity_uid();
		entities[uid] = p_entity;
		// Optionally add to the appropriate solver based on entity type
		assign_entity_to_solver(p_entity);
	}

	void remove_entity(entity_id_t p_uid) {
		entities.erase(p_uid);
		// remove from solver's internal lists – solvers will need a remove method (omitted for brevity)
	}

	Ref<BaseEntity> get_entity(entity_id_t p_uid) const {
		HashMap<entity_id_t, Ref<BaseEntity>>::ConstIterator it = entities.find(p_uid);
		if (it) return it->value;
		return Ref<BaseEntity>();
	}

	void clear_entities() {
		entities.clear();
		// clear solvers
		rigid_solver->clear_entities();
		fem_solver->clear_entities();
		mpm_solver->clear_entities();
		sph_solver->clear_entities();
		pbd_solver->clear_entities();
		sf_solver->clear_entities();
		kinematic_solver->clear_entities();
	}

	// --- Sensor attachment ---
	void add_sensor(const Ref<BaseSensor> &p_sensor, entity_id_t p_attach_to) {
		ERR_FAIL_COND(p_sensor.is_null());
		p_sensor->set_entity_uid(p_attach_to);
		sensors.push_back(p_sensor);
	}

	// --- Save / load simulation state ---
	Ref<SolverState> capture_state() const {
		Ref<SolverState> state = memnew(SolverState);
		state->set_time(simulation_time);
		// capture entities states
		for (const KeyValue<entity_id_t, Ref<BaseEntity>> &kv : entities) {
			Ref<EntityState> ent_state = memnew(EntityState);
			ent_state->capture_from(kv.value);
			state->add_entity_state(kv.key, ent_state);
		}
		return state;
	}

	void restore_state(const Ref<SolverState> &p_state) {
		ERR_FAIL_COND(p_state.is_null());
		simulation_time = p_state->get_time();
		for (KeyValue<entity_id_t, Ref<BaseEntity>> &kv : entities) {
			Ref<EntityState> ent_state = p_state->get_entity_state(kv.key);
			if (ent_state.is_valid()) {
				ent_state->apply_to(kv.value);
			}
		}
	}

	// --- Main simulation interface (called from physics process) ---
	void simulate_step(real_t dt) {
		real_t sub_dt = dt / real_t(sub_steps);
		for (int i = 0; i < sub_steps; ++i) {
			// Kinematic entities first
			kinematic_solver->set_dt(sub_dt);
			kinematic_solver->step();

			// Run all physics solvers
			rigid_solver->set_gravity(gravity);
			rigid_solver->set_dt(sub_dt);
			rigid_solver->step();

			fem_solver->set_gravity(gravity);
			fem_solver->set_dt(sub_dt);
			fem_solver->step();

			mpm_solver->set_gravity(gravity);
			mpm_solver->set_dt(sub_dt);
			mpm_solver->step();

			sph_solver->set_gravity(gravity);
			sph_solver->set_dt(sub_dt);
			sph_solver->step();

			pbd_solver->set_gravity(gravity);
			pbd_solver->step();

			sf_solver->set_gravity(gravity);
			sf_solver->step();

			// Update sensors (every substep, but they will respect their internal update rate)
			update_sensors(sub_dt);

			simulation_time += sub_dt;
		}
	}

	// --- Option loading ---
	void load_options_from_file() {
		options::Options opts;
		Error err = opts.load_from_json(options_path);
		if (err == OK) {
			load_options(opts);
		} else {
			WARN_PRINT("GenesisWorld: failed to load options from " + options_path);
		}
	}

	void load_options(const options::Options &p_opts) {
		gravity = p_opts.get_vector3("gravity", gravity);
		sub_steps = p_opts.get_int("sub_steps", sub_steps);
		physics_dt = p_opts.get_real("dt", physics_dt);
		// Pass to solvers
		rigid_solver->init_from_options(p_opts.get_dict("rigid_solver", Dictionary()));
		fem_solver->init_from_options(p_opts.get_dict("fem_solver", Dictionary()));
		mpm_solver->init_from_options(p_opts.get_dict("mpm_solver", Dictionary()));
		sph_solver->init_from_options(p_opts.get_dict("sph_solver", Dictionary()));
		pbd_solver->init_from_options(p_opts.get_dict("pbd_solver", Dictionary()));
		sf_solver->init_from_options(p_opts.get_dict("sf_solver", Dictionary()));
		kinematic_solver->init_from_options(p_opts.get_dict("kinematic_solver", Dictionary()));
	}

protected:
	void _notification(int p_what) {
		if (p_what == NOTIFICATION_READY) {
			initialize_solvers();
			if (!options_path.is_empty()) {
				load_options_from_file();
			}
		}
		if (p_what == NOTIFICATION_PHYSICS_PROCESS) {
			real_t dt = get_physics_process_delta_time();
			simulate_step(dt);
		}
	}

	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_physics_dt", "dt"), &GenesisWorld::set_physics_dt);
		ClassDB::bind_method(D_METHOD("get_physics_dt"), &GenesisWorld::get_physics_dt);
		ClassDB::bind_method(D_METHOD("set_sub_steps", "sub_steps"), &GenesisWorld::set_sub_steps);
		ClassDB::bind_method(D_METHOD("get_sub_steps"), &GenesisWorld::get_sub_steps);
		ClassDB::bind_method(D_METHOD("set_gravity", "gravity"), &GenesisWorld::set_gravity);
		ClassDB::bind_method(D_METHOD("get_gravity"), &GenesisWorld::get_gravity);
		ClassDB::bind_method(D_METHOD("set_options_path", "path"), &GenesisWorld::set_options_path);
		ClassDB::bind_method(D_METHOD("get_options_path"), &GenesisWorld::get_options_path);

		ClassDB::bind_method(D_METHOD("add_entity", "entity"), &GenesisWorld::add_entity);
		ClassDB::bind_method(D_METHOD("remove_entity", "uid"), &GenesisWorld::remove_entity);
		ClassDB::bind_method(D_METHOD("get_entity", "uid"), &GenesisWorld::get_entity);
		ClassDB::bind_method(D_METHOD("clear_entities"), &GenesisWorld::clear_entities);
		ClassDB::bind_method(D_METHOD("add_sensor", "sensor", "attach_to_uid"), &GenesisWorld::add_sensor);

		ClassDB::bind_method(D_METHOD("capture_state"), &GenesisWorld::capture_state);
		ClassDB::bind_method(D_METHOD("restore_state", "state"), &GenesisWorld::restore_state);
		ClassDB::bind_method(D_METHOD("simulate_step", "dt"), &GenesisWorld::simulate_step);
		ClassDB::bind_method(D_METHOD("load_options_from_file"), &GenesisWorld::load_options_from_file);

		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "physics_dt", PROPERTY_HINT_RANGE, "0.0001,1,0.0001"), "set_physics_dt", "get_physics_dt");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "sub_steps", PROPERTY_HINT_RANGE, "1,100,1"), "set_sub_steps", "get_sub_steps");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "gravity"), "set_gravity", "get_gravity");
		ADD_PROPERTY(PropertyInfo(Variant::STRING, "options_path", PROPERTY_HINT_FILE, "*.json"), "set_options_path", "get_options_path");
	}

private:
	void initialize_solvers() {
		rigid_solver.instantiate();
		fem_solver.instantiate();
		mpm_solver.instantiate();
		sph_solver.instantiate();
		pbd_solver.instantiate();
		sf_solver.instantiate();
		kinematic_solver.instantiate();
	}

	void assign_entity_to_solver(const Ref<BaseEntity> &p_entity) {
		SolverType type = p_entity->get_solver_type();
		BaseSolver *solver = nullptr;
		switch (type) {
			case SolverType::RIGID: solver = rigid_solver.ptr(); break;
			case SolverType::FEM:   solver = fem_solver.ptr(); break;
			case SolverType::MPM:   solver = mpm_solver.ptr(); break;
			case SolverType::SPH:   solver = sph_solver.ptr(); break;
			case SolverType::PBD:   solver = pbd_solver.ptr(); break;
			case SolverType::SF:    solver = sf_solver.ptr(); break;
			case SolverType::TOOL:  solver = kinematic_solver.ptr(); break; // tool uses kinematic
			default: break;
		}
		if (solver) {
			solver->add_entity(p_entity);
		}
	}

	void update_sensors(real_t dt) {
		for (Ref<BaseSensor> &sensor : sensors) {
			if (sensor.is_null() || !sensor->is_enabled()) continue;
			Ref<BaseEntity> entity = get_entity(sensor->get_entity_uid());
			if (entity.is_null()) continue;
			sensor->step(dt, entity);
		}
	}

	// Simulation state
	real_t simulation_time;
	real_t physics_dt;
	int sub_steps;
	Vector3 gravity;
	String options_path;

	// Entities and sensors
	HashMap<entity_id_t, Ref<BaseEntity>> entities;
	LocalVector<Ref<BaseSensor>> sensors;

	// Solvers (all as Ref for safety)
	Ref<RigidSolver> rigid_solver;
	Ref<FEMSolver> fem_solver;
	Ref<MPMSolver> mpm_solver;
	Ref<SPHSolver> sph_solver;
	Ref<GenesisPBDSolver> pbd_solver;
	Ref<SFSolver> sf_solver;
	Ref<KinematicSolver> kinematic_solver;
};

} // namespace genesis

#endif // GENESIS_WORLD_H