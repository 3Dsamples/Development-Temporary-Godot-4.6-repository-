// File 155: modules/genesis/src/genesis_world.cpp
// Implements the GenesisWorld Node3D that runs the multi‑solver simulation
// and integrates with the Godot scene tree. Provides methods to add entities,
// solvers, sensors and advance physics step by step.

#include "genesis_world.h"

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
#include "options/options_system.h"
#include "core/core_string_names.h"

namespace genesis {

void GenesisWorld::_bind_methods() {
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

GenesisWorld::GenesisWorld() :
		simulation_time(0.0),
		physics_dt(1.0 / 60.0),
		sub_steps(1),
		gravity(0.0, -9.81, 0.0),
		options_path("res://genesis_options.json") {
	set_process(true);
	set_physics_process(true);
}

void GenesisWorld::_notification(int p_what) {
	if (p_what == NOTIFICATION_READY) {
		initialize_solvers();                      // create all default solver instances
		if (!options_path.is_empty()) {
			load_options_from_file();              // apply global options
		}
	}
	if (p_what == NOTIFICATION_PHYSICS_PROCESS) {
		real_t dt = get_physics_process_delta_time();
		simulate_step(dt);                         // run one or more physics substeps
	}
}

void GenesisWorld::set_physics_dt(real_t p_dt) { physics_dt = MAX(p_dt, 1e-6); }
real_t GenesisWorld::get_physics_dt() const { return physics_dt; }

void GenesisWorld::set_sub_steps(int p_sub) { sub_steps = MAX(p_sub, 1); }
int GenesisWorld::get_sub_steps() const { return sub_steps; }

void GenesisWorld::set_gravity(const Vector3 &p_g) { gravity = p_g; }
Vector3 GenesisWorld::get_gravity() const { return gravity; }

void GenesisWorld::set_options_path(const String &p_path) { options_path = p_path; }
String GenesisWorld::get_options_path() const { return options_path; }

void GenesisWorld::add_entity(const Ref<BaseEntity> &p_entity) {
	ERR_FAIL_COND(p_entity.is_null());
	entity_id_t uid = p_entity->get_entity_uid();
	entities[uid] = p_entity;
	assign_entity_to_solver(p_entity);           // route the entity to the correct solver
}

void GenesisWorld::remove_entity(entity_id_t p_uid) {
	if (!entities.has(p_uid)) return;
	Ref<BaseEntity> ent = entities[p_uid];
	entities.erase(p_uid);
	// Remove from the solver that owns it (each solver has its own remove method)
	SolverType type = ent->get_solver_type();
	BaseSolver *solver = get_solver_for_type(type);
	if (solver) solver->remove_entity(p_uid);
}

Ref<BaseEntity> GenesisWorld::get_entity(entity_id_t p_uid) const {
	HashMap<entity_id_t, Ref<BaseEntity>>::ConstIterator it = entities.find(p_uid);
	return it ? it->value : Ref<BaseEntity>();
}

void GenesisWorld::clear_entities() {
	entities.clear();
	rigid_solver->clear_entities();
	fem_solver->clear_entities();
	mpm_solver->clear_entities();
	sph_solver->clear_entities();
	pbd_solver->clear_entities();
	sf_solver->clear_entities();
	kinematic_solver->clear_entities();
}

void GenesisWorld::add_sensor(const Ref<BaseSensor> &p_sensor, entity_id_t p_attach_to) {
	ERR_FAIL_COND(p_sensor.is_null());
	p_sensor->set_entity_uid(p_attach_to);
	sensors.push_back(p_sensor);
}

Ref<SolverState> GenesisWorld::capture_state() const {
	Ref<SolverState> state; state.instantiate();
	state->set_time(simulation_time);
	for (const KeyValue<entity_id_t, Ref<BaseEntity>> &kv : entities) {
		Ref<EntityState> ent_state; ent_state.instantiate();
		ent_state->capture_from(kv.value);
		state->add_entity_state(kv.key, ent_state);
	}
	return state;
}

void GenesisWorld::restore_state(const Ref<SolverState> &p_state) {
	ERR_FAIL_COND(p_state.is_null());
	simulation_time = p_state->get_time();
	for (KeyValue<entity_id_t, Ref<BaseEntity>> &kv : entities) {
		Ref<EntityState> ent_state = p_state->get_entity_state(kv.key);
		if (ent_state.is_valid()) {
			ent_state->apply_to(kv.value);
		}
	}
}

void GenesisWorld::simulate_step(real_t dt) {
	real_t sub_dt = dt / real_t(sub_steps);
	for (int i = 0; i < sub_steps; ++i) {
		// Kinematic solver first (prescribed trajectories)
		kinematic_solver->set_dt(sub_dt);
		kinematic_solver->set_gravity(gravity);
		kinematic_solver->step();

		// Multi‑solver pipeline
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
		pbd_solver->step();                      // PBD step uses its own dt only for compliance/lambda

		sf_solver->set_gravity(gravity);
		sf_solver->step();

		// Update all sensors attached to entities
		update_sensors(sub_dt);

		simulation_time += sub_dt;
	}
}

void GenesisWorld::load_options_from_file() {
	options::Options opts;
	Error err = opts.load_from_json(options_path);
	if (err == OK) {
		load_options(opts);
	} else {
		WARN_PRINT(vformat("GenesisWorld: failed to load options from %s", options_path));
	}
}

void GenesisWorld::load_options(const options::Options &p_opts) {
	gravity = p_opts.get_vector3("gravity", gravity);
	sub_steps = p_opts.get_int("sub_steps", sub_steps);
	physics_dt = p_opts.get_real("dt", physics_dt);
	// Feed sub‑dictionaries to each solver
	rigid_solver->init_from_options(p_opts.get_dict("rigid_solver", Dictionary()));
	fem_solver->init_from_options(p_opts.get_dict("fem_solver", Dictionary()));
	mpm_solver->init_from_options(p_opts.get_dict("mpm_solver", Dictionary()));
	sph_solver->init_from_options(p_opts.get_dict("sph_solver", Dictionary()));
	pbd_solver->init_from_options(p_opts.get_dict("pbd_solver", Dictionary()));
	sf_solver->init_from_options(p_opts.get_dict("sf_solver", Dictionary()));
	kinematic_solver->init_from_options(p_opts.get_dict("kinematic_solver", Dictionary()));
}

void GenesisWorld::initialize_solvers() {
	rigid_solver.instantiate();
	fem_solver.instantiate();
	mpm_solver.instantiate();
	sph_solver.instantiate();
	pbd_solver.instantiate();
	sf_solver.instantiate();
	kinematic_solver.instantiate();
}

void GenesisWorld::assign_entity_to_solver(const Ref<BaseEntity> &p_entity) {
	SolverType type = p_entity->get_solver_type();
	BaseSolver *solver = get_solver_for_type(type);
	if (solver) solver->add_entity(p_entity);
}

BaseSolver *GenesisWorld::get_solver_for_type(SolverType p_type) {
	switch (p_type) {
		case SolverType::RIGID: return rigid_solver.ptr();
		case SolverType::FEM:   return fem_solver.ptr();
		case SolverType::MPM:   return mpm_solver.ptr();
		case SolverType::SPH:   return sph_solver.ptr();
		case SolverType::PBD:   return pbd_solver.ptr();
		case SolverType::SF:    return sf_solver.ptr();
		case SolverType::TOOL:  return kinematic_solver.ptr();   // tool uses kinematic solver
		default: return nullptr;
	}
}

void GenesisWorld::update_sensors(real_t dt) {
	for (Ref<BaseSensor> &sensor : sensors) {
		if (sensor.is_null() || !sensor->is_enabled()) continue;
		Ref<BaseEntity> entity = get_entity(sensor->get_entity_uid());
		if (entity.is_null()) continue;
		sensor->step(dt, entity);
	}
}

} // namespace genesis