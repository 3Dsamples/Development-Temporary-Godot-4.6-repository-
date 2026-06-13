// File 83: modules/genesis/src/states/solver_state.h
// Solver state – stores the complete state of a solver for checkpointing.
// Includes entity states and solver-specific parameters.

#ifndef GENESIS_STATES_SOLVER_STATE_H
#define GENESIS_STATES_SOLVER_STATE_H

#include "core/io/resource.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "entity_state.h"
#include "../core/genesis_types.h"

namespace genesis {

/**
 * A snapshot of the entire solver at a given time.
 * Contains a collection of entity states and solver metadata.
 */
class SolverState : public Resource {
	GDCLASS(SolverState, Resource);

public:
	SolverState() : time(0.0), solver_type(SolverType::CUSTOM) {}

	// --- Time ---
	void set_time(real_t p_t) { time = p_t; }
	real_t get_time() const { return time; }

	// --- Solver identification ---
	void set_solver_type(SolverType p_type) { solver_type = p_type; }
	SolverType get_solver_type() const { return solver_type; }

	// --- Entity states ---
	void add_entity_state(entity_id_t p_uid, const Ref<EntityState> &p_state) {
		entity_states[p_uid] = p_state;
	}
	Ref<EntityState> get_entity_state(entity_id_t p_uid) const {
		HashMap<entity_id_t, Ref<EntityState>>::ConstIterator it = entity_states.find(p_uid);
		if (it) return it->value;
		return Ref<EntityState>();
	}
	HashMap<entity_id_t, Ref<EntityState>> &get_entity_states() { return entity_states; }
	void clear_entity_states() { entity_states.clear(); }

	// --- Solver-specific parameters (stored as Variant dictionary) ---
	void set_parameters(const Dictionary &p_params) { parameters = p_params; }
	Dictionary get_parameters() const { return parameters; }

	// Serialisation helpers (JSON/binary via Godot's ResourceSaver)
	virtual void _get_property_list(List<PropertyInfo> *p_list) const override {
		Resource::_get_property_list(p_list);
	}
	virtual bool _get(const StringName &p_name, Variant &r_ret) const override {
		if (p_name == "time") { r_ret = time; return true; }
		if (p_name == "solver_type") { r_ret = int(solver_type); return true; }
		if (p_name == "entity_states") {
			Array arr;
			for (const KeyValue<entity_id_t, Ref<EntityState>> &kv : entity_states) {
				Dictionary entry;
				entry["uid"] = kv.key;
				entry["state"] = kv.value;
				arr.push_back(entry);
			}
			r_ret = arr;
			return true;
		}
		if (p_name == "parameters") { r_ret = parameters; return true; }
		return false;
	}
	virtual bool _set(const StringName &p_name, const Variant &p_value) override {
		if (p_name == "time") { time = p_value; return true; }
		if (p_name == "solver_type") { solver_type = SolverType(int(p_value)); return true; }
		if (p_name == "entity_states") {
			entity_states.clear();
			Array arr = p_value;
			for (int i = 0; i < arr.size(); ++i) {
				Dictionary entry = arr[i];
				entity_id_t uid = entry["uid"];
				Ref<EntityState> state = entry["state"];
				if (state.is_valid()) entity_states[uid] = state;
			}
			return true;
		}
		if (p_name == "parameters") { parameters = p_value; return true; }
		return false;
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_time", "t"), &SolverState::set_time);
		ClassDB::bind_method(D_METHOD("get_time"), &SolverState::get_time);
		ClassDB::bind_method(D_METHOD("set_solver_type", "type"), &SolverState::set_solver_type);
		ClassDB::bind_method(D_METHOD("get_solver_type"), &SolverState::get_solver_type);
		ClassDB::bind_method(D_METHOD("add_entity_state", "uid", "state"), &SolverState::add_entity_state);
		ClassDB::bind_method(D_METHOD("get_entity_state", "uid"), &SolverState::get_entity_state);
		ClassDB::bind_method(D_METHOD("clear_entity_states"), &SolverState::clear_entity_states);
		ClassDB::bind_method(D_METHOD("set_parameters", "params"), &SolverState::set_parameters);
		ClassDB::bind_method(D_METHOD("get_parameters"), &SolverState::get_parameters);

		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "time"), "set_time", "get_time");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "solver_type"), "set_solver_type", "get_solver_type");
		ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "parameters"), "set_parameters", "get_parameters");
	}

private:
	real_t time;
	SolverType solver_type;
	HashMap<entity_id_t, Ref<EntityState>> entity_states;
	Dictionary parameters;
};

} // namespace genesis

#endif // GENESIS_STATES_SOLVER_STATE_H