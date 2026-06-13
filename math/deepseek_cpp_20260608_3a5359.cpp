// File 92: modules/genesis/src/genesis_configuration.h
// SimulationConfiguration – a Godot Resource that bundles all entities, solvers,
// materials, sensors, and options into a single asset for easy loading/saving.

#ifndef GENESIS_CONFIGURATION_H
#define GENESIS_CONFIGURATION_H

#include "core/io/resource.h"
#include "core/templates/local_vector.h"
#include "entities/base_entity.h"
#include "materials/material_base.h"
#include "sensors/base_sensor.h"
#include "options/options_system.h"
#include "core/genesis_types.h"

namespace genesis {

class SimulationConfiguration : public Resource {
	GDCLASS(SimulationConfiguration, Resource);

public:
	SimulationConfiguration() {}

	// --- Entities ---
	void add_entity(const Ref<BaseEntity> &p_entity) { entities.push_back(p_entity); }
	int get_entity_count() const { return entities.size(); }
	Ref<BaseEntity> get_entity(int p_idx) const { return entities[p_idx]; }
	void clear_entities() { entities.clear(); }

	// --- Materials (shared library) ---
	void add_material(const Ref<GenesisMaterial> &p_mat) {
		// Use name as key, overwriting existing
		materials[p_mat->get_material_type()] = p_mat;
	}
	Ref<GenesisMaterial> get_material(const String &p_name) const {
		HashMap<String, Ref<GenesisMaterial>>::ConstIterator it = materials.find(p_name);
		if (it) return it->value;
		return Ref<GenesisMaterial>();
	}
	HashMap<String, Ref<GenesisMaterial>> &get_materials() { return materials; }

	// --- Sensors ---
	void add_sensor(const Ref<BaseSensor> &p_sensor) { sensors.push_back(p_sensor); }
	LocalVector<Ref<BaseSensor>> &get_sensors() { return sensors; }

	// --- Global options (gravity, dt, etc.) ---
	void set_options(const options::Options &p_opts) { global_options = p_opts; }
	options::Options &get_options() { return global_options; }

	// Save / Load (JSON format) – uses options system for simplicity
	Error save_to_file(const String &p_path) {
		Dictionary dict;
		// Serialize entities (very basic: store their type and serialized properties)
		Array ent_arr;
		for (int i = 0; i < entities.size(); ++i) {
			Ref<BaseEntity> ent = entities[i];
			if (ent.is_null()) continue;
			Dictionary ed;
			ed["type"] = ent->get_class_name(); // Godot class name
			// Save all properties via _get_property_list (not implemented fully, rely on serializer)
			// Simplified: we only save basic common data
			ed["uid"] = ent->get_entity_uid();
			ed["transform"] = ent->get_transform();
			ed["active"] = ent->is_active();
			ent_arr.push_back(ed);
		}
		dict["entities"] = ent_arr;
		// Save materials similarly (omitted for brevity)
		// Save global options dictionary
		dict["options"] = global_options.get_data();
		// Write JSON
		Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
		ERR_FAIL_COND_V(f.is_null(), ERR_FILE_CANT_WRITE);
		JSON json;
		String text = json.stringify(dict, "\t");
		f->store_string(text);
		return OK;
	}

	Error load_from_file(const String &p_path) {
		options::Options opts;
		Error err = opts.load_from_json(p_path); // loads into dictionary
		if (err != OK) return err;
		// Parse entities etc. from the dictionary
		// Not fully implemented, but could reconstruct objects.
		return OK;
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("add_entity", "entity"), &SimulationConfiguration::add_entity);
		ClassDB::bind_method(D_METHOD("get_entity_count"), &SimulationConfiguration::get_entity_count);
		ClassDB::bind_method(D_METHOD("get_entity", "idx"), &SimulationConfiguration::get_entity);
		ClassDB::bind_method(D_METHOD("clear_entities"), &SimulationConfiguration::clear_entities);
		ClassDB::bind_method(D_METHOD("add_material", "material"), &SimulationConfiguration::add_material);
		ClassDB::bind_method(D_METHOD("get_material", "name"), &SimulationConfiguration::get_material);
		ClassDB::bind_method(D_METHOD("add_sensor", "sensor"), &SimulationConfiguration::add_sensor);
		ClassDB::bind_method(D_METHOD("save_to_file", "path"), &SimulationConfiguration::save_to_file);
		ClassDB::bind_method(D_METHOD("load_from_file", "path"), &SimulationConfiguration::load_from_file);
	}

private:
	LocalVector<Ref<BaseEntity>> entities;
	HashMap<String, Ref<GenesisMaterial>> materials;
	LocalVector<Ref<BaseSensor>> sensors;
	options::Options global_options;
};

} // namespace genesis

#endif // GENESIS_CONFIGURATION_H