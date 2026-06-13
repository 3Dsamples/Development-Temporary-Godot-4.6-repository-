// File 54: modules/genesis/src/materials/material_base.h
// Base material class for all Genesis solver materials. Derived from Godot Resource
// for serialisation and editor integration.

#ifndef GENESIS_MATERIALS_MATERIAL_BASE_H
#define GENESIS_MATERIALS_MATERIAL_BASE_H

#include "core/io/resource.h"
#include "core/variant/variant.h"
#include "core/math/vector3.h"
#include "../core/genesis_types.h"
#include "../options/options_system.h"

namespace genesis {

/**
 * Abstract base for all physics materials (FEM, MPM, PBD, SPH, SF).
 * Stores common physical properties and provides a virtual interface
 * to retrieve solver‑specific parameters.
 */
class GenesisMaterial : public Resource {
	GDCLASS(GenesisMaterial, Resource);

public:
	GenesisMaterial() :
		density(1000.0),
		young_modulus(1e6),
		poisson_ratio(0.3),
		friction(0.5),
		restitution(0.0),
		damping(0.001),
		material_type("generic") {}

	// --- Common properties ---
	void set_density(real_t p_val) { density = MAX(p_val, 0.0); }
	real_t get_density() const { return density; }

	void set_young_modulus(real_t p_val) { young_modulus = MAX(p_val, 0.0); }
	real_t get_young_modulus() const { return young_modulus; }

	void set_poisson_ratio(real_t p_val) { poisson_ratio = CLAMP(p_val, 0.0, 0.49); }
	real_t get_poisson_ratio() const { return poisson_ratio; }

	void set_friction(real_t p_val) { friction = CLAMP(p_val, 0.0, 1.0); }
	real_t get_friction() const { return friction; }

	void set_restitution(real_t p_val) { restitution = CLAMP(p_val, 0.0, 1.0); }
	real_t get_restitution() const { return restitution; }

	void set_damping(real_t p_val) { damping = MAX(p_val, 0.0); }
	real_t get_damping() const { return damping; }

	void set_material_type(const String &p_type) { material_type = p_type; }
	String get_material_type() const { return material_type; }

	// --- Lame parameters (used by FEM, MPM) ---
	real_t get_lame_lambda() const {
		real_t nu = poisson_ratio;
		return young_modulus * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
	}
	real_t get_lame_mu() const {
		return young_modulus / (2.0 * (1.0 + poisson_ratio));
	}

	// --- Bulk modulus (SPH, SF) ---
	real_t get_bulk_modulus() const {
		return young_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio));
	}

	// --- Virtual method to return a sub‑Options of solver‑specific parameters ---
	// Derived classes override this to expose their unique settings.
	virtual void get_solver_options(genesis::options::Options &opts) const {
		opts.set("density", density);
		opts.set("young_modulus", young_modulus);
		opts.set("poisson_ratio", poisson_ratio);
	}

	// --- Clone (required for Resource) ---
	virtual Ref<GenesisMaterial> duplicate(bool p_subresources = false) const {
		Ref<GenesisMaterial> mat = memnew(GenesisMaterial);
		mat->density = density;
		mat->young_modulus = young_modulus;
		mat->poisson_ratio = poisson_ratio;
		mat->friction = friction;
		mat->restitution = restitution;
		mat->damping = damping;
		mat->material_type = material_type;
		return mat;
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_density", "density"), &GenesisMaterial::set_density);
		ClassDB::bind_method(D_METHOD("get_density"), &GenesisMaterial::get_density);
		ClassDB::bind_method(D_METHOD("set_young_modulus", "young_modulus"), &GenesisMaterial::set_young_modulus);
		ClassDB::bind_method(D_METHOD("get_young_modulus"), &GenesisMaterial::get_young_modulus);
		ClassDB::bind_method(D_METHOD("set_poisson_ratio", "poisson_ratio"), &GenesisMaterial::set_poisson_ratio);
		ClassDB::bind_method(D_METHOD("get_poisson_ratio"), &GenesisMaterial::get_poisson_ratio);
		ClassDB::bind_method(D_METHOD("set_friction", "friction"), &GenesisMaterial::set_friction);
		ClassDB::bind_method(D_METHOD("get_friction"), &GenesisMaterial::get_friction);
		ClassDB::bind_method(D_METHOD("set_restitution", "restitution"), &GenesisMaterial::set_restitution);
		ClassDB::bind_method(D_METHOD("get_restitution"), &GenesisMaterial::get_restitution);
		ClassDB::bind_method(D_METHOD("set_damping", "damping"), &GenesisMaterial::set_damping);
		ClassDB::bind_method(D_METHOD("get_damping"), &GenesisMaterial::get_damping);
		ClassDB::bind_method(D_METHOD("set_material_type", "type"), &GenesisMaterial::set_material_type);
		ClassDB::bind_method(D_METHOD("get_material_type"), &GenesisMaterial::get_material_type);
		ClassDB::bind_method(D_METHOD("get_lame_lambda"), &GenesisMaterial::get_lame_lambda);
		ClassDB::bind_method(D_METHOD("get_lame_mu"), &GenesisMaterial::get_lame_mu);
		ClassDB::bind_method(D_METHOD("get_bulk_modulus"), &GenesisMaterial::get_bulk_modulus);

		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "density", PROPERTY_HINT_RANGE, "0,100000,0.1"), "set_density", "get_density");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "young_modulus", PROPERTY_HINT_RANGE, "0,1e12,1"), "set_young_modulus", "get_young_modulus");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "poisson_ratio", PROPERTY_HINT_RANGE, "0,0.49,0.01"), "set_poisson_ratio", "get_poisson_ratio");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "friction", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_friction", "get_friction");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "restitution", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_restitution", "get_restitution");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damping", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_damping", "get_damping");
		ADD_PROPERTY(PropertyInfo(Variant::STRING, "material_type"), "set_material_type", "get_material_type");
	}

private:
	real_t density;
	real_t young_modulus;
	real_t poisson_ratio;
	real_t friction;
	real_t restitution;
	real_t damping;
	String material_type;
};

} // namespace genesis

#endif // GENESIS_MATERIALS_MATERIAL_BASE_H