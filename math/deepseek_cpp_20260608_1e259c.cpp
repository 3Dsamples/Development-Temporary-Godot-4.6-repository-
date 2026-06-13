// File 60: modules/genesis/src/materials/mpm_material.h
// MPM material with elastoplasticity, hardening, and damage models.
// Supports Drucker-Prager, von Mises, and custom yield surfaces.

#ifndef GENESIS_MATERIALS_MPM_MATERIAL_H
#define GENESIS_MATERIALS_MPM_MATERIAL_H

#include "material_base.h"
#include "../options/options_system.h"

namespace genesis {

class MPMMaterial : public GenesisMaterial {
	GDCLASS(MPMMaterial, GenesisMaterial);

public:
	enum YieldSurface {
		VON_MISES = 0,
		DRUCKER_PRAGER,
		MOHR_COULOMB,
		CAM_CLAY,
		NONE
	};

	MPMMaterial() :
		yield_surface(VON_MISES),
		cohesion(1e4),
		friction_angle(30.0),   // degrees
		dilatancy_angle(0.0),
		hardening_modulus(1e5),
		softening_modulus(0.0),
		damage_threshold(1.0),
		damage_rate(0.0),
		max_damage(0.99),
		tensile_strength(1e6),
		compressive_strength(1e7),
		flow_viscosity(0.0),
		use_implicit_yield(false) {}

	// --- Yield surface ---
	void set_yield_surface(YieldSurface p_surface) { yield_surface = p_surface; }
	YieldSurface get_yield_surface() const { return yield_surface; }

	// --- Strength parameters ---
	void set_cohesion(real_t p_c) { cohesion = MAX(p_c, 0.0); }
	real_t get_cohesion() const { return cohesion; }

	void set_friction_angle_degrees(real_t p_phi) { friction_angle = p_phi; }
	real_t get_friction_angle_degrees() const { return friction_angle; }
	real_t get_friction_angle_radians() const { return Math::deg_to_rad(friction_angle); }

	void set_dilatancy_angle_degrees(real_t p_psi) { dilatancy_angle = p_psi; }
	real_t get_dilatancy_angle_degrees() const { return dilatancy_angle; }

	void set_hardening_modulus(real_t p_H) { hardening_modulus = MAX(p_H, 0.0); }
	real_t get_hardening_modulus() const { return hardening_modulus; }

	void set_softening_modulus(real_t p_S) { softening_modulus = MAX(p_S, 0.0); }
	real_t get_softening_modulus() const { return softening_modulus; }

	// --- Damage ---
	void set_damage_threshold(real_t p_thresh) { damage_threshold = MAX(p_thresh, 0.0); }
	real_t get_damage_threshold() const { return damage_threshold; }

	void set_damage_rate(real_t p_rate) { damage_rate = MAX(p_rate, 0.0); }
	real_t get_damage_rate() const { return damage_rate; }

	void set_max_damage(real_t p_max) { max_damage = CLAMP(p_max, 0.0, 1.0); }
	real_t get_max_damage() const { return max_damage; }

	// --- Strength limits ---
	void set_tensile_strength(real_t p_sigma) { tensile_strength = MAX(p_sigma, 0.0); }
	real_t get_tensile_strength() const { return tensile_strength; }

	void set_compressive_strength(real_t p_sigma) { compressive_strength = MAX(p_sigma, 0.0); }
	real_t get_compressive_strength() const { return compressive_strength; }

	// --- Viscosity ---
	void set_flow_viscosity(real_t p_eta) { flow_viscosity = MAX(p_eta, 0.0); }
	real_t get_flow_viscosity() const { return flow_viscosity; }

	void set_use_implicit_yield(bool p_implicit) { use_implicit_yield = p_implicit; }
	bool is_implicit_yield() const { return use_implicit_yield; }

	// --- Solver options ---
	virtual void get_solver_options(genesis::options::Options &opts) const override {
		GenesisMaterial::get_solver_options(opts);
		opts.set("mpm.yield_surface", int(yield_surface));
		opts.set("mpm.cohesion", cohesion);
		opts.set("mpm.friction_angle", friction_angle);
		opts.set("mpm.dilatancy_angle", dilatancy_angle);
		opts.set("mpm.hardening_modulus", hardening_modulus);
		opts.set("mpm.softening_modulus", softening_modulus);
		opts.set("mpm.damage_threshold", damage_threshold);
		opts.set("mpm.damage_rate", damage_rate);
		opts.set("mpm.max_damage", max_damage);
		opts.set("mpm.tensile_strength", tensile_strength);
		opts.set("mpm.compressive_strength", compressive_strength);
		opts.set("mpm.flow_viscosity", flow_viscosity);
		opts.set("mpm.implicit_yield", use_implicit_yield);
	}

	// --- Duplicate ---
	virtual Ref<GenesisMaterial> duplicate(bool p_subresources = false) const override {
		Ref<MPMMaterial> mat = memnew(MPMMaterial);
		_copy_base(mat);
		return mat;
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_yield_surface", "surface"), &MPMMaterial::set_yield_surface);
		ClassDB::bind_method(D_METHOD("get_yield_surface"), &MPMMaterial::get_yield_surface);
		ClassDB::bind_method(D_METHOD("set_cohesion", "cohesion"), &MPMMaterial::set_cohesion);
		ClassDB::bind_method(D_METHOD("get_cohesion"), &MPMMaterial::get_cohesion);
		ClassDB::bind_method(D_METHOD("set_friction_angle_degrees", "phi"), &MPMMaterial::set_friction_angle_degrees);
		ClassDB::bind_method(D_METHOD("get_friction_angle_degrees"), &MPMMaterial::get_friction_angle_degrees);
		ClassDB::bind_method(D_METHOD("set_dilatancy_angle_degrees", "psi"), &MPMMaterial::set_dilatancy_angle_degrees);
		ClassDB::bind_method(D_METHOD("get_dilatancy_angle_degrees"), &MPMMaterial::get_dilatancy_angle_degrees);
		ClassDB::bind_method(D_METHOD("set_hardening_modulus", "H"), &MPMMaterial::set_hardening_modulus);
		ClassDB::bind_method(D_METHOD("get_hardening_modulus"), &MPMMaterial::get_hardening_modulus);
		ClassDB::bind_method(D_METHOD("set_softening_modulus", "S"), &MPMMaterial::set_softening_modulus);
		ClassDB::bind_method(D_METHOD("get_softening_modulus"), &MPMMaterial::get_softening_modulus);
		ClassDB::bind_method(D_METHOD("set_damage_threshold", "thresh"), &MPMMaterial::set_damage_threshold);
		ClassDB::bind_method(D_METHOD("get_damage_threshold"), &MPMMaterial::get_damage_threshold);
		ClassDB::bind_method(D_METHOD("set_damage_rate", "rate"), &MPMMaterial::set_damage_rate);
		ClassDB::bind_method(D_METHOD("get_damage_rate"), &MPMMaterial::get_damage_rate);
		ClassDB::bind_method(D_METHOD("set_max_damage", "max_damage"), &MPMMaterial::set_max_damage);
		ClassDB::bind_method(D_METHOD("get_max_damage"), &MPMMaterial::get_max_damage);
		ClassDB::bind_method(D_METHOD("set_tensile_strength", "sigma_t"), &MPMMaterial::set_tensile_strength);
		ClassDB::bind_method(D_METHOD("get_tensile_strength"), &MPMMaterial::get_tensile_strength);
		ClassDB::bind_method(D_METHOD("set_compressive_strength", "sigma_c"), &MPMMaterial::set_compressive_strength);
		ClassDB::bind_method(D_METHOD("get_compressive_strength"), &MPMMaterial::get_compressive_strength);
		ClassDB::bind_method(D_METHOD("set_flow_viscosity", "eta"), &MPMMaterial::set_flow_viscosity);
		ClassDB::bind_method(D_METHOD("get_flow_viscosity"), &MPMMaterial::get_flow_viscosity);
		ClassDB::bind_method(D_METHOD("set_use_implicit_yield", "implicit"), &MPMMaterial::set_use_implicit_yield);
		ClassDB::bind_method(D_METHOD("is_implicit_yield"), &MPMMaterial::is_implicit_yield);

		BIND_ENUM_CONSTANT(VON_MISES);
		BIND_ENUM_CONSTANT(DRUCKER_PRAGER);
		BIND_ENUM_CONSTANT(MOHR_COULOMB);
		BIND_ENUM_CONSTANT(CAM_CLAY);
		BIND_ENUM_CONSTANT(NONE);

		ADD_PROPERTY(PropertyInfo(Variant::INT, "yield_surface", PROPERTY_HINT_ENUM, "VonMises,DruckerPrager,MohrCoulomb,CamClay,None"), "set_yield_surface", "get_yield_surface");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cohesion", PROPERTY_HINT_RANGE, "0,1e9,0.1"), "set_cohesion", "get_cohesion");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "friction_angle", PROPERTY_HINT_RANGE, "0,89,0.1"), "set_friction_angle_degrees", "get_friction_angle_degrees");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dilatancy_angle", PROPERTY_HINT_RANGE, "0,89,0.1"), "set_dilatancy_angle_degrees", "get_dilatancy_angle_degrees");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "hardening_modulus", PROPERTY_HINT_RANGE, "0,1e9,0.1"), "set_hardening_modulus", "get_hardening_modulus");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "softening_modulus", PROPERTY_HINT_RANGE, "0,1e9,0.1"), "set_softening_modulus", "get_softening_modulus");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damage_threshold", PROPERTY_HINT_RANGE, "0,1e10,0.1"), "set_damage_threshold", "get_damage_threshold");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damage_rate", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_damage_rate", "get_damage_rate");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_damage", PROPERTY_HINT_RANGE, "0,0.99,0.01"), "set_max_damage", "get_max_damage");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "tensile_strength", PROPERTY_HINT_RANGE, "0,1e10,0.1"), "set_tensile_strength", "get_tensile_strength");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "compressive_strength", PROPERTY_HINT_RANGE, "0,1e10,0.1"), "set_compressive_strength", "get_compressive_strength");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "flow_viscosity", PROPERTY_HINT_RANGE, "0,1000,0.001"), "set_flow_viscosity", "get_flow_viscosity");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "implicit_yield"), "set_use_implicit_yield", "is_implicit_yield");
	}

private:
	void _copy_base(Ref<MPMMaterial> &mat) const {
		mat->set_density(get_density());
		mat->set_young_modulus(get_young_modulus());
		mat->set_poisson_ratio(get_poisson_ratio());
		mat->set_friction(get_friction());
		mat->set_restitution(get_restitution());
		mat->set_damping(get_damping());
		mat->set_material_type(get_material_type());
		mat->yield_surface = yield_surface;
		mat->cohesion = cohesion;
		mat->friction_angle = friction_angle;
		mat->dilatancy_angle = dilatancy_angle;
		mat->hardening_modulus = hardening_modulus;
		mat->softening_modulus = softening_modulus;
		mat->damage_threshold = damage_threshold;
		mat->damage_rate = damage_rate;
		mat->max_damage = max_damage;
		mat->tensile_strength = tensile_strength;
		mat->compressive_strength = compressive_strength;
		mat->flow_viscosity = flow_viscosity;
		mat->use_implicit_yield = use_implicit_yield;
	}

	YieldSurface yield_surface;
	real_t cohesion;
	real_t friction_angle;   // degrees
	real_t dilatancy_angle;
	real_t hardening_modulus;
	real_t softening_modulus;
	real_t damage_threshold;
	real_t damage_rate;
	real_t max_damage;
	real_t tensile_strength;
	real_t compressive_strength;
	real_t flow_viscosity;
	bool use_implicit_yield;
};

} // namespace genesis

#endif // GENESIS_MATERIALS_MPM_MATERIAL_H