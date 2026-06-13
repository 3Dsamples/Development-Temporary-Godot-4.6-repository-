// File 61: modules/genesis/src/materials/pbd_material.h
// PBD material with compliance, damping, and collision stiffness parameters.

#ifndef GENESIS_MATERIALS_PBD_MATERIAL_H
#define GENESIS_MATERIALS_PBD_MATERIAL_H

#include "material_base.h"
#include "../options/options_system.h"

namespace genesis {

class PBDMaterial : public GenesisMaterial {
	GDCLASS(PBDMaterial, GenesisMaterial);

public:
	enum StiffnessModel {
		CONSTANT_COMPLIANCE = 0,
		YOUNG_MODULUS_BASED,
		BENDING_MODULUS
	};

	PBDMaterial() :
		compliance(1e-7),
		damping_compliance(0.0),
		bending_compliance(1e-5),
		volume_compliance(1e-6),
		collision_compliance(1e-9),
		friction_coefficient(0.5),
		restitution_coefficient(0.0),
		stiffness_model(YOUNG_MODULUS_BASED),
		bend_stiffness_scale(0.01) {}

	// --- Compliance / stiffness ---
	void set_compliance(real_t p_c) { compliance = MAX(p_c, 0.0); }
	real_t get_compliance() const { return compliance; }

	void set_damping_compliance(real_t p_c) { damping_compliance = MAX(p_c, 0.0); }
	real_t get_damping_compliance() const { return damping_compliance; }

	void set_bending_compliance(real_t p_c) { bending_compliance = MAX(p_c, 0.0); }
	real_t get_bending_compliance() const { return bending_compliance; }

	void set_volume_compliance(real_t p_c) { volume_compliance = MAX(p_c, 0.0); }
	real_t get_volume_compliance() const { return volume_compliance; }

	void set_collision_compliance(real_t p_c) { collision_compliance = MAX(p_c, 0.0); }
	real_t get_collision_compliance() const { return collision_compliance; }

	// --- Friction / restitution ---
	void set_friction_coefficient(real_t p_f) { friction_coefficient = CLAMP(p_f, 0.0, 1.0); }
	real_t get_friction_coefficient() const { return friction_coefficient; }

	void set_restitution_coefficient(real_t p_r) { restitution_coefficient = CLAMP(p_r, 0.0, 1.0); }
	real_t get_restitution_coefficient() const { return restitution_coefficient; }

	// --- Stiffness model ---
	void set_stiffness_model(StiffnessModel p_model) { stiffness_model = p_model; }
	StiffnessModel get_stiffness_model() const { return stiffness_model; }

	void set_bend_stiffness_scale(real_t p_scale) { bend_stiffness_scale = MAX(p_scale, 0.0); }
	real_t get_bend_stiffness_scale() const { return bend_stiffness_scale; }

	// --- Compute compliance from Young's modulus (if using that model) ---
	real_t compute_geometric_stiffness(real_t rest_length, real_t cross_area = 1.0) const {
		// Simple linear spring: k = E * A / L
		return get_young_modulus() * cross_area / rest_length;
	}

	// --- Solver options ---
	virtual void get_solver_options(genesis::options::Options &opts) const override {
		GenesisMaterial::get_solver_options(opts);
		opts.set("pbd.compliance", compliance);
		opts.set("pbd.damping_compliance", damping_compliance);
		opts.set("pbd.bending_compliance", bending_compliance);
		opts.set("pbd.volume_compliance", volume_compliance);
		opts.set("pbd.collision_compliance", collision_compliance);
		opts.set("pbd.friction", friction_coefficient);
		opts.set("pbd.restitution", restitution_coefficient);
		opts.set("pbd.stiffness_model", int(stiffness_model));
		opts.set("pbd.bend_scale", bend_stiffness_scale);
	}

	virtual Ref<GenesisMaterial> duplicate(bool p_subresources = false) const override {
		Ref<PBDMaterial> mat = memnew(PBDMaterial);
		mat->set_density(get_density());
		mat->set_young_modulus(get_young_modulus());
		mat->set_poisson_ratio(get_poisson_ratio());
		mat->set_friction(get_friction());
		mat->set_restitution(get_restitution());
		mat->set_damping(get_damping());
		mat->set_material_type(get_material_type());
		mat->compliance = compliance;
		mat->damping_compliance = damping_compliance;
		mat->bending_compliance = bending_compliance;
		mat->volume_compliance = volume_compliance;
		mat->collision_compliance = collision_compliance;
		mat->friction_coefficient = friction_coefficient;
		mat->restitution_coefficient = restitution_coefficient;
		mat->stiffness_model = stiffness_model;
		mat->bend_stiffness_scale = bend_stiffness_scale;
		return mat;
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_compliance", "c"), &PBDMaterial::set_compliance);
		ClassDB::bind_method(D_METHOD("get_compliance"), &PBDMaterial::get_compliance);
		ClassDB::bind_method(D_METHOD("set_damping_compliance", "c"), &PBDMaterial::set_damping_compliance);
		ClassDB::bind_method(D_METHOD("get_damping_compliance"), &PBDMaterial::get_damping_compliance);
		ClassDB::bind_method(D_METHOD("set_bending_compliance", "c"), &PBDMaterial::set_bending_compliance);
		ClassDB::bind_method(D_METHOD("get_bending_compliance"), &PBDMaterial::get_bending_compliance);
		ClassDB::bind_method(D_METHOD("set_volume_compliance", "c"), &PBDMaterial::set_volume_compliance);
		ClassDB::bind_method(D_METHOD("get_volume_compliance"), &PBDMaterial::get_volume_compliance);
		ClassDB::bind_method(D_METHOD("set_collision_compliance", "c"), &PBDMaterial::set_collision_compliance);
		ClassDB::bind_method(D_METHOD("get_collision_compliance"), &PBDMaterial::get_collision_compliance);
		ClassDB::bind_method(D_METHOD("set_friction_coefficient", "f"), &PBDMaterial::set_friction_coefficient);
		ClassDB::bind_method(D_METHOD("get_friction_coefficient"), &PBDMaterial::get_friction_coefficient);
		ClassDB::bind_method(D_METHOD("set_restitution_coefficient", "r"), &PBDMaterial::set_restitution_coefficient);
		ClassDB::bind_method(D_METHOD("get_restitution_coefficient"), &PBDMaterial::get_restitution_coefficient);
		ClassDB::bind_method(D_METHOD("set_stiffness_model", "model"), &PBDMaterial::set_stiffness_model);
		ClassDB::bind_method(D_METHOD("get_stiffness_model"), &PBDMaterial::get_stiffness_model);
		ClassDB::bind_method(D_METHOD("set_bend_stiffness_scale", "scale"), &PBDMaterial::set_bend_stiffness_scale);
		ClassDB::bind_method(D_METHOD("get_bend_stiffness_scale"), &PBDMaterial::get_bend_stiffness_scale);

		BIND_ENUM_CONSTANT(CONSTANT_COMPLIANCE);
		BIND_ENUM_CONSTANT(YOUNG_MODULUS_BASED);
		BIND_ENUM_CONSTANT(BENDING_MODULUS);

		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "compliance", PROPERTY_HINT_RANGE, "0,1,1e-12"), "set_compliance", "get_compliance");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damping_compliance", PROPERTY_HINT_RANGE, "0,1,1e-12"), "set_damping_compliance", "get_damping_compliance");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bending_compliance", PROPERTY_HINT_RANGE, "0,1,1e-12"), "set_bending_compliance", "get_bending_compliance");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "volume_compliance", PROPERTY_HINT_RANGE, "0,1,1e-12"), "set_volume_compliance", "get_volume_compliance");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "collision_compliance", PROPERTY_HINT_RANGE, "0,1,1e-12"), "set_collision_compliance", "get_collision_compliance");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "friction_coefficient", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_friction_coefficient", "get_friction_coefficient");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "restitution_coefficient", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_restitution_coefficient", "get_restitution_coefficient");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "stiffness_model", PROPERTY_HINT_ENUM, "ConstantCompliance,YoungModulusBased,BendingModulus"), "set_stiffness_model", "get_stiffness_model");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bend_stiffness_scale", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_bend_stiffness_scale", "get_bend_stiffness_scale");
	}

private:
	real_t compliance;
	real_t damping_compliance;
	real_t bending_compliance;
	real_t volume_compliance;
	real_t collision_compliance;
	real_t friction_coefficient;
	real_t restitution_coefficient;
	StiffnessModel stiffness_model;
	real_t bend_stiffness_scale;
};

} // namespace genesis

#endif // GENESIS_MATERIALS_PBD_MATERIAL_H