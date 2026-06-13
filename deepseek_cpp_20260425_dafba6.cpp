// File 55: modules/genesis/src/materials/fem_material.h
// FEM material with hyperelastic constitutive models and plasticity.

#ifndef GENESIS_MATERIALS_FEM_MATERIAL_H
#define GENESIS_MATERIALS_FEM_MATERIAL_H

#include "material_base.h"
#include "../options/options_system.h"

namespace genesis {

/**
 * A material for FEM (Finite Element Method) simulations.
 * Supports several constitutive models (St. Venant-Kirchhoff, Neo-Hookean,
 * linear elasticity) with optional von Mises plasticity.
 */
class FEMMaterial : public GenesisMaterial {
	GDCLASS(FEMMaterial, GenesisMaterial);

public:
	enum ConstitutiveModel {
		ST_VENANT_KIRCHHOFF = 0,
		NEO_HOOKEAN,
		LINEAR_ELASTIC
	};

	FEMMaterial() :
		model(NEO_HOOKEAN),
		plasticity_enabled(false),
		yield_stress(1e6),
		hardening(0.1),
		max_plastic_strain(0.05),
		creep_rate(0.0),
		thermal_expansion(0.0) {}

	// --- Constitutive model ---
	void set_constitutive_model(ConstitutiveModel p_model) { model = p_model; }
	ConstitutiveModel get_constitutive_model() const { return model; }

	// --- Plasticity ---
	void set_plasticity_enabled(bool p_enabled) { plasticity_enabled = p_enabled; }
	bool is_plasticity_enabled() const { return plasticity_enabled; }

	void set_yield_stress(real_t p_val) { yield_stress = MAX(p_val, 0.0); }
	real_t get_yield_stress() const { return yield_stress; }

	void set_hardening(real_t p_val) { hardening = MAX(p_val, 0.0); }
	real_t get_hardening() const { return hardening; }

	void set_max_plastic_strain(real_t p_val) { max_plastic_strain = MAX(p_val, 0.0); }
	real_t get_max_plastic_strain() const { return max_plastic_strain; }

	void set_creep_rate(real_t p_val) { creep_rate = MAX(p_val, 0.0); }
	real_t get_creep_rate() const { return creep_rate; }

	void set_thermal_expansion(real_t p_val) { thermal_expansion = p_val; }
	real_t get_thermal_expansion() const { return thermal_expansion; }

	// --- Compute first Piola-Kirchhoff stress for a given deformation gradient F ---
	// F is a 3x3 matrix stored as Basis (columns are basis vectors).
	Basis compute_pk1_stress(const Basis &F) const {
		real_t mu = get_lame_mu();
		real_t lambda = get_lame_lambda();
		Basis P; // zero

		switch (model) {
			case ST_VENANT_KIRCHHOFF: {
				// StVK: P = F * (2 mu E + lambda tr(E) I)
				// E = 0.5 * (F^T F - I)
				Basis E = 0.5 * (F.transposed() * F - Basis());
				real_t trace = E[0][0] + E[1][1] + E[2][2];
				Basis S = 2.0 * mu * E;
				// add lambda tr(E) to diagonal of S
				for (int i = 0; i < 3; ++i) S[i][i] += lambda * trace;
				P = F * S;
			} break;

			case NEO_HOOKEAN: {
				// Neo-Hookean: P = mu * (F - F^-T) + lambda * log(J) F^-T
				real_t J = F.determinant();
				ERR_FAIL_COND_V_MSG(J <= 0, Basis(), "F has zero or negative determinant (inverted element)");
				Basis F_inv_T = F.inverse().transposed();
				P = mu * (F - F_inv_T) + lambda * Math::log(J) * F_inv_T;
			} break;

			case LINEAR_ELASTIC: {
				// Linear: P = mu * (F + F^T - 2I) + lambda * tr(F - I) * I
				// Approximation for small strains; not fully rotation-invariant.
				Basis H = F - Basis(); // displacement gradient
				real_t trace_H = H[0][0] + H[1][1] + H[2][2];
				for (int i = 0; i < 3; ++i) {
					for (int j = 0; j < 3; ++j) {
						P[i][j] = mu * (H[i][j] + H[j][i]);
					}
					P[i][i] += lambda * trace_H;
				}
			} break;
		}
		return P;
	}

	// --- Plasticity update (returns new F, and plastic deformation gradient Fp, and equivalent plastic strain) ---
	// This is a simple return mapping for von Mises plasticity with isotropic hardening.
	// Input: trial F (deformation gradient), previous Fp, previous eq_plastic_strain.
	// Output: updated F, Fp, eq_plastic_strain, and whether plastic flow occurred.
	bool apply_plasticity(Basis &F, Basis &Fp, real_t &eq_plastic_strain) const {
		if (!plasticity_enabled) return false;

		// Compute elastic deformation gradient Fe = F * Fp^(-1)
		Basis Fp_inv = Fp.inverse();
		Basis Fe = F * Fp_inv;
		Basis Ee = 0.5 * (Fe.transposed() * Fe - Basis());
		// deviatoric part of Ee
		real_t trace3 = (Ee[0][0] + Ee[1][1] + Ee[2][2]) / 3.0;
		Basis Ee_dev = Ee;
		for (int i = 0; i < 3; ++i) Ee_dev[i][i] -= trace3;

		real_t mu = get_lame_mu();
		real_t norm_dev = Ee_dev[0].length() + Ee_dev[1].length() + Ee_dev[2].length(); // rough norm
		// von Mises stress: sqrt(3) * mu * ||Ee_dev||  (approximation)
		real_t vm_stress = Math::sqrt(3.0) * mu * norm_dev;

		real_t yield = yield_stress + hardening * eq_plastic_strain;
		if (vm_stress <= yield || yield <= 0) return false;

		// plastic flow direction (deviatoric part of Ee)
		Basis N = Ee_dev / norm_dev;
		real_t dlambda = (vm_stress - yield) / (2.0 * mu + hardening);
		dlambda = MIN(dlambda, max_plastic_strain); // cap
		Basis delta_Ep = dlambda * N;
		// Update plastic strain
		eq_plastic_strain += dlambda;

		// Update Fp (multiplicative update): Fp_new = exp(delta_Ep) * Fp
		// Approximate exponential using small strain: exp(delta_Ep) ~ I + delta_Ep
		Basis exp_delta = Basis() + delta_Ep;
		Fp = exp_delta * Fp;
		Fp.orthonormalize(); // keep Fp orientation clean
		Fp = Fp; // already orthonormalized? better just keep as is after multiplication.

		// Update F (total deformation) to reflect plastic correction: F remains the same,
		// but after updating Fp, the elastic part Fe changes. Actually we don't change F,
		// we only change internal variables (Fp). The solver will use the new Fp for next steps.
		return true;
	}

	// --- Override solver options ---
	virtual void get_solver_options(genesis::options::Options &opts) const override {
		GenesisMaterial::get_solver_options(opts);
		opts.set("fem.model", int(model));
		opts.set("fem.plasticity", plasticity_enabled);
		opts.set("fem.yield_stress", yield_stress);
		opts.set("fem.hardening", hardening);
		opts.set("fem.max_plastic_strain", max_plastic_strain);
		opts.set("fem.creep_rate", creep_rate);
		opts.set("fem.thermal_expansion", thermal_expansion);
	}

	// --- Duplicate ---
	virtual Ref<GenesisMaterial> duplicate(bool p_subresources = false) const override {
		Ref<FEMMaterial> mat = memnew(FEMMaterial);
		_copy_base(mat);
		return mat;
	}

protected:
	static void _bind_methods() {
		// Inherits from GenesisMaterial
		ClassDB::bind_method(D_METHOD("set_constitutive_model", "model"), &FEMMaterial::set_constitutive_model);
		ClassDB::bind_method(D_METHOD("get_constitutive_model"), &FEMMaterial::get_constitutive_model);

		ClassDB::bind_method(D_METHOD("set_plasticity_enabled", "enabled"), &FEMMaterial::set_plasticity_enabled);
		ClassDB::bind_method(D_METHOD("is_plasticity_enabled"), &FEMMaterial::is_plasticity_enabled);

		ClassDB::bind_method(D_METHOD("set_yield_stress", "yield_stress"), &FEMMaterial::set_yield_stress);
		ClassDB::bind_method(D_METHOD("get_yield_stress"), &FEMMaterial::get_yield_stress);

		ClassDB::bind_method(D_METHOD("set_hardening", "hardening"), &FEMMaterial::set_hardening);
		ClassDB::bind_method(D_METHOD("get_hardening"), &FEMMaterial::get_hardening);

		ClassDB::bind_method(D_METHOD("set_max_plastic_strain", "max_plastic_strain"), &FEMMaterial::set_max_plastic_strain);
		ClassDB::bind_method(D_METHOD("get_max_plastic_strain"), &FEMMaterial::get_max_plastic_strain);

		ClassDB::bind_method(D_METHOD("set_creep_rate", "creep_rate"), &FEMMaterial::set_creep_rate);
		ClassDB::bind_method(D_METHOD("get_creep_rate"), &FEMMaterial::get_creep_rate);

		ClassDB::bind_method(D_METHOD("set_thermal_expansion", "thermal_expansion"), &FEMMaterial::set_thermal_expansion);
		ClassDB::bind_method(D_METHOD("get_thermal_expansion"), &FEMMaterial::get_thermal_expansion);

		BIND_ENUM_CONSTANT(ST_VENANT_KIRCHHOFF);
		BIND_ENUM_CONSTANT(NEO_HOOKEAN);
		BIND_ENUM_CONSTANT(LINEAR_ELASTIC);

		ADD_PROPERTY(PropertyInfo(Variant::INT, "constitutive_model", PROPERTY_HINT_ENUM, "StVK,NeoHookean,LinearElastic"), "set_constitutive_model", "get_constitutive_model");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "plasticity_enabled"), "set_plasticity_enabled", "is_plasticity_enabled");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "yield_stress", PROPERTY_HINT_RANGE, "0,1e12,1"), "set_yield_stress", "get_yield_stress");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "hardening", PROPERTY_HINT_RANGE, "0,1e12,1"), "set_hardening", "get_hardening");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_plastic_strain", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_max_plastic_strain", "get_max_plastic_strain");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "creep_rate", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_creep_rate", "get_creep_rate");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "thermal_expansion", PROPERTY_HINT_RANGE, "-0.1,0.1,0.0001"), "set_thermal_expansion", "get_thermal_expansion");
	}

private:
	void _copy_base(Ref<FEMMaterial> &mat) const {
		mat->set_density(get_density());
		mat->set_young_modulus(get_young_modulus());
		mat->set_poisson_ratio(get_poisson_ratio());
		mat->set_friction(get_friction());
		mat->set_restitution(get_restitution());
		mat->set_damping(get_damping());
		mat->set_material_type(get_material_type());
		mat->model = model;
		mat->plasticity_enabled = plasticity_enabled;
		mat->yield_stress = yield_stress;
		mat->hardening = hardening;
		mat->max_plastic_strain = max_plastic_strain;
		mat->creep_rate = creep_rate;
		mat->thermal_expansion = thermal_expansion;
	}

	ConstitutiveModel model;
	bool plasticity_enabled;
	real_t yield_stress;
	real_t hardening;
	real_t max_plastic_strain;
	real_t creep_rate;
	real_t thermal_expansion;
};

} // namespace genesis

#endif // GENESIS_MATERIALS_FEM_MATERIAL_H