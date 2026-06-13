// File 65: modules/genesis/src/materials/sph_material.h
// SPH (Smoothed Particle Hydrodynamics) material with equation-of-state,
// viscosity, surface tension, and multiphase support.

#ifndef GENESIS_MATERIALS_SPH_MATERIAL_H
#define GENESIS_MATERIALS_SPH_MATERIAL_H

#include "material_base.h"
#include "../options/options_system.h"

namespace genesis {

class SPHMaterial : public GenesisMaterial {
	GDCLASS(SPHMaterial, GenesisMaterial);

public:
	enum EOSModel {
		TAIT = 0,          // p = B * ((rho/rho0)^gamma - 1)
		IDEAL_GAS,         // p = (gamma-1) * rho * e
		BAROTROPIC_STIFFENED_GAS
	};

	SPHMaterial() :
		eos_model(TAIT),
		speed_of_sound(1500.0),
		rest_density(1000.0),
		gamma_eos(7.0),
		viscosity_mu(0.001),        // dynamic viscosity
		viscosity_alpha(0.01),      // artificial viscosity (Monaghan)
		viscosity_beta(0.0),
		surface_tension_gamma(0.072), // N/m (water)
		surface_tension_coeff(0.01),  // scaling in CSF model
		turbulence_coeff(0.0),
		smoothing_length(0.1),       // h
		is_compressible(true) {}

	// --- Equation of state ---
	void set_eos_model(EOSModel p_model) { eos_model = p_model; }
	EOSModel get_eos_model() const { return eos_model; }

	void set_speed_of_sound(real_t p_c) { speed_of_sound = MAX(p_c, 1.0); }
	real_t get_speed_of_sound() const { return speed_of_sound; }

	void set_rest_density(real_t p_rho0) { rest_density = MAX(p_rho0, 0.1); }
	real_t get_rest_density() const { return rest_density; }

	void set_gamma_eos(real_t p_gamma) { gamma_eos = MAX(p_gamma, 1.0); }
	real_t get_gamma_eos() const { return gamma_eos; }

	// --- Viscosity ---
	void set_viscosity_mu(real_t p_mu) { viscosity_mu = MAX(p_mu, 0.0); }
	real_t get_viscosity_mu() const { return viscosity_mu; }

	void set_viscosity_alpha(real_t p_alpha) { viscosity_alpha = MAX(p_alpha, 0.0); }
	real_t get_viscosity_alpha() const { return viscosity_alpha; }

	void set_viscosity_beta(real_t p_beta) { viscosity_beta = MAX(p_beta, 0.0); }
	real_t get_viscosity_beta() const { return viscosity_beta; }

	// --- Surface tension ---
	void set_surface_tension_gamma(real_t p_gamma) { surface_tension_gamma = MAX(p_gamma, 0.0); }
	real_t get_surface_tension_gamma() const { return surface_tension_gamma; }

	void set_surface_tension_coeff(real_t p_coeff) { surface_tension_coeff = MAX(p_coeff, 0.0); }
	real_t get_surface_tension_coeff() const { return surface_tension_coeff; }

	void set_turbulence_coeff(real_t p_coeff) { turbulence_coeff = MAX(p_coeff, 0.0); }
	real_t get_turbulence_coeff() const { return turbulence_coeff; }

	// --- Kernel parameters ---
	void set_smoothing_length(real_t p_h) { smoothing_length = MAX(p_h, 1e-6); }
	real_t get_smoothing_length() const { return smoothing_length; }

	void set_is_compressible(bool p_comp) { is_compressible = p_comp; }
	bool is_compressible() const { return is_compressible; }

	// --- Compute pressure from density using EOS ---
	real_t compute_pressure(real_t rho, real_t rho0 = -1.0) const {
		if (rho0 <= 0.0) rho0 = rest_density;
		switch (eos_model) {
			case TAIT: {
				real_t B = speed_of_sound * speed_of_sound * rest_density / gamma_eos;
				return B * (Math::pow(rho / rho0, gamma_eos) - 1.0);
			}
			case IDEAL_GAS: {
				// requires internal energy, fallback to p = c^2 * (rho - rho0)
				return speed_of_sound * speed_of_sound * (rho - rho0);
			}
			case BAROTROPIC_STIFFENED_GAS:
			default:
				return speed_of_sound * speed_of_sound * (rho - rho0);
		}
	}

	// --- Speed of sound squared from bulk modulus ---
	real_t get_bulk_modulus() const {
		return speed_of_sound * speed_of_sound * rest_density;
	}

	// --- Solver options ---
	virtual void get_solver_options(genesis::options::Options &opts) const override {
		GenesisMaterial::get_solver_options(opts);
		opts.set("sph.eos_model", int(eos_model));
		opts.set("sph.speed_of_sound", speed_of_sound);
		opts.set("sph.rest_density", rest_density);
		opts.set("sph.gamma", gamma_eos);
		opts.set("sph.viscosity_mu", viscosity_mu);
		opts.set("sph.viscosity_alpha", viscosity_alpha);
		opts.set("sph.viscosity_beta", viscosity_beta);
		opts.set("sph.surface_tension_gamma", surface_tension_gamma);
		opts.set("sph.surface_tension_coeff", surface_tension_coeff);
		opts.set("sph.turbulence_coeff", turbulence_coeff);
		opts.set("sph.smoothing_length", smoothing_length);
		opts.set("sph.compressible", is_compressible);
	}

	virtual Ref<GenesisMaterial> duplicate(bool p_subresources = false) const override {
		Ref<SPHMaterial> mat = memnew(SPHMaterial);
		_copy_base(mat);
		return mat;
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_eos_model", "model"), &SPHMaterial::set_eos_model);
		ClassDB::bind_method(D_METHOD("get_eos_model"), &SPHMaterial::get_eos_model);
		ClassDB::bind_method(D_METHOD("set_speed_of_sound", "c"), &SPHMaterial::set_speed_of_sound);
		ClassDB::bind_method(D_METHOD("get_speed_of_sound"), &SPHMaterial::get_speed_of_sound);
		ClassDB::bind_method(D_METHOD("set_rest_density", "rho0"), &SPHMaterial::set_rest_density);
		ClassDB::bind_method(D_METHOD("get_rest_density"), &SPHMaterial::get_rest_density);
		ClassDB::bind_method(D_METHOD("set_gamma_eos", "gamma"), &SPHMaterial::set_gamma_eos);
		ClassDB::bind_method(D_METHOD("get_gamma_eos"), &SPHMaterial::get_gamma_eos);
		ClassDB::bind_method(D_METHOD("set_viscosity_mu", "mu"), &SPHMaterial::set_viscosity_mu);
		ClassDB::bind_method(D_METHOD("get_viscosity_mu"), &SPHMaterial::get_viscosity_mu);
		ClassDB::bind_method(D_METHOD("set_viscosity_alpha", "alpha"), &SPHMaterial::set_viscosity_alpha);
		ClassDB::bind_method(D_METHOD("get_viscosity_alpha"), &SPHMaterial::get_viscosity_alpha);
		ClassDB::bind_method(D_METHOD("set_viscosity_beta", "beta"), &SPHMaterial::set_viscosity_beta);
		ClassDB::bind_method(D_METHOD("get_viscosity_beta"), &SPHMaterial::get_viscosity_beta);
		ClassDB::bind_method(D_METHOD("set_surface_tension_gamma", "gamma"), &SPHMaterial::set_surface_tension_gamma);
		ClassDB::bind_method(D_METHOD("get_surface_tension_gamma"), &SPHMaterial::get_surface_tension_gamma);
		ClassDB::bind_method(D_METHOD("set_surface_tension_coeff", "coeff"), &SPHMaterial::set_surface_tension_coeff);
		ClassDB::bind_method(D_METHOD("get_surface_tension_coeff"), &SPHMaterial::get_surface_tension_coeff);
		ClassDB::bind_method(D_METHOD("set_turbulence_coeff", "coeff"), &SPHMaterial::set_turbulence_coeff);
		ClassDB::bind_method(D_METHOD("get_turbulence_coeff"), &SPHMaterial::get_turbulence_coeff);
		ClassDB::bind_method(D_METHOD("set_smoothing_length", "h"), &SPHMaterial::set_smoothing_length);
		ClassDB::bind_method(D_METHOD("get_smoothing_length"), &SPHMaterial::get_smoothing_length);
		ClassDB::bind_method(D_METHOD("set_is_compressible", "compressible"), &SPHMaterial::set_is_compressible);
		ClassDB::bind_method(D_METHOD("is_compressible"), &SPHMaterial::is_compressible);

		BIND_ENUM_CONSTANT(TAIT);
		BIND_ENUM_CONSTANT(IDEAL_GAS);
		BIND_ENUM_CONSTANT(BAROTROPIC_STIFFENED_GAS);

		ADD_PROPERTY(PropertyInfo(Variant::INT, "eos_model", PROPERTY_HINT_ENUM, "Tait,IdealGas,StiffenedGas"), "set_eos_model", "get_eos_model");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "speed_of_sound", PROPERTY_HINT_RANGE, "1,5000,1"), "set_speed_of_sound", "get_speed_of_sound");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rest_density", PROPERTY_HINT_RANGE, "0.1,20000,0.1"), "set_rest_density", "get_rest_density");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "gamma_eos", PROPERTY_HINT_RANGE, "1,10,0.1"), "set_gamma_eos", "get_gamma_eos");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "viscosity_mu", PROPERTY_HINT_RANGE, "0,10,0.0001"), "set_viscosity_mu", "get_viscosity_mu");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "viscosity_alpha", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_viscosity_alpha", "get_viscosity_alpha");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "viscosity_beta", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_viscosity_beta", "get_viscosity_beta");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "surface_tension_gamma", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_surface_tension_gamma", "get_surface_tension_gamma");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "surface_tension_coeff", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_surface_tension_coeff", "get_surface_tension_coeff");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "turbulence_coeff", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_turbulence_coeff", "get_turbulence_coeff");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "smoothing_length", PROPERTY_HINT_RANGE, "0,10,0.01"), "set_smoothing_length", "get_smoothing_length");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_compressible"), "set_is_compressible", "is_compressible");
	}

private:
	void _copy_base(Ref<SPHMaterial> &mat) const {
		mat->set_density(get_density());
		mat->set_young_modulus(get_young_modulus());
		mat->set_poisson_ratio(get_poisson_ratio());
		mat->set_friction(get_friction());
		mat->set_restitution(get_restitution());
		mat->set_damping(get_damping());
		mat->set_material_type(get_material_type());
		mat->eos_model = eos_model;
		mat->speed_of_sound = speed_of_sound;
		mat->rest_density = rest_density;
		mat->gamma_eos = gamma_eos;
		mat->viscosity_mu = viscosity_mu;
		mat->viscosity_alpha = viscosity_alpha;
		mat->viscosity_beta = viscosity_beta;
		mat->surface_tension_gamma = surface_tension_gamma;
		mat->surface_tension_coeff = surface_tension_coeff;
		mat->turbulence_coeff = turbulence_coeff;
		mat->smoothing_length = smoothing_length;
		mat->is_compressible = is_compressible;
	}

	EOSModel eos_model;
	real_t speed_of_sound;
	real_t rest_density;
	real_t gamma_eos;
	real_t viscosity_mu;
	real_t viscosity_alpha;
	real_t viscosity_beta;
	real_t surface_tension_gamma;
	real_t surface_tension_coeff;
	real_t turbulence_coeff;
	real_t smoothing_length;
	bool is_compressible;
};

} // namespace genesis

#endif // GENESIS_MATERIALS_SPH_MATERIAL_H