// File 66: modules/genesis/src/materials/sf_material.h
// Stable Fluids (Eulerian grid) material with advection, diffusion, and projection.
// Used by the SF solver for smoke, fire, and liquid simulation on a grid.

#ifndef GENESIS_MATERIALS_SF_MATERIAL_H
#define GENESIS_MATERIALS_SF_MATERIAL_H

#include "material_base.h"
#include "../options/options_system.h"

namespace genesis {

class SFMaterial : public GenesisMaterial {
	GDCLASS(SFMaterial, GenesisMaterial);

public:
	enum DiffusionModel {
		CONSTANT_DIFFUSION = 0,
		VARIABLE_DIFFUSION
	};

	SFMaterial() :
		kinematic_viscosity(0.0001),
		diffusivity(0.0),
		density_diffusion(0.0),
		thermal_diffusivity(0.0),
		buoyancy_alpha(1.0),
		buoyancy_beta(0.0),
		ambient_temperature(300.0),
		vorticity_confinement(0.0),
		grid_resolution(64),
		domain_size(1.0),
		advection_method(0), // 0: semi-Lagrange, 1: MacCormack
		pressure_solver_iter(20)
	{
		set_material_type("stable_fluid");
	}

	// --- Viscosity / diffusion ---
	void set_kinematic_viscosity(real_t p_nu) { kinematic_viscosity = MAX(p_nu, 0.0); }
	real_t get_kinematic_viscosity() const { return kinematic_viscosity; }

	void set_diffusivity(real_t p_k) { diffusivity = MAX(p_k, 0.0); }
	real_t get_diffusivity() const { return diffusivity; }

	void set_density_diffusion(real_t p_k) { density_diffusion = MAX(p_k, 0.0); }
	real_t get_density_diffusion() const { return density_diffusion; }

	void set_thermal_diffusivity(real_t p_k) { thermal_diffusivity = MAX(p_k, 0.0); }
	real_t get_thermal_diffusivity() const { return thermal_diffusivity; }

	// --- Buoyancy ---
	void set_buoyancy_alpha(real_t p_a) { buoyancy_alpha = p_a; }
	real_t get_buoyancy_alpha() const { return buoyancy_alpha; }

	void set_buoyancy_beta(real_t p_b) { buoyancy_beta = p_b; }
	real_t get_buoyancy_beta() const { return buoyancy_beta; }

	void set_ambient_temperature(real_t p_T) { ambient_temperature = p_T; }
	real_t get_ambient_temperature() const { return ambient_temperature; }

	// --- Vorticity confinement ---
	void set_vorticity_confinement(real_t p_eps) { vorticity_confinement = MAX(p_eps, 0.0); }
	real_t get_vorticity_confinement() const { return vorticity_confinement; }

	// --- Grid settings ---
	void set_grid_resolution(int p_res) { grid_resolution = MAX(p_res, 4); }
	int get_grid_resolution() const { return grid_resolution; }

	void set_domain_size(real_t p_L) { domain_size = MAX(p_L, 0.1); }
	real_t get_domain_size() const { return domain_size; }

	// --- Advection ---
	void set_advection_method(int p_method) { advection_method = CLAMP(p_method, 0, 1); }
	int get_advection_method() const { return advection_method; }

	// --- Pressure solver ---
	void set_pressure_solver_iter(int p_iter) { pressure_solver_iter = MAX(p_iter, 1); }
	int get_pressure_solver_iter() const { return pressure_solver_iter; }

	virtual void get_solver_options(genesis::options::Options &opts) const override {
		GenesisMaterial::get_solver_options(opts);
		opts.set("sf.viscosity", kinematic_viscosity);
		opts.set("sf.diffusivity", diffusivity);
		opts.set("sf.density_diffusion", density_diffusion);
		opts.set("sf.thermal_diffusivity", thermal_diffusivity);
		opts.set("sf.buoyancy_alpha", buoyancy_alpha);
		opts.set("sf.buoyancy_beta", buoyancy_beta);
		opts.set("sf.ambient_temp", ambient_temperature);
		opts.set("sf.vorticity", vorticity_confinement);
		opts.set("sf.resolution", grid_resolution);
		opts.set("sf.domain_size", domain_size);
		opts.set("sf.advection", advection_method);
		opts.set("sf.pressure_iter", pressure_solver_iter);
	}

	virtual Ref<GenesisMaterial> duplicate(bool p_subresources = false) const override {
		Ref<SFMaterial> mat = memnew(SFMaterial);
		mat->set_density(get_density());
		mat->set_young_modulus(get_young_modulus());
		mat->set_poisson_ratio(get_poisson_ratio());
		mat->set_friction(get_friction());
		mat->set_restitution(get_restitution());
		mat->set_damping(get_damping());
		mat->set_material_type(get_material_type());
		mat->kinematic_viscosity = kinematic_viscosity;
		mat->diffusivity = diffusivity;
		mat->density_diffusion = density_diffusion;
		mat->thermal_diffusivity = thermal_diffusivity;
		mat->buoyancy_alpha = buoyancy_alpha;
		mat->buoyancy_beta = buoyancy_beta;
		mat->ambient_temperature = ambient_temperature;
		mat->vorticity_confinement = vorticity_confinement;
		mat->grid_resolution = grid_resolution;
		mat->domain_size = domain_size;
		mat->advection_method = advection_method;
		mat->pressure_solver_iter = pressure_solver_iter;
		return mat;
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_kinematic_viscosity", "nu"), &SFMaterial::set_kinematic_viscosity);
		ClassDB::bind_method(D_METHOD("get_kinematic_viscosity"), &SFMaterial::get_kinematic_viscosity);
		ClassDB::bind_method(D_METHOD("set_diffusivity", "k"), &SFMaterial::set_diffusivity);
		ClassDB::bind_method(D_METHOD("get_diffusivity"), &SFMaterial::get_diffusivity);
		ClassDB::bind_method(D_METHOD("set_density_diffusion", "k"), &SFMaterial::set_density_diffusion);
		ClassDB::bind_method(D_METHOD("get_density_diffusion"), &SFMaterial::get_density_diffusion);
		ClassDB::bind_method(D_METHOD("set_thermal_diffusivity", "k"), &SFMaterial::set_thermal_diffusivity);
		ClassDB::bind_method(D_METHOD("get_thermal_diffusivity"), &SFMaterial::get_thermal_diffusivity);
		ClassDB::bind_method(D_METHOD("set_buoyancy_alpha", "alpha"), &SFMaterial::set_buoyancy_alpha);
		ClassDB::bind_method(D_METHOD("get_buoyancy_alpha"), &SFMaterial::get_buoyancy_alpha);
		ClassDB::bind_method(D_METHOD("set_buoyancy_beta", "beta"), &SFMaterial::set_buoyancy_beta);
		ClassDB::bind_method(D_METHOD("get_buoyancy_beta"), &SFMaterial::get_buoyancy_beta);
		ClassDB::bind_method(D_METHOD("set_ambient_temperature", "T"), &SFMaterial::set_ambient_temperature);
		ClassDB::bind_method(D_METHOD("get_ambient_temperature"), &SFMaterial::get_ambient_temperature);
		ClassDB::bind_method(D_METHOD("set_vorticity_confinement", "eps"), &SFMaterial::set_vorticity_confinement);
		ClassDB::bind_method(D_METHOD("get_vorticity_confinement"), &SFMaterial::get_vorticity_confinement);
		ClassDB::bind_method(D_METHOD("set_grid_resolution", "res"), &SFMaterial::set_grid_resolution);
		ClassDB::bind_method(D_METHOD("get_grid_resolution"), &SFMaterial::get_grid_resolution);
		ClassDB::bind_method(D_METHOD("set_domain_size", "L"), &SFMaterial::set_domain_size);
		ClassDB::bind_method(D_METHOD("get_domain_size"), &SFMaterial::get_domain_size);
		ClassDB::bind_method(D_METHOD("set_advection_method", "method"), &SFMaterial::set_advection_method);
		ClassDB::bind_method(D_METHOD("get_advection_method"), &SFMaterial::get_advection_method);
		ClassDB::bind_method(D_METHOD("set_pressure_solver_iter", "iter"), &SFMaterial::set_pressure_solver_iter);
		ClassDB::bind_method(D_METHOD("get_pressure_solver_iter"), &SFMaterial::get_pressure_solver_iter);

		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "kinematic_viscosity", PROPERTY_HINT_RANGE, "0,1,0.0001"), "set_kinematic_viscosity", "get_kinematic_viscosity");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "diffusivity", PROPERTY_HINT_RANGE, "0,1,0.0001"), "set_diffusivity", "get_diffusivity");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "density_diffusion", PROPERTY_HINT_RANGE, "0,1,0.0001"), "set_density_diffusion", "get_density_diffusion");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "thermal_diffusivity", PROPERTY_HINT_RANGE, "0,1,0.0001"), "set_thermal_diffusivity", "get_thermal_diffusivity");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "buoyancy_alpha"), "set_buoyancy_alpha", "get_buoyancy_alpha");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "buoyancy_beta"), "set_buoyancy_beta", "get_buoyancy_beta");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ambient_temperature"), "set_ambient_temperature", "get_ambient_temperature");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "vorticity_confinement", PROPERTY_HINT_RANGE, "0,10,0.01"), "set_vorticity_confinement", "get_vorticity_confinement");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "grid_resolution", PROPERTY_HINT_RANGE, "4,1024,1"), "set_grid_resolution", "get_grid_resolution");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "domain_size", PROPERTY_HINT_RANGE, "0.1,100,0.1"), "set_domain_size", "get_domain_size");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "advection_method", PROPERTY_HINT_ENUM, "SemiLagrange,MacCormack"), "set_advection_method", "get_advection_method");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "pressure_solver_iter", PROPERTY_HINT_RANGE, "1,200,1"), "set_pressure_solver_iter", "get_pressure_solver_iter");
	}

private:
	real_t kinematic_viscosity;
	real_t diffusivity;
	real_t density_diffusion;
	real_t thermal_diffusivity;
	real_t buoyancy_alpha;
	real_t buoyancy_beta;
	real_t ambient_temperature;
	real_t vorticity_confinement;
	int grid_resolution;
	real_t domain_size;
	int advection_method;
	int pressure_solver_iter;
};

} // namespace genesis

#endif // GENESIS_MATERIALS_SF_MATERIAL_H