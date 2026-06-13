// File 108: modules/gaia/src/parameters/physics_parameters.h
// Central physics parameter store – holds all global simulation settings
// for Gaia solvers (VBD, PBD, FEM, MPM, etc.). Replaces Gaia's PhysicsParameters.h.

#ifndef GAIA_PARAMETERS_PHYSICS_PARAMETERS_H
#define GAIA_PARAMETERS_PHYSICS_PARAMETERS_H

#include "core/variant/variant.h"
#include "core/string/ustring.h"
#include "../io/parameter_reader.h"

namespace gaia::parameters {

class PhysicsParameters {
public:
	// --- Time stepping ---
	real_t dt = 1.0 / 60.0;
	int sub_steps = 1;
	int iterations = 5;

	// --- Gravity ---
	Vector3 gravity = Vector3(0, -9.81, 0);

	// --- Damping (global defaults) ---
	real_t velocity_damping = 0.001;
	real_t collision_damping = 0.0;

	// --- Solver flags ---
	bool solve_statics = false;
	bool use_vbd = false;
	bool use_pbd = true;

	// --- Collision ---
	bool collision_enabled = true;
	real_t collision_margin = 0.01;
	real_t friction_coefficient = 0.5;
	real_t restitution_coefficient = 0.0;

	// --- Constraints (PBD/XPBD) ---
	real_t distance_compliance = 1e-7;
	real_t bending_compliance = 1e-5;
	real_t volume_compliance = 1e-6;
	real_t collision_compliance = 1e-8;

	// --- FEM ---
	real_t fem_amount = 0.0;
	real_t fem_stiffness = 1e4;
	real_t fem_poisson = 0.3;

	// --- MPM grid ---
	int mpm_grid_res = 32;
	real_t mpm_cell_size = 0.05;

	// --- SPH ---
	real_t sph_smoothing_length = 0.1;
	real_t sph_rest_density = 1000.0;
	real_t sph_viscosity = 0.001;

	// Load from a Gaia-style parameter file (compatible with ParameterReader / JSON)
	void load_from_reader(const io::ParameterReader &reader) {
		dt = reader.get_real("dt", dt);
		sub_steps = reader.get_int("sub_steps", sub_steps);
		iterations = reader.get_int("iterations", iterations);
		gravity = reader.get_vector3("gravity", gravity);
		velocity_damping = reader.get_real("velocity_damping", velocity_damping);
		collision_damping = reader.get_real("collision_damping", collision_damping);
		solve_statics = reader.get_bool("solve_statics", solve_statics);
		use_vbd = reader.get_bool("use_vbd", use_vbd);
		use_pbd = reader.get_bool("use_pbd", use_pbd);
		collision_enabled = reader.get_bool("collision_enabled", collision_enabled);
		collision_margin = reader.get_real("collision_margin", collision_margin);
		friction_coefficient = reader.get_real("friction_coefficient", friction_coefficient);
		restitution_coefficient = reader.get_real("restitution_coefficient", restitution_coefficient);
		distance_compliance = reader.get_real("distance_compliance", distance_compliance);
		bending_compliance = reader.get_real("bending_compliance", bending_compliance);
		volume_compliance = reader.get_real("volume_compliance", volume_compliance);
		collision_compliance = reader.get_real("collision_compliance", collision_compliance);
		fem_amount = reader.get_real("fem_amount", fem_amount);
		fem_stiffness = reader.get_real("fem_stiffness", fem_stiffness);
		fem_poisson = reader.get_real("fem_poisson", fem_poisson);
		mpm_grid_res = reader.get_int("mpm_grid_res", mpm_grid_res);
		mpm_cell_size = reader.get_real("mpm_cell_size", mpm_cell_size);
		sph_smoothing_length = reader.get_real("sph_smoothing_length", sph_smoothing_length);
		sph_rest_density = reader.get_real("sph_rest_density", sph_rest_density);
		sph_viscosity = reader.get_real("sph_viscosity", sph_viscosity);
	}
};

} // namespace gaia::parameters

#endif // GAIA_PARAMETERS_PHYSICS_PARAMETERS_H