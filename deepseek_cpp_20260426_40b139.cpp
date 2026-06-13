// File 112: modules/gaia/src/vbd_physics/vbd_physics_parameters.h
// Parameters for the Vertex Block Descent (VBD) solver.
// Controls convergence, line search, damping, and parallelisation.

#ifndef GAIA_VBD_PHYSICS_PARAMETERS_H
#define GAIA_VBD_PHYSICS_PARAMETERS_H

#include "core/typedefs.h"
#include "core/string/ustring.h"

namespace gaia::vbd {

class VBDPhysicsParameters {
public:
	// --- Time integration ---
	real_t dt = 1.0 / 60.0;
	int    sub_steps = 1;
	int    max_iterations = 50;
	bool   use_warm_start = true;

	// --- Convergence ---
	real_t energy_convergence_tolerance = 1e-6;
	real_t residual_tolerance = 1e-6;

	// --- Line search ---
	bool   enable_line_search = true;
	real_t line_search_c1 = 1e-4;           // Armijo condition
	real_t line_search_backtracking = 0.5;
	int    line_search_max_steps = 20;
	real_t line_search_min_step = 0.001;

	// --- Damping (Rayleigh) ---
	real_t damping_alpha = 0.0;
	real_t damping_beta  = 0.0;

	// --- Parallelisation ---
	int    num_threads = -1;                // -1 = auto detect
	int    color_partition_size = 256;

	// --- Collision ---
	bool   enable_ipc = true;
	real_t ipc_distance = 0.001;
	real_t ipc_stiffness = 1e6;
	real_t ipc_friction = 0.5;

	// --- Solver override flags ---
	bool   use_chebyshev_preconditioner = false;
	real_t chebyshev_lambda_min = 0.1;
	real_t chebyshev_lambda_max = 1.0;

	void load_from(const Dictionary &p_dict) {
		dt = p_dict.get("dt", dt);
		sub_steps = p_dict.get("sub_steps", sub_steps);
		max_iterations = p_dict.get("max_iterations", max_iterations);
		use_warm_start = p_dict.get("use_warm_start", use_warm_start);
		energy_convergence_tolerance = p_dict.get("energy_tol", energy_convergence_tolerance);
		residual_tolerance = p_dict.get("residual_tol", residual_tolerance);
		enable_line_search = p_dict.get("line_search", enable_line_search);
		line_search_c1 = p_dict.get("ls_c1", line_search_c1);
		line_search_backtracking = p_dict.get("ls_backtrack", line_search_backtracking);
		line_search_max_steps = p_dict.get("ls_max_steps", line_search_max_steps);
		line_search_min_step = p_dict.get("ls_min_step", line_search_min_step);
		damping_alpha = p_dict.get("damping_alpha", damping_alpha);
		damping_beta = p_dict.get("damping_beta", damping_beta);
		num_threads = p_dict.get("num_threads", num_threads);
		color_partition_size = p_dict.get("color_partition", color_partition_size);
		enable_ipc = p_dict.get("enable_ipc", enable_ipc);
		ipc_distance = p_dict.get("ipc_distance", ipc_distance);
		ipc_stiffness = p_dict.get("ipc_stiffness", ipc_stiffness);
		ipc_friction = p_dict.get("ipc_friction", ipc_friction);
		use_chebyshev_preconditioner = p_dict.get("chebyshev", use_chebyshev_preconditioner);
		chebyshev_lambda_min = p_dict.get("chebyshev_lambda_min", chebyshev_lambda_min);
		chebyshev_lambda_max = p_dict.get("chebyshev_lambda_max", chebyshev_lambda_max);
	}
};

} // namespace gaia::vbd

#endif // GAIA_VBD_PHYSICS_PARAMETERS_H