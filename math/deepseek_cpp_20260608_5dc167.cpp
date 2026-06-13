// File 257: modules/newton/src/world/newton_world_config.cpp
// NewtonWorldConfig implementation – binds properties and provides
// default values for global simulation settings.

#include "newton_world_config.h"
#include "../core/newton_constants.h"
#include "core/object/class_db.h"

namespace newton {

NewtonWorldConfig::NewtonWorldConfig() :
	gravity(0.0, DEFAULT_GRAVITY, 0.0),
	solver_iterations(DEFAULT_SOLVER_ITERATIONS),
	solver_method(SolverMethod::ITERATIVE_ACCELERATED),
	sleep_linear_speed(0.01),
	sleep_angular_speed(0.01),
	sleep_frames(10),
	broad_phase_algorithm(BroadPhaseAlgorithm::BVH),
	ccd_enabled(false),
	ccd_max_iterations(50) {}

void NewtonWorldConfig::set_gravity(const Vector3 &p_g) { gravity = p_g; }
Vector3 NewtonWorldConfig::get_gravity() const { return gravity; }

void NewtonWorldConfig::set_solver_iterations(int p_iter) { solver_iterations = CLAMP(p_iter, 1, MAX_SOLVER_ITERATIONS); }
int NewtonWorldConfig::get_solver_iterations() const { return solver_iterations; }

void NewtonWorldConfig::set_solver_method(SolverMethod p_method) { solver_method = p_method; }
SolverMethod NewtonWorldConfig::get_solver_method() const { return solver_method; }

void NewtonWorldConfig::set_sleep_linear_speed(real_t p_v) { sleep_linear_speed = MAX(p_v, 0.0); }
real_t NewtonWorldConfig::get_sleep_linear_speed() const { return sleep_linear_speed; }

void NewtonWorldConfig::set_sleep_angular_speed(real_t p_w) { sleep_angular_speed = MAX(p_w, 0.0); }
real_t NewtonWorldConfig::get_sleep_angular_speed() const { return sleep_angular_speed; }

void NewtonWorldConfig::set_sleep_frames(int p_frames) { sleep_frames = MAX(p_frames, 1); }
int NewtonWorldConfig::get_sleep_frames() const { return sleep_frames; }

void NewtonWorldConfig::set_broad_phase_algorithm(BroadPhaseAlgorithm p_algo) { broad_phase_algorithm = p_algo; }
BroadPhaseAlgorithm NewtonWorldConfig::get_broad_phase_algorithm() const { return broad_phase_algorithm; }

void NewtonWorldConfig::set_ccd_enabled(bool p_en) { ccd_enabled = p_en; }
bool NewtonWorldConfig::is_ccd_enabled() const { return ccd_enabled; }

void NewtonWorldConfig::set_ccd_max_iterations(int p_iter) { ccd_max_iterations = MAX(p_iter, 1); }
int NewtonWorldConfig::get_ccd_max_iterations() const { return ccd_max_iterations; }

// Already bound in header; this file exists to keep the module's source
// files consistent and avoid linker errors when the header's inline
// methods are not fully inlined.

} // namespace newton