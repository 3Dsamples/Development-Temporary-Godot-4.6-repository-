// File 229: modules/newton/src/world/newton_world_config.h
// NewtonWorldConfig – serialisable resource holding global Newton Dynamics
// simulation settings (gravity, solver iterations, sleep thresholds, etc.)
// so that a world can be set up from a .tres file or GDScript.

#ifndef NEWTON_WORLD_CONFIG_H
#define NEWTON_WORLD_CONFIG_H

#include "core/io/resource.h"
#include "core/math/vector3.h"
#include "../core/newton_types.h"
#include "../core/newton_constants.h"

namespace newton {

class NewtonWorldConfig : public Resource {
	GDCLASS(NewtonWorldConfig, Resource);

public:
	NewtonWorldConfig() :
		gravity(0.0, DEFAULT_GRAVITY, 0.0),
		solver_iterations(DEFAULT_SOLVER_ITERATIONS),
		solver_method(SolverMethod::ITERATIVE_ACCELERATED),
		sleep_linear_speed(0.01),
		sleep_angular_speed(0.01),
		sleep_frames(10),
		broad_phase_algorithm(BroadPhaseAlgorithm::BVH),
		ccd_enabled(false),
		ccd_max_iterations(50) {}

	void set_gravity(const Vector3 &p_g) { gravity = p_g; }
	Vector3 get_gravity() const { return gravity; }

	void set_solver_iterations(int p_iter) { solver_iterations = CLAMP(p_iter, 1, MAX_SOLVER_ITERATIONS); }
	int get_solver_iterations() const { return solver_iterations; }

	void set_solver_method(SolverMethod p_method) { solver_method = p_method; }
	SolverMethod get_solver_method() const { return solver_method; }

	void set_sleep_linear_speed(real_t p_v) { sleep_linear_speed = MAX(p_v, 0.0); }
	real_t get_sleep_linear_speed() const { return sleep_linear_speed; }

	void set_sleep_angular_speed(real_t p_w) { sleep_angular_speed = MAX(p_w, 0.0); }
	real_t get_sleep_angular_speed() const { return sleep_angular_speed; }

	void set_sleep_frames(int p_frames) { sleep_frames = MAX(p_frames, 1); }
	int get_sleep_frames() const { return sleep_frames; }

	void set_broad_phase_algorithm(BroadPhaseAlgorithm p_algo) { broad_phase_algorithm = p_algo; }
	BroadPhaseAlgorithm get_broad_phase_algorithm() const { return broad_phase_algorithm; }

	void set_ccd_enabled(bool p_en) { ccd_enabled = p_en; }
	bool is_ccd_enabled() const { return ccd_enabled; }

	void set_ccd_max_iterations(int p_iter) { ccd_max_iterations = MAX(p_iter, 1); }
	int get_ccd_max_iterations() const { return ccd_max_iterations; }

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_gravity", "gravity"), &NewtonWorldConfig::set_gravity);
		ClassDB::bind_method(D_METHOD("get_gravity"), &NewtonWorldConfig::get_gravity);
		ClassDB::bind_method(D_METHOD("set_solver_iterations", "iter"), &NewtonWorldConfig::set_solver_iterations);
		ClassDB::bind_method(D_METHOD("get_solver_iterations"), &NewtonWorldConfig::get_solver_iterations);
		ClassDB::bind_method(D_METHOD("set_solver_method", "method"), &NewtonWorldConfig::set_solver_method);
		ClassDB::bind_method(D_METHOD("get_solver_method"), &NewtonWorldConfig::get_solver_method);
		ClassDB::bind_method(D_METHOD("set_sleep_linear_speed", "speed"), &NewtonWorldConfig::set_sleep_linear_speed);
		ClassDB::bind_method(D_METHOD("get_sleep_linear_speed"), &NewtonWorldConfig::get_sleep_linear_speed);
		ClassDB::bind_method(D_METHOD("set_sleep_angular_speed", "speed"), &NewtonWorldConfig::set_sleep_angular_speed);
		ClassDB::bind_method(D_METHOD("get_sleep_angular_speed"), &NewtonWorldConfig::get_sleep_angular_speed);
		ClassDB::bind_method(D_METHOD("set_sleep_frames", "frames"), &NewtonWorldConfig::set_sleep_frames);
		ClassDB::bind_method(D_METHOD("get_sleep_frames"), &NewtonWorldConfig::get_sleep_frames);
		ClassDB::bind_method(D_METHOD("set_broad_phase_algorithm", "algo"), &NewtonWorldConfig::set_broad_phase_algorithm);
		ClassDB::bind_method(D_METHOD("get_broad_phase_algorithm"), &NewtonWorldConfig::get_broad_phase_algorithm);
		ClassDB::bind_method(D_METHOD("set_ccd_enabled", "enabled"), &NewtonWorldConfig::set_ccd_enabled);
		ClassDB::bind_method(D_METHOD("is_ccd_enabled"), &NewtonWorldConfig::is_ccd_enabled);
		ClassDB::bind_method(D_METHOD("set_ccd_max_iterations", "iter"), &NewtonWorldConfig::set_ccd_max_iterations);
		ClassDB::bind_method(D_METHOD("get_ccd_max_iterations"), &NewtonWorldConfig::get_ccd_max_iterations);

		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "gravity"), "set_gravity", "get_gravity");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "solver_iterations", PROPERTY_HINT_RANGE, "1,256,1"), "set_solver_iterations", "get_solver_iterations");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "solver_method", PROPERTY_HINT_ENUM, "IterativeGS,Accelerated,Direct"), "set_solver_method", "get_solver_method");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sleep_linear_speed"), "set_sleep_linear_speed", "get_sleep_linear_speed");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sleep_angular_speed"), "set_sleep_angular_speed", "get_sleep_angular_speed");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "sleep_frames", PROPERTY_HINT_RANGE, "1,100,1"), "set_sleep_frames", "get_sleep_frames");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "broad_phase_algorithm", PROPERTY_HINT_ENUM, "BruteForce,SAP,BVH"), "set_broad_phase_algorithm", "get_broad_phase_algorithm");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ccd_enabled"), "set_ccd_enabled", "is_ccd_enabled");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "ccd_max_iterations", PROPERTY_HINT_RANGE, "1,200,1"), "set_ccd_max_iterations", "get_ccd_max_iterations");
	}

private:
	Vector3 gravity;
	int solver_iterations;
	SolverMethod solver_method;
	real_t sleep_linear_speed;
	real_t sleep_angular_speed;
	int sleep_frames;
	BroadPhaseAlgorithm broad_phase_algorithm;
	bool ccd_enabled;
	int ccd_max_iterations;
};

} // namespace newton

#endif // NEWTON_WORLD_CONFIG_H