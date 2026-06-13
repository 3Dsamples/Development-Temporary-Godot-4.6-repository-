// File 313: modules/vienna/src/settings/vienna_physics_settings.h
// ViennaPhysicsSettings – a Godot Resource that stores global physics
// parameters (gravity, solver iterations, sleep thresholds, broad‑phase
// algorithm, etc.) for the ViennaPhysicsEngine.  Can be saved to disk and
// used to configure a ViennaWorld or ViennaWorldNode3D.

#ifndef VIENNA_SETTINGS_PHYSICS_SETTINGS_H
#define VIENNA_SETTINGS_PHYSICS_SETTINGS_H

#include "core/io/resource.h"
#include "core/math/vector3.h"
#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"

namespace vienna {

class ViennaPhysicsSettings : public Resource {
	GDCLASS(ViennaPhysicsSettings, Resource);

public:
	ViennaPhysicsSettings();

	// Gravity
	void set_gravity(const Vector3 &p_g);
	Vector3 get_gravity() const { return gravity; }

	// Solver iterations (per substep)
	void set_solver_iterations(int p_iters);
	int get_solver_iterations() const { return solver_iterations; }

	// Broad‑phase algorithm
	void set_broad_phase_algorithm(int p_algo); // 0=BruteForce,1=SAP,2=BVH
	int get_broad_phase_algorithm() const { return (int)broad_phase_algo; }

	// Sleep
	void set_sleep_linear_speed(real_t p_speed);
	real_t get_sleep_linear_speed() const { return sleep_linear_speed; }

	void set_sleep_angular_speed(real_t p_speed);
	real_t get_sleep_angular_speed() const { return sleep_angular_speed; }

	void set_sleep_frames(int p_frames);
	int get_sleep_frames() const { return sleep_frames; }

	// Sub‑steps per frame
	void set_sub_steps(int p_sub);
	int get_sub_steps() const { return sub_steps; }

	// Collision margin (skin thickness used in GJK)
	void set_collision_margin(real_t p_margin);
	real_t get_collision_margin() const { return collision_margin; }

	// Default material properties (used when no material is assigned)
	void set_default_friction(real_t p_fric);
	real_t get_default_friction() const { return default_friction; }

	void set_default_restitution(real_t p_rest);
	real_t get_default_restitution() const { return default_restitution; }

	void set_default_softness(real_t p_soft);
	real_t get_default_softness() const { return default_softness; }

protected:
	static void _bind_methods();

private:
	Vector3 gravity;
	int solver_iterations;
	BroadPhaseAlgorithm broad_phase_algo;
	real_t sleep_linear_speed;
	real_t sleep_angular_speed;
	int sleep_frames;
	int sub_steps;
	real_t collision_margin;
	real_t default_friction;
	real_t default_restitution;
	real_t default_softness;
};

} // namespace vienna

#endif // VIENNA_SETTINGS_PHYSICS_SETTINGS_H