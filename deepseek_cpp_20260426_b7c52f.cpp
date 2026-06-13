// File 175: modules/newton/src/world/newton_world.h
// NewtonWorld – the main physics world that owns all bodies, joints, materials,
// and drives the high‑performance rigid‑body simulation step.
// Integrates Gaia's BVH for broad‑phase collision detection.

#ifndef NEWTON_WORLD_NEWTON_WORLD_H
#define NEWTON_WORLD_NEWTON_WORLD_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "../core/newton_types.h"
#include "../core/newton_constants.h"

// Gaia broad‑phase integration
#include "../../../gaia/src/collision_detector/broad_phase.h"

namespace newton {

// Forward declarations
class NewtonBody;
class NewtonJoint;
class NewtonMaterial;
class NewtonSolver;
class NewtonIsland;

class NewtonWorld : public RefCounted {
	GDCLASS(NewtonWorld, RefCounted);

public:
	NewtonWorld();
	virtual ~NewtonWorld();

	// --- Time step ---
	void step(real_t p_dt);
	real_t get_time() const { return world_time; }

	// --- Gravity ---
	void set_gravity(const vec3 &p_gravity);
	vec3 get_gravity() const { return gravity; }

	// --- Solver settings ---
	void set_solver_iterations(int p_iter);
	int get_solver_iterations() const { return solver_iterations; }
	void set_solver_method(SolverMethod p_method) { solver_method = p_method; }
	SolverMethod get_solver_method() const { return solver_method; }

	// --- Sleep ---
	void set_sleep_speed_thresholds(real_t linear, real_t angular);
	void set_sleep_frames(int p_frames) { sleep_frames = MAX(p_frames, 1); }

	// --- Body management ---
	body_id create_body(const Ref<NewtonBody> &p_body);
	void destroy_body(body_id p_id);
	Ref<NewtonBody> get_body(body_id p_id) const;
	int get_body_count() const { return bodies.size(); }

	// --- Joint management ---
	joint_id create_joint(const Ref<NewtonJoint> &p_joint);
	void destroy_joint(joint_id p_id);
	Ref<NewtonJoint> get_joint(joint_id p_id) const;

	// --- Material management ---
	material_id create_material(const Ref<NewtonMaterial> &p_material);
	void destroy_material(material_id p_id);
	Ref<NewtonMaterial> get_material(material_id p_id) const;

	// --- Callback for custom ray‑cast / contact filtering ---
	// Not implemented in this header but can be added later.

private:
	// Main steps of the simulation pipeline
	void detect_collisions();
	void resolve_contacts_and_joints(real_t dt);
	void integrate(real_t dt);
	void update_sleep_state();

	// Island management
	void build_islands();
	void solve_islands(real_t dt);

	// Contact generation between two bodies using narrow‑phase
	void generate_contacts(body_id a, body_id b);

	// Internal containers
	HashMap<body_id, Ref<NewtonBody>> bodies;
	HashMap<joint_id, Ref<NewtonJoint>> joints;
	HashMap<material_id, Ref<NewtonMaterial>> materials;

	// Next IDs
	body_id next_body_id = 1;
	joint_id next_joint_id = 1;
	material_id next_material_id = 1;

	// World state
	vec3 gravity = vec3(0, DEFAULT_GRAVITY, 0);
	real_t world_time = 0.0;
	int solver_iterations = DEFAULT_SOLVER_ITERATIONS;
	SolverMethod solver_method = SolverMethod::ITERATIVE_ACCELERATED;
	int sleep_frames = DEFAULT_SLEEP_FRAMES;
	real_t sleep_linear_threshold = DEFAULT_SLEEP_LINEAR;
	real_t sleep_angular_threshold = DEFAULT_SLEEP_ANGULAR;

	// Broad‑phase (Gaia)
	gaia::collision::BroadPhase broad_phase;

	// Solver and island instances
	Ref<NewtonSolver> solver;
	Ref<NewtonIsland> island_manager;

	// Per‑body sleep counter
	HashMap<body_id, int> sleep_counters;

protected:
	static void _bind_methods();
};

} // namespace newton

#endif // NEWTON_WORLD_NEWTON_WORLD_H