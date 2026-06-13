// File 270: modules/vienna/src/world/vienna_world.h
// ViennaWorld – the main physics world that owns bodies, joints, materials,
// cloths, and particle systems. It drives physics stepping, collision detection,
// and constraint solving, integrated with Gaia's BVH broad‑phase.

#ifndef VIENNA_WORLD_VIENNA_WORLD_H
#define VIENNA_WORLD_VIENNA_WORLD_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"

// Gaia broad‑phase integration
#include "../../../gaia/src/collision_detector/broad_phase.h"

namespace vienna {

// Forward declarations
class ViennaBody;
class ViennaJoint;
class ViennaMaterial;
class ViennaCloth;
class ViennaParticleSystem;
class ViennaSolver;
class ViennaIsland;

class ViennaWorld : public RefCounted {
	GDCLASS(ViennaWorld, RefCounted);

public:
	ViennaWorld();
	virtual ~ViennaWorld();

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
	body_id create_body(const Ref<ViennaBody> &p_body);
	void destroy_body(body_id p_id);
	Ref<ViennaBody> get_body(body_id p_id) const;
	int get_body_count() const;
	LocalVector<body_id> get_body_ids() const;

	// --- Joint management ---
	joint_id create_joint(const Ref<ViennaJoint> &p_joint);
	void destroy_joint(joint_id p_id);
	Ref<ViennaJoint> get_joint(joint_id p_id) const;
	LocalVector<joint_id> get_joint_ids() const;

	// --- Material management ---
	material_id create_material(const Ref<ViennaMaterial> &p_material);
	void destroy_material(material_id p_id);
	Ref<ViennaMaterial> get_material(material_id p_id) const;
	LocalVector<material_id> get_material_ids() const;

	// --- Cloth management ---
	cloth_id create_cloth(const Ref<ViennaCloth> &p_cloth);
	void destroy_cloth(cloth_id p_id);
	Ref<ViennaCloth> get_cloth(cloth_id p_id) const;
	LocalVector<cloth_id> get_cloth_ids() const;

	// --- Particle system management ---
	cloth_id create_particle_system(const Ref<ViennaParticleSystem> &p_system);
	void destroy_particle_system(cloth_id p_id);
	Ref<ViennaParticleSystem> get_particle_system(cloth_id p_id) const;

	// --- Broad‑phase access ---
	gaia::collision::BroadPhase &get_broad_phase() { return broad_phase; }

	// --- Contact reporting ---
	void set_contact_callback(Callable p_callback);
	Callable get_contact_callback() const;

private:
	// Main steps of the simulation pipeline
	void apply_forces(real_t dt);
	void detect_collisions();
	void build_islands();
	void solve_islands(real_t dt);
	void integrate(real_t dt);
	void update_sleep_state();

	// Step cloth and particles
	void step_cloths(real_t dt);
	void step_particle_systems(real_t dt);

	// Internal containers
	HashMap<body_id, Ref<ViennaBody>> bodies;
	HashMap<joint_id, Ref<ViennaJoint>> joints;
	HashMap<material_id, Ref<ViennaMaterial>> materials;
	HashMap<cloth_id, Ref<ViennaCloth>> cloths;
	HashMap<cloth_id, Ref<ViennaParticleSystem>> particle_systems;

	// Next IDs
	body_id next_body_id = 1;
	joint_id next_joint_id = 1;
	material_id next_material_id = 1;
	cloth_id next_cloth_id = 1;

	// World state
	vec3 gravity = vec3(0, DEFAULT_GRAVITY, 0);
	real_t world_time = 0.0;
	int solver_iterations = DEFAULT_SOLVER_ITERATIONS;
	SolverMethod solver_method = SolverMethod::SEQUENTIAL_IMPULSES;
	int sleep_frames = DEFAULT_SLEEP_FRAMES;
	real_t sleep_linear_threshold = DEFAULT_SLEEP_LINEAR;
	real_t sleep_angular_threshold = DEFAULT_SLEEP_ANGULAR;

	// Broad‑phase (Gaia)
	gaia::collision::BroadPhase broad_phase;

	// Solver and island instances
	Ref<ViennaSolver> solver;
	Ref<ViennaIsland> island_manager;

	// Per‑body sleep counter
	HashMap<body_id, int> sleep_counters;

	// Contact tracking
	LocalVector<std::pair<body_id, body_id>> contact_pairs;
	Callable contact_callback;

protected:
	static void _bind_methods();
};

} // namespace vienna

#endif // VIENNA_WORLD_VIENNA_WORLD_H