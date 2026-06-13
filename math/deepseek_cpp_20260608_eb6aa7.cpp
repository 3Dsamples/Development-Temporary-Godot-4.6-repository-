// File 182: modules/newton/src/solver/newton_solver.h
// NewtonSolver – the core constraint solver for Newton Dynamics.
// Implements an iterative projected Gauss‑Seidel (PGS) / sequential‑impulse
// solver with Baumgarte stabilisation, friction, restitution, and warm‑starting.
// Operates on islands built by NewtonIsland.

#ifndef NEWTON_SOLVER_NEWTON_SOLVER_H
#define NEWTON_SOLVER_NEWTON_SOLVER_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "../core/newton_types.h"
#include "../core/newton_constants.h"
#include "../bodies/newton_body.h"
#include "../materials/newton_material.h"
#include "../joints/newton_joint.h"

namespace newton {

// Forward
class NewtonIsland;

class NewtonSolver : public RefCounted {
	GDCLASS(NewtonSolver, RefCounted);

public:
	NewtonSolver() : iterations(16), warm_starting(true) {}
	virtual ~NewtonSolver() {}

	void set_iterations(int p_iter) { iterations = CLAMP(p_iter, 1, MAX_SOLVER_ITERATIONS); }
	int get_iterations() const { return iterations; }

	// Main entry: solve all islands extracted from the world.
	void solve_islands(Ref<NewtonIsland> p_island_manager,
					   const HashMap<body_id, Ref<NewtonBody>> &p_bodies,
					   const HashMap<joint_id, Ref<NewtonJoint>> &p_joints,
					   const HashMap<material_id, Ref<NewtonMaterial>> &p_materials,
					   real_t p_dt);

private:
	// Solve a single island (bodies, contacts, joints) using PGS.
	void solve_island(NewtonIsland *island, real_t dt);

	// Solve a single contact constraint (normal + friction).
	void solve_contact(const NewtonContactPoint &contact,
					   NewtonBody *bodyA, NewtonBody *bodyB,
					   const NewtonMaterial *mat,
					   real_t dt,
					   real_t &normal_impulse,
					   vec3 &friction_impulse1,
					   vec3 &friction_impulse2,
					   int iteration);

	// Solve a single joint constraint (position / velocity level).
	void solve_joint(const Ref<NewtonJoint> &joint, real_t dt);

	// Compute effective mass for a unit impulse along `dir` at `world_point`
	// applied to bodies A and B. Returns the inverse effective mass.
	static real_t compute_effective_mass(const NewtonBody *bodyA,
										 const NewtonBody *bodyB,
										 const vec3 &world_point_a,
										 const vec3 &world_point_b,
										 const vec3 &dir);

	// Apply an impulse to two bodies at given points.
	static void apply_impulse(NewtonBody *bodyA, NewtonBody *bodyB,
							  const vec3 &impulse,
							  const vec3 &world_point_a,
							  const vec3 &world_point_b);

	int iterations;
	bool warm_starting;
};

// -----------------------------------------------------------------------
// Contact point (used during solving)
// -----------------------------------------------------------------------
struct NewtonContactPoint {
	body_id body_a;
	body_id body_b;
	vec3 point_a;       // world‑space contact point on A
	vec3 point_b;       // world‑space contact point on B
	vec3 normal;        // from B to A
	real_t penetration; // positive = interpenetration
	real_t friction;
	real_t restitution;
	// Warm‑start accumulators
	real_t normal_impulse;
	vec3 friction_impulse1; // tangential direction 1
	vec3 friction_impulse2; // tangential direction 2
};

} // namespace newton

#endif // NEWTON_SOLVER_NEWTON_SOLVER_H