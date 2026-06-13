// File 294: modules/vienna/src/world/vienna_world_physics.h
// ViennaWorldPhysics – high‑performance physics step controller that uses
// Gaia BVH broad‑phase, Gaia GJK narrow‑phase, constraint island builder,
// and ViennaSolver for sequential‑impulse solving.  Handles contact
// generation, warm‑starting, friction, restitution, and joint constraints.

#ifndef VIENNA_WORLD_PHYSICS_H
#define VIENNA_WORLD_PHYSICS_H

#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"
#include "../bodies/vienna_body.h"
#include "../joints/vienna_joint.h"
#include "../materials/vienna_material.h"
#include "../solver/vienna_solver.h"
#include "../solver/vienna_island.h"
#include "../../../gaia/src/collision_detector/broad_phase.h"
#include "../../../gaia/src/collision_detector/narrow_phase.h"  // GJK::collide
#include "../../../gaia/src/collision_detector/contact.h"       // gaia::collision::GJK::Result
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"

namespace vienna {

class ViennaWorldPhysics {
public:
	// Step the entire simulation for time dt, using the passed body/joint/material
	// containers.  All active dynamics are advanced, contacts detected and solved.
	static void step(real_t p_dt,
					 HashMap<body_id, Ref<ViennaBody>> &p_bodies,
					 HashMap<joint_id, Ref<ViennaJoint>> &p_joints,
					 HashMap<material_id, Ref<ViennaMaterial>> &p_materials,
					 const vec3 &p_gravity,
					 int p_solver_iterations,
					 gaia::collision::BroadPhase &p_broad_phase,
					 Ref<ViennaSolver> &p_solver,
					 Ref<ViennaIsland> &p_island_manager,
					 LocalVector<ViennaContactPoint> &p_previous_frame_contacts);  // warm‑start data

private:
	// Apply gravity and external forces, integrate velocities.
	static void apply_forces(real_t dt, HashMap<body_id, Ref<ViennaBody>> &bodies, const vec3 &gravity);
	// Broad‑phase: update AABBs, find overlapping pairs.
	static void detect_collisions(HashMap<body_id, Ref<ViennaBody>> &bodies,
								  gaia::collision::BroadPhase &broad_phase,
								  LocalVector<std::pair<body_id, body_id>> &pairs);
	// Narrow‑phase: for each pair, run GJK, produce contact points.
	static void generate_contacts(const LocalVector<std::pair<body_id, body_id>> &pairs,
								  HashMap<body_id, Ref<ViennaBody>> &bodies,
								  HashMap<material_id, Ref<ViennaMaterial>> &materials,
								  LocalVector<ViennaContactPoint> &contacts);
	// Build islands from bodies, joints, and distribute contacts.
	static void build_islands(const HashMap<body_id, Ref<ViennaBody>> &bodies,
							  const HashMap<joint_id, Ref<ViennaJoint>> &joints,
							  const LocalVector<std::pair<body_id, body_id>> &contact_pairs,
							  LocalVector<ViennaContactPoint> &contacts,
							  Ref<ViennaIsland> &island_manager);
	// Run the solver over all islands.
	static void solve_islands(Ref<ViennaIsland> &island_manager,
							  Ref<ViennaSolver> &solver,
							  real_t dt);
	// Integrate positions.
	static void integrate_positions(real_t dt, HashMap<body_id, Ref<ViennaBody>> &bodies);
};

} // namespace vienna

#endif // VIENNA_WORLD_PHYSICS_H