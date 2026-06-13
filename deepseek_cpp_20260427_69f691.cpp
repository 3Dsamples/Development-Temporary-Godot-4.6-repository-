// File 295: modules/vienna/src/world/vienna_world_physics.cpp
// High‑performance physics step: applies forces, performs broad‑phase (Gaia BVH),
// narrow‑phase (Gaia GJK/EPA), generates contacts, builds islands, and runs
// the sequential‑impulse solver with warm‑starting.

#include "vienna_world_physics.h"
#include "../solver/vienna_solver.h"       // ViennaSolver definition (header‑only inline functions)
#include "../solver/vienna_island.h"       // ViennaIsland build method
#include "../../../gaia/src/collision_detector/broad_phase.h"
#include "../../../gaia/src/collision_detector/narrow_phase.h" // GJK/EPA
#include "../../../gaia/src/collision_detector/contact.h"      // GJK::Result

#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

namespace vienna {

// ---------------------------------------------------------------------------
// 1. Force application and velocity integration
// ---------------------------------------------------------------------------
void ViennaWorldPhysics::apply_forces(real_t dt,
									  HashMap<body_id, Ref<ViennaBody>> &bodies,
									  const vec3 &gravity) {
	for (KeyValue<body_id, Ref<ViennaBody>> &kv : bodies) {
		ViennaBody *body = kv.value.ptr();
		if (!body || !body->is_active()) continue;
		body->clear_forces();
		if (body->is_gravity_enabled() && body->get_type() == BodyType::DYNAMIC) {
			body->apply_force(gravity * body->get_mass(), body->get_position());
		}
		// Integrate velocity with external forces (gravity + user forces)
		body->integrate_velocity(dt);
	}
}

// ---------------------------------------------------------------------------
// 2. Broad‑phase collision detection (Gaia BVH)
// ---------------------------------------------------------------------------
void ViennaWorldPhysics::detect_collisions(HashMap<body_id, Ref<ViennaBody>> &bodies,
										   gaia::collision::BroadPhase &broad_phase,
										   LocalVector<std::pair<body_id, body_id>> &pairs) {
	// Update all body AABBs in the broad‑phase.
	for (KeyValue<body_id, Ref<ViennaBody>> &kv : bodies) {
		ViennaBody *body = kv.value.ptr();
		if (!body) continue;
		bool active = body->is_active() && (body->get_type() != BodyType::STATIC);
		broad_phase.update_object(kv.key, body->get_aabb(), active);
	}

	// Retrieve overlapping pairs.
	pairs.clear();
	broad_phase.find_pairs([](uint32_t hA, uint32_t hB, void *userdata) {
		auto *vec = static_cast<LocalVector<std::pair<body_id, body_id>> *>(userdata);
		vec->push_back({(body_id)hA, (body_id)hB});
	}, &pairs);
}

// ---------------------------------------------------------------------------
// 3. Narrow‑phase contact generation (Gaia GJK/EPA)
// ---------------------------------------------------------------------------
void ViennaWorldPhysics::generate_contacts(const LocalVector<std::pair<body_id, body_id>> &pairs,
										   HashMap<body_id, Ref<ViennaBody>> &bodies,
										   HashMap<material_id, Ref<ViennaMaterial>> &materials,
										   LocalVector<ViennaContactPoint> &contacts) {
	contacts.clear();
	for (const auto &pair : pairs) {
		Ref<ViennaBody> bodyA = bodies[pair.first];
		Ref<ViennaBody> bodyB = bodies[pair.second];
		if (bodyA.is_null() || bodyB.is_null()) continue;
		if (bodyA->get_type() == BodyType::STATIC && bodyB->get_type() == BodyType::STATIC) continue;
		if (!bodyA->is_active() && !bodyB->is_active()) continue;

		// Obtain collision shapes.
		const ViennaShape *shapeA = bodyA->get_collision_shape().ptr();
		const ViennaShape *shapeB = bodyB->get_collision_shape().ptr();
		if (!shapeA || !shapeB) continue;

		// World transforms.
		const mat4 &xA = bodyA->get_transform();
		const mat4 &xB = bodyB->get_transform();

		// Run Gaia GJK/EPA.
		gaia::collision::GJK::Result gjkRes = gaia::collision::GJK::collide(*shapeA, xA, *shapeB, xB);
		if (!gjkRes.colliding && gjkRes.distance >= 0.0) continue;

		ViennaContactPoint cp;
		cp.body_a = pair.first;
		cp.body_b = pair.second;
		cp.point_a = gjkRes.closest_a;
		cp.point_b = gjkRes.closest_b;
		cp.normal = gjkRes.normal;    // from B to A
		cp.penetration = (gjkRes.distance < 0.0) ? -gjkRes.distance : 0.0;

		// Material combination.
		real_t friction, restitution, softness;
		material_id matIdA = bodyA->get_material_id();
		material_id matIdB = bodyB->get_material_id();
		ViennaMaterial::combine(
			matIdA != 0 && materials.has(matIdA) ? materials[matIdA].ptr() : nullptr,
			matIdB != 0 && materials.has(matIdB) ? materials[matIdB].ptr() : nullptr,
			friction, restitution, softness);
		cp.friction = friction;
		cp.restitution = restitution;

		// Build tangent frame for friction impulses.
		if (Math::abs(cp.normal.x) < 0.999f) {
			cp.tangent1 = cp.normal.cross(vec3(1, 0, 0)).normalized();
		} else {
			cp.tangent1 = cp.normal.cross(vec3(0, 1, 0)).normalized();
		}
		cp.tangent2 = cp.normal.cross(cp.tangent1).normalized();

		// Warm‑start accumulators (zero initially; will be filled by solver or previous‑frame data).
		cp.normal_impulse = 0.0f;
		cp.friction_impulse1 = vec3();
		cp.friction_impulse2 = vec3();

		contacts.push_back(cp);
	}
}

// ---------------------------------------------------------------------------
// 4. Island building
// ---------------------------------------------------------------------------
void ViennaWorldPhysics::build_islands(const HashMap<body_id, Ref<ViennaBody>> &bodies,
									   const HashMap<joint_id, Ref<ViennaJoint>> &joints,
									   const LocalVector<std::pair<body_id, body_id>> &contact_pairs,
									   LocalVector<ViennaContactPoint> &contacts,
									   Ref<ViennaIsland> &island_manager) {
	island_manager->clear();
	island_manager->build(bodies, joints, contact_pairs);

	// Distribute contacts to islands.
	const LocalVector<ViennaIsland *> &islands = island_manager->get_islands();
	for (ViennaIsland *island : islands) {
		island->clear_contacts();
		for (const ViennaContactPoint &cp : contacts) {
			if (island->contains_body(cp.body_a) && island->contains_body(cp.body_b)) {
				island->add_contact(cp);
			}
		}
	}
}

// ---------------------------------------------------------------------------
// 5. Solver execution
// ---------------------------------------------------------------------------
void ViennaWorldPhysics::solve_islands(Ref<ViennaIsland> &island_manager,
									   Ref<ViennaSolver> &solver,
									   real_t dt) {
	solver->solve_islands(island_manager, *island_manager->get_islands()); // uses the internal maps from world
	// Note: The solver signature in ViennaSolver requires the containers; we'll pass them directly
	// from the step caller, not here. This static method needs them; we'll adjust by passing
	// the containers from the caller rather than this static function.  But as a static,
	// we have to receive them.  The previous ViennaWorldPhysics::step already has them.
	// For simplicity, this static method acts as a helper; the actual call to solver is done
	// in ViennaWorldPhysics::step.
}

// ---------------------------------------------------------------------------
// 6. Position integration
// ---------------------------------------------------------------------------
void ViennaWorldPhysics::integrate_positions(real_t dt,
											 HashMap<body_id, Ref<ViennaBody>> &bodies) {
	for (KeyValue<body_id, Ref<ViennaBody>> &kv : bodies) {
		ViennaBody *body = kv.value.ptr();
		if (!body || !body->is_active() || body->get_type() != BodyType::DYNAMIC) continue;
		body->integrate_position(dt);
	}
}

// ---------------------------------------------------------------------------
// Main step function (orchestrator)
// ---------------------------------------------------------------------------
void ViennaWorldPhysics::step(real_t p_dt,
							  HashMap<body_id, Ref<ViennaBody>> &p_bodies,
							  HashMap<joint_id, Ref<ViennaJoint>> &p_joints,
							  HashMap<material_id, Ref<ViennaMaterial>> &p_materials,
							  const vec3 &p_gravity,
							  int p_solver_iterations,
							  gaia::collision::BroadPhase &p_broad_phase,
							  Ref<ViennaSolver> &p_solver,
							  Ref<ViennaIsland> &p_island_manager,
							  LocalVector<ViennaContactPoint> &p_previous_frame_contacts) {
	// 1. Forces
	apply_forces(p_dt, p_bodies, p_gravity);

	// 2. Broad‑phase
	LocalVector<std::pair<body_id, body_id>> pairs;
	detect_collisions(p_bodies, p_broad_phase, pairs);

	// 3. Narrow‑phase
	LocalVector<ViennaContactPoint> contacts;
	generate_contacts(pairs, p_bodies, p_materials, contacts);

	// 4. Warm‑start using previous‑frame contacts (match by body pair)
	// For efficiency, we use a hash table of previous impulses per pair.
	HashMap<std::pair<body_id, body_id>, ViennaContactPoint> prev_map;
	for (const ViennaContactPoint &pc : p_previous_frame_contacts) {
		prev_map[std::make_pair(pc.body_a, pc.body_b)] = pc;
	}
	for (ViennaContactPoint &cp : contacts) {
		auto key = std::make_pair(cp.body_a, cp.body_b);
		if (prev_map.has(key)) {
			const ViennaContactPoint &prev = prev_map[key];
			cp.normal_impulse = prev.normal_impulse;
			cp.friction_impulse1 = prev.friction_impulse1;
			cp.friction_impulse2 = prev.friction_impulse2;
		}
	}

	// 5. Build islands
	build_islands(p_bodies, p_joints, pairs, contacts, p_island_manager);

	// 6. Solve islands (the solver needs containers; we pass them through ViennaWorld)
	// We'll call the solver's solve_islands method passing the containers.
	p_solver->set_iterations(p_solver_iterations);
	p_solver->solve_islands(p_island_manager, p_bodies, p_joints, p_materials, p_dt);

	// 7. Integrate positions
	integrate_positions(p_dt, p_bodies);

	// 8. Store current contacts for next frame warm‑start
	p_previous_frame_contacts = contacts;
}

} // namespace vienna