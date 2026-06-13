// File 232: modules/newton/src/world/newton_world_islands.cpp
// Contains the broad‑phase contact detection, narrow‑phase contact generation,
// and island building routines for NewtonWorld.  These functions are called
// every physics step to produce the set of islands the solver will process.

#include "newton_world.h"

#include "../bodies/newton_body.h"
#include "../joints/newton_joint.h"
#include "../materials/newton_material.h"
#include "../collision/newton_collision.h"
#include "../collision/newton_contact.h"
#include "../solver/newton_solver.h"
#include "../solver/newton_island.h"

// Gaia broad‑phase and narrow‑phase (GJK/EPA)
#include "../../../gaia/src/collision_detector/broad_phase.h"
#include "../../../gaia/src/collision_detector/narrow_phase.h"

#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

namespace newton {

// ---------------------------------------------------------------------------
// detect_collisions
//   - Update AABBs of all active bodies in the Gaia broad‑phase.
//   - Query overlapping pairs.
//   - Generate contact manifolds for each pair using Gaia GJK.
// ---------------------------------------------------------------------------
void NewtonWorld::detect_collisions() {
	// Update AABBs in broad‑phase
	for (KeyValue<body_id, Ref<NewtonBody>> &kv : bodies) {
		NewtonBody *body = kv.value.ptr();
		if (!body) continue;
		bool active = body->is_active() && (body->get_type() != BodyType::STATIC);
		broad_phase.update_object(kv.key, body->get_aabb(), active);
	}

	// Query overlapping pairs
	contact_pairs.clear();
	broad_phase.find_pairs([](uint32_t hA, uint32_t hB, void *userdata) {
		auto *vec = static_cast<LocalVector<std::pair<body_id, body_id>>*>(userdata);
		vec->push_back({(body_id)hA, (body_id)hB});
	}, &contact_pairs);

	// Generate contacts using narrow‑phase
	generated_contacts.clear();
	for (const auto &pair : contact_pairs) {
		generate_contacts_for_pair(pair.first, pair.second);
	}
}

// ---------------------------------------------------------------------------
// generate_contacts_for_pair
//   Runs GJK on the two bodies and if colliding, adds one contact point
//   to the generated_contacts list.  Friction and restitution are taken
//   from the body's material (or defaults).
// ---------------------------------------------------------------------------
void NewtonWorld::generate_contacts_for_pair(body_id a, body_id b) {
	Ref<NewtonBody> bodyA = get_body(a);
	Ref<NewtonBody> bodyB = get_body(b);
	if (bodyA.is_null() || bodyB.is_null()) return;
	if (bodyA->get_type() == BodyType::STATIC && bodyB->get_type() == BodyType::STATIC) return;
	if (!bodyA->is_active() && !bodyB->is_active()) return;

	const NewtonCollision *shapeA = bodyA->get_collision_shape().ptr();
	const NewtonCollision *shapeB = bodyB->get_collision_shape().ptr();
	if (!shapeA || !shapeB) return;

	const mat4 &xA = bodyA->get_transform();
	const mat4 &xB = bodyB->get_transform();

	gaia::collision::GJK::Result gjkRes = gaia::collision::GJK::collide(*shapeA, xA, *shapeB, xB);
	if (!gjkRes.colliding && gjkRes.distance >= 0.0) return;

	NewtonContactPoint cp;
	cp.body_a = a;
	cp.body_b = b;
	cp.point_a = gjkRes.closest_a;
	cp.point_b = gjkRes.closest_b;
	cp.normal = gjkRes.normal;
	cp.penetration = (gjkRes.distance < 0.0) ? -gjkRes.distance : 0.0;

	// Pick friction/restitution from materials (use max of the two)
	real_t friction = DEFAULT_FRICTION;
	real_t restitution = DEFAULT_RESTITUTION;
	if (bodyA->get_material_id() != 0) {
		Ref<NewtonMaterial> mat = get_material(bodyA->get_material_id());
		if (mat.is_valid()) {
			friction = mat->get_dynamic_friction();
			restitution = mat->get_restitution();
		}
	}
	if (bodyB->get_material_id() != 0) {
		Ref<NewtonMaterial> mat = get_material(bodyB->get_material_id());
		if (mat.is_valid()) {
			friction = MAX(friction, mat->get_dynamic_friction());
			restitution = MAX(restitution, mat->get_restitution());
		}
	}
	cp.friction = friction;
	cp.restitution = restitution;

	// Build tangent frame for friction impulses
	if (Math::abs(cp.normal.x) < 0.999f) {
		cp.tangent1 = cp.normal.cross(vec3(1, 0, 0)).normalized();
	} else {
		cp.tangent1 = cp.normal.cross(vec3(0, 1, 0)).normalized();
	}
	cp.tangent2 = cp.normal.cross(cp.tangent1).normalized();

	cp.normal_impulse = 0.0f;
	cp.friction_impulse = vec3();
	cp.friction_impulse_cache = vec3();

	generated_contacts.push_back(cp);
}

// ---------------------------------------------------------------------------
// build_islands
//   Passes all active bodies, joints, and contact pairs to the island
//   manager.  The manager uses union‑find to partition connected components.
//   After building, each island holds its own contacts (transferred from
//   generated_contacts) and joints.
// ---------------------------------------------------------------------------
void NewtonWorld::build_islands() {
	// Reset island manager state
	island_manager->clear();

	// Send bodies and joints to the island builder
	island_manager->build(bodies, joints, contact_pairs);

	// Distribute generated contacts to the appropriate islands.
	// We iterate over islands and, for each contact, find the island that
	// contains both bodies and add it.
	const LocalVector<NewtonIsland *> &islands = island_manager->get_islands();
	for (NewtonIsland *island : islands) {
		island->clear_contacts();
		for (const NewtonContactPoint &cp : generated_contacts) {
			if (island->contains_body(cp.body_a) && island->contains_body(cp.body_b)) {
				island->add_contact(cp);
			}
		}
	}
}

} // namespace newton