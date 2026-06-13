// File 221: modules/newton/src/world/newton_world_contact.cpp
// Additional NewtonWorld methods: body/joint/material ID retrieval,
// contact pair management, and narrow‑phase contact generation using
// Gaia's GJK/EPA narrow‑phase.

#include "newton_world.h"
#include "../bodies/newton_body.h"
#include "../joints/newton_joint.h"
#include "../materials/newton_material.h"
#include "../collision/newton_collision.h"
#include "../solver/newton_solver.h"
#include "../solver/newton_island.h"

// Gaia narrow‑phase (GJK)
#include "../../../gaia/src/collision_detector/narrow_phase.h"

// Godot
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

namespace newton {

// ---------------------------------------------------------------------------
// Body / joint / material ID iterators
// ---------------------------------------------------------------------------

LocalVector<body_id> NewtonWorld::get_body_ids() const {
	LocalVector<body_id> ids;
	for (const KeyValue<body_id, Ref<NewtonBody>> &kv : bodies) {
		ids.push_back(kv.key);
	}
	return ids;
}

int NewtonWorld::get_body_count() const {
	return bodies.size();
}

LocalVector<joint_id> NewtonWorld::get_joint_ids() const {
	LocalVector<joint_id> ids;
	for (const KeyValue<joint_id, Ref<NewtonJoint>> &kv : joints) {
		ids.push_back(kv.key);
	}
	return ids;
}

LocalVector<material_id> NewtonWorld::get_material_ids() const {
	LocalVector<material_id> ids;
	for (const KeyValue<material_id, Ref<NewtonMaterial>> &kv : materials) {
		ids.push_back(kv.key);
	}
	return ids;
}

// ---------------------------------------------------------------------------
// Contact pair storage (broad‑phase output)
// ---------------------------------------------------------------------------

void NewtonWorld::add_contact_pair(body_id a, body_id b) {
	// Avoid duplicates by ensuring a < b
	if (a > b) SWAP(a, b);
	contact_pairs.push_back({a, b});
}

void NewtonWorld::clear_contact_pairs() {
	contact_pairs.clear();
}

const LocalVector<std::pair<body_id, body_id>> &NewtonWorld::get_contact_pairs() const {
	return contact_pairs;
}

// ---------------------------------------------------------------------------
// Narrow‑phase contact generation for a single pair
// ---------------------------------------------------------------------------

void NewtonWorld::generate_contacts_for_pair(body_id a, body_id b) {
	Ref<NewtonBody> bodyA = get_body(a);
	Ref<NewtonBody> bodyB = get_body(b);
	if (bodyA.is_null() || bodyB.is_null()) return;
	// Only collide dynamic/kinematic pairs where at least one is dynamic
	if (bodyA->get_type() == BodyType::STATIC && bodyB->get_type() == BodyType::STATIC) return;
	if (!bodyA->is_active() && !bodyB->is_active()) return;

	// Retrieve collision shapes from the bodies.
	// NewtonBody must expose a method get_collision_shape() – we assume it exists.
	const NewtonCollision *shapeA = bodyA->get_collision_shape();
	const NewtonCollision *shapeB = bodyB->get_collision_shape();
	if (!shapeA || !shapeB) return;

	// Transforms
	const mat4 &xA = bodyA->get_transform();
	const mat4 &xB = bodyB->get_transform();

	// Run Gaia's GJK/EPA to get contact information
	gaia::collision::GJK::Result gjkRes = gaia::collision::GJK::collide(*shapeA, xA, *shapeB, xB);
	if (!gjkRes.colliding && gjkRes.distance >= 0.0) return; // not touching

	// Build a contact point
	NewtonContactPoint cp;
	cp.body_a = a;
	cp.body_b = b;
	cp.point_a = gjkRes.closest_a;
	cp.point_b = gjkRes.closest_b;
	cp.normal = gjkRes.normal; // from B to A
	cp.penetration = (gjkRes.distance < 0.0) ? -gjkRes.distance : 0.0;
	// Material properties: use combined friction/restitution from material pair.
	// For simplicity, we use the first valid material we find.
	real_t friction = 0.5;
	real_t restitution = 0.0;
	material_id matIdA = bodyA->get_material_id();
	material_id matIdB = bodyB->get_material_id();
	if (material_idA != 0) {
		Ref<NewtonMaterial> mat = get_material(matIdA);
		if (mat.is_valid()) {
			friction = mat->get_dynamic_friction();
			restitution = mat->get_restitution();
		}
	}
	if (matIdB != 0) {
		Ref<NewtonMaterial> mat = get_material(material_idB);
		if (mat.is_valid()) {
			friction = MAX(friction, mat->get_dynamic_friction());
			restitution = MAX(restitution, mat->get_restitution());
		}
	}
	cp.friction = friction;
	cp.restitution = restitution;

	// Compute tangent vectors for friction
	if (Math::abs(cp.normal.x) < 0.999) {
		cp.tangent1 = cp.normal.cross(vec3(1, 0, 0)).normalized();
	} else {
		cp.tangent1 = cp.normal.cross(vec3(0, 1, 0)).normalized();
	}
	cp.tangent2 = cp.normal.cross(cp.tangent1).normalized();

	// Warm-start accumulators start at zero
	cp.normal_impulse = 0.0;
	cp.friction_impulse = vec3();
	cp.friction_impulse_cache = vec3();

	generated_contacts.push_back(cp);
}

} // namespace newton