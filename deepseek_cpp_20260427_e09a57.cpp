// File 204: modules/newton/src/collision/newton_contact.h
// Contact point structure used during constraint solving.
// Stores world-space points, normal, penetration, friction, restitution,
// and warm-starting accumulators for sequential impulses.

#ifndef NEWTON_COLLISION_CONTACT_H
#define NEWTON_COLLISION_CONTACT_H

#include "../core/newton_types.h"

namespace newton {

struct NewtonContactPoint {
	body_id body_a;             // body A ID
	body_id body_b;             // body B ID
	vec3 point_a;               // contact point on body A (world)
	vec3 point_b;               // contact point on body B (world)
	vec3 normal;                // normal from B to A
	real_t penetration;         // positive = interpenetration depth
	real_t friction;            // combined friction coefficient
	real_t restitution;         // coefficient of restitution
	// Warm-starting accumulators
	real_t normal_impulse;      // accumulated normal impulse
	vec3 friction_impulse;      // accumulated tangential impulse vector (2D but stored as 3D)
	vec3 friction_impulse_cache; // secondary tangent (unused if only 1D friction? We store 2 tangents)
	vec3 tangent1;              // first tangent direction (world)
	vec3 tangent2;              // second tangent direction (world)

	NewtonContactPoint() :
		body_a(0), body_b(0), point_a(), point_b(), normal(),
		penetration(0.0), friction(0.5), restitution(0.0),
		normal_impulse(0.0), friction_impulse(), friction_impulse_cache(),
		tangent1(), tangent2() {}
};

} // namespace newton

#endif // NEWTON_COLLISION_CONTACT_H