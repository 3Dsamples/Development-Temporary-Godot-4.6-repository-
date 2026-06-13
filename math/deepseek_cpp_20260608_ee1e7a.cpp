// File 201: modules/newton/src/collision/newton_ccd.h
// Continuous Collision Detection (CCD) for Newton Dynamics.
// Uses conservative advancement (CA) and ray-casts between convex shapes
// to prevent tunneling for fast-moving bodies.

#ifndef NEWTON_COLLISION_CCD_H
#define NEWTON_COLLISION_CCD_H

#include "../core/newton_types.h"
#include "../core/newton_constants.h"
#include "newton_collision.h"
#include "../bodies/newton_body.h"

namespace newton {

class CCDResult {
public:
	bool hit = false;
	real_t toi = 1.0;               // time of impact [0..1]
	vec3 contact_point_a;           // on body A at TOI
	vec3 contact_point_b;           // on body B at TOI
	vec3 normal;                    // from B to A
	real_t depth = 0.0;             // penetration at TOI (optional)
};

class NewtonCCD {
public:
	// Maximum number of conservative-advancement iterations.
	static constexpr int MAX_ITER = 50;
	// Tolerance for TOI resolution.
	static constexpr real_t TOL = 1e-4;

	/**
	 * Perform CCD between two moving rigid bodies using conservative advancement.
	 * Returns the earliest time of impact and the contact information.
	 *
	 * @param bodyA      First body (must have a convex shape).
	 * @param bodyB      Second body.
	 * @param velA       Linear velocity of body A over the step.
	 * @param velB       Linear velocity of body B over the step.
	 * @param dt         Time step.
	 * @param maxDist    Maximum allowed separation to consider a contact.
	 * @return           CCDResult with hit info.
	 */
	static CCDResult compute_ccd(const NewtonBody &bodyA,
								 const NewtonBody &bodyB,
								 const vec3 &velA,
								 const vec3 &velB,
								 real_t dt,
								 real_t maxDist);
};

} // namespace newton

#endif // NEWTON_COLLISION_CCD_H