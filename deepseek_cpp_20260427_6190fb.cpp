// File 219: modules/newton/src/collision/newton_ccd.cpp
// Full continuous collision detection using conservative advancement.
// Iteratively advances the time of impact by the distance to the closest
// point divided by the relative velocity magnitude, until the shapes touch.

#include "newton_ccd.h"
#include "newton_collision.h"
#include "../bodies/newton_body.h"
#include "../../../gaia/src/collision_detector/narrow_phase.h"   // GJK::collide from Gaia
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"

namespace newton {

CCDResult NewtonCCD::compute_ccd(const NewtonBody &bodyA,
								 const NewtonBody &bodyB,
								 const vec3 &velA,
								 const vec3 &velB,
								 real_t dt,
								 real_t maxDist) {
	CCDResult result;
	result.hit = false;
	result.toi = 1.0;

	// Retrieve the collision shapes from the bodies.
	// For CCD we need the convex shapes.  We'll assume the body stores a single
	// collision shape via a method get_collision_shape() that we'll add to NewtonBody.
	const NewtonCollision *shapeA = bodyA.get_collision_shape();
	const NewtonCollision *shapeB = bodyB.get_collision_shape();
	if (!shapeA || !shapeB) return result;

	// Current transforms
	mat4 xformA0 = bodyA.get_transform();
	mat4 xformB0 = bodyB.get_transform();

	// Relative velocity (B relative to A)
	vec3 relVel = (velB - velA);

	real_t relSpeed = relVel.length();
	if (relSpeed < CMP_EPSILON) {
		// No relative motion: perform static GJK at t=0
		GJK::Result staticRes = GJK::collide(*shapeA, xformA0, *shapeB, xformB0);
		if (staticRes.colliding || staticRes.distance < 0.0) {
			result.hit = true;
			result.toi = 0.0;
			result.contact_point_a = staticRes.closest_a;
			result.contact_point_b = staticRes.closest_b;
			result.normal = staticRes.normal;
		}
		return result;
	}

	// Conservative advancement loop
	real_t t = 0.0;
	real_t lambda = 0.0;
	int iter = 0;
	while (iter < MAX_ITER) {
		// Compute current transforms at time t
		mat4 xA = xformA0;
		xA.origin = xformA0.origin + velA * (t * dt);
		mat4 xB = xformB0;
		xB.origin = xformB0.origin + velB * (t * dt);

		// Compute closest distance between shapes using GJK
		GJK::Result gjkRes = GJK::collide(*shapeA, xA, *shapeB, xB);
		real_t dist = gjkRes.distance;
		if (gjkRes.colliding || dist < 0.0) {
			// Penetration detected; refine toi by binary search or return.
			result.hit = true;
			result.toi = t;
			result.contact_point_a = gjkRes.closest_a;
			result.contact_point_b = gjkRes.closest_b;
			result.normal = gjkRes.normal;
			// Optionally refine toi by shrinking step (binary search)
			// For simplicity we accept this time.
			return result;
		}

		// If distance is less than maxDist, we consider it a contact.
		if (dist < maxDist) {
			result.hit = true;
			result.toi = t;
			result.contact_point_a = gjkRes.closest_a;
			result.contact_point_b = gjkRes.closest_b;
			result.normal = (shapeB->get_support(gjkRes.normal, xB) - shapeA->get_support(-gjkRes.normal, xA)).normalized();
			return result;
		}

		// Advance time by the safe fraction: dt_step = dist / relSpeed
		real_t dt_step = dist / relSpeed;
		real_t t_new = t + dt_step;
		if (t_new >= 1.0) {
			// No impact within the time step
			break;
		}
		t = t_new;
		iter++;
	}

	// If we exited loop, no hit within the time step.
	result.toi = 1.0;
	return result;
}

} // namespace newton