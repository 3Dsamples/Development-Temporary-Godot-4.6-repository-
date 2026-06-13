// File 247: modules/newton/src/joints/newton_gear_joint.cpp
// Gear joint implementation – couples the rotation of body A around axis_a
// to the rotation of body B around axis_b with ratio: angleB = ratio * angleA.
// Solves velocity errors and enforces relative angle limits.

#include "newton_gear_joint.h"
#include "../bodies/newton_body.h"
#include "core/math/vector3.h"
#include "core/math/quaternion.h"
#include "core/math/basis.h"
#include "core/typedefs.h"

namespace newton {

void NewtonGearJoint::solve(NewtonBody *a, NewtonBody *b, real_t dt) {
	if (!enabled || !a || !b) return;
	if (a->get_type() == BodyType::STATIC && b->get_type() == BodyType::STATIC) return;

	const mat4 &xA = a->get_transform();
	const mat4 &xB = b->get_transform();

	// World axes
	vec3 worldAxisA = xA.basis.xform(axis_a).normalized();
	vec3 worldAxisB = xB.basis.xform(axis_b).normalized();

	// Current angular velocities around the respective axes
	vec3 omegaA = a->get_angular_velocity();
	vec3 omegaB = b->get_angular_velocity();
	real_t speedA = omegaA.dot(worldAxisA);
	real_t speedB = omegaB.dot(worldAxisB);

	// Desired relationship: speedB = ratio * speedA
	real_t speedError = speedB - ratio * speedA;

	// Effective inverse inertias around the gear axes
	vec3 invIA = a->get_inverse_inertia_world().xform(worldAxisA);
	vec3 invIB = b->get_inverse_inertia_world().xform(worldAxisB);
	real_t effA = invIA.dot(worldAxisA);
	real_t effB = invIB.dot(worldAxisB);

	// Total effective inverse inertia for the coupled system
	real_t denom = effA * ratio * ratio + effB;
	if (denom < CMP_EPSILON) return;

	// Torque on B to correct the velocity error
	real_t torqueB = -speedError / denom;
	// Corresponding torque on A (reaction)
	real_t torqueA = -ratio * torqueB;

	// Apply angular impulses
	vec3 impulseA = worldAxisA * torqueA;
	vec3 impulseB = worldAxisB * torqueB;
	if (a->get_inverse_mass() > 0.0) a->apply_impulse(vec3(), impulseA);
	if (b->get_inverse_mass() > 0.0) b->apply_impulse(vec3(), impulseB);

	// --- Angle limits ---
	if (limit_enabled) {
		// Approximate the accumulated relative angle by integrating the velocity difference
		real_t deltaAngle = (speedB - ratio * speedA) * dt;
		accumulated_angle += deltaAngle;

		real_t lower = limit_min;
		real_t upper = limit_max;
		if (accumulated_angle < lower || accumulated_angle > upper) {
			// Determine the limit error
			real_t limitError = (accumulated_angle < lower) ? (lower - accumulated_angle) : (upper - accumulated_angle);
			// Desired velocity correction to eliminate limit error
			real_t correction = limitError * 0.5f / dt;
			// Re‑compute torques for the position correction
			real_t torqueB_limit = correction / denom;
			real_t torqueA_limit = -ratio * torqueB_limit;
			vec3 impulseA_limit = worldAxisA * torqueA_limit;
			vec3 impulseB_limit = worldAxisB * torqueB_limit;
			if (a->get_inverse_mass() > 0.0) a->apply_impulse(vec3(), impulseA_limit);
			if (b->get_inverse_mass() > 0.0) b->apply_impulse(vec3(), impulseB_limit);
			// Clamp accumulated angle to avoid windup
			accumulated_angle = CLAMP(accumulated_angle, lower, upper);
		}
	}
}

} // namespace newton