// File 229: modules/newton/src/joints/newton_gear_joint.cpp
// Gear joint implementation: enforces a fixed ratio between the rotation of
// body A around axis_a and body B around axis_b. Uses angular impulses.

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

	// Compute current angular velocities around the respective axes
	vec3 omegaA = a->get_angular_velocity();
	vec3 omegaB = b->get_angular_velocity();
	real_t speedA = omegaA.dot(worldAxisA);
	real_t speedB = omegaB.dot(worldAxisB);

	// Desired relationship: speedB = ratio * speedA
	// Error in angular velocity coupling
	real_t speedError = speedB - ratio * speedA;

	// Effective inertias along the gear axes
	vec3 invIA = a->get_inverse_inertia_world().xform(worldAxisA);
	vec3 invIB = b->get_inverse_inertia_world().xform(worldAxisB);
	real_t invEffA = invIA.dot(worldAxisA);
	real_t invEffB = invIB.dot(worldAxisB);

	// Apply correction: we want to equalize angular accelerations.
	// A torque around worldAxisA and an opposite-signed torque around worldAxisB
	// (scaled by ratio) will produce the necessary corrections.
	real_t denom = invEffA * ratio * ratio + invEffB;
	if (denom < CMP_EPSILON) return;

	// Desired velocity correction along axis B (-speedError)
	real_t correctionB = -speedError;
	real_t torqueB = correctionB / denom;          // torque on B
	real_t torqueA = -ratio * torqueB;             // torque on A (reaction)

	// Clamp to some maximum? Not specified, but could add max_torque later.

	// Apply angular impulses
	vec3 impulseA = worldAxisA * torqueA;
	vec3 impulseB = worldAxisB * torqueB;
	if (a->get_inverse_mass() > 0.0) a->apply_impulse(vec3(), impulseA);
	if (b->get_inverse_mass() > 0.0) b->apply_impulse(vec3(), impulseB);

	// --- Limit enforcement (accumulated angle) ---
	if (limit_enabled) {
		// Approximate the angle accumulation over recent frames.
		// We can estimate delta angle = current speed * dt, but this is noisy.
		// For robustness, we might integrate the angular velocities.
		// A simple method: track the difference in accumulated rotations by
		// storing the integral of speedB - ratio*speedA over time.
		real_t deltaAngle = (speedB - ratio * speedA) * dt;
		accumulated_angle += deltaAngle;

		// Check limits
		real_t lower = limit_min;
		real_t upper = limit_max;
		if (accumulated_angle < lower || accumulated_angle > upper) {
			// Apply a stronger correction to push back into limits
			real_t limitError = (accumulated_angle < lower) ? (lower - accumulated_angle) : (upper - accumulated_angle);
			real_t correction = limitError / dt;
			real_t torqueB_limit = correction / denom;
			real_t torqueA_limit = -ratio * torqueB_limit;
			vec3 impulseA_limit = worldAxisA * torqueA_limit;
			vec3 impulseB_limit = worldAxisB * torqueB_limit;
			if (a->get_inverse_mass() > 0.0) a->apply_impulse(vec3(), impulseA_limit);
			if (b->get_inverse_mass() > 0.0) b->apply_impulse(vec3(), impulseB_limit);
			// Clamp accumulated angle to the nearest limit to avoid windup
			accumulated_angle = CLAMP(accumulated_angle, lower, upper);
		}
	}
}

} // namespace newton