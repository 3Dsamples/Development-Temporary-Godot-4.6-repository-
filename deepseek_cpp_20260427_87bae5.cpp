// File 229: modules/newton/src/joints/newton_gear_joint.cpp
// Gear joint implementation: constrains the angular velocities of two
// bodies around their local axes to maintain a ratio (angleB = ratio * angleA).
// Applies corrective torques and enforces optional limits on relative angle.

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

	// World axes for each body
	vec3 worldAxisA = xA.basis.xform(axis_a).normalized();
	vec3 worldAxisB = xB.basis.xform(axis_b).normalized();

	// Compute current angular velocities around each axis
	vec3 omegaA = a->get_angular_velocity();
	vec3 omegaB = b->get_angular_velocity();

	real_t omegaA_axial = omegaA.dot(worldAxisA);
	real_t omegaB_axial = omegaB.dot(worldAxisB);

	// Desired gear relation: omegaB - ratio * omegaA = 0
	real_t gearError = omegaB_axial - ratio * omegaA_axial;

	// Compute effective inverse inertia for the gear constraint.
	// For an impulse lambda applied to body A as torque = worldAxisA * (ratio * lambda) 
	// and to body B as torque = -worldAxisB * lambda, the resulting relative velocity change
	// is (axisA^T I_A^{-1} axisA * ratio^2 + axisB^T I_B^{-1} axisB) * lambda.
	vec3 invIA_axisA = a->get_inverse_inertia_world().xform(worldAxisA);
	vec3 invIB_axisB = b->get_inverse_inertia_world().xform(worldAxisB);

	real_t effInertiaA = worldAxisA.dot(invIA_axisA);
	real_t effInertiaB = worldAxisB.dot(invIB_axisB);

	real_t invEffInertia = ratio * ratio * effInertiaA + effInertiaB;

	if (invEffInertia > CMP_EPSILON) {
		// Desired velocity correction (no restitution for gear constraint)
		real_t targetDV = -gearError;

		// Compute the impulse lambda
		real_t lambda = targetDV / invEffInertia;

		// Apply torques: body A receives torque += worldAxisA * (ratio * lambda)
		// body B receives torque -= worldAxisB * lambda
		vec3 torqueImpulseA = worldAxisA * (ratio * lambda);
		vec3 torqueImpulseB = -worldAxisB * lambda;

		// Apply angular impulses
		if (a->get_inverse_mass() > 0.0) a->apply_impulse(vec3(), torqueImpulseA);
		if (b->get_inverse_mass() > 0.0) b->apply_impulse(vec3(), torqueImpulseB);
	}

	// ---- Enforce angle limits (position-level correction) ----
	if (limit_enabled) {
		// Compute current relative angle: we need to integrate the angular velocities to
		// track angleB - ratio * angleA.  The joint stores an accumulated_angle that is
		// updated each substep by (omegaB_axial - ratio * omegaA_axial) * dt.
		accumulated_angle += gearError * dt;

		real_t lower = limit_min;
		real_t upper = limit_max;
		real_t limitError = 0.0f;
		if (accumulated_angle < lower) {
			limitError = lower - accumulated_angle;
		} else if (accumulated_angle > upper) {
			limitError = upper - accumulated_angle;
		}

		if (Math::abs(limitError) > CMP_EPSILON) {
			// Use Baumgarte stabilisation to bring the angle back into limits.
			// Desired angular velocity correction = limitError * erp / dt.
			real_t erp = 0.2f;
			real_t targetOmegaCorrection = limitError * erp / dt;

			// The same effective inertia as above can be used (assuming small correction).
			if (invEffInertia > CMP_EPSILON) {
				real_t lambda_limit = targetOmegaCorrection / invEffInertia;
				vec3 torqueImpulseA_limit = worldAxisA * (ratio * lambda_limit);
				vec3 torqueImpulseB_limit = -worldAxisB * lambda_limit;

				if (a->get_inverse_mass() > 0.0) a->apply_impulse(vec3(), torqueImpulseA_limit);
				if (b->get_inverse_mass() > 0.0) b->apply_impulse(vec3(), torqueImpulseB_limit);
			}
		}
	}
}

} // namespace newton