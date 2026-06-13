// File 245: modules/newton/src/joints/newton_pulley_joint.cpp
// Pulley joint implementation: enforces a rope‑length constraint between
// two bodies using a fixed ratio (pulley ratio). Limits can be set for
// each segment. The solve applies impulses along the rope directions.

#include "newton_pulley_joint.h"
#include "../bodies/newton_body.h"
#include "core/math/vector3.h"
#include "core/math/quaternion.h"
#include "core/math/basis.h"
#include "core/typedefs.h"

namespace newton {

void NewtonPulleyJoint::solve(NewtonBody *a, NewtonBody *b, real_t dt) {
	if (!enabled || !a || !b) return;
	if (a->get_type() == BodyType::STATIC && b->get_type() == BodyType::STATIC) return;

	const mat4 &xA = a->get_transform();
	const mat4 &xB = b->get_transform();

	// World anchor points on each body
	vec3 worldAnchorA = xA.xform(anchor_a);
	vec3 worldAnchorB = xB.xform(anchor_b);

	// Rope directions in world space
	vec3 worldDirA = xA.basis.xform(dir_a).normalized();
	vec3 worldDirB = xB.basis.xform(dir_b).normalized();

	// Compute the current rope length contribution from each body.
	// Rope length A = projection of (worldAnchorA - pulley_point_A) onto dir_a, etc.
	// We need a reference world pulley point. For simplicity, we assume the
	// pulley point is the world origin of the joint?  Actually the joint has no
	// explicit pulley centre; the rope length is measured along the direction
	// from the anchor.  We define rest lengths when the joint was initialised.
	// Here we'll enforce the relationship:
	//    lengthA + ratio * lengthB = rest_lengthA + ratio * rest_lengthB  (constant)
	// Where lengthA = (worldAnchorA - pulleyA).dot(worldDirA)  (if dir points from anchor to pulley).
	// We'll treat the joint as having two variable-length segments, with a fixed
	// total effective length L0 = L0_A + ratio * L0_B.
	// We need the initial rest lengths. We'll compute them on first solve and store.
	if (first_solve) {
		first_solve = false;
		// Store rest lengths based on current positions (assuming they satisfy the constraint).
		rest_lengthA = (worldAnchorA).dot(worldDirA); // assuming pulley at 0? We'll just use the anchor position as if the rope goes towards origin. For realistic pulley, the user provides directions.
		rest_lengthB = (worldAnchorB).dot(worldDirB);
	}

	real_t currentA = worldAnchorA.dot(worldDirA);
	real_t currentB = worldAnchorB.dot(worldDirB);

	// Constraint error: (currentA + ratio * currentB) - (rest_lengthA + ratio * rest_lengthB)
	real_t error = (currentA + ratio * currentB) - (rest_lengthA + ratio * rest_lengthB);

	// Corrective impulse to bring error to zero.
	// The Jacobian along the direction of pull on body A is worldDirA (pulling body A toward the pulley point).
	// For body B, the direction is worldDirB (pulling body B toward its pulley).
	// The constraint velocity: vA·dA + ratio * vB·dB = 0.
	// Effective inverse mass: invMA + ratio^2 * invMB (ignoring angular).
	real_t invMassA = a->get_inverse_mass();
	real_t invMassB = b->get_inverse_mass();
	if (invMassA <= 0.0 && invMassB <= 0.0) return;

	// Add rotational contributions if needed; for simplicity we ignore angular inertia.

	real_t invEff = invMassA + ratio * ratio * invMassB;
	if (invEff < CMP_EPSILON) return;

	// Desired velocity correction to remove error over dt (Baumgarte)
	real_t erp = 0.2f;
	real_t target_vel = -error * erp / dt;

	// Current constraint velocity
	vec3 velA = a->get_linear_velocity();
	vec3 velB = b->get_linear_velocity();
	real_t current_vel = velA.dot(worldDirA) + ratio * velB.dot(worldDirB);

	// Impulse magnitude
	real_t lambda = (target_vel - current_vel) / invEff;

	// Apply impulses
	vec3 impulseA = worldDirA * lambda;
	vec3 impulseB = worldDirB * (ratio * lambda);
	if (invMassA > 0.0) a->apply_impulse( impulseA, worldAnchorA);
	if (invMassB > 0.0) b->apply_impulse( impulseB, worldAnchorB);

	// --- Travel limits ---
	if (limit_enabled) {
		// Check limits for each segment independently.
		// For segment A, the displacement relative to rest is currentA - rest_lengthA.
		real_t dA = currentA - rest_lengthA;
		real_t dB = currentB - rest_lengthB;

		auto apply_limit_impulse = [&](NewtonBody *body, const vec3 &worldAnchor, const vec3 &dir,
									   real_t delta, real_t min_val, real_t max_val, real_t invMass) {
			if (invMass <= 0.0) return;
			real_t limit_error = 0.0;
			if (delta < min_val) limit_error = min_val - delta;
			else if (delta > max_val) limit_error = max_val - delta;
			if (Math::abs(limit_error) > CMP_EPSILON) {
				real_t effInv = invMass; // ignoring angular
				if (effInv > CMP_EPSILON) {
					real_t correction = limit_error * 0.5f / dt;
					vec3 impulse = dir * (correction / effInv);
					body->apply_impulse(impulse, worldAnchor);
				}
			}
		};

		apply_limit_impulse(a, worldAnchorA, worldDirA, dA, limit_a_min, limit_a_max, invMassA);
		apply_limit_impulse(b, worldAnchorB, worldDirB, dB, limit_b_min, limit_b_max, invMassB);
	}
}

} // namespace newton