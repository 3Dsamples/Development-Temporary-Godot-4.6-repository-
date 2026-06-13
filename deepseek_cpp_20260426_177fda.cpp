// File 196: modules/newton/src/joints/newton_universal_joint.cpp
// Universal joint solver: constrains two rotation axes, leaving one free
// rotational DOF. Supports swing limits and motor on the primary axis.

#include "newton_universal_joint.h"
#include "../bodies/newton_body.h"
#include "core/math/vector3.h"
#include "core/math/quaternion.h"
#include "core/math/basis.h"
#include "core/typedefs.h"

namespace newton {

void NewtonUniversalJoint::solve(real_t dt) {
	if (!enabled) return;

	NewtonBody *bodyA = get_body_a_ptr();
	NewtonBody *bodyB = get_body_b_ptr();
	if (!bodyA || !bodyB) return;
	if (bodyA->get_type() == BodyType::STATIC && bodyB->get_type() == BodyType::STATIC) return;

	const mat4 &xA = bodyA->get_transform();
	const mat4 &xB = bodyB->get_transform();

	// World pivot points
	vec3 worldPivotA = xA.xform(pivot_a);
	vec3 worldPivotB = xB.xform(pivot_a); // symmetric local pivot

	// World axes from body A
	vec3 worldAxisA1 = xA.basis.xform(axis_a1).normalized();
	vec3 worldAxisA2 = xA.basis.xform(axis_a2).normalized();

	// Corresponding axes in body B (same local definitions)
	vec3 worldAxisB1 = xB.basis.xform(axis_a1).normalized();
	vec3 worldAxisB2 = xB.basis.xform(axis_a2).normalized();

	// Inv masses
	real_t invMassA = bodyA->get_inverse_mass();
	real_t invMassB = bodyB->get_inverse_mass();
	real_t invMassSum = invMassA + invMassB;

	// ---- PIVOT CONSTRAINT: keep anchors coincident ----
	vec3 posError = worldPivotB - worldPivotA;
	if (invMassSum > CMP_EPSILON) {
		real_t baumgarte = 0.2f;
		vec3 correction = posError * (baumgarte / dt);
		vec3 impulse = correction / invMassSum;
		if (invMassA > 0.0) bodyA->apply_impulse( impulse, worldPivotA);
		if (invMassB > 0.0) bodyB->apply_impulse(-impulse, worldPivotB);
	}

	// ---- AXIS CONSTRAINT: make worldAxisB2 perpendicular to worldAxisA1 ----
	// The universal joint ensures that axis2 of B stays orthogonal to axis1 of A.
	// This is enforced by projecting error and applying an angular correction.
	real_t dotAxis12 = worldAxisB2.dot(worldAxisA1);
	if (Math::abs(dotAxis12) > CMP_EPSILON) {
		vec3 correctionDir = worldAxisA1.cross(worldAxisB2).normalized();
		real_t angleError = Math::asin(dotAxis12); // small angle approximation
		angleError = CLAMP(angleError, -0.5f, 0.5f);

		vec3 invInertiaA = bodyA->get_inverse_inertia_world().xform(correctionDir);
		vec3 invInertiaB = bodyB->get_inverse_inertia_world().xform(correctionDir);
		real_t invEffInertia = invInertiaA.dot(correctionDir) + invInertiaB.dot(correctionDir);
		if (invEffInertia > CMP_EPSILON) {
			real_t angularCorrection = angleError * 0.5f / dt;
			vec3 angularImpulse = correctionDir * (angularCorrection / invEffInertia);
			if (invMassA > 0.0) bodyA->apply_impulse(vec3(),  angularImpulse);
			if (invMassB > 0.0) bodyB->apply_impulse(vec3(), -angularImpulse);
		}
	}

	// Also ensure axis1 of B stays perpendicular to axis2 of A (symmetry)
	real_t dotAxis21 = worldAxisB1.dot(worldAxisA2);
	if (Math::abs(dotAxis21) > CMP_EPSILON) {
		vec3 correctionDir = worldAxisA2.cross(worldAxisB1).normalized();
		real_t angleError = Math::asin(dotAxis21);
		angleError = CLAMP(angleError, -0.5f, 0.5f);

		vec3 invInertiaA = bodyA->get_inverse_inertia_world().xform(correctionDir);
		vec3 invInertiaB = bodyB->get_inverse_inertia_world().xform(correctionDir);
		real_t invEffInertia = invInertiaA.dot(correctionDir) + invInertiaB.dot(correctionDir);
		if (invEffInertia > CMP_EPSILON) {
			real_t angularCorrection = angleError * 0.5f / dt;
			vec3 angularImpulse = correctionDir * (angularCorrection / invEffInertia);
			if (invMassA > 0.0) bodyA->apply_impulse(vec3(),  angularImpulse);
			if (invMassB > 0.0) bodyB->apply_impulse(vec3(), -angularImpulse);
		}
	}

	// ---- SWING LIMIT ----
	if (limit_enabled) {
		// Swing angle: the angle between worldAxisA2 and worldAxisB2 after projecting 
		// out the primary axis (worldAxisA1).  We measure this by constructing a frame.
		// Compute the projection of worldAxisB2 onto the plane perpendicular to worldAxisA1.
		vec3 projB2 = worldAxisB2 - worldAxisA1 * worldAxisB2.dot(worldAxisA1);
		real_t projLen = projB2.length();
		if (projLen > CMP_EPSILON) {
			projB2 /= projLen;
			// Reference direction for body A is worldAxisA2 (which is perpendicular to worldAxisA1)
			// Compute signed angle from worldAxisA2 to projB2 around worldAxisA1
			vec3 crossRef = projB2.cross(worldAxisA2);
			real_t dotRef   = projB2.dot(worldAxisA2);
			real_t swingAngle = Math::atan2(crossRef.dot(worldAxisA1), dotRef);

			real_t lower = swing_min;
			real_t upper = swing_max;
			real_t limitError = 0.0f;
			if (swingAngle < lower) {
				limitError = lower - swingAngle;
			} else if (swingAngle > upper) {
				limitError = upper - swingAngle;
			}

			if (Math::abs(limitError) > CMP_EPSILON) {
				// Correct the swing angle by applying torque around worldAxisA1.
				vec3 invInertiaA = bodyA->get_inverse_inertia_world().xform(worldAxisA1);
				vec3 invInertiaB = bodyB->get_inverse_inertia_world().xform(worldAxisA1);
				real_t invEff = invInertiaA.dot(worldAxisA1) + invInertiaB.dot(worldAxisA1);
				if (invEff > CMP_EPSILON) {
					real_t speed = limitError * 0.3f / dt;
					vec3 limitImpulse = worldAxisA1 * (speed / invEff);
					if (invMassA > 0.0) bodyA->apply_impulse(vec3(),  limitImpulse);
					if (invMassB > 0.0) bodyB->apply_impulse(vec3(), -limitImpulse);
				}
			}
		}
	}

	// ---- MOTOR (around primary axis) ----
	if (motor_enabled) {
		// Relative angular velocity around worldAxisA1
		vec3 omegaA = bodyA->get_angular_velocity();
		vec3 omegaB = bodyB->get_angular_velocity();
		real_t currentOmega = (omegaB - omegaA).dot(worldAxisA1);
		real_t omegaError = motor_target_vel - currentOmega;

		vec3 invInertiaA = bodyA->get_inverse_inertia_world().xform(worldAxisA1);
		vec3 invInertiaB = bodyB->get_inverse_inertia_world().xform(worldAxisA1);
		real_t invEff = invInertiaA.dot(worldAxisA1) + invInertiaB.dot(worldAxisA1);
		if (invEff > CMP_EPSILON) {
			real_t motorAccel = omegaError / dt;
			real_t motorTorque = motorAccel / invEff;
			motorTorque = CLAMP(motorTorque, -motor_max_torque, motor_max_torque);
			vec3 motorImpulse = worldAxisA1 * motorTorque;
			if (invMassA > 0.0) bodyA->apply_impulse(vec3(), -motorImpulse);
			if (invMassB > 0.0) bodyB->apply_impulse(vec3(),  motorImpulse);
		}
	}
}

} // namespace newton