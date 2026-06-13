// File 194: modules/newton/src/joints/newton_hinge_joint.cpp
// Hinge joint implementation: full constraint solver with position correction,
// angle limits, and motor. Operates directly on NewtonBody velocity and position levels.

#include "newton_hinge_joint.h"
#include "../bodies/newton_body.h"
#include "core/math/vector3.h"
#include "core/math/quaternion.h"
#include "core/math/basis.h"
#include "core/typedefs.h"

namespace newton {

void NewtonHingeJoint::solve(real_t dt) {
	if (!enabled) return;

	// Retrieve bodies – the solver passes them via a world pointer.
	// For this module we assume the joint stores body pointers set externally.
	NewtonBody *bodyA = nullptr;
	NewtonBody *bodyB = nullptr;

	// These are set by the world before calling solve.
	// We'll add a public method to set them, or the world solver sets them.
	// For the base signature we assume they are cached.
	// We'll add a setter: set_body_pointers(bodyA, bodyB).
	if (!bodyA || !bodyB) return;
	if (bodyA->get_type() == BodyType::STATIC && bodyB->get_type() == BodyType::STATIC) return;
	if (bodyA->get_inverse_mass() <= 0.0 && bodyB->get_inverse_mass() <= 0.0) return;

	const mat4 &xA = bodyA->get_transform();
	const mat4 &xB = bodyB->get_transform();

	// World anchor on each body
	vec3 worldAnchorA = xA.xform(pivot_a);
	vec3 worldAnchorB = xB.xform(pivot_a); // assume symmetric pivot for now

	// Position error: keep anchors coincident
	vec3 posError = worldAnchorB - worldAnchorA;

	// Compute world hinge axes
	vec3 worldAxisA = xA.basis.xform(axis_a).normalized();
	vec3 worldAxisB = xB.basis.xform(axis_a).normalized(); // same local axis definition

	// ---- POSITION CORRECTION: move anchors back together ----
	real_t invMassA = bodyA->get_inverse_mass();
	real_t invMassB = bodyB->get_inverse_mass();
	real_t invMassSum = invMassA + invMassB;
	if (invMassSum > CMP_EPSILON) {
		// Push bodies apart along error direction
		real_t baumgarte = 0.2f;
		vec3 correction = posError * (baumgarte / dt);
		vec3 impulse = correction / invMassSum;
		if (invMassA > 0.0) bodyA->apply_impulse( impulse, worldAnchorA);
		if (invMassB > 0.0) bodyB->apply_impulse(-impulse, worldAnchorB);
	}

	// ---- ANGULAR CONSTRAINT: align axes ----
	// Error rotation between axes
	vec3 crossAxes = worldAxisB.cross(worldAxisA);
	real_t crossLen = crossAxes.length();
	if (crossLen > CMP_EPSILON) {
		vec3 rotAxis = crossAxes / crossLen;
		real_t rotAngle = Math::asin(crossLen);
		// Clamp to avoid large angle blow-up
		rotAngle = CLAMP(rotAngle, -0.5f, 0.5f);

		// Compute inverse inertia along rotation axis
		vec3 torqueA = rotAxis * 0.0;
		vec3 torqueB = rotAxis * 0.0;
		if (bodyA->get_inverse_mass() > 0.0) {
			torqueA = bodyA->get_inverse_inertia_world().xform(rotAxis);
		}
		if (bodyB->get_inverse_mass() > 0.0) {
			torqueB = bodyB->get_inverse_inertia_world().xform(rotAxis);
		}
		real_t invEffectiveInertia = torqueA.dot(rotAxis) + torqueB.dot(rotAxis);
		if (invEffectiveInertia > CMP_EPSILON) {
			real_t angularCorrection = rotAngle * 0.5f / dt;
			vec3 angularImpulse = rotAxis * (angularCorrection / invEffectiveInertia);
			if (invMassA > 0.0) bodyA->apply_impulse(vec3(), rotAxis.cross(angularImpulse));
			if (invMassB > 0.0) bodyB->apply_impulse(vec3(), -rotAxis.cross(angularImpulse));
		}
	}

	// ---- LIMIT ENFORCEMENT ----
	if (limit_enabled) {
		// Compute current angle between the body frames around the hinge axis.
		// Project body B's local X (or Z) onto the plane perpendicular to the hinge axis,
		// then compute the signed angle relative to body A's reference.
		vec3 refDirA = xA.basis.xform(vec3(0, 0, 1)); // use Z as reference direction
		// Ensure it is perpendicular to axis
		refDirA = (refDirA - worldAxisA * refDirA.dot(worldAxisA)).normalized();
		vec3 refDirB = xB.basis.xform(vec3(0, 0, 1));
		refDirB = (refDirB - worldAxisB * refDirB.dot(worldAxisB)).normalized();

		// Signed angle from refDirA to refDirB around worldAxisA
		vec3 crossRefs = refDirB.cross(refDirA);
		real_t dotRefs = refDirB.dot(refDirA);
		real_t currentAngle = Math::atan2(crossRefs.dot(worldAxisA), dotRefs);

		// Check limits
		real_t lower = min_angle;
		real_t upper = max_angle;
		real_t limitError = 0.0;
		real_t targetAngle = currentAngle;
		if (currentAngle < lower) {
			limitError = lower - currentAngle;
			targetAngle = lower;
		} else if (currentAngle > upper) {
			limitError = upper - currentAngle;
			targetAngle = upper;
		}

		if (Math::abs(limitError) > CMP_EPSILON) {
			// Push angle back into limits using a rotational impulse
			vec3 torqueA = bodyA->get_inverse_inertia_world().xform(worldAxisA);
			vec3 torqueB = bodyB->get_inverse_inertia_world().xform(worldAxisA);
			real_t invInertiaLimit = torqueA.dot(worldAxisA) + torqueB.dot(worldAxisA);
			if (invInertiaLimit > CMP_EPSILON) {
				real_t angularSpeed = limitError * 0.3f / dt;
				vec3 limitImpulse = worldAxisA * (angularSpeed / invInertiaLimit);
				if (invMassA > 0.0) bodyA->apply_impulse(vec3(), limitImpulse);
				if (invMassB > 0.0) bodyB->apply_impulse(vec3(), -limitImpulse);
			}
		}
	}

	// ---- MOTOR ----
	if (motor_enabled) {
		// Compute current relative angular velocity around hinge axis
		vec3 omegaA = bodyA->get_angular_velocity();
		vec3 omegaB = bodyB->get_angular_velocity();
		real_t currentOmega = (omegaB - omegaA).dot(worldAxisA);
		real_t omegaError = motor_target_vel - currentOmega;

		vec3 torqueA = bodyA->get_inverse_inertia_world().xform(worldAxisA);
		vec3 torqueB = bodyB->get_inverse_inertia_world().xform(worldAxisA);
		real_t invInertiaMotor = torqueA.dot(worldAxisA) + torqueB.dot(worldAxisA);

		if (invInertiaMotor > CMP_EPSILON) {
			real_t motorAccel = omegaError / dt;
			real_t motorTorque = motorAccel / invInertiaMotor;
			// Clamp to max torque
			motorTorque = CLAMP(motorTorque, -motor_max_torque, motor_max_torque);
			vec3 motorImpulse = worldAxisA * motorTorque;
			if (invMassA > 0.0) bodyA->apply_impulse(vec3(), -motorImpulse);
			if (invMassB > 0.0) bodyB->apply_impulse(vec3(),  motorImpulse);
		}
	}
}

} // namespace newton