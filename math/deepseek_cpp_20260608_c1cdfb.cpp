// File 209: modules/newton/src/joints/newton_ball_joint.cpp
// Ball‑and‑socket joint implementation: constrains the pivot points of
// two bodies to remain coincident.  Supports cone limits, twist limits,
// and an optional angular motor.

#include "newton_ball_joint.h"
#include "../bodies/newton_body.h"
#include "core/math/vector3.h"
#include "core/math/quaternion.h"
#include "core/math/basis.h"
#include "core/typedefs.h"

namespace newton {

void NewtonBallJoint::solve(NewtonBody *a, NewtonBody *b, real_t dt) {
	if (!enabled || !a || !b) return;
	if (a->get_type() == BodyType::STATIC && b->get_type() == BodyType::STATIC) return;

	const mat4 &xA = a->get_transform();
	const mat4 &xB = b->get_transform();

	// World pivot on each body
	vec3 worldPivotA = xA.xform(pivot_a);
	vec3 worldPivotB = xB.xform(pivot_a); // symmetric local pivot

	real_t invMassA = a->get_inverse_mass();
	real_t invMassB = b->get_inverse_mass();
	real_t invMassSum = invMassA + invMassB;

	// ---- POVOT CONSTRAINT: keep anchors coincident ----
	vec3 posError = worldPivotB - worldPivotA;
	if (invMassSum > CMP_EPSILON) {
		// Baumgarte stabilisation (0.2)
		real_t erp = 0.2f;
		vec3 correction = posError * (erp / dt);
		vec3 impulse = correction / invMassSum;
		if (invMassA > 0.0) a->apply_impulse( impulse, worldPivotA);
		if (invMassB > 0.0) b->apply_impulse(-impulse, worldPivotB);
	}

	// ---- CONE LIMIT ----
	if (cone_limit_enabled) {
		// Define a reference axis in each body – use local Z as the cone axis.
		vec3 coneAxisA = xA.basis.get_column(2).normalized();
		vec3 coneAxisB = xB.basis.get_column(2).normalized();

		// Angle between the two cone axes
		real_t dotAxes = coneAxisA.dot(coneAxisB);
		real_t angle = Math::acos(CLAMP(dotAxes, -1.0, 1.0));
		if (angle > cone_angle) {
			// Correction direction: perpendicular to both axes (to bring them together)
			vec3 rotAxis = coneAxisB.cross(coneAxisA);
			real_t rotAxisLen = rotAxis.length();
			if (rotAxisLen > CMP_EPSILON) {
				rotAxis /= rotAxisLen;
				real_t error = angle - cone_angle;
				// Apply an angular impulse around rotAxis
				vec3 invIA = a->get_inverse_inertia_world().xform(rotAxis);
				vec3 invIB = b->get_inverse_inertia_world().xform(rotAxis);
				real_t invEffInertia = invIA.dot(rotAxis) + invIB.dot(rotAxis);
				if (invEffInertia > CMP_EPSILON) {
					real_t angularSpeed = error * 0.5f / dt;
					vec3 angularImpulse = rotAxis * (angularSpeed / invEffInertia);
					if (invMassA > 0.0) a->apply_impulse(vec3(),  angularImpulse);
					if (invMassB > 0.0) b->apply_impulse(vec3(), -angularImpulse);
				}
			}
		}
	}

	// ---- TWIST LIMIT ----
	if (twist_limit_enabled) {
		// Compute the twist angle: we choose a reference direction perpendicular
		// to the cone axis (let's use local X) and measure the angle between the
		// two bodies around the cone axis.
		vec3 coneAxisA = xA.basis.get_column(2).normalized();
		vec3 coneAxisB = xB.basis.get_column(2).normalized();
		vec3 refDirA = xA.basis.get_column(0).normalized();
		vec3 refDirB = xB.basis.get_column(0).normalized();
		// Project reference directions onto the plane perpendicular to the cone axis
		refDirA = (refDirA - coneAxisA * refDirA.dot(coneAxisA)).normalized();
		refDirB = (refDirB - coneAxisB * refDirB.dot(coneAxisB)).normalized();
		// Signed angle from refDirA to refDirB around coneAxisA
		vec3 crossRef = refDirB.cross(refDirA);
		real_t dotRef   = refDirB.dot(refDirA);
		real_t twistAngle = Math::atan2(crossRef.dot(coneAxisA), dotRef);

		real_t lower = twist_min;
		real_t upper = twist_max;
		real_t limitError = 0.0f;
		if (twistAngle < lower) {
			limitError = lower - twistAngle;
		} else if (twistAngle > upper) {
			limitError = upper - twistAngle;
		}
		if (Math::abs(limitError) > CMP_EPSILON) {
			// Apply correction around the cone axis
			vec3 rotAxis = coneAxisA;
			vec3 invIA = a->get_inverse_inertia_world().xform(rotAxis);
			vec3 invIB = b->get_inverse_inertia_world().xform(rotAxis);
			real_t invEffInertia = invIA.dot(rotAxis) + invIB.dot(rotAxis);
			if (invEffInertia > CMP_EPSILON) {
				real_t angularSpeed = limitError * 0.5f / dt;
				vec3 angularImpulse = rotAxis * (angularSpeed / invEffInertia);
				if (invMassA > 0.0) a->apply_impulse(vec3(),  angularImpulse);
				if (invMassB > 0.0) b->apply_impulse(vec3(), -angularImpulse);
			}
		}
	}

	// ---- MOTOR ----
	if (motor_enabled) {
		// Motor drives relative angular velocity around the cone axis of body A.
		vec3 coneAxisA = xA.basis.get_column(2).normalized();
		vec3 omegaA = a->get_angular_velocity();
		vec3 omegaB = b->get_angular_velocity();
		real_t currentOmega = (omegaB - omegaA).dot(coneAxisA);
		real_t omegaError = motor_target_vel - currentOmega;

		vec3 invIA = a->get_inverse_inertia_world().xform(coneAxisA);
		vec3 invIB = b->get_inverse_inertia_world().xform(coneAxisA);
		real_t invEffInertia = invIA.dot(coneAxisA) + invIB.dot(coneAxisA);
		if (invEffInertia > CMP_EPSILON) {
			real_t motorAccel = omegaError / dt;
			real_t motorTorque = motorAccel / invEffInertia;
			motorTorque = CLAMP(motorTorque, -motor_max_torque, motor_max_torque);
			vec3 motorImpulse = coneAxisA * motorTorque;
			if (invMassA > 0.0) a->apply_impulse(vec3(), -motorImpulse);
			if (invMassB > 0.0) b->apply_impulse(vec3(),  motorImpulse);
		}
	}
}

} // namespace newton