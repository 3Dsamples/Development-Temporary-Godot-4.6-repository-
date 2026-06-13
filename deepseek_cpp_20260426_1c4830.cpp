// File 198: modules/newton/src/joints/newton_corkscrew_joint.cpp
// Corkscrew joint solver: enforces the screw constraint coupling translation
// and rotation along a common axis, with limits and motor.

#include "newton_corkscrew_joint.h"
#include "../bodies/newton_body.h"
#include "core/math/vector3.h"
#include "core/math/quaternion.h"
#include "core/math/basis.h"
#include "core/typedefs.h"

namespace newton {

void NewtonCorkscrewJoint::solve(NewtonBody *bodyA, NewtonBody *bodyB, real_t dt) {
	if (!enabled || !bodyA || !bodyB) return;
	if (bodyA->get_type() == BodyType::STATIC && bodyB->get_type() == BodyType::STATIC) return;

	const mat4 &xA = bodyA->get_transform();
	const mat4 &xB = bodyB->get_transform();

	// World anchor on each body (symmetric local pivot)
	vec3 worldAnchorA = xA.xform(pivot_a);
	vec3 worldAnchorB = xB.xform(pivot_a);

	// World screw axis from body A
	vec3 worldAxis = xA.basis.xform(axis_a).normalized();

	real_t invMassA = bodyA->get_inverse_mass();
	real_t invMassB = bodyB->get_inverse_mass();
	real_t invMassSum = invMassA + invMassB;

	// ---- LATERAL POSITION CONSTRAINT: anchors may not separate perpendicular to axis ----
	vec3 posError = worldAnchorB - worldAnchorA;
	real_t parallelError = posError.dot(worldAxis);
	vec3 perpendicularError = posError - worldAxis * parallelError;

	if (invMassSum > CMP_EPSILON) {
		real_t baumgarte = 0.2f;
		vec3 correction = perpendicularError * (baumgarte / dt);
		vec3 impulse = correction / invMassSum;
		if (invMassA > 0.0) bodyA->apply_impulse( impulse, worldAnchorA);
		if (invMassB > 0.0) bodyB->apply_impulse(-impulse, worldAnchorB);
	}

	// ---- ANGULAR CONSTRAINT: align rotations except around the axis ----
	vec3 perpDirA = (Math::abs(worldAxis.x) < 0.999f) ? worldAxis.cross(vec3(1, 0, 0)).normalized()
	                                                    : worldAxis.cross(vec3(0, 1, 0)).normalized();
	vec3 refA1 = perpDirA;
	vec3 refA2 = worldAxis.cross(refA1).normalized();
	mat3 refFrameA(refA1, refA2, worldAxis);

	vec3 worldAxisB = xB.basis.xform(axis_a).normalized();
	vec3 perpDirB = (Math::abs(worldAxisB.x) < 0.999f) ? worldAxisB.cross(vec3(1, 0, 0)).normalized()
	                                                    : worldAxisB.cross(vec3(0, 1, 0)).normalized();
	vec3 refB1 = perpDirB;
	vec3 refB2 = worldAxisB.cross(refB1).normalized();
	mat3 refFrameB(refB1, refB2, worldAxisB);

	mat3 R_err = refFrameB * refFrameA.transposed();
	quat q_err(R_err);
	vec3 rotAxis;
	real_t rotAngle;
	q_err.get_axis_angle(rotAxis, rotAngle);
	if (Math::abs(rotAngle) > 0.01f) {
		vec3 angularTorqueA = bodyA->get_inverse_inertia_world().xform(rotAxis);
		vec3 angularTorqueB = bodyB->get_inverse_inertia_world().xform(rotAxis);
		real_t invEffectiveInertia = angularTorqueA.dot(rotAxis) + angularTorqueB.dot(rotAxis);
		if (invEffectiveInertia > CMP_EPSILON) {
			real_t angularCorrection = rotAngle * 0.5f / dt;
			vec3 angularImpulse = rotAxis * (angularCorrection / invEffectiveInertia);
			if (invMassA > 0.0) bodyA->apply_impulse(vec3(),  angularImpulse);
			if (invMassB > 0.0) bodyB->apply_impulse(vec3(), -angularImpulse);
		}
	}

	// ---- SCREW COUPLING: translation = pitch * angle ----
	// Compute the current angle between the two bodies around the axis.
	vec3 refDirA = perpDirA;
	vec3 refDirB = perpDirB;
	vec3 crossRefs = refDirB.cross(refDirA);
	real_t dotRefs = refDirB.dot(refDirA);
	real_t currentAngle = Math::atan2(crossRefs.dot(worldAxis), dotRefs);

	// Desired translation from the screw relation: translation = pitch * angle
	real_t desiredTranslation = pitch * currentAngle;
	real_t translationError = parallelError - desiredTranslation;

	// Correct translation along the axis using impulses at the anchor points
	if (Math::abs(translationError) > CMP_EPSILON) {
		// Effective mass along axis
		vec3 rA = worldAnchorA - bodyA->get_position();
		vec3 rB = worldAnchorB - bodyB->get_position();
		real_t invEffMassAlong = invMassSum +
			worldAxis.dot(bodyA->get_inverse_inertia_world().xform(rA.cross(worldAxis)).cross(rA)) +
			worldAxis.dot(bodyB->get_inverse_inertia_world().xform(rB.cross(worldAxis)).cross(rB));
		if (invEffMassAlong > CMP_EPSILON) {
			real_t baumgarteScrew = 0.3f;
			vec3 correctionVec = worldAxis * (translationError * baumgarteScrew / dt);
			vec3 screwImpulse = correctionVec / invEffMassAlong;
			if (invMassA > 0.0) bodyA->apply_impulse( screwImpulse, worldAnchorA);
			if (invMassB > 0.0) bodyB->apply_impulse(-screwImpulse, worldAnchorB);
		}
	}

	// Also apply angular correction to couple translation error to rotation (dual)
	// DeltaAngle = translationError / pitch (if pitch > 0)
	if (Math::abs(pitch) > CMP_EPSILON) {
		real_t angleError = translationError / pitch;
		vec3 angularTorqueA = bodyA->get_inverse_inertia_world().xform(worldAxis);
		vec3 angularTorqueB = bodyB->get_inverse_inertia_world().xform(worldAxis);
		real_t invEff = angularTorqueA.dot(worldAxis) + angularTorqueB.dot(worldAxis);
		if (invEff > CMP_EPSILON) {
			real_t angularSpeed = angleError * 0.3f / dt;
			vec3 angularImpulse = worldAxis * (angularSpeed / invEff);
			if (invMassA > 0.0) bodyA->apply_impulse(vec3(), -angularImpulse);
			if (invMassB > 0.0) bodyB->apply_impulse(vec3(),  angularImpulse);
		}
	}

	// ---- TRANSLATION LIMITS ----
	if (limit_enabled) {
		real_t currentTrans = parallelError;
		real_t lower = trans_min;
		real_t upper = trans_max;
		real_t limitError = 0.0f;
		if (currentTrans < lower) {
			limitError = lower - currentTrans;
		} else if (currentTrans > upper) {
			limitError = upper - currentTrans;
		}
		if (Math::abs(limitError) > CMP_EPSILON) {
			vec3 rA = worldAnchorA - bodyA->get_position();
			vec3 rB = worldAnchorB - bodyB->get_position();
			real_t invEffMassAlong = invMassSum +
				worldAxis.dot(bodyA->get_inverse_inertia_world().xform(rA.cross(worldAxis)).cross(rA)) +
				worldAxis.dot(bodyB->get_inverse_inertia_world().xform(rB.cross(worldAxis)).cross(rB));
			if (invEffMassAlong > CMP_EPSILON) {
				vec3 correctionVec = worldAxis * (limitError * 0.3f / dt);
				vec3 limitImpulse = correctionVec / invEffMassAlong;
				if (invMassA > 0.0) bodyA->apply_impulse( limitImpulse, worldAnchorA);
				if (invMassB > 0.0) bodyB->apply_impulse(-limitImpulse, worldAnchorB);
			}
		}
	}

	// ---- MOTOR ----
	if (motor_enabled) {
		vec3 rA = worldAnchorA - bodyA->get_position();
		vec3 rB = worldAnchorB - bodyB->get_position();
		vec3 velA = bodyA->get_linear_velocity();
		vec3 velB = bodyB->get_linear_velocity();
		vec3 omegaA = bodyA->get_angular_velocity();
		vec3 omegaB = bodyB->get_angular_velocity();
		vec3 vAnchorA = velA + omegaA.cross(rA);
		vec3 vAnchorB = velB + omegaB.cross(rB);

		// If the motor drives rotation around the axis, we apply torque.
		real_t currentOmega = (omegaB - omegaA).dot(worldAxis);
		real_t omegaError = motor_target_vel - currentOmega;
		vec3 angularTorqueA = bodyA->get_inverse_inertia_world().xform(worldAxis);
		vec3 angularTorqueB = bodyB->get_inverse_inertia_world().xform(worldAxis);
		real_t invEffRot = angularTorqueA.dot(worldAxis) + angularTorqueB.dot(worldAxis);
		if (invEffRot > CMP_EPSILON) {
			real_t motorAccel = omegaError / dt;
			real_t motorTorqueVal = motorAccel / invEffRot;
			motorTorqueVal = CLAMP(motorTorqueVal, -motor_max_torque, motor_max_torque);
			vec3 motorImpulseRot = worldAxis * motorTorqueVal;
			if (invMassA > 0.0) bodyA->apply_impulse(vec3(), -motorImpulseRot);
			if (invMassB > 0.0) bodyB->apply_impulse(vec3(),  motorImpulseRot);
		}

		// If the motor also drives translation (or the translation limit motor), we apply force.
		if (motor_max_force > 0.0f) {
			real_t currentSpeed = (vAnchorB - vAnchorA).dot(worldAxis);
			real_t speedError = motor_target_vel * pitch - currentSpeed; // target translation speed
			real_t invEffMassAlong = invMassSum +
				worldAxis.dot(bodyA->get_inverse_inertia_world().xform(rA.cross(worldAxis)).cross(rA)) +
				worldAxis.dot(bodyB->get_inverse_inertia_world().xform(rB.cross(worldAxis)).cross(rB));
			if (invEffMassAlong > CMP_EPSILON) {
				real_t motorForceVal = speedError / (invEffMassAlong * dt);
				motorForceVal = CLAMP(motorForceVal, -motor_max_force, motor_max_force);
				vec3 motorImpulseForce = worldAxis * motorForceVal;
				if (invMassA > 0.0) bodyA->apply_impulse( motorImpulseForce, worldAnchorA);
				if (invMassB > 0.0) bodyB->apply_impulse(-motorImpulseForce, worldAnchorB);
			}
		}
	}
}

} // namespace newton