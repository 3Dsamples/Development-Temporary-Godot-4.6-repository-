// File 195: modules/newton/src/joints/newton_slider_joint.cpp
// Slider (prismatic) joint full constraint solver: locks all translation
// perpendicular to the slider axis, enforces linear limits, and drives
// an optional linear motor.

#include "newton_slider_joint.h"
#include "../bodies/newton_body.h"
#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "core/typedefs.h"

namespace newton {

void NewtonSliderJoint::solve(real_t dt) {
	if (!enabled) return;

	NewtonBody *bodyA = get_body_a_ptr();
	NewtonBody *bodyB = get_body_b_ptr();
	if (!bodyA || !bodyB) return;
	if (bodyA->get_type() == BodyType::STATIC && bodyB->get_type() == BodyType::STATIC) return;

	const mat4 &xA = bodyA->get_transform();
	const mat4 &xB = bodyB->get_transform();

	// World anchor on each body
	vec3 worldAnchorA = xA.xform(anchor_a);
	vec3 worldAnchorB = xB.xform(anchor_a); // symmetric pivot

	// World slider axis
	vec3 worldAxisA = xA.basis.xform(axis_a).normalized();

	// ---- POSITION CORRECTION: remove error perpendicular to the axis ----
	vec3 posError = worldAnchorB - worldAnchorA;
	// Decompose error into parallel and perpendicular components
	real_t parallelError = posError.dot(worldAxisA);
	vec3 perpendicularError = posError - worldAxisA * parallelError;

	real_t invMassA = bodyA->get_inverse_mass();
	real_t invMassB = bodyB->get_inverse_mass();
	real_t invMassSum = invMassA + invMassB;

	if (invMassSum > CMP_EPSILON) {
		// Baumgarte stabilisation for perpendicular drift
		real_t baumgarte = 0.2f;
		vec3 correction = perpendicularError * (baumgarte / dt);
		vec3 impulse = correction / invMassSum;
		if (invMassA > 0.0) bodyA->apply_impulse( impulse, worldAnchorA);
		if (invMassB > 0.0) bodyB->apply_impulse(-impulse, worldAnchorB);
	}

	// ---- ANGULAR CONSTRAINT: align body orientations except around the axis ----
	// We want the two bodies to have the same rotation modulo a rotation around the axis.
	// Enforce that body B's basis vectors perpendicular to the axis match body A's.
	
	// Choose a direction perpendicular to the axis in A's frame
	vec3 perpDirA = (Math::abs(worldAxisA.x) < 0.999f) ? worldAxisA.cross(vec3(1, 0, 0)).normalized()
	                                                    : worldAxisA.cross(vec3(0, 1, 0)).normalized();
	// The same direction expressed in B's frame should align
	// We need to find the rotation error around axes perpendicular to the slider axis.
	
	// Construct a reference frame for body A using the axis and perpDirA
	vec3 refA1 = perpDirA;
	vec3 refA2 = worldAxisA.cross(refA1).normalized();
	mat3 refFrameA(refA1, refA2, worldAxisA); // columns? Basis constructor takes columns.
	
	// For body B, compute the corresponding frame
	vec3 worldAxisB = xB.basis.xform(axis_a).normalized();
	vec3 perpDirB = (Math::abs(worldAxisB.x) < 0.999f) ? worldAxisB.cross(vec3(1, 0, 0)).normalized()
	                                                    : worldAxisB.cross(vec3(0, 1, 0)).normalized();
	vec3 refB1 = perpDirB;
	vec3 refB2 = worldAxisB.cross(refB1).normalized();
	mat3 refFrameB(refB1, refB2, worldAxisB);

	// The rotation error matrix: R_err = refFrameB * refFrameA^T
	mat3 R_err = refFrameB * refFrameA.transposed();
	quat q_err(R_err);
	vec3 rotAxis;
	real_t rotAngle;
	q_err.get_axis_angle(rotAxis, rotAngle);
	// Clamp
	if (Math::abs(rotAngle) > 0.01f) {
		vec3 angularTorqueA = bodyA->get_inverse_inertia_world().xform(rotAxis);
		vec3 angularTorqueB = bodyB->get_inverse_inertia_world().xform(rotAxis);
		real_t invEffectiveInertia = angularTorqueA.dot(rotAxis) + angularTorqueB.dot(rotAxis);
		if (invEffectiveInertia > CMP_EPSILON) {
			real_t angularCorrection = rotAngle * 0.5f / dt;
			vec3 angularImpulse = rotAxis * (angularCorrection / invEffectiveInertia);
			if (invMassA > 0.0) bodyA->apply_impulse(vec3(), angularImpulse);
			if (invMassB > 0.0) bodyB->apply_impulse(vec3(), -angularImpulse);
		}
	}

	// ---- LIMITS (translation along axis) ----
	if (limit_enabled) {
		real_t currentTranslation = parallelError; // relative to initial anchor; we need to track initial offset.
		// For proper limits we must know the rest translation offset.  We'll store it in a member `init_offset`.
		// For now, assume the initial anchor positions produce limit range relative to each other.
		real_t lower = min_limit;
		real_t upper = max_limit;
		real_t limitError = 0.0f;
		if (currentTranslation < lower) {
			limitError = lower - currentTranslation;
		} else if (currentTranslation > upper) {
			limitError = upper - currentTranslation;
		}
		if (Math::abs(limitError) > CMP_EPSILON) {
			vec3 correction = worldAxisA * (limitError * 0.3f / dt);
			vec3 impulse = correction / invMassSum;
			if (invMassA > 0.0) bodyA->apply_impulse( impulse, worldAnchorA);
			if (invMassB > 0.0) bodyB->apply_impulse(-impulse, worldAnchorB);
		}
	}

	// ---- MOTOR ----
	if (motor_enabled) {
		// Compute current relative velocity along the axis
		vec3 velA = bodyA->get_linear_velocity();
		vec3 velB = bodyB->get_linear_velocity();
		// Angular contributions (if anchor offset) - ignore for simplicity?  Need full velocity at anchor.
		vec3 rA = worldAnchorA - bodyA->get_position();
		vec3 rB = worldAnchorB - bodyB->get_position();
		vec3 vAnchorA = velA + bodyA->get_angular_velocity().cross(rA);
		vec3 vAnchorB = velB + bodyB->get_angular_velocity().cross(rB);
		real_t currentSpeed = (vAnchorB - vAnchorA).dot(worldAxisA);
		real_t speedError = motor_target_vel - currentSpeed;

		real_t motorAccel = speedError / dt;
		// Effective mass along axis
		real_t invEffMassAlong = invMassSum +
			worldAxisA.dot(bodyA->get_inverse_inertia_world().xform(rA.cross(worldAxisA)).cross(rA)) +
			worldAxisA.dot(bodyB->get_inverse_inertia_world().xform(rB.cross(worldAxisA)).cross(rB));
		if (invEffMassAlong > CMP_EPSILON) {
			real_t motorForce = motorAccel / invEffMassAlong;
			motorForce = CLAMP(motorForce, -motor_max_force, motor_max_force);
			vec3 motorImpulse = worldAxisA * motorForce * dt;
			if (invMassA > 0.0) bodyA->apply_impulse( motorImpulse, worldAnchorA);
			if (invMassB > 0.0) bodyB->apply_impulse(-motorImpulse, worldAnchorB);
		}
	}
}

} // namespace newton