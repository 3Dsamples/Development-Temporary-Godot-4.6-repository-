// File 199: modules/newton/src/joints/newton_kinematic_controller.cpp
// Kinematic controller joint solver: applies proportional-derivative (PD) forces
// and torques to drive a dynamic body towards a target transform. Respects
// max force and torque limits.

#include "newton_kinematic_controller.h"
#include "../bodies/newton_body.h"
#include "core/math/vector3.h"
#include "core/math/quaternion.h"
#include "core/math/basis.h"
#include "core/typedefs.h"

namespace newton {

void NewtonKinematicController::solve(NewtonBody *bodyA, NewtonBody *bodyB, real_t dt) {
	// The controlled body is bodyB (bodyA is often the world / static).
	// We drive bodyB towards target_xform.
	if (!enabled || !bodyB) return;
	if (bodyB->get_type() != BodyType::DYNAMIC) return;

	const mat4 &currentXform = bodyB->get_transform();
	const vec3 &currentPos = currentXform.origin;
	const mat3 &currentRot = currentXform.basis;

	// --- Linear PD controller ---
	vec3 posError = target_xform.origin - currentPos;
	vec3 velError = bodyB->get_linear_velocity();
	// Target velocity is zero (or could be set separately) – we drive to zero velocity at target.
	vec3 linearForce = pos_kp * posError - pos_kd * velError;

	// Clamp force magnitude
	real_t forceMag = linearForce.length();
	if (forceMag > max_force && max_force > 0.0) {
		linearForce = linearForce / forceMag * max_force;
	}

	// Apply linear impulse
	real_t invMass = bodyB->get_inverse_mass();
	if (invMass > 0.0) {
		bodyB->apply_impulse(linearForce * dt, currentPos);
	}

	// --- Angular PD controller ---
	// Compute rotation error: target_rotation * current_rotation^T
	mat3 targetRot = target_xform.basis;
	mat3 rotErrorMat = targetRot * currentRot.transposed();
	quat rotErrorQuat(rotErrorMat);
	vec3 rotAxis;
	real_t rotAngle;
	rotErrorQuat.get_axis_angle(rotAxis, rotAngle);

	// Avoid instability for small errors
	if (Math::abs(rotAngle) > CMP_EPSILON) {
		vec3 angularVel = bodyB->get_angular_velocity();
		vec3 torque = rot_kp * rotAxis * rotAngle - rot_kd * angularVel;

		real_t torqueMag = torque.length();
		if (torqueMag > max_torque && max_torque > 0.0) {
			torque = torque / torqueMag * max_torque;
		}

		// Apply torque as angular impulse
		if (invMass > 0.0) {
			vec3 angularImpulse = torque * dt;
			bodyB->apply_impulse(vec3(), angularImpulse);
		}
	}
}

} // namespace newton