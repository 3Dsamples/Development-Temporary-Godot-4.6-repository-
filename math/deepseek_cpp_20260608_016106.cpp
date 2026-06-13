// File 251: modules/newton/src/joints/newton_d6_joint.cpp
// D6 Joint implementation: enforces linear and angular constraints on six
// axes with optional limits, motors, and springs. Each axis is solved
// independently using sequential impulses.

#include "newton_d6_joint.h"
#include "../bodies/newton_body.h"
#include "core/math/vector3.h"
#include "core/math/quaternion.h"
#include "core/math/basis.h"
#include "core/typedefs.h"

namespace newton {

void NewtonD6Joint::solve(NewtonBody *a, NewtonBody *b, real_t dt) {
	if (!enabled || !a || !b) return;
	if (a->get_type() == BodyType::STATIC && b->get_type() == BodyType::STATIC) return;

	const mat4 xA = a->get_transform() * frame_a; // world frame of joint on A
	const mat4 xB = b->get_transform() * frame_b; // world frame of joint on B

	// Solve each linear axis (0=X, 1=Y, 2=Z)
	for (int i = 0; i < 3; ++i) {
		solve_linear_axis(i, a, b, xA, xB, dt);
	}
	// Solve each angular axis (3=rotX, 4=rotY, 5=rotZ)
	for (int i = 3; i < 6; ++i) {
		solve_angular_axis(i, a, b, xA, xB, dt);
	}
}

void NewtonD6Joint::solve_linear_axis(int p_axis, NewtonBody *a, NewtonBody *b,
									  const mat4 &xA, const mat4 &xB, real_t dt) {
	const AxisParams &ax = axes[p_axis];

	// World axis direction (X, Y, or Z of the joint frame A)
	vec3 worldAxis = xA.basis.get_column(p_axis).normalized();
	// Anchor points: origin of each frame
	vec3 anchorA = xA.origin;
	vec3 anchorB = xB.origin;

	// Current relative position along the axis
	vec3 relPos = anchorB - anchorA;
	real_t current = relPos.dot(worldAxis);

	// Velocity correction
	vec3 velA = a->get_linear_velocity() + a->get_angular_velocity().cross(anchorA - a->get_position());
	vec3 velB = b->get_linear_velocity() + b->get_angular_velocity().cross(anchorB - b->get_position());
	real_t relVel = (velB - velA).dot(worldAxis);

	// Compute effective inverse mass along this axis
	real_t invMA = a->get_inverse_mass();
	real_t invMB = b->get_inverse_mass();
	vec3 rA = anchorA - a->get_position();
	vec3 rB = anchorB - b->get_position();
	real_t invEff = invMA + invMB +
		worldAxis.dot(a->get_inverse_inertia_world().xform(rA.cross(worldAxis)).cross(rA)) +
		worldAxis.dot(b->get_inverse_inertia_world().xform(rB.cross(worldAxis)).cross(rB));

	if (invEff < CMP_EPSILON) return;

	// Limit enforcement
	real_t limitError = 0.0;
	if (ax.limited) {
		if (current < ax.lower_limit) limitError = ax.lower_limit - current;
		else if (current > ax.upper_limit) limitError = ax.upper_limit - current;
	}

	// Motor
	real_t motorTarget = 0.0;
	if (ax.motor_enabled) {
		motorTarget = ax.motor_target_velocity;
		// If limits exist and we are within limits, motor can be clamped
	}

	// Spring
	real_t springForce = 0.0;
	real_t damperForce = 0.0;
	if (ax.spring_enabled) {
		springForce = -ax.spring_stiffness * current; // return to zero? relative to initial frame offset? assume initial = 0
		damperForce = -ax.spring_damper * relVel;
	}

	// Desired velocity change (Baumgarte + motor + spring)
	real_t targetSpeed = motorTarget;
	real_t baumgarteVel = 0.0;
	if (ax.limited && Math::abs(limitError) > CMP_EPSILON) {
		baumgarteVel = limitError * 0.5f / dt; // erp = 0.5
	}
	real_t springVel = 0.0;
	if (ax.spring_enabled) {
		springVel = (springForce + damperForce) * invEff; // acceleration to add
	}
	real_t desiredAccel = (targetSpeed - relVel) / dt + baumgarteVel + springVel;

	real_t lambda = desiredAccel * dt / invEff; // impulse = force * dt

	// Clamp motor force
	if (ax.motor_enabled && ax.motor_max_force > 0.0) {
		real_t maxImpulse = ax.motor_max_force * dt;
		lambda = CLAMP(lambda, -maxImpulse, maxImpulse);
	}

	vec3 impulse = worldAxis * lambda;
	if (invMA > 0.0) a->apply_impulse( impulse, anchorA);
	if (invMB > 0.0) b->apply_impulse(-impulse, anchorB);
}

void NewtonD6Joint::solve_angular_axis(int p_axis, NewtonBody *a, NewtonBody *b,
									   const mat4 &xA, const mat4 &xB, real_t dt) {
	const AxisParams &ax = axes[p_axis];
	int angIdx = p_axis - 3; // 0,1,2

	// The constraint enforces that the relative rotation between the two
	// frames around the axis is zero (or within limits).
	mat3 RA = xA.basis;
	mat3 RB = xB.basis;

	// Compute the relative rotation error
	mat3 Rerr = RB * RA.transposed();
	quat qerr(Rerr);
	vec3 rotAxis;
	real_t rotAngle;
	qerr.get_axis_angle(rotAxis, rotAngle);

	// Project onto the desired angular axis
	vec3 worldAxis = xA.basis.get_column(angIdx).normalized();
	real_t angleError = rotAngle * (rotAxis.dot(worldAxis)); // signed projection
	// Clamp angle error to near PI to avoid large impulses
	angleError = CLAMP(angleError, -Math_PI * 0.5f, Math_PI * 0.5f);

	// Relative angular velocity around the axis
	vec3 omegaA = a->get_angular_velocity();
	vec3 omegaB = b->get_angular_velocity();
	real_t relOmega = (omegaB - omegaA).dot(worldAxis);

	// Effective inverse inertia around this axis
	vec3 invIA = a->get_inverse_inertia_world().xform(worldAxis);
	vec3 invIB = b->get_inverse_inertia_world().xform(worldAxis);
	real_t invEff = invIA.dot(worldAxis) + invIB.dot(worldAxis);

	if (invEff < CMP_EPSILON) return;

	// Limit check
	real_t limitError = 0.0;
	if (ax.limited) {
		if (angleError < ax.lower_limit) limitError = ax.lower_limit - angleError;
		else if (angleError > ax.upper_limit) limitError = ax.upper_limit - angleError;
	}

	// Motor
	real_t motorTarget = 0.0;
	if (ax.motor_enabled) {
		motorTarget = ax.motor_target_velocity;
	}

	// Spring
	real_t springTorque = 0.0;
	real_t damperTorque = 0.0;
	if (ax.spring_enabled) {
		springTorque = -ax.spring_stiffness * angleError;
		damperTorque = -ax.spring_damper * relOmega;
	}

	// Desired angular acceleration
	real_t targetAccel = 0.0;
	real_t baumgarteAccel = 0.0;
	if (ax.limited && Math::abs(limitError) > CMP_EPSILON) {
		baumgarteAccel = limitError * 0.5f / dt;
	}
	real_t springAccel = 0.0;
	if (ax.spring_enabled) {
		springAccel = (springTorque + damperTorque) * invEff;
	}
	targetAccel = (motorTarget - relOmega) / dt + baumgarteAccel + springAccel;

	real_t lambda = targetAccel * dt / invEff;

	// Clamp motor torque
	if (ax.motor_enabled && ax.motor_max_force > 0.0) {
		real_t maxImpulse = ax.motor_max_force * dt;
		lambda = CLAMP(lambda, -maxImpulse, maxImpulse);
	}

	vec3 angularImpulse = worldAxis * lambda;
	if (a->get_inverse_mass() > 0.0) a->apply_impulse(vec3(),  angularImpulse);
	if (b->get_inverse_mass() > 0.0) b->apply_impulse(vec3(), -angularImpulse);
}

} // namespace newton