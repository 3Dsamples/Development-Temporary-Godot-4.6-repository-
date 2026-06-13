// File 223: modules/newton/src/joints/newton_up_vector_joint.cpp
// UpVector joint implementation: constrains a body's local axis to stay
// aligned with a world-space up direction by applying corrective torques.
// Uses a spring-damper model with optional max torque limit.

#include "newton_up_vector_joint.h"
#include "../bodies/newton_body.h"
#include "core/math/vector3.h"
#include "core/math/quaternion.h"
#include "core/math/basis.h"
#include "core/typedefs.h"

namespace newton {

void NewtonUpVectorJoint::solve(NewtonBody *a, NewtonBody *b, real_t dt) {
	if (!enabled || !b) return;
	// The joint applies torque only to the dynamic body (body_b).
	NewtonBody *body = b;
	if (body->get_type() != BodyType::DYNAMIC) return;
	if (body->get_inverse_mass() <= 0.0f) return;

	const mat4 &xB = body->get_transform();

	// World-space direction of the body's local axis.
	vec3 worldAxis = xB.basis.xform(local_axis).normalized();

	// Angle between the world axis and the desired up direction.
	real_t dot = worldAxis.dot(world_up);
	dot = CLAMP(dot, -1.0f, 1.0f);
	real_t angle = Math::acos(dot);
	if (Math::abs(angle) < CMP_EPSILON) return;

	// Torque axis: perpendicular to both vectors (rotate worldAxis into world_up).
	vec3 torqueAxis = worldAxis.cross(world_up);
	real_t axisLen = torqueAxis.length();
	if (axisLen < CMP_EPSILON) {
		// The vectors are collinear (parallel or anti-parallel).
		// If anti-parallel, we need to pick any perpendicular axis.
		if (dot < 0.0f) {
			// Choose an arbitrary perpendicular axis.
			torqueAxis = (Math::abs(world_up.x) < 0.999f) ? world_up.cross(vec3(1, 0, 0)).normalized()
			                                                : world_up.cross(vec3(0, 1, 0)).normalized();
		} else {
			// Perfectly aligned, no torque needed.
			return;
		}
	} else {
		torqueAxis /= axisLen;
	}

	// Spring torque: stiffness * angle (restoring).
	vec3 torque = torqueAxis * stiffness * angle;

	// Damping torque: opposes angular velocity around the torque axis.
	vec3 omega = body->get_angular_velocity();
	real_t omegaAlong = omega.dot(torqueAxis);
	torque -= torqueAxis * damping * omegaAlong;

	// Clamp to maximum allowed torque.
	real_t torqueMag = torque.length();
	if (torqueMag > max_torque && max_torque > 0.0f) {
		torque = torque * (max_torque / torqueMag);
	}

	// Apply as angular impulse.
	vec3 angularImpulse = torque * dt;
	body->apply_impulse(vec3(), angularImpulse);
}

} // namespace newton