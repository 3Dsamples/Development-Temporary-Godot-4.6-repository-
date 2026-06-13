// File 214: modules/newton/src/joints/newton_fixed_joint.cpp
// Fixed joint implementation: constrains the relative transform between two
// bodies to remain constant. Uses position-level correction (Baumgarte) and
// velocity-level damping. Supports breakable force/torque thresholds.

#include "newton_fixed_joint.h"
#include "../bodies/newton_body.h"
#include "core/math/vector3.h"
#include "core/math/quaternion.h"
#include "core/math/basis.h"
#include "core/typedefs.h"

namespace newton {

void NewtonFixedJoint::solve(NewtonBody *a, NewtonBody *b, real_t dt) {
	if (!enabled || !a || !b) return;
	if (a->get_type() == BodyType::STATIC && b->get_type() == BodyType::STATIC) return;

	const mat4 &xA = a->get_transform();
	const mat4 &xB = b->get_transform();

	// Desired relative transform: B = A * relative_xform  (in world space)
	// So target world transform of B is A * relative_xform.
	mat4 targetB = xA * relative_xform;

	// Current relative error
	vec3 posError = targetB.origin - xB.origin;

	// Rotational error: targetB.basis * currentB.basis^T
	mat3 rotErrorMat = targetB.basis * xB.basis.transposed();
	quat rotErrorQuat(rotErrorMat);
	vec3 rotAxis;
	real_t rotAngle;
	rotErrorQuat.get_axis_angle(rotAxis, rotAngle);

	real_t invMassA = a->get_inverse_mass();
	real_t invMassB = b->get_inverse_mass();
	real_t invMassSum = invMassA + invMassB;

	// Gather the current constraint force / torque to check breakability.
	vec3 totalForce(0, 0, 0);
	vec3 totalTorque(0, 0, 0);

	// ---- POSITION CORRECTION (translation) ----
	if (invMassSum > CMP_EPSILON) {
		// Baumgarte stabilisation for position error
		real_t erp = 0.2f;
		vec3 correction = posError * (erp / dt);
		vec3 impulse = correction / invMassSum;
		// Apply impulses - we apply at the body's origins for simplicity.
		if (invMassA > 0.0) a->apply_impulse( impulse, xA.origin);
		if (invMassB > 0.0) b->apply_impulse(-impulse, xB.origin);
		totalForce += impulse / dt;  // force = impulse/dt
	}

	// ---- ANGULAR CORRECTION ----
	if (Math::abs(rotAngle) > CMP_EPSILON) {
		vec3 invIA = a->get_inverse_inertia_world().xform(rotAxis);
		vec3 invIB = b->get_inverse_inertia_world().xform(rotAxis);
		real_t invEffInertia = invIA.dot(rotAxis) + invIB.dot(rotAxis);
		if (invEffInertia > CMP_EPSILON) {
			real_t angularSpeed = rotAngle * 0.5f / dt; // factor 0.5 to avoid overshoot
			vec3 angularImpulse = rotAxis * (angularSpeed / invEffInertia);
			if (invMassA > 0.0) a->apply_impulse(vec3(),  angularImpulse);
			if (invMassB > 0.0) b->apply_impulse(vec3(), -angularImpulse);
			totalTorque += angularImpulse / dt;
		}
	}

	// ---- BREAKABLE CHECK ----
	if (breakable) {
		// Check if force or torque exceeds limits
		real_t forceMag = totalForce.length();
		real_t torqueMag = totalTorque.length();
		if (forceMag > break_force || torqueMag > break_torque) {
			enabled = false;  // joint breaks
		}
	}
}

} // namespace newton