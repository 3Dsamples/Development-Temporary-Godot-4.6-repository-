// File 249: modules/newton/src/bodies/newton_body_damping.h
// Adds linear and angular damping to NewtonBody.  Damping reduces velocity
// over time, mimicking air resistance or joint friction.  Implemented as
// a multiplier applied each substep after force integration.

#ifndef NEWTON_BODIES_DAMPING_H
#define NEWTON_BODIES_DAMPING_H

#include "newton_body.h"

namespace newton {

// Damping parameters can be stored directly in NewtonBody; this header
// provides the getters/setters and integration logic that NewtonWorld
// calls during the velocity integration phase.

class NewtonBodyDamping {
public:
	static void set_linear_damping(NewtonBody *p_body, real_t p_damping) {
		if (p_body) p_body->set_linear_damping(CLAMP(p_damping, 0.0, 1.0));
	}
	static real_t get_linear_damping(const NewtonBody *p_body) {
		return p_body ? p_body->get_linear_damping() : 0.0;
	}

	static void set_angular_damping(NewtonBody *p_body, real_t p_damping) {
		if (p_body) p_body->set_angular_damping(CLAMP(p_damping, 0.0, 1.0));
	}
	static real_t get_angular_damping(const NewtonBody *p_body) {
		return p_body ? p_body->get_angular_damping() : 0.0;
	}

	// Apply damping to linear and angular velocities (called each substep).
	static void apply_damping(NewtonBody *p_body, real_t dt) {
		if (!p_body || p_body->get_type() != BodyType::DYNAMIC) return;
		real_t lin_damp = 1.0 - p_body->get_linear_damping() * dt;
		real_t ang_damp = 1.0 - p_body->get_angular_damping() * dt;
		lin_damp = CLAMP(lin_damp, 0.0, 1.0);
		ang_damp = CLAMP(ang_damp, 0.0, 1.0);
		p_body->set_linear_velocity(p_body->get_linear_velocity() * lin_damp);
		p_body->set_angular_velocity(p_body->get_angular_velocity() * ang_damp);
	}
};

} // namespace newton

#endif // NEWTON_BODIES_DAMPING_H