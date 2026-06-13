// File 21: modules/gaia/src/pbd/distance_constraint.h

#ifndef GAIA_PBD_DISTANCE_CONSTRAINT_H
#define GAIA_PBD_DISTANCE_CONSTRAINT_H

#include "../framework/constraint.h"
#include "../framework/body.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"

namespace gaia {

// Forward declaration
class PBDSolver;

/**
 * XPBD distance constraint between two particles (vertices) of a soft body.
 *
 * Satisfies |p2 - p1| == rest_length, using XPBD compliance.
 */
class DistanceConstraint : public Constraint {
public:
	DistanceConstraint() :
		Constraint(),
		soft_body(nullptr),
		idx0(-1),
		idx1(-1),
		rest_length(0.0),
		lambda(0.0) {
		type = DISTANCE;
	}

	void set_body(SoftBody *p_body) { soft_body = p_body; }
	SoftBody *get_body() const { return soft_body; }

	void set_indices(int p_idx0, int p_idx1) {
		idx0 = p_idx0;
		idx1 = p_idx1;
	}
	void set_rest_length(real_t p_len) { rest_length = MAX(p_len, 0.0); }

	// Initialize from current positions (call after body setup).
	void init_from_positions() {
		ERR_FAIL_COND(!soft_body);
		ERR_FAIL_INDEX(idx0, soft_body->positions.size());
		ERR_FAIL_INDEX(idx1, soft_body->positions.size());
		rest_length = soft_body->positions[idx0].distance_to(soft_body->positions[idx1]);
	}

	virtual void solve_position(real_t dt) override {
		ERR_FAIL_COND(!soft_body);
		ERR_FAIL_INDEX(idx0, soft_body->positions.size());
		ERR_FAIL_INDEX(idx1, soft_body->positions.size());

		Vector3 &p0 = soft_body->positions[idx0];
		Vector3 &p1 = soft_body->positions[idx1];
		Vector3 dir = p1 - p0;
		real_t dist = dir.length();
		if (dist < CMP_EPSILON) {
			// avoid division by zero, skip
			return;
		}
		Vector3 grad = dir / dist; // normalized

		real_t inv_mass0 = 1.0 / soft_body->get_total_mass(); // simplified: uniform mass
		real_t inv_mass1 = inv_mass0;
		real_t w_sum = inv_mass0 + inv_mass1;

		// XPBD stiffness from compliance
		real_t alpha = compliance;
		real_t alpha_tilde = alpha / (dt * dt);
		real_t C = dist - rest_length;

		real_t delta_lambda = -(C + alpha_tilde * lambda) / (w_sum + alpha_tilde);
		lambda += delta_lambda;

		Vector3 correction = delta_lambda * grad;
		p0 -= inv_mass0 * correction;
		p1 += inv_mass1 * correction;
	}

	virtual void solve_velocity(real_t dt) override {
		// Optional damping (simplified: no separate damping pass)
		if (damping <= 0.0) return;
		ERR_FAIL_COND(!soft_body);
		ERR_FAIL_INDEX(idx0, soft_body->velocities.size());
		ERR_FAIL_INDEX(idx1, soft_body->velocities.size());

		Vector3 &v0 = soft_body->velocities[idx0];
		Vector3 &v1 = soft_body->velocities[idx1];
		Vector3 n = (soft_body->positions[idx1] - soft_body->positions[idx0]).normalized();
		real_t rel_v = (v1 - v0).dot(n);
		if (rel_v > 0.0) return; // moving apart, no damping

		real_t inv_mass = 1.0 / soft_body->get_total_mass();
		real_t w_sum = 2.0 * inv_mass;
		real_t impulse = damping * rel_v / w_sum;
		v0 += inv_mass * impulse * n;
		v1 -= inv_mass * impulse * n;
	}

	// For solver access
	PBDSolver *solver;

private:
	SoftBody *soft_body;
	int idx0;
	int idx1;
	real_t rest_length;
	real_t lambda; // XPBD lagrange multiplier
};

} // namespace gaia

#endif // GAIA_PBD_DISTANCE_CONSTRAINT_H