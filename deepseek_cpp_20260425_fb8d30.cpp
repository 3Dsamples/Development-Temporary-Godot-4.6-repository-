// File 23: modules/gaia/src/pbd/volume_constraint.h

#ifndef GAIA_PBD_VOLUME_CONSTRAINT_H
#define GAIA_PBD_VOLUME_CONSTRAINT_H

#include "../framework/constraint.h"
#include "../framework/body.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"

namespace gaia {

// Forward declaration
class PBDSolver;

/**
 * XPBD volume preservation constraint for a tetrahedron.
 *
 * Enforces that the signed volume of the tetrahedron matches the rest volume.
 * Indices: idx0, idx1, idx2, idx3 are the four vertices.
 */
class VolumeConstraint : public Constraint {
public:
	VolumeConstraint() :
		Constraint(),
		soft_body(nullptr),
		idx0(-1), idx1(-1), idx2(-1), idx3(-1),
		rest_volume(0.0),
		lambda(0.0) {
		type = VOLUME;
	}

	void set_body(SoftBody *p_body) { soft_body = p_body; }
	SoftBody *get_body() const { return soft_body; }

	void set_indices(int p_idx0, int p_idx1, int p_idx2, int p_idx3) {
		idx0 = p_idx0;
		idx1 = p_idx1;
		idx2 = p_idx2;
		idx3 = p_idx3;
	}

	// Compute rest volume from current positions (call after body setup).
	void init_from_positions() {
		ERR_FAIL_COND(!soft_body);
		ERR_FAIL_INDEX(idx0, soft_body->positions.size());
		ERR_FAIL_INDEX(idx1, soft_body->positions.size());
		ERR_FAIL_INDEX(idx2, soft_body->positions.size());
		ERR_FAIL_INDEX(idx3, soft_body->positions.size());

		const Vector3 &p0 = soft_body->positions[idx0];
		const Vector3 &p1 = soft_body->positions[idx1];
		const Vector3 &p2 = soft_body->positions[idx2];
		const Vector3 &p3 = soft_body->positions[idx3];

		Vector3 e1 = p1 - p0;
		Vector3 e2 = p2 - p0;
		Vector3 e3 = p3 - p0;
		rest_volume = Math::abs(e1.cross(e2).dot(e3)) / 6.0;
	}

	virtual void solve_position(real_t dt) override {
		ERR_FAIL_COND(!soft_body);
		ERR_FAIL_INDEX(idx0, soft_body->positions.size());
		ERR_FAIL_INDEX(idx1, soft_body->positions.size());
		ERR_FAIL_INDEX(idx2, soft_body->positions.size());
		ERR_FAIL_INDEX(idx3, soft_body->positions.size());

		Vector3 &p0 = soft_body->positions[idx0];
		Vector3 &p1 = soft_body->positions[idx1];
		Vector3 &p2 = soft_body->positions[idx2];
		Vector3 &p3 = soft_body->positions[idx3];

		// Compute current volume
		Vector3 e1 = p1 - p0;
		Vector3 e2 = p2 - p0;
		Vector3 e3 = p3 - p0;
		real_t current_volume = e1.cross(e2).dot(e3) / 6.0;

		if (Math::abs(current_volume) < CMP_EPSILON)
			return; // degenerate, can't correct

		// Constraint function: C = current_volume - rest_volume
		real_t C = current_volume - rest_volume;

		// Gradients w.r.t. vertices (derived from volume formula)
		// Volume = (1/6) * ( (p1-p0) x (p2-p0) ) · (p3-p0)
		Vector3 grad0 = (p2 - p1).cross(p3 - p1) / 6.0;
		Vector3 grad1 = (p2 - p0).cross(p3 - p0) / 6.0;
		Vector3 grad2 = (p3 - p0).cross(p1 - p0) / 6.0;
		Vector3 grad3 = (p1 - p0).cross(p2 - p0) / 6.0;

		// Uniform inverse mass per vertex
		real_t inv_mass = 1.0 / soft_body->get_total_mass();
		real_t w_sum = inv_mass * (grad0.length_squared() +
								   grad1.length_squared() +
								   grad2.length_squared() +
								   grad3.length_squared());

		// XPBD: compute delta_lambda
		real_t alpha = compliance;
		real_t alpha_tilde = alpha / (dt * dt);
		real_t delta_lambda = -(C + alpha_tilde * lambda) / (w_sum + alpha_tilde);
		lambda += delta_lambda;

		// Apply position corrections
		p0 -= inv_mass * delta_lambda * grad0;
		p1 -= inv_mass * delta_lambda * grad1;
		p2 -= inv_mass * delta_lambda * grad2;
		p3 -= inv_mass * delta_lambda * grad3;
	}

	virtual void solve_velocity(real_t dt) override {
		// No velocity-level correction for volume constraint.
	}

	PBDSolver *solver;

private:
	SoftBody *soft_body;
	int idx0, idx1, idx2, idx3;
	real_t rest_volume;
	real_t lambda;
};

} // namespace gaia

#endif // GAIA_PBD_VOLUME_CONSTRAINT_H