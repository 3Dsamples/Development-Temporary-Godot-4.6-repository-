// File 22: modules/gaia/src/pbd/bending_constraint.h

#ifndef GAIA_PBD_BENDING_CONSTRAINT_H
#define GAIA_PBD_BENDING_CONSTRAINT_H

#include "../framework/constraint.h"
#include "../framework/body.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"

namespace gaia {

// Forward declaration
class PBDSolver;

/**
 * XPBD bending constraint for a pair of triangles sharing an edge.
 *
 * Enforces the dihedral angle between the two triangles.
 * Indices: p1,p2 are the shared edge, p3 and p4 are the opposite vertices.
 */
class BendingConstraint : public Constraint {
public:
	BendingConstraint() :
		Constraint(),
		soft_body(nullptr),
		idx1(-1), idx2(-1), idx3(-1), idx4(-1),
		rest_angle(0.0),
		lambda(0.0) {
		type = BENDING;
	}

	void set_body(SoftBody *p_body) { soft_body = p_body; }
	SoftBody *get_body() const { return soft_body; }

	void set_indices(int p_idx1, int p_idx2, int p_idx3, int p_idx4) {
		idx1 = p_idx1;
		idx2 = p_idx2;
		idx3 = p_idx3;
		idx4 = p_idx4;
	}

	// Compute rest angle from current positions.
	void init_from_positions() {
		ERR_FAIL_COND(!soft_body);
		ERR_FAIL_INDEX(idx1, soft_body->positions.size());
		ERR_FAIL_INDEX(idx2, soft_body->positions.size());
		ERR_FAIL_INDEX(idx3, soft_body->positions.size());
		ERR_FAIL_INDEX(idx4, soft_body->positions.size());

		const Vector3 &p1 = soft_body->positions[idx1];
		const Vector3 &p2 = soft_body->positions[idx2];
		const Vector3 &p3 = soft_body->positions[idx3];
		const Vector3 &p4 = soft_body->positions[idx4];

		Vector3 n1 = (p2 - p1).cross(p3 - p1);
		Vector3 n2 = (p2 - p1).cross(p4 - p1);
		real_t len_n1 = n1.length();
		real_t len_n2 = n2.length();
		if (len_n1 < CMP_EPSILON || len_n2 < CMP_EPSILON) {
			rest_angle = 0.0;
			return;
		}
		n1 /= len_n1;
		n2 /= len_n2;
		real_t cos_angle = CLAMP(n1.dot(n2), -1.0, 1.0);
		rest_angle = Math::acos(cos_angle);
	}

	virtual void solve_position(real_t dt) override {
		ERR_FAIL_COND(!soft_body);
		ERR_FAIL_INDEX(idx1, soft_body->positions.size());
		ERR_FAIL_INDEX(idx2, soft_body->positions.size());
		ERR_FAIL_INDEX(idx3, soft_body->positions.size());
		ERR_FAIL_INDEX(idx4, soft_body->positions.size());

		Vector3 &p1 = soft_body->positions[idx1];
		Vector3 &p2 = soft_body->positions[idx2];
		Vector3 &p3 = soft_body->positions[idx3];
		Vector3 &p4 = soft_body->positions[idx4];

		// Compute current normals
		Vector3 n1 = (p2 - p1).cross(p3 - p1);
		Vector3 n2 = (p2 - p1).cross(p4 - p1);
		real_t len_n1 = n1.length();
		real_t len_n2 = n2.length();
		if (len_n1 < CMP_EPSILON || len_n2 < CMP_EPSILON) return;

		n1 /= len_n1;
		n2 /= len_n2;

		// Cosine dihedral constraint: C = acos(n1·n2) - rest_angle
		real_t d = CLAMP(n1.dot(n2), -1.0, 1.0);
		// To avoid singularities when angle ≈ 0 or π, we clamp derivative.
		real_t angle = Math::acos(d);
		real_t C = angle - rest_angle;
		if (Math::is_nan(C) || Math::is_inf(C)) return;

		// Compute gradients (using the formula from "PBD: The Dihedral Angle Constraint")
		Vector3 e = p2 - p1;
		real_t len_e = e.length();
		if (len_e < CMP_EPSILON) return;
		e /= len_e;

		Vector3 grad1, grad2, grad3, grad4;
		// Simplified: use approximate gradients from bending energy.
		// A common implementation computes the angle gradient w.r.t. vertices,
		// and then moves vertices proportionally.
		// We'll implement the standard dihedral bending constraint from
		// Muller et al. Position Based Dynamics.

		// Compute cotangent weights for the edge.
		real_t cot1 = cotangent(p1, p3, p2); // angle at p3
		real_t cot2 = cotangent(p1, p2, p4); // angle at p4
		real_t w_sum = cot1 + cot2;

		// Gradient: for p1..p4 (see Muller et al.)
		Vector3 grad_p1 = (cot1 * n1 + cot2 * n2) / w_sum;
		Vector3 grad_p2 = -grad_p1;
		Vector3 grad_p3 = (cot1 * (p1 - p2).cross(n1)) / (2.0 * len_n1);
		Vector3 grad_p4 = (cot2 * (p2 - p1).cross(n2)) / (2.0 * len_n2);

		// Simplified: use a single effective gradient magnitude.
		// Standard PBD bending: delta = -C / (sum wi |grad_i|^2) * wi * grad_i
		// We'll compute inverse masses (uniform per vertex)
		real_t inv_mass = 1.0 / soft_body->get_total_mass(); // uniform
		real_t grad_sq = inv_mass * (grad_p1.length_squared() + grad_p2.length_squared() +
									 grad_p3.length_squared() + grad_p4.length_squared());

		real_t alpha = compliance;
		real_t alpha_tilde = alpha / (dt * dt);
		real_t delta_lambda = -(C + alpha_tilde * lambda) / (grad_sq + alpha_tilde);
		lambda += delta_lambda;

		// Apply corrections
		p1 -= inv_mass * delta_lambda * grad_p1;
		p2 -= inv_mass * delta_lambda * grad_p2;
		p3 -= inv_mass * delta_lambda * grad_p3;
		p4 -= inv_mass * delta_lambda * grad_p4;
	}

	virtual void solve_velocity(real_t dt) override {
		// Optional damping based on relative angular velocity (stub)
	}

	PBDSolver *solver;

private:
	// Helper: cotangent of angle at vertex `v` of triangle (a, b, v)
	static real_t cotangent(const Vector3 &a, const Vector3 &b, const Vector3 &v) {
		Vector3 edge1 = a - v;
		Vector3 edge2 = b - v;
		real_t dot = edge1.dot(edge2);
		Vector3 cross = edge1.cross(edge2);
		real_t len_cross = cross.length();
		if (len_cross < CMP_EPSILON) return 0.0;
		return dot / len_cross;
	}

	SoftBody *soft_body;
	int idx1, idx2, idx3, idx4;
	real_t rest_angle;
	real_t lambda;
};

} // namespace gaia

#endif // GAIA_PBD_BENDING_CONSTRAINT_H