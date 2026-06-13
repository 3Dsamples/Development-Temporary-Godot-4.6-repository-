// File 24: modules/gaia/src/pbd/collision_constraint.h

#ifndef GAIA_PBD_COLLISION_CONSTRAINT_H
#define GAIA_PBD_COLLISION_CONSTRAINT_H

#include "../framework/constraint.h"
#include "../framework/body.h"
#include "../collision_detector/contact.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"

namespace gaia {

// Forward declaration
class PBDSolver;

/**
 * XPBD collision constraint between a soft body vertex and a rigid body
 * (or another soft body vertex), using the contact information from
 * the narrow phase.
 *
 * For simplicity, this implementation handles vertex vs. static plane/wall,
 * but can be extended for general contacts.
 */
class CollisionConstraint : public Constraint {
public:
	CollisionConstraint() :
		Constraint(),
		soft_body(nullptr),
		vertex_idx(-1),
		contact_point(Vector3()),
		contact_normal(Vector3()),
		penetration(0.0),
		lambda(0.0) {
		type = COLLISION;
	}

	void set_body(SoftBody *p_body) { soft_body = p_body; }
	SoftBody *get_body() const { return soft_body; }

	void set_vertex_index(int p_idx) { vertex_idx = p_idx; }
	int get_vertex_index() const { return vertex_idx; }

	// Configure the constraint from a contact (world-space).
	void set_contact(const Vector3 &p_contact_point, const Vector3 &p_normal, real_t p_penetration) {
		contact_point = p_contact_point;
		contact_normal = p_normal.normalized();
		penetration = MAX(p_penetration, 0.0);
	}

	virtual void solve_position(real_t dt) override {
		ERR_FAIL_COND(!soft_body);
		ERR_FAIL_INDEX(vertex_idx, soft_body->positions.size());

		Vector3 &p = soft_body->positions[vertex_idx];
		// Constraint: C = (p - contact_point) · n >= 0  (no penetration)
		// We enforce: C = (p - contact_point).dot(n)   (if negative, penetration)
		real_t C = (p - contact_point).dot(contact_normal);
		if (C >= 0.0) return; // No penetration

		// Gradient: n
		real_t inv_mass = 1.0 / soft_body->get_total_mass();
		real_t w = inv_mass; // only one vertex

		// XPBD
		real_t alpha = compliance;
		real_t alpha_tilde = alpha / (dt * dt);
		// If we want hard collision, set compliance = 0.
		real_t delta_lambda = -(C + alpha_tilde * lambda) / (w + alpha_tilde);
		lambda += delta_lambda;

		// Correction: move vertex out
		p += inv_mass * delta_lambda * contact_normal;
	}

	virtual void solve_velocity(real_t dt) override {
		if (damping <= 0.0) return;
		ERR_FAIL_COND(!soft_body);
		ERR_FAIL_INDEX(vertex_idx, soft_body->velocities.size());

		Vector3 &v = soft_body->velocities[vertex_idx];
		// Relative velocity along normal
		real_t vn = v.dot(contact_normal);
		if (vn >= 0.0) return; // moving away

		real_t inv_mass = 1.0 / soft_body->get_total_mass();
		real_t w = inv_mass;
		// Damped impulse: reduce normal velocity by damping factor
		real_t impulse = -damping * vn / w;
		v += inv_mass * impulse * contact_normal;
	}

	PBDSolver *solver;

private:
	SoftBody *soft_body;
	int vertex_idx;
	Vector3 contact_point;
	Vector3 contact_normal;
	real_t penetration;
	real_t lambda;
};

} // namespace gaia

#endif // GAIA_PBD_COLLISION_CONSTRAINT_H