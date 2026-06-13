// File 26: modules/gaia/src/vbd/vbd_constraint.h

#ifndef GAIA_VBD_CONSTRAINT_H
#define GAIA_VBD_CONSTRAINT_H

#include "../framework/constraint.h"
#include "../framework/body.h"
#include "../vbd/vbd_element.h"

#include "core/typedefs.h"

namespace gaia {

// Forward declaration
class VBDSolver;

/**
 * A VBD constraint encapsulates a single deformable element
 * (tetrahedron or triangle) and performs a local block‑coordinate
 * energy minimisation step on its vertices.
 */
class VBDConstraint : public Constraint {
public:
	VBDConstraint() :
		Constraint(),
		soft_body(nullptr),
		element(nullptr),
		solver(nullptr) {
		type = CUSTOM; // VBD is a custom constraint
	}

	~VBDConstraint() {
		if (element) {
			memdelete(element);
			element = nullptr;
		}
	}

	void set_body(SoftBody *p_body) { soft_body = p_body; }
	SoftBody *get_body() const { return soft_body; }

	// Set the underlying element (takes ownership).
	void set_element(VBDElement *p_element) {
		if (element) memdelete(element);
		element = p_element;
	}
	VBDElement *get_element() const { return element; }

	// Initialize rest state from the body's current positions.
	void init_from_positions() {
		ERR_FAIL_COND(!soft_body);
		ERR_FAIL_COND(!element);
		element->compute_rest_state(soft_body);
	}

	// Perform one block descent step: for each vertex of the element,
	// solve for its optimal position holding others fixed.
	virtual void solve_position(real_t dt) override {
		ERR_FAIL_COND(!soft_body);
		ERR_FAIL_COND(!element);

		// The step size is governed by compliance (alpha) and dt.
		real_t alpha = compliance;
		// VBD typically uses a blending factor between current and target
		// positions. Here we compute a per‑vertex descent direction and
		// move a fraction based on alpha and dt.
		element->solve_block_descent(soft_body, alpha, dt);
	}

	virtual void solve_velocity(real_t dt) override {
		// Velocities are derived from the position updates that happened
		// during solve_position. VBD usually doesn't have a separate
		// velocity pass; velocities are recomputed after integration.
	}

	VBDSolver *solver;

private:
	SoftBody *soft_body;
	VBDElement *element;
};

} // namespace gaia

#endif // GAIA_VBD_CONSTRAINT_H