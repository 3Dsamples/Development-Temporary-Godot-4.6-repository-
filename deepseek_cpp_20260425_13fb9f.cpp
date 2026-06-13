// File 25: modules/gaia/src/vbd/vbd_solver.h

#ifndef GAIA_VBD_SOLVER_H
#define GAIA_VBD_SOLVER_H

#include "../framework/solver.h"
#include "../framework/body.h"
#include "../vbd/vbd_constraint.h"
#include "../vbd/vbd_element.h"

#include "core/templates/local_vector.h"
#include "core/typedefs.h"

namespace gaia {

/**
 * VBD (Vertex Block Descent) solver.
 *
 * An energy‑based solver that minimises a non‑linear elastic energy
 * for each element (tetrahedron or triangle). It uses a block‑coordinate
 * descent approach.
 */
class VBDSolver : public Solver {
public:
	VBDSolver() : Solver() {
		iteration_count = 3; // VBD typically uses fewer outer iterations
	}
	virtual ~VBDSolver() { clear_constraints(); }

	virtual void solve(real_t dt) override {
		for (int iter = 0; iter < iteration_count; ++iter) {
			// Solve each element independently
			for (VBDConstraint *c : vbd_constraints) {
				c->solve_position(dt);
				c->solve_velocity(dt);
			}
		}
	}

	virtual void add_constraint(Constraint *p_constraint) override {
		VBDConstraint *vbd = dynamic_cast<VBDConstraint *>(p_constraint);
		ERR_FAIL_COND(!vbd);
		vbd_constraints.push_back(vbd);
		vbd->solver = this;
	}

	virtual void remove_constraint(Constraint *p_constraint) override {
		VBDConstraint *vbd = dynamic_cast<VBDConstraint *>(p_constraint);
		if (!vbd) return;
		for (int i = 0; i < vbd_constraints.size(); ++i) {
			if (vbd_constraints[i] == vbd) {
				vbd_constraints.remove_at_unordered(i);
				return;
			}
		}
	}

	virtual void clear_constraints() override {
		vbd_constraints.clear();
	}

	virtual int get_constraint_count() const override {
		return vbd_constraints.size();
	}

private:
	LocalVector<VBDConstraint *> vbd_constraints;
};

} // namespace gaia

#endif // GAIA_VBD_SOLVER_H