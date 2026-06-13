// File 13: modules/gaia/src/framework/solver.h

#ifndef GAIA_FRAMEWORK_SOLVER_H
#define GAIA_FRAMEWORK_SOLVER_H

#include "../framework/constraint.h"

#include "core/templates/local_vector.h"
#include "core/typedefs.h"

namespace gaia {

/**
 * Abstract solver interface. Concrete solvers (PBD, VBD) implement
 * the solve() method and manage their constraints.
 */
class Solver {
public:
	Solver() : iteration_count(5) {}
	virtual ~Solver() {}

	// Set the number of constraint projection iterations per substep.
	void set_iterations(int p_iterations) { iteration_count = MAX(p_iterations, 1); }
	int get_iterations() const { return iteration_count; }

	// Main solve step: enforce all constraints for the given timestep.
	virtual void solve(real_t dt) = 0;

	// Add/remove constraints (the solver owns them if not destroyed externally).
	virtual void add_constraint(Constraint *p_constraint) = 0;
	virtual void remove_constraint(Constraint *p_constraint) = 0;

	// Clear all internal constraints.
	virtual void clear_constraints() = 0;

	// Get total constraint count.
	virtual int get_constraint_count() const = 0;

protected:
	int iteration_count;
};

} // namespace gaia

#endif // GAIA_FRAMEWORK_SOLVER_H