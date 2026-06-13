// File 20: modules/gaia/src/pbd/pbd_solver.h

#ifndef GAIA_PBD_SOLVER_H
#define GAIA_PBD_SOLVER_H

#include "../framework/solver.h"
#include "../framework/body.h"
#include "../pbd/distance_constraint.h"
#include "../pbd/bending_constraint.h"
#include "../pbd/volume_constraint.h"
#include "../pbd/collision_constraint.h"

#include "core/templates/local_vector.h"
#include "core/typedefs.h"

namespace gaia {

/**
 * PBD (Position Based Dynamics) / XPBD solver.
 *
 * Ensures all registered constraints are satisfied iteratively.
 * Supports XPBD compliance for soft constraints.
 */
class PBDSolver : public Solver {
public:
	PBDSolver() : Solver() {
		iteration_count = 5;
	}
	virtual ~PBDSolver() { clear_constraints(); }

	// Solve all constraints for the given time step.
	virtual void solve(real_t dt) override {
		// XPBD: compute Lagrange multiplier accumulators from compliance.
		// For simplicity, we assume each constraint handles its own lambda.
		for (int iter = 0; iter < iteration_count; ++iter) {
			for (DistanceConstraint *c : distance_constraints) {
				c->solve_position(dt);
				c->solve_velocity(dt);
			}
			for (BendingConstraint *c : bending_constraints) {
				c->solve_position(dt);
				c->solve_velocity(dt);
			}
			for (VolumeConstraint *c : volume_constraints) {
				c->solve_position(dt);
				c->solve_velocity(dt);
			}
			for (CollisionConstraint *c : collision_constraints) {
				c->solve_position(dt);
				c->solve_velocity(dt);
			}
		}
	}

	virtual void add_constraint(Constraint *p_constraint) override {
		ERR_FAIL_COND(!p_constraint);
		switch (p_constraint->get_type()) {
			case Constraint::DISTANCE:
				distance_constraints.push_back(static_cast<DistanceConstraint *>(p_constraint));
				distance_constraints.back()->solver = this;
				break;
			case Constraint::BENDING:
				bending_constraints.push_back(static_cast<BendingConstraint *>(p_constraint));
				bending_constraints.back()->solver = this;
				break;
			case Constraint::VOLUME:
				volume_constraints.push_back(static_cast<VolumeConstraint *>(p_constraint));
				volume_constraints.back()->solver = this;
				break;
			case Constraint::COLLISION:
				collision_constraints.push_back(static_cast<CollisionConstraint *>(p_constraint));
				collision_constraints.back()->solver = this;
				break;
			default:
				// Unsupported, leak? We'll ignore for now.
				break;
		}
	}

	virtual void remove_constraint(Constraint *p_constraint) override {
		// Linear search and remove
		// We'll implement for each list
		remove_from_list(distance_constraints, p_constraint);
		remove_from_list(bending_constraints, p_constraint);
		remove_from_list(volume_constraints, p_constraint);
		remove_from_list(collision_constraints, p_constraint);
	}

	virtual void clear_constraints() override {
		distance_constraints.clear();
		bending_constraints.clear();
		volume_constraints.clear();
		collision_constraints.clear();
	}

	virtual int get_constraint_count() const override {
		return distance_constraints.size() + bending_constraints.size() +
			   volume_constraints.size() + collision_constraints.size();
	}

private:
	template <typename T>
	void remove_from_list(LocalVector<T *> &p_list, Constraint *p_constraint) {
		for (int i = 0; i < p_list.size(); ++i) {
			if (p_list[i] == p_constraint) {
				p_list.remove_at_unordered(i);
				return;
			}
		}
	}

	LocalVector<DistanceConstraint *> distance_constraints;
	LocalVector<BendingConstraint *> bending_constraints;
	LocalVector<VolumeConstraint *> volume_constraints;
	LocalVector<CollisionConstraint *> collision_constraints;
};

} // namespace gaia

#endif // GAIA_PBD_SOLVER_H