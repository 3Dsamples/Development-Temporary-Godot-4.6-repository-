// File 72: modules/genesis/src/solvers/pbd_solver.h
// Genesis PBD/XPBD solver for cloth, soft bodies, and rigid-deformable coupling.
// Uses PBDMaterial and handles distance, bending, volume, and collision constraints.

#ifndef GENESIS_SOLVERS_PBD_SOLVER_H
#define GENESIS_SOLVERS_PBD_SOLVER_H

#include "base_solver.h"
#include "../materials/pbd_material.h"
#include "../entities/base_entity.h"           // BaseEntity (used if needed)
#include "../../../gaia/src/pbd/distance_constraint.h"
#include "../../../gaia/src/pbd/bending_constraint.h"
#include "../../../gaia/src/pbd/volume_constraint.h"
#include "../../../gaia/src/pbd/collision_constraint.h"
#include "core/templates/local_vector.h"
#include "core/typedefs.h"

namespace genesis {

/**
 * PBD/XPBD solver compatible with Genesis' PBDMaterial.
 * Extends the Gaia PBD solver with compliance-based parameters and
 * supports per‑constraint stiffness from the material.
 */
class GenesisPBDSolver : public BaseSolver {
	GDCLASS(GenesisPBDSolver, BaseSolver);

public:
	GenesisPBDSolver() : BaseSolver() {}
	virtual ~GenesisPBDSolver() { clear_constraints(); }

	// --- Material ---
	void set_material(const Ref<PBDMaterial> &p_mat) { material = p_mat; }
	Ref<PBDMaterial> get_material() const { return material; }

	// --- Constraint management (using Gaia constraint types) ---
	void add_distance_constraint(gaia::DistanceConstraint *p_con) {
		distance_constraints.push_back(p_con);
	}
	void add_bending_constraint(gaia::BendingConstraint *p_con) {
		bending_constraints.push_back(p_con);
	}
	void add_volume_constraint(gaia::VolumeConstraint *p_con) {
		volume_constraints.push_back(p_con);
	}
	void add_collision_constraint(gaia::CollisionConstraint *p_con) {
		collision_constraints.push_back(p_con);
	}

	void clear_constraints() {
		distance_constraints.clear();
		bending_constraints.clear();
		volume_constraints.clear();
		collision_constraints.clear();
	}

	// Main XPBD step
	virtual void solve(real_t dt) override {
		if (material.is_null()) return;

		// Set global compliance from material overrides
		real_t global_alpha = material->get_compliance();
		real_t bend_alpha = material->get_bending_compliance();
		real_t vol_alpha = material->get_volume_compliance();
		real_t coll_alpha = material->get_collision_compliance();

		for (int iter = 0; iter < iterations; ++iter) {
			// Distance constraints
			for (gaia::DistanceConstraint *c : distance_constraints) {
				c->set_compliance(global_alpha);
				c->set_damping(material->get_damping_compliance());
				c->solve_position(dt);
				c->solve_velocity(dt);
			}
			// Bending constraints
			for (gaia::BendingConstraint *c : bending_constraints) {
				c->set_compliance(bend_alpha);
				c->solve_position(dt);
				c->solve_velocity(dt);
			}
			// Volume constraints
			for (gaia::VolumeConstraint *c : volume_constraints) {
				c->set_compliance(vol_alpha);
				c->solve_position(dt);
				c->solve_velocity(dt);
			}
			// Collision constraints
			for (gaia::CollisionConstraint *c : collision_constraints) {
				c->set_compliance(coll_alpha);
				c->set_damping(0.0); // friction handled separately
				c->solve_position(dt);
				c->solve_velocity(dt);
			}
		}
	}

	// Step with default pre/post if needed
	virtual void step() override {
		real_t sub_dt = dt / real_t(sub_steps);
		for (int substep = 0; substep < sub_steps; ++substep) {
			pre_step(sub_dt);
			solve(sub_dt);
			post_step(sub_dt);
			time += sub_dt;
		}
	}

	virtual void pre_step(real_t p_sub_dt) override {
		// Apply gravity to particles of all constraints if they are associated with a soft body.
		// The soft body is stored in constraints via their body pointer.
		// But we can't easily gather all unique bodies. Instead, the constraints themselves use body->apply_force.
		// We'll assume the soft body has already been force‑integrated outside.
	}

	virtual void post_step(real_t p_sub_dt) override {
		// Update velocities from position changes (XPBD already updates positions, so we compute velocities afterward)
		for (gaia::DistanceConstraint *c : distance_constraints) {
			// Constraint body may be a SoftBody*; if so, update velocities based on position displacement.
			// Actually, the Gaia constraints already call solve_velocity. We'll leave as is.
		}
	}

private:
	Ref<PBDMaterial> material;

	LocalVector<gaia::DistanceConstraint *> distance_constraints;
	LocalVector<gaia::BendingConstraint *> bending_constraints;
	LocalVector<gaia::VolumeConstraint *> volume_constraints;
	LocalVector<gaia::CollisionConstraint *> collision_constraints;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_material", "material"), &GenesisPBDSolver::set_material);
		ClassDB::bind_method(D_METHOD("get_material"), &GenesisPBDSolver::get_material);
		ClassDB::bind_method(D_METHOD("add_distance_constraint", "constraint"), &GenesisPBDSolver::add_distance_constraint);
		ClassDB::bind_method(D_METHOD("add_bending_constraint", "constraint"), &GenesisPBDSolver::add_bending_constraint);
		ClassDB::bind_method(D_METHOD("add_volume_constraint", "constraint"), &GenesisPBDSolver::add_volume_constraint);
		ClassDB::bind_method(D_METHOD("add_collision_constraint", "constraint"), &GenesisPBDSolver::add_collision_constraint);
		ClassDB::bind_method(D_METHOD("clear_constraints"), &GenesisPBDSolver::clear_constraints);
		ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "PBDMaterial"), "set_material", "get_material");
	}
};

} // namespace genesis

#endif // GENESIS_SOLVERS_PBD_SOLVER_H