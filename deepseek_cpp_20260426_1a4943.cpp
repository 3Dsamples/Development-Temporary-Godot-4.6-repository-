// File 144: modules/genesis/src/solvers/solver_registry.h
// SolverRegistry – a factory that maps solver type identifiers to concrete
// solver instances. Used by GenesisWorld and GenesisPhysicsServer to
// instantiate the correct solver per entity or simulation group.

#ifndef GENESIS_SOLVERS_SOLVER_REGISTRY_H
#define GENESIS_SOLVERS_SOLVER_REGISTRY_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "base_solver.h"
#include "rigid_solver.h"
#include "fem_solver.h"
#include "mpm_solver.h"
#include "pbd_solver.h"
#include "sph_solver.h"
#include "sf_solver.h"
#include "kinematic_solver.h"

namespace genesis {

class SolverRegistry : public RefCounted {
	GDCLASS(SolverRegistry, RefCounted);

public:
	SolverRegistry() { register_default_solvers(); }

	// Register a solver instance for a given solver type.
	void register_solver(SolverType p_type, const Ref<BaseSolver> &p_solver) {
		ERR_FAIL_COND(p_solver.is_null());
		solvers[(int)p_type] = p_solver;
	}

	// Retrieve a solver by type. Returns null if not registered.
	Ref<BaseSolver> get_solver(SolverType p_type) const {
		HashMap<int, Ref<BaseSolver>>::ConstIterator it = solvers.find((int)p_type);
		if (it) return it->value;
		return Ref<BaseSolver>();
	}

	// Create a default solver of a given type (not already registered).
	Ref<BaseSolver> create_default(SolverType p_type) {
		Ref<BaseSolver> existing = get_solver(p_type);
		if (existing.is_valid()) return existing;

		Ref<BaseSolver> new_solver;
		switch (p_type) {
			case SolverType::RIGID:  new_solver.instantiate(); break; // RigidSolver
			case SolverType::FEM:    new_solver.instantiate(); break; // FEMSolver
			case SolverType::MPM:    new_solver.instantiate(); break; // MPMSolver
			case SolverType::PBD:    new_solver.instantiate(); break; // GenesisPBDSolver
			case SolverType::SPH:    new_solver.instantiate(); break; // SPHSolver
			case SolverType::SF:     new_solver.instantiate(); break; // SFSolver
			case SolverType::TOOL:   new_solver.instantiate(); break; // KinematicSolver
			default: return Ref<BaseSolver>();
		}
		register_solver(p_type, new_solver);
		return new_solver;
	}

	// Return all registered solver types.
	LocalVector<SolverType> get_registered_types() const {
		LocalVector<SolverType> types;
		for (const KeyValue<int, Ref<BaseSolver>> &kv : solvers) {
			types.push_back((SolverType)kv.key);
		}
		return types;
	}

	// Step all registered solvers by dt (substepping is handled internally).
	void step_all(real_t dt) {
		for (KeyValue<int, Ref<BaseSolver>> &kv : solvers) {
			if (kv.value.is_valid()) {
				kv.value->set_dt(dt);
				kv.value->step();
			}
		}
	}

	// Set a parameter globally on all solvers (e.g., gravity).
	void set_global_gravity(const Vector3 &p_gravity) {
		for (KeyValue<int, Ref<BaseSolver>> &kv : solvers) {
			if (kv.value.is_valid()) kv.value->set_gravity(p_gravity);
		}
	}

	// Clear all constraints / entities from all solvers.
	void clear_all() {
		for (KeyValue<int, Ref<BaseSolver>> &kv : solvers) {
			if (kv.value.is_valid()) kv.value->clear_entities();
		}
	}

private:
	void register_default_solvers() {
		create_default(SolverType::RIGID);
		create_default(SolverType::FEM);
		create_default(SolverType::MPM);
		create_default(SolverType::PBD);
		create_default(SolverType::SPH);
		create_default(SolverType::SF);
		create_default(SolverType::TOOL);
	}

	HashMap<int, Ref<BaseSolver>> solvers;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("register_solver", "type", "solver"), &SolverRegistry::register_solver);
		ClassDB::bind_method(D_METHOD("get_solver", "type"), &SolverRegistry::get_solver);
		ClassDB::bind_method(D_METHOD("create_default", "type"), &SolverRegistry::create_default);
		ClassDB::bind_method(D_METHOD("step_all", "dt"), &SolverRegistry::step_all);
		ClassDB::bind_method(D_METHOD("set_global_gravity", "gravity"), &SolverRegistry::set_global_gravity);
		ClassDB::bind_method(D_METHOD("clear_all"), &SolverRegistry::clear_all);
	}
};

} // namespace genesis

#endif // GENESIS_SOLVERS_SOLVER_REGISTRY_H