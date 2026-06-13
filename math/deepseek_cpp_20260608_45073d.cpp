// File 232: modules/newton/src/solver/newton_parallel_solver.h
// NewtonParallelSolver – parallelises the constraint solving across islands
// using Godot's WorkerThreadPool.  Each island can be solved independently
// because islands have no shared bodies or joints.  This accelerates large
// scenes with many disjoint groups.

#ifndef NEWTON_SOLVER_PARALLEL_SOLVER_H
#define NEWTON_SOLVER_PARALLEL_SOLVER_H

#include "core/object/ref_counted.h"
#include "core/object/worker_thread_pool.h"
#include "core/templates/local_vector.h"
#include "../core/newton_types.h"
#include "../core/newton_constants.h"
#include "newton_solver.h"        // base solver definitions (contact, island, etc.)
#include "newton_island.h"

namespace newton {

class NewtonParallelSolver : public RefCounted {
	GDCLASS(NewtonParallelSolver, RefCounted);

public:
	NewtonParallelSolver() {}
	virtual ~NewtonParallelSolver() {}

	/**
	 * Solve all islands in parallel.  Each island is dispatched to a worker
	 * thread that executes the sequential solver on that island.  The method
	 * blocks until all islands are solved.
	 *
	 * @param islands        Vector of island pointers to be solved.
	 * @param contacts       All contact points (owned by islands).
	 * @param joints         All joints (owned by islands).
	 * @param bodies         Global body map (read-only for effective mass lookups).
	 * @param materials      Global material map (read-only).
	 * @param dt             Time step.
	 */
	void solve_parallel(const LocalVector<NewtonIsland *> &islands,
						const HashMap<body_id, Ref<NewtonBody>> &bodies,
						const HashMap<material_id, Ref<NewtonMaterial>> &materials,
						real_t dt);

private:
	// A task that solves a single island.
	struct IslandTask {
		NewtonIsland *island;
		const HashMap<body_id, Ref<NewtonBody>> *bodies;
		const HashMap<material_id, Ref<NewtonMaterial>> *materials;
		real_t dt;
		int iterations;
		bool warm_starting;
	};

	static void _solve_island_task(void *p_userdata);
};

} // namespace newton

#endif // NEWTON_SOLVER_PARALLEL_SOLVER_H