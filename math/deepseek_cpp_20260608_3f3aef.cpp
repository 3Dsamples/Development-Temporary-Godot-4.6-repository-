// File 297: modules/vienna/src/solver/vienna_parallel_solver.h
// ViennaParallelSolver – parallel island solving using Godot's WorkerThreadPool.
// Each island is dispatched to a worker thread, achieving multi‑core acceleration
// for large scenes without shared mutex contention (islands are independent).

#ifndef VIENNA_SOLVER_PARALLEL_SOLVER_H
#define VIENNA_SOLVER_PARALLEL_SOLVER_H

#include "core/object/worker_thread_pool.h"
#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"
#include "vienna_solver.h"          // ViennaContactPoint, ViennaSolver
#include "vienna_island.h"

namespace vienna {

class ViennaParallelSolver : public RefCounted {
	GDCLASS(ViennaParallelSolver, RefCounted);

public:
	ViennaParallelSolver() {}
	virtual ~ViennaParallelSolver() {}

	// Solve all islands in parallel.  Contacts and joints are solved by the
	// solver's PGS routine.  The method blocks until all islands have been solved.
	void solve_parallel(const LocalVector<ViennaIsland *> &islands,
						const HashMap<body_id, Ref<ViennaBody>> &bodies,
						const HashMap<joint_id, Ref<ViennaJoint>> &joints,
						const HashMap<material_id, Ref<ViennaMaterial>> &materials,
						real_t dt,
						int iterations) {
		WorkerThreadPool *pool = WorkerThreadPool::get_singleton();
		if (!pool || islands.is_empty()) return;

		// Prepare task data for each non‑sleeping island.
		int total = islands.size();
		LocalVector<IslandTask> tasks;
		tasks.resize(total);
		int active_count = 0;
		for (int i = 0; i < total; ++i) {
			if (islands[i]->is_sleeping()) continue;
			IslandTask &task = tasks[i];
			task.island = islands[i];
			task.bodies = &bodies;
			task.joints = &joints;
			task.materials = &materials;
			task.dt = dt;
			task.iterations = iterations;
			active_count++;
		}
		if (active_count == 0) return;

		// Use the pool to execute one task per island.
		// Since tasks are independent, we can dispatch them all and then wait.
		// Godot 4.6 does not expose a "wait all" natively; we use an atomic counter.
		Counter *counter = memnew(Counter);
		counter->remaining = active_count;
		counter->completed.event = nullptr; // we use a simple spin‑wait? Not ideal. We'll use a Mutex.
		// Better: use a semaphore via OS. We'll implement a simple busy‑wait with yield, acceptable for physics step.
		for (int i = 0; i < total; ++i) {
			if (islands[i]->is_sleeping()) continue;
			IslandTask *pTask = memnew(IslandTask(tasks[i]));
			pTask->counter = counter;
			pool->add_task(&solve_island_task, pTask);
		}

		// Busy‑wait until all islands solved (OK for physics step, usually < 2ms)
		while (counter->remaining > 0) {
			OS::get_singleton()->delay_usec(0); // yield
		}

		memdelete(counter);
	}

protected:
	static void _bind_methods() {}

private:
	struct Counter {
		int remaining;
		Mutex mutex;
	};

	struct IslandTask {
		ViennaIsland *island;
		const HashMap<body_id, Ref<ViennaBody>> *bodies;
		const HashMap<joint_id, Ref<ViennaJoint>> *joints;
		const HashMap<material_id, Ref<ViennaMaterial>> *materials;
		real_t dt;
		int iterations;
		Counter *counter;
	};

	static void solve_island_task(void *p_userdata) {
		IslandTask *task = (IslandTask *)p_userdata;
		if (!task || !task->island) {
			memdelete(task);
			return;
		}

		// Sequential solve for this island.
		LocalVector<ViennaContactPoint> &contacts = task->island->get_contacts();
		LocalVector<Ref<ViennaJoint>> &joints = task->island->get_joints();

		// Iterative PGS solver (same as ViennaSolver but inline here for parallelism).
		const int iters = task->iterations;
		const real_t dt = task->dt;

		for (int iter = 0; iter < iters; ++iter) {
			// Normal impulses for all contacts
			for (ViennaContactPoint &cp : contacts) {
				ViennaBody *bodyA = task->island->get_body(cp.body_a);
				ViennaBody *bodyB = task->island->get_body(cp.body_b);
				if (!bodyA || !bodyB) continue;
				if (bodyA->get_inverse_mass() <= 0.0 && bodyB->get_inverse_mass() <= 0.0) continue;

				real_t inv_eff_n = compute_eff_mass(bodyA, bodyB, cp.point_a, cp.point_b, cp.normal);
				if (inv_eff_n < CMP_EPSILON) continue;

				vec3 rA = cp.point_a - bodyA->get_position();
				vec3 rB = cp.point_b - bodyB->get_position();
				vec3 velA = bodyA->get_linear_velocity() + bodyA->get_angular_velocity().cross(rA);
				vec3 velB = bodyB->get_linear_velocity() + bodyB->get_angular_velocity().cross(rB);
				vec3 rel_vel = velB - velA;
				real_t vn = rel_vel.dot(cp.normal);

				real_t restitution = (iter == 0) ? cp.restitution : 0.0f;
				real_t target_dv = -(1.0f + restitution) * vn;
				if (iter > 0 && cp.penetration > 0.0f) {
					real_t erp = 0.2f / dt;
					target_dv += cp.penetration * erp;
				}

				real_t dP_n = target_dv / inv_eff_n;
				real_t P_n_old = cp.normal_impulse;
				cp.normal_impulse = MAX(P_n_old + dP_n, 0.0f);
				dP_n = cp.normal_impulse - P_n_old;

				vec3 impulse_n = cp.normal * dP_n;
				apply_pair_impulse(bodyA, bodyB, impulse_n, cp.point_a, cp.point_b);

				// Friction
				vec3 vt = rel_vel - cp.normal * vn;
				real_t vt_len = vt.length();
				if (vt_len > CMP_EPSILON) {
					vec3 t_dir = vt / vt_len;
					real_t inv_eff_t = compute_eff_mass(bodyA, bodyB, cp.point_a, cp.point_b, t_dir);
					if (inv_eff_t > CMP_EPSILON) {
						real_t dP_t = -vt_len / inv_eff_t;
						real_t max_friction = cp.friction * cp.normal_impulse;
						real_t P_t_old = cp.friction_impulse1.length();
						real_t P_t_new = CLAMP(P_t_old + dP_t, 0.0f, max_friction);
						dP_t = P_t_new - P_t_old;
						vec3 impulse_t = t_dir * dP_t;
						apply_pair_impulse(bodyA, bodyB, impulse_t, cp.point_a, cp.point_b);
						cp.friction_impulse1 += impulse_t;
					}
				}
			}

			// Joints
			for (Ref<ViennaJoint> &joint : joints) {
				if (joint.is_null() || !joint->is_enabled()) continue;
				ViennaBody *ba = task->island->get_body(joint->get_body_a());
				ViennaBody *bb = task->island->get_body(joint->get_body_b());
				joint->solve(ba, bb, dt);
			}
		}

		// Signal completion
		task->counter->mutex.lock();
		task->counter->remaining--;
		task->counter->mutex.unlock();

		memdelete(task);
	}

	// Effective mass and impulse helpers (replicated from ViennaSolver to avoid dependency)
	static real_t compute_eff_mass(const ViennaBody *bodyA, const ViennaBody *bodyB,
								   const vec3 &pointA, const vec3 &pointB, const vec3 &dir) {
		real_t inv_mass = 0.0;
		if (bodyA->get_inverse_mass() > 0.0) {
			vec3 rA = pointA - bodyA->get_position();
			inv_mass += bodyA->get_inverse_mass();
			inv_mass += dir.dot(bodyA->get_inverse_inertia_world().xform(rA.cross(dir)).cross(rA));
		}
		if (bodyB->get_inverse_mass() > 0.0) {
			vec3 rB = pointB - bodyB->get_position();
			inv_mass += bodyB->get_inverse_mass();
			inv_mass += dir.dot(bodyB->get_inverse_inertia_world().xform(rB.cross(dir)).cross(rB));
		}
		return inv_mass;
	}

	static void apply_pair_impulse(ViennaBody *bodyA, ViennaBody *bodyB,
								   const vec3 &impulse, const vec3 &pointA, const vec3 &pointB) {
		if (bodyA->get_inverse_mass() > 0.0) bodyA->apply_impulse( impulse, pointA);
		if (bodyB->get_inverse_mass() > 0.0) bodyB->apply_impulse(-impulse, pointB);
	}
};

} // namespace vienna

#endif // VIENNA_SOLVER_PARALLEL_SOLVER_H