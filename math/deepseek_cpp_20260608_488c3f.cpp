// File 148: modules/genesis/src/solvers/rigid_solver.cpp
// Implementation of RigidSolver that uses the Gaia broad‑phase and the
// ConstraintIslandSolver for resolving collisions and constraints.
// The step() method from the base class is overridden by the full pipeline.

#include "rigid_solver.h"
#include "constraint_island.h"          // island builder and contact solver
#include "../entities/rigid_entity.h"
#include "../../../gaia/src/collision_detector/broad_phase.h"
#include "../../../gaia/src/collision_detector/narrow_phase.h"
#include "core/templates/local_vector.h"

namespace genesis {

void RigidSolver::step() {
	real_t sub_dt = dt / real_t(sub_steps);
	for (int substep = 0; substep < sub_steps; ++substep) {
		// 1. Apply external forces (gravity, user forces) and integrate velocities
		pre_step(sub_dt);

		// 2. Broad‑phase + narrow‑phase → contacts fed to constraint island solver
		detect_collisions(sub_dt);

		// 3. Solve islands (position and velocity iterations)
		solve(sub_dt);

		// 4. Integrate positions and update world transforms
		post_step(sub_dt);

		time += sub_dt;
	}
}

void RigidSolver::pre_step(real_t p_sub_dt) {
	// Gather active rigid entities
	active_rigid_bodies.clear();
	for (KeyValue<entity_id_t, Ref<BaseEntity>> &kv : entities) {
		Ref<RigidEntity> rigid = kv.value;
		if (rigid.is_valid() && rigid->is_active()) {
			active_rigid_bodies.push_back(rigid);
		}
	}
	// Apply gravity and external forces, then integrate velocities
	for (Ref<RigidEntity> &body : active_rigid_bodies) {
		if (body->is_gravity_enabled()) {
			body->apply_force(gravity * body->get_mass(), body->get_position());
		}
		body->integrate_velocity(p_sub_dt);
	}
}

void RigidSolver::detect_collisions(real_t p_sub_dt) {
	// Build broad‑phase with current AABBs
	gaia::collision::BroadPhase broad;
	LocalVector<int> broad_to_body;
	for (int i = 0; i < active_rigid_bodies.size(); ++i) {
		if (active_rigid_bodies[i]->is_active()) {
			broad.add_object(i, active_rigid_bodies[i]->get_aabb());
		}
	}

	// Collect pairs
	LocalVector<Pair> overlapping_pairs;
	broad.find_pairs([](uint32_t hA, uint32_t hB, void *userdata) {
		auto *pairs = static_cast<LocalVector<Pair>*>(userdata);
		pairs->push_back({(int)hA, (int)hB});
	}, &overlapping_pairs);

	// Narrow‑phase: for each pair, compute contact manifolds via GJK/EPA
	manifolds.clear();
	for (const Pair &pair : overlapping_pairs) {
		if (pair.a >= active_rigid_bodies.size() || pair.b >= active_rigid_bodies.size()) continue;
		const Ref<RigidEntity> &bodyA = active_rigid_bodies[pair.a];
		const Ref<RigidEntity> &bodyB = active_rigid_bodies[pair.b];
		if (bodyA.is_null() || bodyB.is_null()) continue;

		GJK::Result gjk_res = GJK::collide(bodyA->get_collider(), bodyA->get_transform(),
										   bodyB->get_collider(), bodyB->get_transform());
		if (!gjk_res.colliding) continue;

		ContactManifold m;
		m.body_a = pair.a;
		m.body_b = pair.b;
		m.point_a = gjk_res.closest_a;
		m.point_b = gjk_res.closest_b;
		m.normal = gjk_res.normal;          // from B to A
		m.distance = gjk_res.distance;      // negative for penetration
		m.friction = 0.5;                   // should come from material
		m.restitution = 0.0;
		m.normal_impulse = 0.0;
		m.tangent_impulse = Vector3();
		manifolds.push_back(m);
	}
}

void RigidSolver::solve(real_t p_sub_dt) {
	if (active_rigid_bodies.is_empty() || manifolds.is_empty()) return;

	// Build contact islands and solve them with sequential impulses
	ConstraintIslandSolver island_solver;
	island_solver.velocity_iterations = 2;
	island_solver.position_iterations = 4;

	// Prepare an array of Ref<RigidEntity> for island_solver
	LocalVector<Ref<RigidEntity>> refs;
	for (Ref<RigidEntity> &b : active_rigid_bodies) refs.push_back(b);

	// Transfer manifolds into a format that island_solver expects
	LocalVector<ConstraintIslandSolver::ContactManifold> island_manifolds;
	for (ContactManifold &m : manifolds) {
		ConstraintIslandSolver::ContactManifold im;
		im.body_a = m.body_a;
		im.body_b = m.body_b;
		im.point_a = m.point_a;
		im.point_b = m.point_b;
		im.normal = m.normal;
		im.distance = m.distance;
		im.friction = m.friction;
		im.restitution = m.restitution;
		im.normal_impulse = m.normal_impulse;
		im.tangent_impulse = m.tangent_impulse;
		island_manifolds.push_back(im);
	}

	// Solve islands (the function expects the entities list and manifolds)
	// We'll call a method that builds islands internally using the manifolds.
	// (ConstraintIslandSolver::solve is designed for full pipeline, but we
	// can directly use its internal methods by copying the logic here.)
	// For brevity, we construct the Island manually:
	ConstraintIslandSolver::Island island;
	island.bodies = refs;
	island.contacts = island_manifolds;
	island_solver.solve_island(island, p_sub_dt);

	// Copy back warm‑starting impulses
	for (int i = 0; i < island_manifolds.size(); ++i) {
		if (i < manifolds.size()) {
			manifolds[i].normal_impulse = island_manifolds[i].normal_impulse;
			manifolds[i].tangent_impulse = island_manifolds[i].tangent_impulse;
		}
	}
}

void RigidSolver::post_step(real_t p_sub_dt) {
	for (Ref<RigidEntity> &body : active_rigid_bodies) {
		body->integrate_position(p_sub_dt);
	}
}

} // namespace genesis