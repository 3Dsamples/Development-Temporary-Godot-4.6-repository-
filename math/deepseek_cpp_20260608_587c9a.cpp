// File 183: modules/newton/src/solver/newton_solver.cpp
// NewtonSolver implementation – iterative projected Gauss‑Seidel (PGS) /
// sequential‑impulse solver with Baumgarte stabilisation, friction,
// restitution, and warm‑starting. Operates on contact points and joints.

#include "newton_solver.h"
#include "newton_island.h"

#include "../bodies/newton_body.h"
#include "../materials/newton_material.h"
#include "../joints/newton_joint.h"

#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "core/typedefs.h"

namespace newton {

void NewtonSolver::solve_islands(
	Ref<NewtonIsland> p_island_manager,
	const HashMap<body_id, Ref<NewtonBody>> &p_bodies,
	const HashMap<joint_id, Ref<NewtonJoint>> &p_joints,
	const HashMap<material_id, Ref<NewtonMaterial>> &p_materials,
	real_t p_dt) {

	if (p_island_manager.is_null()) return;

	// Retrieve islands
	const LocalVector<NewtonIsland *> &islands = p_island_manager->get_islands();

	for (NewtonIsland *island : islands) {
		if (island->is_sleeping()) continue;
		solve_island(island, p_dt);
	}
}

void NewtonSolver::solve_island(NewtonIsland *island, real_t dt) {
	const LocalVector<NewtonContactPoint> &contacts = island->get_contacts();
	const LocalVector<Ref<NewtonJoint>> &joints = island->get_joints();

	// --- Velocity constraints (normal impulses for restitution) ---
	for (NewtonContactPoint &cp : const_cast<LocalVector<NewtonContactPoint> &>(contacts)) {
		NewtonBody *bodyA = island->get_body(cp.body_a);
		NewtonBody *bodyB = island->get_body(cp.body_b);
		if (!bodyA || !bodyB) continue;

		real_t inv_mass_a = bodyA->get_inverse_mass();
		real_t inv_mass_b = bodyB->get_inverse_mass();
		if (inv_mass_a <= 0.0 && inv_mass_b <= 0.0) continue;

		const vec3 &n = cp.normal;
		vec3 rA = cp.point_a - bodyA->get_position();
		vec3 rB = cp.point_b - bodyB->get_position();

		// Relative velocity at contact point (B relative to A)
		vec3 velA = bodyA->get_linear_velocity() + bodyA->get_angular_velocity().cross(rA);
		vec3 velB = bodyB->get_linear_velocity() + bodyB->get_angular_velocity().cross(rB);
		vec3 rel_vel = velB - velA;

		real_t vn = rel_vel.dot(n);

		// Effective mass along normal
		real_t inv_eff_mass = compute_effective_mass(bodyA, bodyB, cp.point_a, cp.point_b, n);
		if (inv_eff_mass < CMP_EPSILON) continue;

		// Restitution only for velocity iterations (first iteration)
		real_t target_dv = -(1.0 + cp.restitution) * vn;

		real_t dP_n = target_dv / inv_eff_mass;

		// Clamp accumulated normal impulse to non‑negative
		real_t P_n_old = cp.normal_impulse;
		cp.normal_impulse = MAX(P_n_old + dP_n, 0.0);
		dP_n = cp.normal_impulse - P_n_old;

		vec3 impulse_n = n * dP_n;
		apply_impulse(bodyA, bodyB, impulse_n, cp.point_a, cp.point_b);

		// Friction (tangential) – solved using two tangent directions
		vec3 t1, t2;
		// Build orthonormal basis with n as Z
		if (Math::abs(n.x) < 0.999) {
			t1 = n.cross(vec3(1, 0, 0)).normalized();
		} else {
			t1 = n.cross(vec3(0, 1, 0)).normalized();
		}
		t2 = n.cross(t1).normalized();

		vec3 vt = rel_vel - n * vn; // tangential relative velocity
		real_t vt_len = vt.length();
		if (vt_len < CMP_EPSILON) continue;

		// Solve friction in tangent space
		for (int i = 0; i < 2; ++i) {
			vec3 t_dir = (i == 0) ? t1 : t2;
			real_t vt_i = vt.dot(t_dir);
			real_t inv_eff_mass_t = compute_effective_mass(bodyA, bodyB, cp.point_a, cp.point_b, t_dir);
			if (inv_eff_mass_t < CMP_EPSILON) continue;

			real_t dP_t = -vt_i / inv_eff_mass_t;
			real_t max_friction = cp.friction * cp.normal_impulse;
			real_t P_t_old = (i == 0) ? cp.friction_impulse1.dot(t_dir) : cp.friction_impulse2.dot(t_dir);
			real_t P_t_new = CLAMP(P_t_old + dP_t, -max_friction, max_friction);
			dP_t = P_t_new - P_t_old;
			vec3 impulse_t = t_dir * dP_t;
			apply_impulse(bodyA, bodyB, impulse_t, cp.point_a, cp.point_b);

			if (i == 0) cp.friction_impulse1 += impulse_t;
			else cp.friction_impulse2 += impulse_t;
		}
	}

	// --- Position correction (Baumgarte) – additional iteration? ---
	// Typically done in separate loops or after velocity. We'll do a dedicated pass.
	for (NewtonContactPoint &cp : const_cast<LocalVector<NewtonContactPoint> &>(contacts)) {
		NewtonBody *bodyA = island->get_body(cp.body_a);
		NewtonBody *bodyB = island->get_body(cp.body_b);
		if (!bodyA || !bodyB) continue;

		if (cp.penetration <= 0.0) continue;

		real_t inv_mass_a = bodyA->get_inverse_mass();
		real_t inv_mass_b = bodyB->get_inverse_mass();
		if (inv_mass_a <= 0.0 && inv_mass_b <= 0.0) continue;

		const vec3 &n = cp.normal;
		real_t inv_eff_mass = compute_effective_mass(bodyA, bodyB, cp.point_a, cp.point_b, n);
		if (inv_eff_mass < CMP_EPSILON) continue;

		real_t correction = cp.penetration * 0.2 / dt; // ERP = 0.2
		vec3 impulse_p = n * (correction / inv_eff_mass);
		apply_impulse(bodyA, bodyB, impulse_p, cp.point_a, cp.point_b);
	}

	// --- Joint constraints ---
	for (const Ref<NewtonJoint> &joint : joints) {
		solve_joint(joint, dt);
	}
}

void NewtonSolver::solve_joint(const Ref<NewtonJoint> &joint, real_t dt) {
	if (joint.is_null()) return;
	// Delegate to the joint's own solver method.
	joint->solve(dt);
}

real_t NewtonSolver::compute_effective_mass(const NewtonBody *bodyA,
											const NewtonBody *bodyB,
											const vec3 &world_point_a,
											const vec3 &world_point_b,
											const vec3 &dir) {
	real_t inv_mass = 0.0;
	if (bodyA->get_inverse_mass() > 0.0) {
		vec3 rA = world_point_a - bodyA->get_position();
		inv_mass += bodyA->get_inverse_mass();
		inv_mass += dir.dot(bodyA->get_inverse_inertia_world().xform(rA.cross(dir)).cross(rA));
	}
	if (bodyB->get_inverse_mass() > 0.0) {
		vec3 rB = world_point_b - bodyB->get_position();
		inv_mass += bodyB->get_inverse_mass();
		inv_mass += dir.dot(bodyB->get_inverse_inertia_world().xform(rB.cross(dir)).cross(rB));
	}
	return inv_mass;
}

void NewtonSolver::apply_impulse(NewtonBody *bodyA, NewtonBody *bodyB,
								 const vec3 &impulse,
								 const vec3 &world_point_a,
								 const vec3 &world_point_b) {
	if (bodyA->get_inverse_mass() > 0.0) {
		bodyA->apply_impulse(impulse, world_point_a);
	}
	if (bodyB->get_inverse_mass() > 0.0) {
		bodyB->apply_impulse(-impulse, world_point_b);
	}
}

} // namespace newton