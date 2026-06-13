// File 227: modules/newton/src/solver/newton_solver_island.cpp
// NewtonSolver island solving implementation – iterates over all contacts
// and joints in each island, applying sequential impulses for convergence.

#include "newton_solver.h"
#include "newton_island.h"
#include "../bodies/newton_body.h"
#include "../materials/newton_material.h"
#include "../joints/newton_joint.h"
#include "../collision/newton_contact.h"

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

	const LocalVector<NewtonIsland *> &islands = p_island_manager->get_islands();

	for (NewtonIsland *island : islands) {
		if (island->is_sleeping()) continue;
		solve_island(island, p_dt);
	}
}

void NewtonSolver::solve_island(NewtonIsland *island, real_t dt) {
	const LocalVector<NewtonContactPoint> &contacts = island->get_contacts();
	const LocalVector<Ref<NewtonJoint>> &joints = island->get_joints();

	// Build a set of body pointers for this island for fast lookup.
	// All bodies referenced by contacts and joints must be present.
	// Also grab material IDs for each body pair.
	HashMap<std::pair<body_id, body_id>, material_id> pair_material_map;

	// --- Solver iteration loop ---
	for (int iter = 0; iter < iterations; ++iter) {
		// Process velocity constraints (restitution, friction)
		for (NewtonContactPoint &cp : const_cast<LocalVector<NewtonContactPoint> &>(contacts)) {
			NewtonBody *bodyA = island->get_body(cp.body_a);
			NewtonBody *bodyB = island->get_body(cp.body_b);
			if (!bodyA || !bodyB) continue;
			if (bodyA->get_inverse_mass() <= 0.0 && bodyB->get_inverse_mass() <= 0.0) continue;

			// Compute effective mass for normal direction
			real_t inv_eff_mass = compute_effective_mass(bodyA, bodyB, cp.point_a, cp.point_b, cp.normal);
			if (inv_eff_mass < CMP_EPSILON) continue;

			// Compute relative velocity at contact point
			vec3 rA = cp.point_a - bodyA->get_position();
			vec3 rB = cp.point_b - bodyB->get_position();
			vec3 velA = bodyA->get_linear_velocity() + bodyA->get_angular_velocity().cross(rA);
			vec3 velB = bodyB->get_linear_velocity() + bodyB->get_angular_velocity().cross(rB);
			vec3 rel_vel = velB - velA;

			real_t vn = rel_vel.dot(cp.normal);

			// Restitution is applied on the first iteration only
			real_t restitution = (iter == 0) ? cp.restitution : 0.0f;
			real_t target_dv = -(1.0f + restitution) * vn;

			// Add Baumgarte position correction (ERP) on later iterations
			if (iter > 0 && cp.penetration > 0.0f) {
				real_t erp = 0.2f / dt;
				target_dv += cp.penetration * erp;
			}

			real_t dP_n = target_dv / inv_eff_mass;

			// Clamp accumulated normal impulse
			real_t P_n_old = cp.normal_impulse;
			cp.normal_impulse = MAX(P_n_old + dP_n, 0.0f);
			dP_n = cp.normal_impulse - P_n_old;

			vec3 impulse_n = cp.normal * dP_n;
			apply_impulse(bodyA, bodyB, impulse_n, cp.point_a, cp.point_b);

			// Friction in tangent plane
			vec3 vt = rel_vel - cp.normal * vn;
			real_t vt_len = vt.length();
			if (vt_len > CMP_EPSILON) {
				vec3 tangent_dir = vt / vt_len;
				real_t inv_eff_mass_t = compute_effective_mass(bodyA, bodyB, cp.point_a, cp.point_b, tangent_dir);
				if (inv_eff_mass_t > CMP_EPSILON) {
					real_t dP_t = -vt_len / inv_eff_mass_t;
					real_t max_friction = cp.friction * cp.normal_impulse;
					real_t P_t_old = cp.friction_impulse.length();
					real_t P_t_new = CLAMP(P_t_old + dP_t, 0.0f, max_friction);
					dP_t = P_t_new - P_t_old;
					vec3 impulse_t = tangent_dir * dP_t;
					apply_impulse(bodyA, bodyB, impulse_t, cp.point_a, cp.point_b);
					cp.friction_impulse += impulse_t;
				}
			}
		}

		// Solve all joint constraints in the island
		for (const Ref<NewtonJoint> &joint : joints) {
			if (joint.is_null() || !joint->is_enabled()) continue;
			// Set body pointers from island before solving
			NewtonBody *bodyA = island->get_body(joint->get_body_a());
			NewtonBody *bodyB = island->get_body(joint->get_body_b());
			joint->solve(bodyA, bodyB, dt);
		}
	}
}

} // namespace newton