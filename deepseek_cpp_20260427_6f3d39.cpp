// File 286: modules/vienna/src/solver/vienna_solver.h
// ViennaSolver – solves contacts and joints using sequential impulses
// with warm‑starting, Baumgarte stabilisation, friction, and restitution.
// Operates on islands built by ViennaIsland.

#ifndef VIENNA_SOLVER_VIENNA_SOLVER_H
#define VIENNA_SOLVER_VIENNA_SOLVER_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"
#include "../bodies/vienna_body.h"
#include "../joints/vienna_joint.h"
#include "../materials/vienna_material.h"

namespace vienna {

// Forward declaration
class ViennaIsland;

// ---------------------------------------------------------------------------
// Contact point stored during solving
// ---------------------------------------------------------------------------
struct ViennaContactPoint {
	body_id body_a;
	body_id body_b;
	vec3 point_a;         // world‑space contact point on A
	vec3 point_b;         // world‑space contact point on B
	vec3 normal;          // from B to A
	real_t penetration;   // positive = interpenetration
	real_t friction;      // combined friction coefficient
	real_t restitution;   // coefficient of restitution
	// Warm‑start accumulators
	real_t normal_impulse;
	vec3 friction_impulse1; // tangent 1
	vec3 friction_impulse2; // tangent 2
	// Tangent directions
	vec3 tangent1;
	vec3 tangent2;

	ViennaContactPoint() : body_a(0), body_b(0), point_a(), point_b(), normal(),
		penetration(0.0), friction(0.5), restitution(0.0),
		normal_impulse(0.0), friction_impulse1(), friction_impulse2(),
		tangent1(), tangent2() {}
};

// ---------------------------------------------------------------------------
// ViennaSolver class
// ---------------------------------------------------------------------------
class ViennaSolver : public RefCounted {
	GDCLASS(ViennaSolver, RefCounted);

public:
	ViennaSolver() : iterations(16), warm_starting(true) {}
	virtual ~ViennaSolver() {}

	void set_iterations(int p_iter) { iterations = CLAMP(p_iter, 1, MAX_SOLVER_ITERATIONS); }
	int get_iterations() const { return iterations; }

	// Main entry: solve all islands built by the island manager.
	void solve_islands(Ref<ViennaIsland> p_island_manager,
					   const HashMap<body_id, Ref<ViennaBody>> &p_bodies,
					   const HashMap<joint_id, Ref<ViennaJoint>> &p_joints,
					   const HashMap<material_id, Ref<ViennaMaterial>> &p_materials,
					   real_t p_dt);

private:
	// Solve a single island.  Contacts and joints are processed in PGS order.
	void solve_island(ViennaIsland *island, real_t dt);

	// Effective mass for a unit impulse along `dir` at contact points.
	static real_t compute_effective_mass(const ViennaBody *bodyA,
										const ViennaBody *bodyB,
										const vec3 &pointA,
										const vec3 &pointB,
										const vec3 &dir);

	// Apply an impulse to two bodies at given points.
	static void apply_pair_impulse(ViennaBody *bodyA, ViennaBody *bodyB,
								   const vec3 &impulse,
								   const vec3 &pointA, const vec3 &pointB);

	int iterations;
	bool warm_starting;
};

// ===========================================================================
// Inline implementations
// ===========================================================================

inline void ViennaSolver::solve_islands(
	Ref<ViennaIsland> p_island_manager,
	const HashMap<body_id, Ref<ViennaBody>> &p_bodies,
	const HashMap<joint_id, Ref<ViennaJoint>> &p_joints,
	const HashMap<material_id, Ref<ViennaMaterial>> &p_materials,
	real_t p_dt) {

	if (p_island_manager.is_null()) return;
	const LocalVector<ViennaIsland *> &islands = p_island_manager->get_islands();
	for (ViennaIsland *island : islands) {
		if (island->is_sleeping()) continue;
		solve_island(island, p_dt);
	}
}

inline void ViennaSolver::solve_island(ViennaIsland *island, real_t dt) {
	LocalVector<ViennaContactPoint> &contacts = island->get_contacts();
	LocalVector<Ref<ViennaJoint>> &joints = island->get_joints();

	for (int iter = 0; iter < iterations; ++iter) {
		// --- Normal and friction impulses on all contacts ---
		for (ViennaContactPoint &cp : contacts) {
			ViennaBody *bodyA = island->get_body(cp.body_a);
			ViennaBody *bodyB = island->get_body(cp.body_b);
			if (!bodyA || !bodyB) continue;
			if (bodyA->get_inverse_mass() <= 0.0 && bodyB->get_inverse_mass() <= 0.0) continue;

			real_t inv_eff_n = compute_effective_mass(bodyA, bodyB, cp.point_a, cp.point_b, cp.normal);
			if (inv_eff_n < CMP_EPSILON) continue;

			vec3 rA = cp.point_a - bodyA->get_position();
			vec3 rB = cp.point_b - bodyB->get_position();
			vec3 velA = bodyA->get_linear_velocity() + bodyA->get_angular_velocity().cross(rA);
			vec3 velB = bodyB->get_linear_velocity() + bodyB->get_angular_velocity().cross(rB);
			vec3 rel_vel = velB - velA;
			real_t vn = rel_vel.dot(cp.normal);

			// Restitution only on first iteration
			real_t restitution = (iter == 0) ? cp.restitution : 0.0f;
			real_t target_dv = -(1.0f + restitution) * vn;

			// Baumgarte position correction (ERP) on later iterations
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

			// Friction (tangential)
			vec3 vt = rel_vel - cp.normal * vn;
			real_t vt_len = vt.length();
			if (vt_len > CMP_EPSILON) {
				vec3 t_dir = vt / vt_len;
				real_t inv_eff_t = compute_effective_mass(bodyA, bodyB, cp.point_a, cp.point_b, t_dir);
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

		// --- Joint constraints ---
		for (Ref<ViennaJoint> &joint : joints) {
			if (joint.is_null() || !joint->is_enabled()) continue;
			ViennaBody *bodyA = island->get_body(joint->get_body_a());
			ViennaBody *bodyB = island->get_body(joint->get_body_b());
			joint->solve(bodyA, bodyB, dt);
		}
	}
}

// Compute effective inverse mass for a unit impulse along `dir`.
inline real_t ViennaSolver::compute_effective_mass(const ViennaBody *bodyA,
												   const ViennaBody *bodyB,
												   const vec3 &pointA,
												   const vec3 &pointB,
												   const vec3 &dir) {
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

// Apply impulse to two bodies at given points.
inline void ViennaSolver::apply_pair_impulse(ViennaBody *bodyA, ViennaBody *bodyB,
											 const vec3 &impulse,
											 const vec3 &pointA, const vec3 &pointB) {
	if (bodyA->get_inverse_mass() > 0.0) bodyA->apply_impulse( impulse, pointA);
	if (bodyB->get_inverse_mass() > 0.0) bodyB->apply_impulse(-impulse, pointB);
}

} // namespace vienna

#endif // VIENNA_SOLVER_VIENNA_SOLVER_H