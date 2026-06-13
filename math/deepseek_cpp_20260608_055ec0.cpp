// File 337: modules/wicked/src/solver/wicked_solver.cpp
// Implementation of WickedSolver – sequential‑impulse PGS with warm‑starting,
// Baumgarte ERP/ERP2, CFM, friction, and joint solving. All hot‑path helpers
// (effective mass, relative velocity, impulse application) are defined inline
// in the header to avoid call overhead.

#include "wicked_solver.h"
#include "wicked_island.h"
#include "../bodies/wicked_body.h"
#include "../joints/wicked_joint.h"
#include "../materials/wicked_material.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "core/typedefs.h"

namespace wicked {

WickedSolver::WickedSolver() : iterations(DEFAULT_SOLVER_ITERATIONS), erp(DEFAULT_ERP), erp2(DEFAULT_ERP2), cfm(DEFAULT_TAU) {}
WickedSolver::~WickedSolver() {}

void WickedSolver::set_erp(real_t p_erp, real_t p_erp2, real_t p_cfm) {
    erp  = CLAMP(p_erp,  0.0, 1.0);
    erp2 = CLAMP(p_erp2, 0.0, 1.0);
    cfm  = MAX(p_cfm, 0.0);
}

void WickedSolver::solve_islands(const Ref<WickedIsland> &p_island_manager,
                                 const HashMap<body_id, Ref<WickedBody>> &p_bodies,
                                 const HashMap<joint_id, Ref<WickedJoint>> &p_joints,
                                 const HashMap<material_id, Ref<WickedMaterial>> &p_materials,
                                 real_t p_dt) {
    if (p_island_manager.is_null()) return;
    const LocalVector<WickedIsland *> &islands = p_island_manager->get_islands();
    for (WickedIsland *island : islands) {
        if (!island->is_sleeping()) {
            solve_island(island, p_dt);
        }
    }
}

void WickedSolver::solve_island(WickedIsland *island, real_t dt) {
    LocalVector<WickedContactPoint> &contacts = island->get_contacts();
    LocalVector<Ref<WickedJoint>> &joints = island->get_joints();

    // Sequential impulse iterations (velocity + position)
    for (int iter = 0; iter < iterations; ++iter) {
        // --- Normal and friction impulses for all contacts ---
        for (WickedContactPoint &cp : contacts) {
            WickedBody *bodyA = island->get_body(cp.body_a);
            WickedBody *bodyB = island->get_body(cp.body_b);
            if (!bodyA || !bodyB) continue;
            if (bodyA->get_inverse_mass() <= 0.0 && bodyB->get_inverse_mass() <= 0.0) continue;

            solve_contact(cp, bodyA, bodyB, dt, iter);
        }

        // --- Joint constraints ---
        for (Ref<WickedJoint> &joint : joints) {
            if (joint.is_null() || !joint->is_enabled()) continue;
            WickedBody *bodyA = island->get_body(joint->get_body_a());
            WickedBody *bodyB = island->get_body(joint->get_body_b());
            joint->solve(bodyA, bodyB, dt);
        }
    }
}

void WickedSolver::solve_contact(WickedContactPoint &cp, WickedBody *bodyA, WickedBody *bodyB,
                                 real_t dt, int iteration) {
    // Effective mass along normal
    real_t inv_eff_n = compute_effective_inv_mass(bodyA, bodyB, cp.point_a, cp.point_b, cp.normal);
    if (inv_eff_n < CMP_EPSILON) return;

    // Relative velocity at contact point
    vec3 rel_vel = relative_velocity(bodyA, bodyB, cp.point_a, cp.point_b);
    real_t vn = rel_vel.dot(cp.normal);

    // Restitution only on first iteration
    real_t restitution = (iteration == 0) ? cp.restitution : 0.0f;
    real_t target_dv = -(1.0f + restitution) * vn;

    // Baumgarte position correction (ERP) – apply after first iteration
    if (iteration > 0 && cp.penetration > 0.0f) {
        // Split impulse: erp * penetration / dt + erp2 * vn
        target_dv += erp * cp.penetration / dt;
        // Target also adjusts relative velocity to reduce penetration (ERP2)
        target_dv += erp2 * vn; // erp2 applied to vn (velocity correction)
    }

    // Incorporate CFM (constraint force mixing) – adds compliance
    real_t normal_mass = inv_eff_n + cfm;

    real_t dP_n = target_dv / normal_mass;
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
        real_t inv_eff_t = compute_effective_inv_mass(bodyA, bodyB, cp.point_a, cp.point_b, t_dir);
        if (inv_eff_t > CMP_EPSILON) {
            real_t dP_t = -vt_len / (inv_eff_t + cfm);
            real_t max_friction = cp.friction * cp.normal_impulse;
            real_t P_t_old = cp.friction_impulse.length();
            real_t P_t_new = CLAMP(P_t_old + dP_t, 0.0f, max_friction);
            dP_t = P_t_new - P_t_old;
            vec3 impulse_t = t_dir * dP_t;
            apply_pair_impulse(bodyA, bodyB, impulse_t, cp.point_a, cp.point_b);
            cp.friction_impulse += impulse_t;
        }
    }
}

} // namespace wicked