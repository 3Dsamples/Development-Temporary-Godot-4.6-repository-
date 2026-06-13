// File 336: modules/wicked/src/solver/wicked_solver.h
// WickedSolver – sequential‑impulse constraint solver for WickedEngine.
// Handles contacts (normal + friction) and joints via projected Gauss‑Seidel.
// Uses warm‑starting, Baumgarte ERP, CFM, and split impulse for penetration.
// Small hot‑path helpers are defined inline in the header for maximum speed.

#ifndef WICKED_SOLVER_WICKED_SOLVER_H
#define WICKED_SOLVER_WICKED_SOLVER_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "../core/wicked_types.h"
#include "../core/wicked_constants.h"
#include "../bodies/wicked_body.h"
#include "../joints/wicked_joint.h"
#include "../materials/wicked_material.h"

namespace wicked {

// Forward declaration
class WickedIsland;

// ---------------------------------------------------------------------------
// Inline helper: effective inverse mass along a unit direction (hot path)
// ---------------------------------------------------------------------------
inline real_t compute_effective_inv_mass(const WickedBody *bodyA,
                                         const WickedBody *bodyB,
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

// ---------------------------------------------------------------------------
// Inline helper: apply an impulse to two bodies at given world points
// ---------------------------------------------------------------------------
inline void apply_pair_impulse(WickedBody *bodyA, WickedBody *bodyB,
                                const vec3 &impulse,
                                const vec3 &pointA,
                                const vec3 &pointB) {
    if (bodyA->get_inverse_mass() > 0.0) bodyA->apply_impulse( impulse, pointA);
    if (bodyB->get_inverse_mass() > 0.0) bodyB->apply_impulse(-impulse, pointB);
}

// ---------------------------------------------------------------------------
// Inline helper: relative velocity at contact point (B relative to A)
// ---------------------------------------------------------------------------
inline vec3 relative_velocity(const WickedBody *bodyA, const WickedBody *bodyB,
                               const vec3 &pointA, const vec3 &pointB) {
    vec3 rA = pointA - bodyA->get_position();
    vec3 rB = pointB - bodyB->get_position();
    vec3 velA = bodyA->get_linear_velocity() + bodyA->get_angular_velocity().cross(rA);
    vec3 velB = bodyB->get_linear_velocity() + bodyB->get_angular_velocity().cross(rB);
    return velB - velA;
}

// ---------------------------------------------------------------------------
// Contact point stored during solving (identical to world's contact)
// ---------------------------------------------------------------------------
struct WickedContactPoint {
    body_id body_a, body_b;
    vec3 point_a, point_b;
    vec3 normal;            // from B to A
    real_t penetration;     // positive = interpenetration
    real_t friction;
    real_t restitution;
    // Warm‑start accumulators
    real_t normal_impulse;
    vec3 friction_impulse;
    vec3 tangent1, tangent2;
};

// ---------------------------------------------------------------------------
// WickedSolver class
// ---------------------------------------------------------------------------
class WickedSolver : public RefCounted {
    GDCLASS(WickedSolver, RefCounted);

public:
    WickedSolver();
    virtual ~WickedSolver();

    void set_iterations(int p_iter) { iterations = CLAMP(p_iter, 1, MAX_SOLVER_ITERATIONS); }
    int get_iterations() const { return iterations; }

    // Set solver parameters (ERP, ERP2, CFM)
    void set_erp(real_t p_erp, real_t p_erp2, real_t p_cfm);

    // Main entry: solve all islands built by the island manager.
    void solve_islands(const Ref<WickedIsland> &p_island_manager,
                       const HashMap<body_id, Ref<WickedBody>> &p_bodies,
                       const HashMap<joint_id, Ref<WickedJoint>> &p_joints,
                       const HashMap<material_id, Ref<WickedMaterial>> &p_materials,
                       real_t p_dt);

private:
    // Solve a single island
    void solve_island(WickedIsland *island, real_t dt);
    // Solve a single contact constraint (normal + tangential friction)
    void solve_contact(WickedContactPoint &cp, WickedBody *bodyA, WickedBody *bodyB,
                       real_t dt, int iteration);

    int iterations = DEFAULT_SOLVER_ITERATIONS;
    real_t erp = DEFAULT_ERP;
    real_t erp2 = DEFAULT_ERP2;
    real_t cfm = DEFAULT_TAU;
};

} // namespace wicked

#endif // WICKED_SOLVER_WICKED_SOLVER_H