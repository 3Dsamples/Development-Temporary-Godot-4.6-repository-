// File 10: modules/gaia/src/framework/world.h

#ifndef GAIA_FRAMEWORK_WORLD_H
#define GAIA_FRAMEWORK_WORLD_H

#include "../collision_detector/broad_phase.h"
#include "../collision_detector/narrow_phase.h"
#include "../collision_detector/contact.h"
#include "../collision_detector/collision_object.h"

#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/typedefs.h"

// Forward declarations for body/constraint types
namespace gaia {
    class RigidBody;
    class SoftBody;
    class Constraint;
    class PBDSolver;
    class VBDSolver;
}

namespace gaia::framework {

/**
 * World – the container for all simulation entities and the main
 * simulation loop. It owns the broad phase, triggers collision
 * detection, applies forces, and invokes the solver.
 */
class World {
public:
    World();
    ~World();

    // --- Body management ---
    uint32_t add_body(RigidBody *p_body);
    void remove_body(uint32_t p_handle);
    RigidBody *get_body(uint32_t p_handle) const;

    // --- Constraint management ---
    uint32_t add_constraint(Constraint *p_constraint);
    void remove_constraint(uint32_t p_handle);
    Constraint *get_constraint(uint32_t p_handle) const;

    // --- Global parameters ---
    void set_gravity(const Vector3 &p_gravity) { gravity = p_gravity; }
    Vector3 get_gravity() const { return gravity; }

    void set_sub_steps(int p_count) { sub_step_count = MAX(p_count, 1); }
    int get_sub_steps() const { return sub_step_count; }

    void set_solver_iterations(int p_iter) { solver_iterations = MAX(p_iter, 1); }
    int get_solver_iterations() const { return solver_iterations; }

    // --- Main step ---
    void step(real_t p_dt);

    // --- Debug / query ---
    collision::BroadPhase &get_broad_phase() { return broad_phase; }

private:
    void detect_collisions();
    void apply_forces_and_gravity(real_t sub_dt);
    void update_constraints(real_t sub_dt);
    void solve_constraints(real_t sub_dt);
    void integrate_velocities(real_t sub_dt);
    void integrate_positions(real_t sub_dt);

    // Owner storage for bodies / constraints
    struct BodyEntry {
        RigidBody *body;
        uint32_t broad_handle;
    };
    struct ConstraintEntry {
        Constraint *constraint;
    };

    HashMap<uint32_t, BodyEntry> bodies;
    HashMap<uint32_t, ConstraintEntry> constraints;
    uint32_t next_handle;

    Vector3 gravity;
    int sub_step_count;
    int solver_iterations;

    collision::BroadPhase broad_phase;
    PBDSolver *pbd_solver;   // optionally assigned from outside
    VBDSolver *vbd_solver;
};

} // namespace gaia::framework

#endif // GAIA_FRAMEWORK_WORLD_H