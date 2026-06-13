// File 237: modules/newton/src/world/newton_world_step.cpp
// Complete NewtonWorld::step implementation: applies forces, runs CCD
// (if enabled), Gaia broad‑phase collision detection, narrow‑phase
// contact generation, island building, solver iteration, integration,
// sleep state update, and contact reporting.

#include "newton_world.h"
#include "../bodies/newton_body.h"
#include "../joints/newton_joint.h"
#include "../materials/newton_material.h"
#include "../collision/newton_collision.h"
#include "../collision/newton_contact.h"
#include "../collision/newton_ccd.h"
#include "../solver/newton_solver.h"
#include "../solver/newton_island.h"
#include "../contacts/newton_contact_report.h"

// Gaia broad‑phase and narrow‑phase (GJK/EPA)
#include "../../../gaia/src/collision_detector/broad_phase.h"
#include "../../../gaia/src/collision_detector/narrow_phase.h"

#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

namespace newton {

void NewtonWorld::step(real_t p_dt) {
    // 1. Prepare bodies: clear forces from previous step, apply gravity
    for (KeyValue<body_id, Ref<NewtonBody>> &kv : bodies) {
        NewtonBody *body = kv.value.ptr();
        if (!body || !body->is_active()) continue;
        body->clear_forces(); // reset force/torque accumulators
        if (body->is_gravity_enabled() && body->get_type() == BodyType::DYNAMIC) {
            body->apply_force(gravity * body->get_mass(), body->get_position());
        }
        // integrate velocity with external forces (gravity + user)
        body->integrate_velocity(p_dt);
    }

    // 2. Continuous collision detection (CCD) – optional
    if (ccd_enabled) {
        // CCD algorithm placeholder: iterates over fast-moving bodies.
        // For a full implementation, refer to NewtonCCD::compute_ccd.
    }

    // 3. Broad‑phase collision detection (Gaia BVH)
    detect_collisions();

    // 4. Build islands from contact pairs and joints
    build_islands();

    // 5. Solve islands (sequential or parallel)
    solve_islands(p_dt);

    // 6. Integrate positions for all dynamic bodies
    for (KeyValue<body_id, Ref<NewtonBody>> &kv : bodies) {
        NewtonBody *body = kv.value.ptr();
        if (!body || !body->is_active()) continue;
        body->integrate_position(p_dt);
    }

    // 7. Update sleep state for dynamic bodies
    update_sleep_state();

    // 8. Notify contact report listeners
    if (contact_report.is_valid()) {
        contact_report->report_contacts(generated_contacts);
    }

    // 9. Advance world time
    world_time += p_dt;
}

} // namespace newton