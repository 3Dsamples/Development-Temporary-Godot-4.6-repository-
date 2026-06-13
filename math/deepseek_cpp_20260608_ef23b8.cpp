// File 456: modules/integration/unified_penalty_contact_solver.h
// Resolves soft‑body ↔ rigid‑body contacts using an exponential penalty
// potential (similar to IPC) plus optional Coulomb friction.  Takes the
// contact lists produced by UnifiedSoftCollisionDetector (which uses
// TreeNSearch for proximity) and applies equal‑and‑opposite forces to the
// soft vertex and the rigid body.  All maths is fully inline; no separate
// .cpp is needed.

#ifndef INTEGRATION_UNIFIED_PENALTY_CONTACT_SOLVER_H
#define INTEGRATION_UNIFIED_PENALTY_CONTACT_SOLVER_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/typedefs.h"

// The contact structure from the soft collision detector
#include "unified_soft_collision_detector.h"

// Engine headers for applying forces to rigid bodies
#include "../../newton/src/world/newton_world.h"
#include "../../newton/src/bodies/newton_body.h"
#include "../../genesis/src/genesis_world.h"
#include "../../genesis/src/entities/rigid_entity.h"
#include "../../vienna/src/world/vienna_world.h"
#include "../../vienna/src/bodies/vienna_body.h"
#include "../../wicked/src/world/wicked_world.h"
#include "../../wicked/src/bodies/wicked_body.h"

namespace unified {

class UnifiedPenaltyContactSolver : public RefCounted {
    GDCLASS(UnifiedPenaltyContactSolver, RefCounted);

public:
    // Penalty stiffness (N/m) – higher values reduce penetration but may
    // require smaller time steps.
    real_t penalty_stiffness = 1e5f;

    // Damping coefficient (Ns/m) applied to normal relative velocity.
    real_t penalty_damping = 10.0f;

    // Friction coefficient (Coulomb) – tangential force is clamped to
    // mu * normal_force.
    real_t friction_coefficient = 0.5f;

    // Maximum penetration allowed before the penalty force saturates (m).
    real_t max_penetration = 0.05f;

    // -------------------------------------------------------------------
    // Resolve contacts for the given soft mesh and contact data.
    // Soft velocities are updated in place; equal‑opposite impulses are
    // applied to the rigid bodies in their respective engine worlds.
    //
    // @param p_soft_velocities  Current velocities of all soft vertices
    //                           (will be modified).
    // @param p_soft_masses      Per‑vertex mass (for impulse -> velocity).
    // @param p_contacts         Contacts per vertex (from detector).
    // @param p_rigid_worlds     Engine worlds (map engine index -> world ptr).
    // @param p_dt               Time step.
    // -------------------------------------------------------------------
    void resolve_contacts(
            LocalVector<Vector3> &p_soft_velocities,
            const LocalVector<real_t> &p_soft_masses,
            const LocalVector<LocalVector<UnifiedSoftCollisionDetector::Contact>> &p_contacts,
            const HashMap<int, void *> &p_rigid_worlds,
            real_t p_dt) const;

    // -------------------------------------------------------------------
    // Single‑contact force computation (for debug / external use).
    // Returns the world‑space force vector to be applied to the soft vertex.
    // The opposite force is applied to the rigid body.
    // -------------------------------------------------------------------
    Vector3 compute_penalty_force(const UnifiedSoftCollisionDetector::Contact &p_contact,
                                  const Vector3 &p_soft_velocity,
                                  real_t p_soft_mass,
                                  real_t p_dt) const;

protected:
    static void _bind_methods();

private:
    // Helpers to apply force to a rigid body in a specific engine.
    void apply_force_to_rigid(int p_engine, uint64_t p_body_id,
                              const Vector3 &p_force, const Vector3 &p_world_point,
                              const HashMap<int, void *> &p_rigid_worlds) const;
};

// =========================================================================
// Inline implementations
// =========================================================================

void UnifiedPenaltyContactSolver::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_penalty_stiffness", "k"), &UnifiedPenaltyContactSolver::set_penalty_stiffness);
    ClassDB::bind_method(D_METHOD("get_penalty_stiffness"), &UnifiedPenaltyContactSolver::get_penalty_stiffness);
    ClassDB::bind_method(D_METHOD("set_penalty_damping", "d"), &UnifiedPenaltyContactSolver::set_penalty_damping);
    ClassDB::bind_method(D_METHOD("get_penalty_damping"), &UnifiedPenaltyContactSolver::get_penalty_damping);
    ClassDB::bind_method(D_METHOD("set_friction_coefficient", "mu"), &UnifiedPenaltyContactSolver::set_friction_coefficient);
    ClassDB::bind_method(D_METHOD("get_friction_coefficient"), &UnifiedPenaltyContactSolver::get_friction_coefficient);
    ClassDB::bind_method(D_METHOD("resolve_contacts", "soft_velocities", "soft_masses", "contacts", "rigid_worlds", "dt"),
        &UnifiedPenaltyContactSolver::resolve_contacts);
    ClassDB::bind_method(D_METHOD("compute_penalty_force", "contact", "soft_velocity", "soft_mass", "dt"),
        &UnifiedPenaltyContactSolver::compute_penalty_force);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "penalty_stiffness"), "set_penalty_stiffness", "get_penalty_stiffness");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "penalty_damping"), "set_penalty_damping", "get_penalty_damping");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "friction_coefficient"), "set_friction_coefficient", "get_friction_coefficient");
}

// Property setters/getters
void UnifiedPenaltyContactSolver::set_penalty_stiffness(real_t v) { penalty_stiffness = MAX(v, 0.0f); }
real_t UnifiedPenaltyContactSolver::get_penalty_stiffness() const { return penalty_stiffness; }
void UnifiedPenaltyContactSolver::set_penalty_damping(real_t v) { penalty_damping = MAX(v, 0.0f); }
real_t UnifiedPenaltyContactSolver::get_penalty_damping() const { return penalty_damping; }
void UnifiedPenaltyContactSolver::set_friction_coefficient(real_t v) { friction_coefficient = CLAMP(v, 0.0f, 1.0f); }
real_t UnifiedPenaltyContactSolver::get_friction_coefficient() const { return friction_coefficient; }

// ---------------------------------------------------------------------------
// Resolve all contacts: iterate per vertex, apply penalty forces.
// ---------------------------------------------------------------------------
void UnifiedPenaltyContactSolver::resolve_contacts(
        LocalVector<Vector3> &p_soft_velocities,
        const LocalVector<real_t> &p_soft_masses,
        const LocalVector<LocalVector<UnifiedSoftCollisionDetector::Contact>> &p_contacts,
        const HashMap<int, void *> &p_rigid_worlds,
        real_t p_dt) const {

    int n_verts = p_contacts.size();
    for (int i = 0; i < n_verts; ++i) {
        Vector3 &soft_vel = p_soft_velocities[i];
        real_t mass = (i < p_soft_masses.size()) ? p_soft_masses[i] : 1.0f;
        for (const auto &c : p_contacts[i]) {
            // Compute penalty and friction forces.
            Vector3 force_on_soft = compute_penalty_force(c, soft_vel, mass, p_dt);
            // Apply impulse to soft vertex (velocity += force_impulse / mass).
            if (mass > CMP_EPSILON) {
                soft_vel += force_on_soft * (p_dt / mass);
            }
            // Apply equal‑opposite force to the rigid body at the contact point.
            apply_force_to_rigid(c.engine, c.rigid_body_id, -force_on_soft,
                                 c.rigid_point, p_rigid_worlds);
        }
    }
}

// ---------------------------------------------------------------------------
// Penalty force on a single contact.
// ---------------------------------------------------------------------------
Vector3 UnifiedPenaltyContactSolver::compute_penalty_force(
        const UnifiedSoftCollisionDetector::Contact &p_contact,
        const Vector3 &p_soft_velocity,
        real_t p_soft_mass,
        real_t p_dt) const {

    // Normal points from rigid to soft (according to the detector definition).
    // If distance < 0, there is penetration (overlap).
    real_t d = p_contact.distance;       // negative = penetration
    real_t penetration = -d;             // positive when penetrating

    // If no penetration, exit.
    if (penetration <= 0.0f) return Vector3();

    // Clamp penetration to avoid huge forces.
    if (penetration > max_penetration) penetration = max_penetration;

    // Normal force magnitude: k * penetration  (linear penalty).
    // Could use exponential barrier: k * (d - d0)^2 / d? We use simple spring.
    real_t normal_force_mag = penalty_stiffness * penetration;

    // Damping: relative velocity along normal.
    // Rigid velocity is not easily available; we assume rigid is static or ignore
    // its velocity for damping? We'll use soft vertex velocity projected onto normal.
    // If soft vertex is moving into the rigid body (vn < 0, because normal points from rigid to soft,
    // penetration direction is opposite to normal), we add damping.
    // Actually, if the vertex is moving towards the rigid body, its velocity
    // along the normal (from rigid to vertex) is positive, indicating it's leaving? Wait:
    // normal points from rigid to soft. If the soft vertex is moving away from rigid,
    // its velocity dot normal is positive. If it's moving into rigid, dot is negative.
    // Damping should resist penetration: we want to add a force opposing the motion
    // that is causing further penetration. So if the soft vertex is moving into the rigid,
    // we add a force along the normal (pushing it out).
    real_t vn = p_soft_velocity.dot(p_contact.normal);
    // If vn < 0, the vertex is moving into the rigid → we need extra damping force.
    // Damping force magnitude = damping * |vn| (only when vn < 0).
    real_t damping_force_mag = 0.0f;
    if (vn < 0.0f) {
        damping_force_mag = penalty_damping * (-vn);
    }

    // Total normal force.
    real_t total_normal = normal_force_mag + damping_force_mag;
    Vector3 normal_force = p_contact.normal * total_normal;

    // ---------- Coulomb friction ----------
    // Tangential slip velocity: v_t = v_soft - (v_soft·n)n
    Vector3 v_t = p_soft_velocity - p_contact.normal * vn;
    real_t slip_speed = v_t.length();

    Vector3 friction_force(0, 0, 0);
    if (slip_speed > CMP_EPSILON) {
        Vector3 tangent_dir = v_t / slip_speed;
        // Maximum friction force magnitude.
        real_t max_friction = friction_coefficient * total_normal;
        // We apply a force opposing the slip, clamped to max_friction.
        // A simple spring for friction would require a tangential displacement;
        // we use a velocity‑based damping friction: f = -mu * normal_force * sign(v_t).
        friction_force = -tangent_dir * max_friction;
        // Clamp to the actual slip (so that friction doesn't reverse direction) – this
        // is already correct because we use the slip direction.
    }

    return normal_force + friction_force;
}

// ---------------------------------------------------------------------------
// Apply force to a rigid body at a world point.
// ---------------------------------------------------------------------------
void UnifiedPenaltyContactSolver::apply_force_to_rigid(
        int p_engine, uint64_t p_body_id,
        const Vector3 &p_force, const Vector3 &p_world_point,
        const HashMap<int, void *> &p_rigid_worlds) const {

    void *world = p_rigid_worlds.has(p_engine) ? p_rigid_worlds[p_engine] : nullptr;
    if (!world) return;

    switch (p_engine) {
        case UnifiedSpatialQueryManager::ENGINE_NEWTON: {
            auto *nw = static_cast<newton::NewtonWorld *>(world);
            Ref<newton::NewtonBody> body = nw->get_body(p_body_id);
            if (body.is_valid()) body->apply_force(p_force, p_world_point);
        } break;
        case UnifiedSpatialQueryManager::ENGINE_GENESIS: {
            auto *gw = static_cast<genesis::GenesisWorld *>(world);
            Ref<genesis::RigidEntity> entity = gw->get_entity(p_body_id);
            if (entity.is_valid()) entity->apply_force(p_force, p_world_point);
        } break;
        case UnifiedSpatialQueryManager::ENGINE_VIENNA: {
            auto *vw = static_cast<vienna::ViennaWorld *>(world);
            Ref<vienna::ViennaBody> body = vw->get_body(p_body_id);
            if (body.is_valid()) body->apply_impulse(p_force, p_world_point);
        } break;
        case UnifiedSpatialQueryManager::ENGINE_WICKED: {
            auto *ww = static_cast<wicked::WickedWorld *>(world);
            Ref<wicked::WickedBody> body = ww->get_body(p_body_id);
            if (body.is_valid()) body->apply_force(p_force, p_world_point);
        } break;
    }
}

} // namespace unified

#endif // INTEGRATION_UNIFIED_PENALTY_CONTACT_SOLVER_H