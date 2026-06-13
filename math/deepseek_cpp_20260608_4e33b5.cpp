// File 457: modules/integration/unified_ipc_solver.h
// Incremental Potential Contact (IPC) solver for soft‑body ↔ rigid‑body
// interactions.  Replaces linear penalty with a C²‑continuous log‑barrier
// that guarantees intersection‑free trajectories when used with a
// line‑search.  Uses TreeNSearch for proximity queries (via the unified
// soft‑collision detector) and applies equal‑and‑opposite barrier forces
// plus Coulomb friction.  All kernels are fully inlined; parallel dispatch
// is performed over contact batches using Gaia's CPUParallelization.

#ifndef INTEGRATION_UNIFIED_IPC_SOLVER_H
#define INTEGRATION_UNIFIED_IPC_SOLVER_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/typedefs.h"

#include "unified_soft_collision_detector.h"        // Contact structure

// Engine headers for applying forces to rigid bodies
#include "../../newton/src/world/newton_world.h"
#include "../../newton/src/bodies/newton_body.h"
#include "../../genesis/src/genesis_world.h"
#include "../../genesis/src/entities/rigid_entity.h"
#include "../../vienna/src/world/vienna_world.h"
#include "../../vienna/src/bodies/vienna_body.h"
#include "../../wicked/src/world/wicked_world.h"
#include "../../wicked/src/bodies/wicked_body.h"

// Gaia parallelisation (for chunked loops)
#include "../../gaia/src/parallelization/cpu_parallelization.h"

namespace unified {

class UnifiedIPCSolver : public RefCounted {
    GDCLASS(UnifiedIPCSolver, RefCounted);

public:
    // IPC barrier distance (d̂) – contacts with d < d̂ are subject to the
    // barrier energy.  Typical value 5e-4 m.
    real_t barrier_distance = 0.0005f;

    // Barrier stiffness (kappa) – scales the log‑barrier energy.
    real_t barrier_stiffness = 1e5f;

    // Friction coefficient (Coulomb) – tangential force clamped to μ·F_n.
    real_t friction_coefficient = 0.5f;

    // Number of barrier outer iterations (re‑evaluates gradient).
    int max_barrier_iterations = 1;

    // Number of friction inner iterations (within each barrier iteration).
    int max_friction_iterations = 1;

    // -------------------------------------------------------------------
    // Resolve contacts for the given soft mesh and contact data.
    // Soft velocities are updated in place; equal‑opposite impulses are
    // applied to the rigid bodies.
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
    // Compute barrier force for a single contact.
    // Returns (normal_barrier_force, tangential_friction_force) in world
    // space for the soft vertex.  The opposite force is applied to the rigid.
    // -------------------------------------------------------------------
    Vector3 compute_barrier_force(const UnifiedSoftCollisionDetector::Contact &p_contact,
                                  const Vector3 &p_soft_velocity,
                                  real_t p_dt,
                                  real_t &r_normal_force_magnitude) const;

protected:
    static void _bind_methods();

private:
    // Helpers to apply force to a rigid body.
    void apply_force_to_rigid(int p_engine, uint64_t p_body_id,
                              const Vector3 &p_force, const Vector3 &p_world_point,
                              const HashMap<int, void *> &p_rigid_worlds) const;
};

// =========================================================================
// Inline implementations
// =========================================================================

void UnifiedIPCSolver::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_barrier_distance", "d_hat"), &UnifiedIPCSolver::set_barrier_distance);
    ClassDB::bind_method(D_METHOD("get_barrier_distance"), &UnifiedIPCSolver::get_barrier_distance);
    ClassDB::bind_method(D_METHOD("set_barrier_stiffness", "kappa"), &UnifiedIPCSolver::set_barrier_stiffness);
    ClassDB::bind_method(D_METHOD("get_barrier_stiffness"), &UnifiedIPCSolver::get_barrier_stiffness);
    ClassDB::bind_method(D_METHOD("set_friction_coefficient", "mu"), &UnifiedIPCSolver::set_friction_coefficient);
    ClassDB::bind_method(D_METHOD("get_friction_coefficient"), &UnifiedIPCSolver::get_friction_coefficient);
    ClassDB::bind_method(D_METHOD("set_max_barrier_iterations", "iter"), &UnifiedIPCSolver::set_max_barrier_iterations);
    ClassDB::bind_method(D_METHOD("get_max_barrier_iterations"), &UnifiedIPCSolver::get_max_barrier_iterations);
    ClassDB::bind_method(D_METHOD("set_max_friction_iterations", "iter"), &UnifiedIPCSolver::set_max_friction_iterations);
    ClassDB::bind_method(D_METHOD("get_max_friction_iterations"), &UnifiedIPCSolver::get_max_friction_iterations);
    ClassDB::bind_method(D_METHOD("resolve_contacts", "soft_velocities", "soft_masses", "contacts", "rigid_worlds", "dt"),
        &UnifiedIPCSolver::resolve_contacts);
    ClassDB::bind_method(D_METHOD("compute_barrier_force", "contact", "soft_velocity", "dt"),
        &UnifiedIPCSolver::compute_barrier_force);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "barrier_distance"), "set_barrier_distance", "get_barrier_distance");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "barrier_stiffness"), "set_barrier_stiffness", "get_barrier_stiffness");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "friction_coefficient"), "set_friction_coefficient", "get_friction_coefficient");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_barrier_iterations"), "set_max_barrier_iterations", "get_max_barrier_iterations");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_friction_iterations"), "set_max_friction_iterations", "get_max_friction_iterations");
}

// Property setters/getters
void UnifiedIPCSolver::set_barrier_distance(real_t v) { barrier_distance = MAX(v, 1e-8f); }
real_t UnifiedIPCSolver::get_barrier_distance() const { return barrier_distance; }
void UnifiedIPCSolver::set_barrier_stiffness(real_t v) { barrier_stiffness = MAX(v, 0.0f); }
real_t UnifiedIPCSolver::get_barrier_stiffness() const { return barrier_stiffness; }
void UnifiedIPCSolver::set_friction_coefficient(real_t v) { friction_coefficient = CLAMP(v, 0.0f, 1.0f); }
real_t UnifiedIPCSolver::get_friction_coefficient() const { return friction_coefficient; }
void UnifiedIPCSolver::set_max_barrier_iterations(int v) { max_barrier_iterations = MAX(v, 1); }
int UnifiedIPCSolver::get_max_barrier_iterations() const { return max_barrier_iterations; }
void UnifiedIPCSolver::set_max_friction_iterations(int v) { max_friction_iterations = MAX(v, 1); }
int UnifiedIPCSolver::get_max_friction_iterations() const { return max_friction_iterations; }

// ---------------------------------------------------------------------------
// Resolve all contacts using barrier and friction iterations.
// ---------------------------------------------------------------------------
void UnifiedIPCSolver::resolve_contacts(
        LocalVector<Vector3> &p_soft_velocities,
        const LocalVector<real_t> &p_soft_masses,
        const LocalVector<LocalVector<UnifiedSoftCollisionDetector::Contact>> &p_contacts,
        const HashMap<int, void *> &p_rigid_worlds,
        real_t p_dt) const {

    int n_verts = p_contacts.size();

    // For parallel dispatch, we need to flatten the contact list.
    // We'll build a flat list of contacts with their soft vertex index.
    struct IndexedContact {
        int soft_vertex;
        UnifiedSoftCollisionDetector::Contact contact;
    };
    LocalVector<IndexedContact> flat_contacts;
    for (int i = 0; i < n_verts; ++i) {
        for (const auto &c : p_contacts[i]) {
            flat_contacts.push_back({i, c});
        }
    }

    int n_contacts = flat_contacts.size();
    if (n_contacts == 0) return;

    // Outer barrier iteration loop.
    for (int barrier_iter = 0; barrier_iter < max_barrier_iterations; ++barrier_iter) {
        // Friction inner loop.
        for (int friction_iter = 0; friction_iter < max_friction_iterations; ++friction_iter) {
            // Process contacts in parallel chunks using Gaia's CPUParallelization.
            gaia::parallel::CPUParallelization::parallel_for(n_contacts,
                [this, &flat_contacts, &p_soft_velocities, &p_soft_masses,
                 &p_rigid_worlds, p_dt](int64_t start, int64_t end) {
                    for (int64_t idx = start; idx < end; ++idx) {
                        const IndexedContact &ic = flat_contacts[idx];
                        const auto &contact = ic.contact;
                        int vert_idx = ic.soft_vertex;

                        Vector3 &soft_vel = p_soft_velocities[vert_idx];
                        real_t mass = (vert_idx < p_soft_masses.size()) ? p_soft_masses[vert_idx] : 1.0f;

                        real_t normal_force_mag = 0.0f;
                        Vector3 force_on_soft = compute_barrier_force(contact, soft_vel, p_dt, normal_force_mag);

                        // Apply impulse to soft vertex (velocity += force_impulse / mass).
                        if (mass > CMP_EPSILON) {
                            soft_vel += force_on_soft * (p_dt / mass);
                        }

                        // Apply equal‑opposite force to the rigid body at the contact point.
                        apply_force_to_rigid(contact.engine, contact.rigid_body_id,
                                             -force_on_soft, contact.rigid_point,
                                             p_rigid_worlds);
                    }
                },
                128); // min batch size 128 contacts per thread
        }
    }
}

// ---------------------------------------------------------------------------
// Compute barrier force for a single contact.
// ---------------------------------------------------------------------------
Vector3 UnifiedIPCSolver::compute_barrier_force(
        const UnifiedSoftCollisionDetector::Contact &p_contact,
        const Vector3 &p_soft_velocity,
        real_t p_dt,
        real_t &r_normal_force_magnitude) const {

    // The normal points from rigid to soft (as defined by the detector).
    // d = signed separation: positive = apart, negative = penetration.
    real_t d = p_contact.distance;
    real_t d_hat = barrier_distance;

    // Barrier only active when 0 < d < d_hat.
    if (d <= 0.0f || d >= d_hat) return Vector3();

    // Avoid division by zero.
    if (d < CMP_EPSILON) d = CMP_EPSILON;

    // Classic IPC barrier: b(d) = -κ (d - d_hat)² ln(d / d_hat)
    // The derivative (force magnitude) = κ * [ 2 (d_hat - d) ln(d/d_hat) + (d - d_hat)² / d ]
    // Force direction is along the outward normal (from rigid to soft),
    // pushing the soft vertex away from the rigid body.
    // So the force on the soft vertex is: + F_barrier * n.
    real_t diff = d - d_hat;         // negative
    real_t ratio = d / d_hat;        // < 1
    real_t log_ratio = Math::log(ratio);

    // Barrier force magnitude (positive = repulsion pushing soft away from rigid).
    real_t barrier_force_mag = barrier_stiffness *
        (2.0f * (d_hat - d) * log_ratio + (d - d_hat) * (d - d_hat) / d);

    // Clamp to avoid huge forces when d is extremely small.
    if (barrier_force_mag > 1e10f) barrier_force_mag = 1e10f;

    // Normal force (world space) on soft vertex.
    Vector3 normal_force = p_contact.normal * barrier_force_mag;

    r_normal_force_magnitude = barrier_force_mag;

    // ---------- Coulomb friction ----------
    // Tangential relative velocity: v_t = v_soft - (v_soft·n)n
    real_t vn = p_soft_velocity.dot(p_contact.normal);
    Vector3 v_t = p_soft_velocity - p_contact.normal * vn;
    real_t slip_speed = v_t.length();

    Vector3 friction_force(0, 0, 0);
    if (slip_speed > CMP_EPSILON) {
        Vector3 tangent_dir = v_t / slip_speed;
        // Maximum friction magnitude.
        real_t max_friction = friction_coefficient * barrier_force_mag;
        // Friction opposes tangential slip, clamped to max_friction.
        friction_force = -tangent_dir * max_friction;
    }

    return normal_force + friction_force;
}

// ---------------------------------------------------------------------------
// Apply force to a rigid body at a world point.
// ---------------------------------------------------------------------------
void UnifiedIPCSolver::apply_force_to_rigid(
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

#endif // INTEGRATION_UNIFIED_IPC_SOLVER_H