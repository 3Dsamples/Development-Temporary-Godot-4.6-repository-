// File 459: modules/integration/unified_soft_self_collision_resolver.h
// Resolves self‑collision contacts (vertex‑face) within a single tetrahedral
// mesh using either a linear penalty or an IPC log‑barrier (selectable via
// a flag).  Takes the contact list produced by UnifiedSoftSelfCollisionDetector
// and applies equal‑and‑opposite impulses to the involved vertices.
// All computations are parallelised over contacts using Gaia's CPUParallelization.

#ifndef INTEGRATION_UNIFIED_SOFT_SELF_COLLISION_RESOLVER_H
#define INTEGRATION_UNIFIED_SOFT_SELF_COLLISION_RESOLVER_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"

#include "unified_soft_self_collision_detector.h"   // SelfContact structure
#include "../../gaia/src/parallelization/cpu_parallelization.h"

namespace unified {

class UnifiedSoftSelfCollisionResolver : public RefCounted {
    GDCLASS(UnifiedSoftSelfCollisionResolver, RefCounted);

public:
    // Contact response model.
    enum Model {
        LINEAR_PENALTY = 0,
        IPC_BARRIER    = 1
    };

    // Active model.
    Model model = IPC_BARRIER;

    // -- Penalty parameters --
    real_t penalty_stiffness = 1e5f;        // N/m
    real_t penalty_damping   = 10.0f;       // Ns/m
    real_t max_penetration   = 0.05f;       // m (saturation)

    // -- IPC barrier parameters --
    real_t barrier_distance  = 0.0005f;     // d̂
    real_t barrier_stiffness = 1e5f;        // κ

    // -- Friction (Coulomb) --
    real_t friction_coefficient = 0.5f;

    // -- Iterations (IPC recomputes gradient each outer iteration) --
    int max_barrier_iterations  = 1;
    int max_friction_iterations = 1;

    // -------------------------------------------------------------------
    // Resolve self‑collision contacts.
    //
    // @param p_soft_velocities  Current velocities of all soft vertices
    //                           (will be modified in place).
    // @param p_soft_masses      Per‑vertex mass (for impulse -> velocity).
    // @param p_contacts         Contacts per vertex (from self‑detector).
    // @param p_dt               Time step.
    // -------------------------------------------------------------------
    void resolve_contacts(
            LocalVector<Vector3> &p_soft_velocities,
            const LocalVector<real_t> &p_soft_masses,
            const LocalVector<LocalVector<UnifiedSoftSelfCollisionDetector::SelfContact>> &p_contacts,
            real_t p_dt) const;

    // -------------------------------------------------------------------
    // Compute the force (world space) on the vertex for a single contact,
    // given the current vertex velocity.
    // r_face_force receives the opposite force to be distributed to the
    // three face vertices (barycentric weights not returned).
    // -------------------------------------------------------------------
    Vector3 compute_contact_force(
            const UnifiedSoftSelfCollisionDetector::SelfContact &p_contact,
            const Vector3 &p_vertex_velocity,
            real_t p_dt,
            Vector3 &r_face_force) const;

protected:
    static void _bind_methods();

private:
    // Apply an impulse to a vertex.
    static inline void apply_impulse(int p_vertex_idx, const Vector3 &p_impulse,
                                     LocalVector<Vector3> &p_velocities,
                                     const LocalVector<real_t> &p_masses);
};

// =========================================================================
// Inline implementations
// =========================================================================

void UnifiedSoftSelfCollisionResolver::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_model", "model"), &UnifiedSoftSelfCollisionResolver::set_model);
    ClassDB::bind_method(D_METHOD("get_model"), &UnifiedSoftSelfCollisionResolver::get_model);
    ClassDB::bind_method(D_METHOD("set_penalty_stiffness", "k"), &UnifiedSoftSelfCollisionResolver::set_penalty_stiffness);
    ClassDB::bind_method(D_METHOD("get_penalty_stiffness"), &UnifiedSoftSelfCollisionResolver::get_penalty_stiffness);
    ClassDB::bind_method(D_METHOD("set_penalty_damping", "d"), &UnifiedSoftSelfCollisionResolver::set_penalty_damping);
    ClassDB::bind_method(D_METHOD("get_penalty_damping"), &UnifiedSoftSelfCollisionResolver::get_penalty_damping);
    ClassDB::bind_method(D_METHOD("set_barrier_distance", "d_hat"), &UnifiedSoftSelfCollisionResolver::set_barrier_distance);
    ClassDB::bind_method(D_METHOD("get_barrier_distance"), &UnifiedSoftSelfCollisionResolver::get_barrier_distance);
    ClassDB::bind_method(D_METHOD("set_barrier_stiffness", "kappa"), &UnifiedSoftSelfCollisionResolver::set_barrier_stiffness);
    ClassDB::bind_method(D_METHOD("get_barrier_stiffness"), &UnifiedSoftSelfCollisionResolver::get_barrier_stiffness);
    ClassDB::bind_method(D_METHOD("set_friction_coefficient", "mu"), &UnifiedSoftSelfCollisionResolver::set_friction_coefficient);
    ClassDB::bind_method(D_METHOD("get_friction_coefficient"), &UnifiedSoftSelfCollisionResolver::get_friction_coefficient);
    ClassDB::bind_method(D_METHOD("set_max_barrier_iterations", "iter"), &UnifiedSoftSelfCollisionResolver::set_max_barrier_iterations);
    ClassDB::bind_method(D_METHOD("get_max_barrier_iterations"), &UnifiedSoftSelfCollisionResolver::get_max_barrier_iterations);
    ClassDB::bind_method(D_METHOD("set_max_friction_iterations", "iter"), &UnifiedSoftSelfCollisionResolver::set_max_friction_iterations);
    ClassDB::bind_method(D_METHOD("get_max_friction_iterations"), &UnifiedSoftSelfCollisionResolver::get_max_friction_iterations);
    ClassDB::bind_method(D_METHOD("resolve_contacts", "soft_velocities", "soft_masses", "contacts", "dt"),
        &UnifiedSoftSelfCollisionResolver::resolve_contacts);
    ClassDB::bind_method(D_METHOD("compute_contact_force", "contact", "vertex_velocity", "dt"),
        &UnifiedSoftSelfCollisionResolver::compute_contact_force);

    BIND_ENUM_CONSTANT(LINEAR_PENALTY);
    BIND_ENUM_CONSTANT(IPC_BARRIER);

    ADD_PROPERTY(PropertyInfo(Variant::INT, "model", PROPERTY_HINT_ENUM, "LinearPenalty,IPCBarrier"), "set_model", "get_model");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "penalty_stiffness"), "set_penalty_stiffness", "get_penalty_stiffness");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "penalty_damping"), "set_penalty_damping", "get_penalty_damping");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "barrier_distance"), "set_barrier_distance", "get_barrier_distance");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "barrier_stiffness"), "set_barrier_stiffness", "get_barrier_stiffness");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "friction_coefficient"), "set_friction_coefficient", "get_friction_coefficient");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_barrier_iterations"), "set_max_barrier_iterations", "get_max_barrier_iterations");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_friction_iterations"), "set_max_friction_iterations", "get_max_friction_iterations");
}

// Property setters/getters.
void UnifiedSoftSelfCollisionResolver::set_model(Model v) { model = v; }
UnifiedSoftSelfCollisionResolver::Model UnifiedSoftSelfCollisionResolver::get_model() const { return model; }
void UnifiedSoftSelfCollisionResolver::set_penalty_stiffness(real_t v) { penalty_stiffness = MAX(v, 0.0f); }
real_t UnifiedSoftSelfCollisionResolver::get_penalty_stiffness() const { return penalty_stiffness; }
void UnifiedSoftSelfCollisionResolver::set_penalty_damping(real_t v) { penalty_damping = MAX(v, 0.0f); }
real_t UnifiedSoftSelfCollisionResolver::get_penalty_damping() const { return penalty_damping; }
void UnifiedSoftSelfCollisionResolver::set_barrier_distance(real_t v) { barrier_distance = MAX(v, 1e-8f); }
real_t UnifiedSoftSelfCollisionResolver::get_barrier_distance() const { return barrier_distance; }
void UnifiedSoftSelfCollisionResolver::set_barrier_stiffness(real_t v) { barrier_stiffness = MAX(v, 0.0f); }
real_t UnifiedSoftSelfCollisionResolver::get_barrier_stiffness() const { return barrier_stiffness; }
void UnifiedSoftSelfCollisionResolver::set_friction_coefficient(real_t v) { friction_coefficient = CLAMP(v, 0.0f, 1.0f); }
real_t UnifiedSoftSelfCollisionResolver::get_friction_coefficient() const { return friction_coefficient; }
void UnifiedSoftSelfCollisionResolver::set_max_barrier_iterations(int v) { max_barrier_iterations = MAX(v, 1); }
int UnifiedSoftSelfCollisionResolver::get_max_barrier_iterations() const { return max_barrier_iterations; }
void UnifiedSoftSelfCollisionResolver::set_max_friction_iterations(int v) { max_friction_iterations = MAX(v, 1); }
int UnifiedSoftSelfCollisionResolver::get_max_friction_iterations() const { return max_friction_iterations; }

// ---------------------------------------------------------------------------
// Resolve contacts: iterate per vertex, apply forces to that vertex and
// distribute opposite force to the three face vertices using barycentric
// weights (approximated by equal parts for simplicity, or proper barycentric
// if available).  Parallelised over vertices.
// ---------------------------------------------------------------------------
void UnifiedSoftSelfCollisionResolver::resolve_contacts(
        LocalVector<Vector3> &p_soft_velocities,
        const LocalVector<real_t> &p_soft_masses,
        const LocalVector<LocalVector<UnifiedSoftSelfCollisionDetector::SelfContact>> &p_contacts,
        real_t p_dt) const {

    int n_verts = p_contacts.size();

    // Outer barrier loop (if IPC barrier is selected)
    int outer_iters = (model == IPC_BARRIER) ? max_barrier_iterations : 1;
    int inner_iters = (model == IPC_BARRIER) ? max_friction_iterations : 1;

    for (int barrier_iter = 0; barrier_iter < outer_iters; ++barrier_iter) {
        for (int friction_iter = 0; friction_iter < inner_iters; ++friction_iter) {
            // Parallel loop over vertices.
            gaia::parallel::CPUParallelization::parallel_for(n_verts,
                [this, &p_soft_velocities, &p_soft_masses, &p_contacts, p_dt](int64_t start, int64_t end) {
                    for (int64_t vi = start; vi < end; ++vi) {
                        if (p_contacts[vi].is_empty()) continue;
                        Vector3 &vert_vel = p_soft_velocities[vi];
                        real_t mass_i = (vi < p_soft_masses.size()) ? p_soft_masses[vi] : 1.0f;

                        for (const auto &c : p_contacts[vi]) {
                            Vector3 face_force;
                            Vector3 force_on_vertex = compute_contact_force(c, vert_vel, p_dt, face_force);

                            // Apply impulse to the vertex.
                            if (mass_i > CMP_EPSILON) {
                                vert_vel += force_on_vertex * (p_dt / mass_i);
                            }

                            // Distribute opposite force to the three face vertices.
                            // We use simple equal weighting (1/3 each) for impulse distribution.
                            // A more accurate distribution uses barycentric coordinates,
                            // but they are not stored in the SelfContact.  Equal weighting
                            // conserves momentum and is stable for penalty methods.
                            Vector3 per_face_impulse = -face_force * (p_dt / 3.0f);
                            apply_impulse(c.face_v0, per_face_impulse, p_soft_velocities, p_soft_masses);
                            apply_impulse(c.face_v1, per_face_impulse, p_soft_velocities, p_soft_masses);
                            apply_impulse(c.face_v2, per_face_impulse, p_soft_velocities, p_soft_masses);
                        }
                    }
                },
                128);
        }
    }
}

// ---------------------------------------------------------------------------
// Compute force for a single self‑contact.
// ---------------------------------------------------------------------------
Vector3 UnifiedSoftSelfCollisionResolver::compute_contact_force(
        const UnifiedSoftSelfCollisionDetector::SelfContact &p_contact,
        const Vector3 &p_vertex_velocity,
        real_t p_dt,
        Vector3 &r_face_force) const {

    // The contact gives distance = separation (positive if apart).
    // For self‑collision we want to prevent negative distance (penetration).
    real_t d = p_contact.distance;
    // Normal from face to vertex (as defined in the detector).
    Vector3 n = p_contact.normal;

    // If no proximity, no force.
    if (d >= barrier_distance && model == IPC_BARRIER) {
        r_face_force = Vector3();
        return Vector3();
    }
    if (d >= collision_margin && model == LINEAR_PENALTY) {
        r_face_force = Vector3();
        return Vector3();
    }

    Vector3 force_on_vertex;
    real_t normal_force_mag = 0.0f;

    if (model == LINEAR_PENALTY) {
        real_t penetration = -d;  // positive when overlapping
        if (penetration <= 0.0f) {
            r_face_force = Vector3();
            return Vector3();
        }
        if (penetration > max_penetration) penetration = max_penetration;

        // Normal force magnitude: k * penetration.
        normal_force_mag = penalty_stiffness * penetration;

        // Damping: if vertex is moving toward the face (vn is negative? Actually n points from face to vertex.
        // If the vertex is moving toward the face, its velocity along n is negative.  Damping opposes that.
        real_t vn = p_vertex_velocity.dot(n);
        if (vn < 0.0f) {
            normal_force_mag += penalty_damping * (-vn);
        }

        // Force on vertex: push it away from the face (along +n).
        force_on_vertex = n * normal_force_mag;
    } else { // IPC barrier
        real_t d_hat = barrier_distance;
        if (d <= 0.0f || d >= d_hat) {
            r_face_force = Vector3();
            return Vector3();
        }
        if (d < CMP_EPSILON) d = CMP_EPSILON;

        real_t diff = d - d_hat;
        real_t ratio = d / d_hat;
        real_t log_ratio = Math::log(ratio);

        // Barrier force magnitude (repulsion pushing vertex away from face).
        normal_force_mag = barrier_stiffness *
            (2.0f * (d_hat - d) * log_ratio + (d - d_hat) * (d - d_hat) / d);

        if (normal_force_mag > 1e10f) normal_force_mag = 1e10f;

        // Force on vertex: push it away from the face along +n.
        force_on_vertex = n * normal_force_mag;
    }

    // Friction: tangential relative velocity.
    real_t vn = p_vertex_velocity.dot(n);
    Vector3 v_t = p_vertex_velocity - n * vn;
    real_t slip_speed = v_t.length();
    Vector3 friction_force(0,0,0);
    if (slip_speed > CMP_EPSILON) {
        Vector3 tangent_dir = v_t / slip_speed;
        real_t max_friction = friction_coefficient * normal_force_mag;
        friction_force = -tangent_dir * max_friction;
    }

    Vector3 total_force = force_on_vertex + friction_force;
    // Opposite force is applied to the face.
    r_face_force = -total_force;
    return total_force;
}

// ---------------------------------------------------------------------------
// Helper: apply impulse to a vertex.
// ---------------------------------------------------------------------------
inline void UnifiedSoftSelfCollisionResolver::apply_impulse(
        int p_vertex_idx, const Vector3 &p_impulse,
        LocalVector<Vector3> &p_velocities,
        const LocalVector<real_t> &p_masses) {
    if (p_vertex_idx < 0 || p_vertex_idx >= p_velocities.size()) return;
    real_t mass = (p_vertex_idx < p_masses.size()) ? p_masses[p_vertex_idx] : 1.0f;
    if (mass > CMP_EPSILON) {
        p_velocities[p_vertex_idx] += p_impulse / mass;
    }
}

} // namespace unified

#endif // INTEGRATION_UNIFIED_SOFT_SELF_COLLISION_RESOLVER_H