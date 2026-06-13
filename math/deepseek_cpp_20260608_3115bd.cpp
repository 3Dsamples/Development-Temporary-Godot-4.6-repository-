// File 446: modules/integration/unified_muscle_system.h
// Hill‑type muscle model for the unified physics pipeline.  Each muscle
// attaches to two bodies (across any engine) and produces a contractile
// force driven by an activation signal (0‑1).  The force is modulated by
// length‑tension and force‑velocity relationships using piecewise‑linear
// approximations.  Activation dynamics follow a first‑order ordinary
// differential equation with adjustable rise and fall time constants.
// All methods are fully inlined; no separate .cpp is required.

#ifndef INTEGRATION_UNIFIED_MUSCLE_SYSTEM_H
#define INTEGRATION_UNIFIED_MUSCLE_SYSTEM_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/typedefs.h"

// Forward engine body pointers.
namespace newton   { class NewtonWorld; class NewtonBody; }
namespace genesis  { class GenesisWorld; class BaseEntity; }
namespace vienna   { class ViennaWorld; class ViennaBody; }
namespace wicked   { class WickedWorld; class WickedBody; }

namespace unified {

class UnifiedMuscleSystem : public RefCounted {
    GDCLASS(UnifiedMuscleSystem, RefCounted);

public:
    // -------------------------------------------------------------------
    // Muscle descriptor.
    // -------------------------------------------------------------------
    struct Muscle {
        // Endpoint bodies (can be in different engines).
        int      engine_a = 0;
        uint64_t body_id_a = 0;
        int      engine_b = 0;
        uint64_t body_id_b = 0;

        // Local anchors relative to each body's origin.
        Vector3  local_anchor_a;
        Vector3  local_anchor_b;

        // Optimal fibre length (m) – length at which max force is produced.
        real_t   optimal_length = 0.5f;
        // Maximum isometric force (N) at optimal length and full activation.
        real_t   max_force = 500.0f;
        // Width of the length‑tension curve (controls how quickly force drops off).
        real_t   length_range = 0.3f;         // ± this value from optimal_length produces 0 force
        // Maximum shortening velocity (m/s).  Used in force‑velocity relation.
        real_t   max_shortening_velocity = 1.0f;
        // Activation time constant (s) – rise.
        real_t   tau_activation = 0.01f;
        // Deactivation time constant (s) – fall.
        real_t   tau_deactivation = 0.04f;

        // Current activation level (internal state, set by set_activation).
        real_t   activation = 0.0f;           // 0..1
        // Desired activation (set by user).
        real_t   desired_activation = 0.0f;   // 0..1

        bool     enabled = true;
    };

    // -------------------------------------------------------------------
    // Add a muscle and return its index.
    // -------------------------------------------------------------------
    int add_muscle(const Muscle &p_muscle);
    void remove_muscle(int p_index);
    void remove_muscles_with_body(int p_engine, uint64_t p_body_id);
    void clear_muscles();
    int get_muscle_count() const;
    Muscle &get_muscle(int p_idx);

    // -------------------------------------------------------------------
    // Set the desired activation for a muscle (to be used next step).
    // -------------------------------------------------------------------
    void set_activation(int p_muscle_idx, real_t p_desired);

    // -------------------------------------------------------------------
    // Advance activation dynamics by dt seconds.
    // Must be called before apply_forces() each substep.
    // -------------------------------------------------------------------
    void update_activations(real_t p_dt);

    // -------------------------------------------------------------------
    // Apply muscle forces to the connected bodies.
    // Must be called after update_activations() and after body transforms
    // have been updated.
    // -------------------------------------------------------------------
    void apply_forces(real_t p_dt,
                      newton::NewtonWorld *newton_world,
                      genesis::GenesisWorld *genesis_world,
                      vienna::ViennaWorld *vienna_world,
                      wicked::WickedWorld *wicked_world);

protected:
    static void _bind_methods();

private:
    LocalVector<Muscle> muscles;

    // Engine helpers.
    void apply_body_force(int p_engine, uint64_t p_body_id,
                          const Vector3 &p_force, const Vector3 &p_world_point,
                          newton::NewtonWorld *nw, genesis::GenesisWorld *gw,
                          vienna::ViennaWorld *vw, wicked::WickedWorld *ww) const;

    void get_body_state(int p_engine, uint64_t p_body_id,
                        const newton::NewtonWorld *nw, const genesis::GenesisWorld *gw,
                        const vienna::ViennaWorld *vw, const wicked::WickedWorld *ww,
                        Transform3D &r_xform, Vector3 &r_vel) const;

    // -------------------------------------------------------------------
    // Fibre length‑tension factor: 0 when |L-L0| > length_range,
    // 1 when L == L0, parabolic in between.
    // -------------------------------------------------------------------
    static inline real_t length_tension_factor(real_t p_current_length,
                                                real_t p_optimal_length,
                                                real_t p_range) {
        real_t diff = p_current_length - p_optimal_length;
        if (Math::abs(diff) >= p_range) return 0.0f;
        real_t t = diff / p_range;
        // Parabolic: 1 - t²
        return 1.0f - t * t;
    }

    // -------------------------------------------------------------------
    // Force‑velocity factor (Hill's hyperbolic approximation).
    // v_shortening = -stretch_rate (positive when muscle is shortening).
    // Returns factor in [0, 1.8] typically, but we clamp.
    // -------------------------------------------------------------------
    static inline real_t force_velocity_factor(real_t p_stretch_rate,
                                                real_t p_max_shortening_velocity) {
        // If muscle is lengthening (positive stretch_rate), force increases slightly.
        // If shortening (negative), force decreases.
        if (p_stretch_rate > 0.0f) {
            // Lengthening: force can be up to 1.5 times isometric.
            return 1.5f;
        }
        real_t v = -p_stretch_rate; // shortening speed
        // Hill equation: (F + a)(v + b) = (F0 + a) b.
        // Simplified linear drop: F/F0 = 1 - v / vmax, clamped to 0.
        real_t factor = 1.0f - v / MAX(p_max_shortening_velocity, 1e-6f);
        return CLAMP(factor, 0.0f, 1.5f);
    }
};

// =========================================================================
// Inline implementations
// =========================================================================

void UnifiedMuscleSystem::_bind_methods() {
    ClassDB::bind_method(D_METHOD("add_muscle", "muscle"), &UnifiedMuscleSystem::add_muscle);
    ClassDB::bind_method(D_METHOD("remove_muscle", "index"), &UnifiedMuscleSystem::remove_muscle);
    ClassDB::bind_method(D_METHOD("remove_muscles_with_body", "engine", "body_id"), &UnifiedMuscleSystem::remove_muscles_with_body);
    ClassDB::bind_method(D_METHOD("clear_muscles"), &UnifiedMuscleSystem::clear_muscles);
    ClassDB::bind_method(D_METHOD("get_muscle_count"), &UnifiedMuscleSystem::get_muscle_count);
    ClassDB::bind_method(D_METHOD("get_muscle", "index"), &UnifiedMuscleSystem::get_muscle);
    ClassDB::bind_method(D_METHOD("set_activation", "muscle_idx", "desired"), &UnifiedMuscleSystem::set_activation);
    ClassDB::bind_method(D_METHOD("update_activations", "dt"), &UnifiedMuscleSystem::update_activations);
    ClassDB::bind_method(D_METHOD("apply_forces", "dt", "newton_world", "genesis_world", "vienna_world", "wicked_world"),
        &UnifiedMuscleSystem::apply_forces);
}

int UnifiedMuscleSystem::add_muscle(const Muscle &p_muscle) {
    muscles.push_back(p_muscle);
    return muscles.size() - 1;
}

void UnifiedMuscleSystem::remove_muscle(int p_index) {
    ERR_FAIL_INDEX(p_index, muscles.size());
    muscles.remove_at(p_index);
}

void UnifiedMuscleSystem::remove_muscles_with_body(int p_engine, uint64_t p_body_id) {
    for (int i = muscles.size() - 1; i >= 0; --i) {
        const Muscle &m = muscles[i];
        if ((m.engine_a == p_engine && m.body_id_a == p_body_id) ||
            (m.engine_b == p_engine && m.body_id_b == p_body_id)) {
            muscles.remove_at(i);
        }
    }
}

void UnifiedMuscleSystem::clear_muscles() { muscles.clear(); }
int  UnifiedMuscleSystem::get_muscle_count() const { return muscles.size(); }
UnifiedMuscleSystem::Muscle &UnifiedMuscleSystem::get_muscle(int p_idx) { return muscles[p_idx]; }

void UnifiedMuscleSystem::set_activation(int p_muscle_idx, real_t p_desired) {
    ERR_FAIL_INDEX(p_muscle_idx, muscles.size());
    muscles[p_muscle_idx].desired_activation = CLAMP(p_desired, 0.0f, 1.0f);
}

// ---------------------------------------------------------------------------
// First-order activation dynamics.
// ---------------------------------------------------------------------------
void UnifiedMuscleSystem::update_activations(real_t p_dt) {
    for (Muscle &mus : muscles) {
        real_t tau = (mus.desired_activation > mus.activation) ? mus.tau_activation : mus.tau_deactivation;
        real_t alpha = p_dt / (tau + p_dt);
        mus.activation += alpha * (mus.desired_activation - mus.activation);
        mus.activation = CLAMP(mus.activation, 0.0f, 1.0f);
    }
}

// ---------------------------------------------------------------------------
// Apply muscle forces.
// ---------------------------------------------------------------------------
void UnifiedMuscleSystem::apply_forces(real_t p_dt,
                                        newton::NewtonWorld *newton_world,
                                        genesis::GenesisWorld *genesis_world,
                                        vienna::ViennaWorld *vienna_world,
                                        wicked::WickedWorld *wicked_world) {
    for (Muscle &mus : muscles) {
        if (!mus.enabled || mus.activation < 1e-6f) continue;

        Transform3D xform_a, xform_b;
        Vector3 vel_a, vel_b;
        get_body_state(mus.engine_a, mus.body_id_a, newton_world, genesis_world, vienna_world, wicked_world,
                       xform_a, vel_a);
        get_body_state(mus.engine_b, mus.body_id_b, newton_world, genesis_world, vienna_world, wicked_world,
                       xform_b, vel_b);

        // World anchors.
        Vector3 world_a = xform_a.xform(mus.local_anchor_a);
        Vector3 world_b = xform_b.xform(mus.local_anchor_b);

        Vector3 delta = world_b - world_a;
        real_t length = delta.length();
        if (length < CMP_EPSILON) continue;
        Vector3 dir = delta / length;

        // Length‑tension factor.
        real_t f_L = length_tension_factor(length, mus.optimal_length, mus.length_range);
        if (f_L <= 0.0f) continue;

        // Stretch rate (positive when lengthening).
        Vector3 rel_vel = vel_b - vel_a;
        real_t stretch_rate = rel_vel.dot(dir);
        // Force‑velocity factor.
        real_t f_V = force_velocity_factor(stretch_rate, mus.max_shortening_velocity);

        // Total contractile force magnitude.
        real_t force_mag = mus.activation * mus.max_force * f_L * f_V;
        if (force_mag <= 0.0f) continue;

        Vector3 force_on_a = dir * force_mag;
        Vector3 force_on_b = -force_on_a;

        apply_body_force(mus.engine_a, mus.body_id_a, force_on_a, world_a,
                         newton_world, genesis_world, vienna_world, wicked_world);
        apply_body_force(mus.engine_b, mus.body_id_b, force_on_b, world_b,
                         newton_world, genesis_world, vienna_world, wicked_world);
    }
}

// ---------------------------------------------------------------------------
// Engine helpers – apply force
// ---------------------------------------------------------------------------
void UnifiedMuscleSystem::apply_body_force(int p_engine, uint64_t p_body_id,
                                            const Vector3 &p_force, const Vector3 &p_world_point,
                                            newton::NewtonWorld *nw, genesis::GenesisWorld *gw,
                                            vienna::ViennaWorld *vw, wicked::WickedWorld *ww) const {
    switch (p_engine) {
        case 0: if (nw) { Ref<newton::NewtonBody> b = nw->get_body(p_body_id); if (b.is_valid()) b->apply_force(p_force, p_world_point); } break;
        case 1: if (gw) { Ref<genesis::BaseEntity> e = gw->get_entity(p_body_id); if (e.is_valid()) e->apply_force(p_force, p_world_point); } break;
        case 2: if (vw) { Ref<vienna::ViennaBody> b = vw->get_body(p_body_id); if (b.is_valid()) b->apply_impulse(p_force, p_world_point); } break;
        case 3: if (ww) { Ref<wicked::WickedBody> b = ww->get_body(p_body_id); if (b.is_valid()) b->apply_force(p_force, p_world_point); } break;
    }
}

// ---------------------------------------------------------------------------
// Engine helpers – get transform and velocity
// ---------------------------------------------------------------------------
void UnifiedMuscleSystem::get_body_state(int p_engine, uint64_t p_body_id,
                                          const newton::NewtonWorld *nw, const genesis::GenesisWorld *gw,
                                          const vienna::ViennaWorld *vw, const wicked::WickedWorld *ww,
                                          Transform3D &r_xform, Vector3 &r_vel) const {
    r_xform = Transform3D();
    r_vel = Vector3();
    switch (p_engine) {
        case 0: if (nw) { Ref<newton::NewtonBody> b = nw->get_body(p_body_id); if (b.is_valid()) { r_xform = b->get_transform(); r_vel = b->get_linear_velocity(); } } break;
        case 1: if (gw) { Ref<genesis::BaseEntity> e = gw->get_entity(p_body_id); if (e.is_valid()) { r_xform = e->get_transform(); r_vel = e->get_linear_velocity(); } } break;
        case 2: if (vw) { Ref<vienna::ViennaBody> b = vw->get_body(p_body_id); if (b.is_valid()) { r_xform = b->get_transform(); r_vel = b->get_linear_velocity(); } } break;
        case 3: if (ww) { Ref<wicked::WickedBody> b = ww->get_body(p_body_id); if (b.is_valid()) { r_xform = b->get_transform(); r_vel = b->get_linear_velocity(); } } break;
    }
}

} // namespace unified

#endif // INTEGRATION_UNIFIED_MUSCLE_SYSTEM_H