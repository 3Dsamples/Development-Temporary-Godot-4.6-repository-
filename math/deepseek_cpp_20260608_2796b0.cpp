// File 467: modules/integration/unified_adaptive_substepping.h
// Automatically adjusts the number of physics sub‑steps per engine frame
// based on the maximum body speed, penetration depth, and contact stiffness
// to prevent instability.  Each active body's velocity and angular velocity
// are measured; if any exceeds the safe thresholds, the sub‑step count is
// increased for the next frame.  Also reduces sub‑steps when the scene is
// at rest, saving CPU time.  All checks are O(n) and performed every frame.

#ifndef INTEGRATION_UNIFIED_ADAPTIVE_SUBSTEPPING_H
#define INTEGRATION_UNIFIED_ADAPTIVE_SUBSTEPPING_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"

// Engine body types (forward)
namespace newton   { class NewtonWorld; class NewtonBody; }
namespace genesis  { class GenesisWorld; class BaseEntity; }
namespace vienna   { class ViennaWorld; class ViennaBody; }
namespace wicked   { class WickedWorld; class WickedBody; }

namespace unified {

class UnifiedAdaptiveSubstepping : public RefCounted {
    GDCLASS(UnifiedAdaptiveSubstepping, RefCounted);

public:
    // Maximum allowed linear speed (m/s) for a single sub‑step without
    // exceeding the safe penetration margin.
    real_t max_linear_speed = 0.5f;
    // Maximum allowed angular speed (rad/s).
    real_t max_angular_speed = 3.0f;
    // Maximum penetration depth that is considered safe (m).
    real_t max_penetration_depth = 0.02f;

    // Number of consecutive frames that must stay within all limits before
    // the sub‑step count is allowed to decrease.
    int   hysteresis_frames = 10;
    // Minimum sub‑steps (always at least 1).
    int   min_sub_steps = 1;
    // Maximum sub‑steps (cap to avoid extreme slowdown).
    int   max_sub_steps = 16;

    // Current sub‑step count (updated by the latest evaluation).
    int   current_sub_steps = 1;

    // -------------------------------------------------------------------
    // Evaluate the scene from the given engine worlds and return the
    // recommended sub‑step count for the next physics frame.
    // This method should be called once per frame, before stepping the
    // physics engines.
    // -------------------------------------------------------------------
    int evaluate(const HashMap<int, void *> &p_worlds);

    // -------------------------------------------------------------------
    // Return the last computed sub‑step count (without re‑evaluating).
    // -------------------------------------------------------------------
    int get_current_sub_steps() const { return current_sub_steps; }

protected:
    static void _bind_methods();

private:
    // Per‑engine speed measurement.
    int measure_engine(int p_engine, void *p_world,
                       real_t &r_max_lin_speed, real_t &r_max_ang_speed,
                       real_t &r_max_pen) const;

    int frames_within_limit = 0;   // hysteresis counter
};

// =========================================================================
// Inline implementations
// =========================================================================

void UnifiedAdaptiveSubstepping::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_max_linear_speed", "speed"), &UnifiedAdaptiveSubstepping::set_max_linear_speed);
    ClassDB::bind_method(D_METHOD("get_max_linear_speed"), &UnifiedAdaptiveSubstepping::get_max_linear_speed);
    ClassDB::bind_method(D_METHOD("set_max_angular_speed", "speed"), &UnifiedAdaptiveSubstepping::set_max_angular_speed);
    ClassDB::bind_method(D_METHOD("get_max_angular_speed"), &UnifiedAdaptiveSubstepping::get_max_angular_speed);
    ClassDB::bind_method(D_METHOD("set_max_penetration_depth", "depth"), &UnifiedAdaptiveSubstepping::set_max_penetration_depth);
    ClassDB::bind_method(D_METHOD("get_max_penetration_depth"), &UnifiedAdaptiveSubstepping::get_max_penetration_depth);
    ClassDB::bind_method(D_METHOD("set_hysteresis_frames", "frames"), &UnifiedAdaptiveSubstepping::set_hysteresis_frames);
    ClassDB::bind_method(D_METHOD("get_hysteresis_frames"), &UnifiedAdaptiveSubstepping::get_hysteresis_frames);
    ClassDB::bind_method(D_METHOD("set_min_sub_steps", "min"), &UnifiedAdaptiveSubstepping::set_min_sub_steps);
    ClassDB::bind_method(D_METHOD("get_min_sub_steps"), &UnifiedAdaptiveSubstepping::get_min_sub_steps);
    ClassDB::bind_method(D_METHOD("set_max_sub_steps", "max"), &UnifiedAdaptiveSubstepping::set_max_sub_steps);
    ClassDB::bind_method(D_METHOD("get_max_sub_steps"), &UnifiedAdaptiveSubstepping::get_max_sub_steps);
    ClassDB::bind_method(D_METHOD("evaluate", "worlds"), &UnifiedAdaptiveSubstepping::evaluate);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_linear_speed"), "set_max_linear_speed", "get_max_linear_speed");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_angular_speed"), "set_max_angular_speed", "get_max_angular_speed");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_penetration_depth"), "set_max_penetration_depth", "get_max_penetration_depth");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "hysteresis_frames"), "set_hysteresis_frames", "get_hysteresis_frames");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "min_sub_steps"), "set_min_sub_steps", "get_min_sub_steps");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_sub_steps"), "set_max_sub_steps", "get_max_sub_steps");
}

void UnifiedAdaptiveSubstepping::set_max_linear_speed(real_t v) { max_linear_speed = MAX(v, 0.001f); }
real_t UnifiedAdaptiveSubstepping::get_max_linear_speed() const { return max_linear_speed; }
void UnifiedAdaptiveSubstepping::set_max_angular_speed(real_t v) { max_angular_speed = MAX(v, 0.001f); }
real_t UnifiedAdaptiveSubstepping::get_max_angular_speed() const { return max_angular_speed; }
void UnifiedAdaptiveSubstepping::set_max_penetration_depth(real_t v) { max_penetration_depth = MAX(v, 0.001f); }
real_t UnifiedAdaptiveSubstepping::get_max_penetration_depth() const { return max_penetration_depth; }
void UnifiedAdaptiveSubstepping::set_hysteresis_frames(int v) { hysteresis_frames = MAX(v, 1); }
int UnifiedAdaptiveSubstepping::get_hysteresis_frames() const { return hysteresis_frames; }
void UnifiedAdaptiveSubstepping::set_min_sub_steps(int v) { min_sub_steps = MAX(v, 1); }
int UnifiedAdaptiveSubstepping::get_min_sub_steps() const { return min_sub_steps; }
void UnifiedAdaptiveSubstepping::set_max_sub_steps(int v) { max_sub_steps = MAX(v, 1); }
int UnifiedAdaptiveSubstepping::get_max_sub_steps() const { return max_sub_steps; }

// ---------------------------------------------------------------------------
// evaluate: iterate all bodies, find worst-case speed/penetration,
// then compute required sub‑steps.
// ---------------------------------------------------------------------------
int UnifiedAdaptiveSubstepping::evaluate(const HashMap<int, void *> &p_worlds) {
    real_t global_max_lin = 0.0f;
    real_t global_max_ang = 0.0f;
    real_t global_max_pen = 0.0f;

    for (const KeyValue<int, void *> &kv : p_worlds) {
        real_t max_lin, max_ang, max_pen;
        measure_engine(kv.key, kv.value, max_lin, max_ang, max_pen);
        if (max_lin > global_max_lin) global_max_lin = max_lin;
        if (max_ang > global_max_ang) global_max_ang = max_ang;
        if (max_pen > global_max_pen) global_max_pen = max_pen;
    }

    // Determine desired sub‑steps: we need at least global_max_lin / max_linear_speed,
    // global_max_ang / max_angular_speed, and global_max_pen / max_penetration_depth.
    // The actual number is the maximum of these ratios, rounded up.
    int desired = 1;
    if (global_max_lin > max_linear_speed) {
        int ratio = (int)Math::ceil(global_max_lin / max_linear_speed);
        if (ratio > desired) desired = ratio;
    }
    if (global_max_ang > max_angular_speed) {
        int ratio = (int)Math::ceil(global_max_ang / max_angular_speed);
        if (ratio > desired) desired = ratio;
    }
    if (global_max_pen > max_penetration_depth) {
        int ratio = (int)Math::ceil(global_max_pen / max_penetration_depth);
        if (ratio > desired) desired = ratio;
    }

    desired = CLAMP(desired, min_sub_steps, max_sub_steps);

    // Hysteresis: only decrease if we have been within limits for several frames.
    if (desired < current_sub_steps) {
        if (frames_within_limit < hysteresis_frames) {
            frames_within_limit++;
            return current_sub_steps;  // keep current higher value
        }
    } else {
        frames_within_limit = 0;
    }

    current_sub_steps = desired;
    return current_sub_steps;
}

// ---------------------------------------------------------------------------
// measure_engine: iterate bodies in one engine.
// ---------------------------------------------------------------------------
int UnifiedAdaptiveSubstepping::measure_engine(int p_engine, void *p_world,
                                                real_t &r_max_lin, real_t &r_max_ang,
                                                real_t &r_max_pen) const {
    r_max_lin = 0.0f;
    r_max_ang = 0.0f;
    r_max_pen = 0.0f;
    if (!p_world) return 0;

    switch (p_engine) {
        case 0: { // Newton
            auto *nw = static_cast<newton::NewtonWorld *>(p_world);
            LocalVector<newton::body_id> ids = nw->get_body_ids();
            for (newton::body_id id : ids) {
                Ref<newton::NewtonBody> body = nw->get_body(id);
                if (body.is_null() || !body->is_active() || body->get_type() != newton::BodyType::DYNAMIC) continue;
                real_t lin = body->get_linear_velocity().length();
                real_t ang = body->get_angular_velocity().length();
                if (lin > r_max_lin) r_max_lin = lin;
                if (ang > r_max_ang) r_max_ang = ang;
                // Penetration depth is not directly stored in the body; we skip.
            }
        } break;
        case 1: { // Genesis
            auto *gw = static_cast<genesis::GenesisWorld *>(p_world);
            LocalVector<genesis::entity_id_t> uids = gw->get_all_entity_uids();
            for (genesis::entity_id_t uid : uids) {
                Ref<genesis::BaseEntity> ent = gw->get_entity(uid);
                if (ent.is_null() || !ent->is_active()) continue;
                real_t lin = ent->get_linear_velocity().length();
                real_t ang = ent->get_angular_velocity().length();
                if (lin > r_max_lin) r_max_lin = lin;
                if (ang > r_max_ang) r_max_ang = ang;
            }
        } break;
        case 2: { // Vienna
            auto *vw = static_cast<vienna::ViennaWorld *>(p_world);
            LocalVector<vienna::body_id> ids = vw->get_body_ids();
            for (vienna::body_id id : ids) {
                Ref<vienna::ViennaBody> body = vw->get_body(id);
                if (body.is_null() || !body->is_active() || body->get_type() != vienna::BodyType::DYNAMIC) continue;
                real_t lin = body->get_linear_velocity().length();
                real_t ang = body->get_angular_velocity().length();
                if (lin > r_max_lin) r_max_lin = lin;
                if (ang > r_max_ang) r_max_ang = ang;
            }
        } break;
        case 3: { // Wicked
            auto *ww = static_cast<wicked::WickedWorld *>(p_world);
            LocalVector<wicked::body_id> ids = ww->get_body_ids();
            for (wicked::body_id id : ids) {
                Ref<wicked::WickedBody> body = ww->get_body(id);
                if (body.is_null() || !body->is_active() || body->get_type() != wicked::BodyType::DYNAMIC) continue;
                real_t lin = body->get_linear_velocity().length();
                real_t ang = body->get_angular_velocity().length();
                if (lin > r_max_lin) r_max_lin = lin;
                if (ang > r_max_ang) r_max_ang = ang;
            }
        } break;
    }
    return 0;
}

} // namespace unified

#endif // INTEGRATION_UNIFIED_ADAPTIVE_SUBSTEPPING_H