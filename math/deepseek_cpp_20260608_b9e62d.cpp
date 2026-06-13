// File 441: modules/integration/unified_gravity_field_system.cpp
// Complete implementation of the gravity field system.  All field
// management, per‑body scale storage, and force application loops are
// defined here.

#include "unified_gravity_field_system.h"
#include "core/variant/variant.h"
#include "core/object/class_db.h"

namespace unified {

void UnifiedGravityFieldSystem::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_body_gravity_scale", "engine", "body_id", "scale"), &UnifiedGravityFieldSystem::set_body_gravity_scale);
    ClassDB::bind_method(D_METHOD("get_body_gravity_scale", "engine", "body_id"), &UnifiedGravityFieldSystem::get_body_gravity_scale);
    ClassDB::bind_method(D_METHOD("add_field", "field"), &UnifiedGravityFieldSystem::add_field);
    ClassDB::bind_method(D_METHOD("remove_field", "index"), &UnifiedGravityFieldSystem::remove_field);
    ClassDB::bind_method(D_METHOD("clear_fields"), &UnifiedGravityFieldSystem::clear_fields);
    ClassDB::bind_method(D_METHOD("get_field_count"), &UnifiedGravityFieldSystem::get_field_count);
    ClassDB::bind_method(D_METHOD("get_field", "index"), &UnifiedGravityFieldSystem::get_field);
    ClassDB::bind_method(D_METHOD("apply_forces", "dt", "newton_world", "genesis_world", "vienna_world", "wicked_world"), &UnifiedGravityFieldSystem::apply_forces);
    ClassDB::bind_method(D_METHOD("compute_acceleration", "position", "mass", "scale"), &UnifiedGravityFieldSystem::compute_acceleration);

    BIND_ENUM_CONSTANT(FIELD_ATTRACTOR);
    BIND_ENUM_CONSTANT(FIELD_REPULSOR);
    BIND_ENUM_CONSTANT(FIELD_DIRECTIONAL);
}

// ---------------------------------------------------------------------------
// Per‑body scale
// ---------------------------------------------------------------------------
void UnifiedGravityFieldSystem::set_body_gravity_scale(int p_engine, uint64_t p_body_id, real_t p_scale) {
    BodyKey key{p_engine, p_body_id};
    body_scales[key] = MAX(p_scale, 0.0f);
}

real_t UnifiedGravityFieldSystem::get_body_gravity_scale(int p_engine, uint64_t p_body_id) const {
    BodyKey key{p_engine, p_body_id};
    HashMap<BodyKey, real_t, BodyKey::Hash>::ConstIterator it = body_scales.find(key);
    return it ? it->value : 1.0f;
}

// ---------------------------------------------------------------------------
// Field management
// ---------------------------------------------------------------------------
int UnifiedGravityFieldSystem::add_field(const GravityField &p_field) {
    fields.push_back(p_field);
    return fields.size() - 1;
}

void UnifiedGravityFieldSystem::remove_field(int p_idx) {
    ERR_FAIL_INDEX(p_idx, fields.size());
    fields.remove_at(p_idx);
}

void UnifiedGravityFieldSystem::clear_fields() {
    fields.clear();
}

int UnifiedGravityFieldSystem::get_field_count() const { return fields.size(); }

UnifiedGravityFieldSystem::GravityField &UnifiedGravityFieldSystem::get_field(int p_idx) {
    return fields[p_idx];
}

const UnifiedGravityFieldSystem::GravityField &UnifiedGravityFieldSystem::get_field(int p_idx) const {
    return fields[p_idx];
}

// ---------------------------------------------------------------------------
// Compute acceleration from all fields at a point.
// ---------------------------------------------------------------------------
Vector3 UnifiedGravityFieldSystem::compute_acceleration(const Vector3 &p_body_position,
                                                         real_t p_body_mass,
                                                         real_t p_body_gravity_scale) const {
    Vector3 total_accel(0,0,0);
    for (const GravityField &field : fields) {
        if (!field.enabled) continue;
        Vector3 accel;
        switch (field.type) {
            case FIELD_ATTRACTOR: {
                Vector3 diff = field.position - p_body_position;
                real_t dist2 = diff.length_squared();
                real_t dist = Math::sqrt(dist2);
                // Effective distance with softening.
                real_t eff_dist2 = dist2 + field.softening * field.softening;
                // Gravitational acceleration magnitude = G * mass / eff_dist^2 * falloff.
                // Strength is used as G.
                real_t mag = field.strength * field.mass / eff_dist2;
                // Clamp distance falloff.
                if (field.max_distance > 0.0f && dist > field.max_distance) continue;
                // For falloff power != 2, adjust.
                real_t falloff = Math::pow(MAX(1.0f, dist), field.falloff_power);
                if (field.falloff_power > 0.0f) mag /= falloff / (dist2 + field.softening*field.softening);  // this needs careful maths.
                // Simpler: apply Newton's law directly.
                real_t softening2 = field.softening * field.softening;
                real_t dist_sq = dist2 + softening2;
                mag = field.strength * field.mass / dist_sq;
                // Apply falloff power as extra factor.
                if (field.falloff_power != 2.0f) {
                    real_t inv_dist = 1.0f / MAX(Math::sqrt(dist_sq), 0.001f);
                    mag *= Math::pow(inv_dist, field.falloff_power - 2.0f);
                }
                // Direction towards field.
                Vector3 dir = diff.normalized();
                accel = dir * mag;
            } break;
            case FIELD_REPULSOR: {
                Vector3 diff = p_body_position - field.position;
                real_t dist2 = diff.length_squared();
                real_t dist = Math::sqrt(dist2);
                real_t softening2 = field.softening * field.softening;
                real_t dist_sq = dist2 + softening2;
                real_t mag = field.strength * field.mass / dist_sq;
                if (field.max_distance > 0.0f && dist > field.max_distance) continue;
                if (field.falloff_power != 2.0f) {
                    real_t inv_dist = 1.0f / MAX(Math::sqrt(dist_sq), 0.001f);
                    mag *= Math::pow(inv_dist, field.falloff_power - 2.0f);
                }
                Vector3 dir = diff.normalized();
                accel = dir * mag;
            } break;
            case FIELD_DIRECTIONAL: {
                accel = field.direction.normalized() * field.strength;
            } break;
        }
        total_accel += accel * p_body_gravity_scale;
    }
    return total_accel;
}

// ---------------------------------------------------------------------------
// Apply forces to all bodies in all worlds.
// ---------------------------------------------------------------------------
void UnifiedGravityFieldSystem::apply_forces(real_t p_dt,
                                              newton::NewtonWorld *newton_world,
                                              genesis::GenesisWorld *genesis_world,
                                              vienna::ViennaWorld *vienna_world,
                                              wicked::WickedWorld *wicked_world) const {
    for (int engine = 0; engine < 4; ++engine) {
        LocalVector<uint64_t> body_ids;
        collect_body_ids(engine, newton_world, genesis_world, vienna_world, wicked_world, body_ids);
        for (uint64_t id : body_ids) {
            Vector3 pos;
            real_t mass;
            get_body_state(engine, id, newton_world, genesis_world, vienna_world, wicked_world, pos, mass);
            real_t scale = get_body_gravity_scale(engine, id);
            Vector3 accel = compute_acceleration(pos, mass, scale);
            Vector3 force = accel * mass;
            if (force.length_squared() > CMP_EPSILON) {
                apply_body_force(engine, id, force, newton_world, genesis_world, vienna_world, wicked_world);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Collect all body IDs for a given engine.
// ---------------------------------------------------------------------------
void UnifiedGravityFieldSystem::collect_body_ids(int p_engine,
                                                  const newton::NewtonWorld *nw,
                                                  const genesis::GenesisWorld *gw,
                                                  const vienna::ViennaWorld *vw,
                                                  const wicked::WickedWorld *ww,
                                                  LocalVector<uint64_t> &r_ids) const {
    r_ids.clear();
    switch (p_engine) {
        case 0: if (nw) { auto ids = nw->get_body_ids(); for (auto id : ids) r_ids.push_back(id); } break;
        case 1: if (gw) { auto ids = gw->get_all_entity_uids(); for (auto id : ids) r_ids.push_back(id); } break;
        case 2: if (vw) { auto ids = vw->get_body_ids(); for (auto id : ids) r_ids.push_back(id); } break;
        case 3: if (ww) { auto ids = ww->get_body_ids(); for (auto id : ids) r_ids.push_back(id); } break;
    }
}

// ---------------------------------------------------------------------------
// Get body state.
// ---------------------------------------------------------------------------
void UnifiedGravityFieldSystem::get_body_state(int p_engine, uint64_t p_body_id,
                                                const newton::NewtonWorld *nw,
                                                const genesis::GenesisWorld *gw,
                                                const vienna::ViennaWorld *vw,
                                                const wicked::WickedWorld *ww,
                                                Vector3 &r_pos, real_t &r_mass) const {
    r_pos = Vector3();
    r_mass = 1.0f;
    switch (p_engine) {
        case 0: if (nw) { Ref<newton::NewtonBody> b = nw->get_body(p_body_id); if (b.is_valid()) { r_pos = b->get_position(); r_mass = b->get_mass(); } } break;
        case 1: if (gw) { Ref<genesis::BaseEntity> e = gw->get_entity(p_body_id); if (e.is_valid()) { r_pos = e->get_position(); r_mass = e->get_mass(); } } break;
        case 2: if (vw) { Ref<vienna::ViennaBody> b = vw->get_body(p_body_id); if (b.is_valid()) { r_pos = b->get_position(); r_mass = b->get_mass(); } } break;
        case 3: if (ww) { Ref<wicked::WickedBody> b = ww->get_body(p_body_id); if (b.is_valid()) { r_pos = b->get_position(); r_mass = b->get_mass(); } } break;
    }
}

// ---------------------------------------------------------------------------
// Apply force to a body.
// ---------------------------------------------------------------------------
void UnifiedGravityFieldSystem::apply_body_force(int p_engine, uint64_t p_body_id,
                                                  const Vector3 &p_force,
                                                  newton::NewtonWorld *nw,
                                                  genesis::GenesisWorld *gw,
                                                  vienna::ViennaWorld *vw,
                                                  wicked::WickedWorld *ww) const {
    switch (p_engine) {
        case 0: if (nw) { Ref<newton::NewtonBody> b = nw->get_body(p_body_id); if (b.is_valid()) b->apply_force(p_force, b->get_position()); } break;
        case 1: if (gw) { Ref<genesis::BaseEntity> e = gw->get_entity(p_body_id); if (e.is_valid()) e->apply_force(p_force, e->get_position()); } break;
        case 2: if (vw) { Ref<vienna::ViennaBody> b = vw->get_body(p_body_id); if (b.is_valid()) b->apply_force(p_force, b->get_position()); } break;
        case 3: if (ww) { Ref<wicked::WickedBody> b = ww->get_body(p_body_id); if (b.is_valid()) b->apply_force(p_force, b->get_position()); } break;
    }
}

} // namespace unified