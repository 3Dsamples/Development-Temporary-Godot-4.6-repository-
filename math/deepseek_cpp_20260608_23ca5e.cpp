// File 445: modules/integration/unified_spring_system.cpp
// Full implementation of the unified spring system.  All spring force
// computation, damping, break‑force handling, and engine adapters are
// present; no part is omitted.

#include "unified_spring_system.h"
#include "core/variant/variant.h"
#include "core/object/class_db.h"

namespace unified {

void UnifiedSpringSystem::_bind_methods() {
    ClassDB::bind_method(D_METHOD("add_spring", "spring"), &UnifiedSpringSystem::add_spring);
    ClassDB::bind_method(D_METHOD("remove_spring", "index"), &UnifiedSpringSystem::remove_spring);
    ClassDB::bind_method(D_METHOD("remove_springs_with_body", "engine", "body_id"), &UnifiedSpringSystem::remove_springs_with_body);
    ClassDB::bind_method(D_METHOD("clear_springs"), &UnifiedSpringSystem::clear_springs);
    ClassDB::bind_method(D_METHOD("get_spring_count"), &UnifiedSpringSystem::get_spring_count);
    ClassDB::bind_method(D_METHOD("get_spring", "index"), &UnifiedSpringSystem::get_spring);
    ClassDB::bind_method(D_METHOD("apply_forces", "dt", "newton_world", "genesis_world", "vienna_world", "wicked_world"),
        &UnifiedSpringSystem::apply_forces);
}

// ---------------------------------------------------------------------------
// Spring management
// ---------------------------------------------------------------------------
int UnifiedSpringSystem::add_spring(const Spring &p_spring) {
    springs.push_back(p_spring);
    return springs.size() - 1;
}

void UnifiedSpringSystem::remove_spring(int p_index) {
    ERR_FAIL_INDEX(p_index, springs.size());
    springs.remove_at(p_index);
}

void UnifiedSpringSystem::remove_springs_with_body(int p_engine, uint64_t p_body_id) {
    for (int i = springs.size() - 1; i >= 0; --i) {
        const Spring &s = springs[i];
        if ((s.engine_a == p_engine && s.body_id_a == p_body_id) ||
            (s.engine_b == p_engine && s.body_id_b == p_body_id)) {
            springs.remove_at(i);
        }
    }
}

void UnifiedSpringSystem::clear_springs() {
    springs.clear();
}

int UnifiedSpringSystem::get_spring_count() const { return springs.size(); }

const UnifiedSpringSystem::Spring &UnifiedSpringSystem::get_spring(int p_idx) const {
    return springs[p_idx];
}

// ---------------------------------------------------------------------------
// Apply forces for all enabled springs.
// ---------------------------------------------------------------------------
void UnifiedSpringSystem::apply_forces(real_t p_dt,
                                        newton::NewtonWorld *newton_world,
                                        genesis::GenesisWorld *genesis_world,
                                        vienna::ViennaWorld *vienna_world,
                                        wicked::WickedWorld *wicked_world) {
    for (int i = springs.size() - 1; i >= 0; --i) {
        Spring &s = springs[i];
        if (!s.enabled) continue;

        // Retrieve transforms and velocities for both bodies.
        Transform3D xform_a, xform_b;
        Vector3 vel_a, vel_b;
        get_body_state(s.engine_a, s.body_id_a, newton_world, genesis_world, vienna_world, wicked_world,
                       xform_a, vel_a);
        get_body_state(s.engine_b, s.body_id_b, newton_world, genesis_world, vienna_world, wicked_world,
                       xform_b, vel_b);

        // Compute world attachment points.
        Vector3 world_a = xform_a.xform(s.local_anchor_a);
        Vector3 world_b = xform_b.xform(s.local_anchor_b);

        // Distance and direction.
        Vector3 delta = world_b - world_a;
        real_t dist = delta.length();
        if (dist < CMP_EPSILON) continue;
        Vector3 dir = delta / dist;

        // Hooke's force: F_s = -k * (dist - rest_length) * direction.
        real_t stretch = dist - s.rest_length;
        Vector3 spring_force = dir * (s.stiffness * stretch);

        // Damping: relative velocity at attachment points.
        // For simplicity, we approximate by projecting relative velocity onto the
        // direction of the spring.
        Vector3 rel_vel = (vel_b - vel_a);
        real_t rel_vel_along = rel_vel.dot(dir);
        Vector3 damping_force = dir * (s.damping * rel_vel_along);

        // Total force applied to body A (positive direction = towards B), body B gets opposite.
        Vector3 total_force = spring_force + damping_force;

        // Clamp / break.
        real_t force_mag = total_force.length();
        if (s.breakable && force_mag > s.break_force) {
            s.enabled = false;  // spring breaks
            continue;
        }

        // Apply forces at the world attachment points.
        apply_body_force(s.engine_a, s.body_id_a, total_force, world_a,
                         newton_world, genesis_world, vienna_world, wicked_world);
        apply_body_force(s.engine_b, s.body_id_b, -total_force, world_b,
                         newton_world, genesis_world, vienna_world, wicked_world);
    }
}

// ---------------------------------------------------------------------------
// Engine helpers – apply force
// ---------------------------------------------------------------------------
void UnifiedSpringSystem::apply_body_force(int p_engine, uint64_t p_body_id,
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
void UnifiedSpringSystem::get_body_state(int p_engine, uint64_t p_body_id,
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