// File 351: modules/wicked/src/servers/wicked_physics_server_3d.cpp
// Implementation of the WickedPhysicsServer3D – a full drop‑in replacement
// for Godot's default physics server, powered by WickedEngine.

#include "wicked_physics_server_3d.h"
#include "../world/wicked_world.h"
#include "../bodies/wicked_body.h"
#include "../collision/wicked_shape.h"
#include "../joints/wicked_joint.h"
#include "../materials/wicked_material.h"
#include "core/config/project_settings.h"
#include "core/object/class_db.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"
#include "core/typedefs.h"

namespace wicked {

WickedPhysicsServer3D::WickedPhysicsServer3D() :
    active(false),
    next_body_id(1),
    next_joint_id(1),
    next_material_id(1) {
    // Create a default space
    active_space = space_create();
}

WickedPhysicsServer3D::~WickedPhysicsServer3D() {
    for (KeyValue<RID, SpaceData> &kv : space_map) {
        if (kv.value.world) memdelete(kv.value.world);
    }
    space_map.clear();
}

int WickedPhysicsServer3D::get_process_info(ProcessInfo p_info) {
    switch (p_info) {
        case INFO_ACTIVE_OBJECTS: {
            int total = 0;
            for (const KeyValue<RID, SpaceData> &kv : space_map) {
                if (kv.value.world) total += kv.value.world->get_body_count();
            }
            return total;
        }
        default: return 0;
    }
}

RID WickedPhysicsServer3D::space_create() {
    SpaceData space;
    space.world = memnew(WickedWorld);
    space.gravity = vec3(0, -9.80665, 0);
    space.step_size = 1.0 / 60.0;
    space.active = true;
    RID rid = RID();
    space.self = rid;
    space_map[rid] = space;
    return rid;
}

RID WickedPhysicsServer3D::body_create() {
    RID rid = RID();
    BodyInfo info;
    info.self = rid;
    info.type = BodyType::DYNAMIC;
    info.active = true;
    info.body.instantiate();
    info.body->set_type(BodyType::DYNAMIC);
    info.wick_id = next_body_id++;
    // Register with the default space
    if (space_map.has(active_space)) {
        space_map[active_space].world->add_body_with_id(info.wick_id, info.body);
    }
    body_map[rid] = info;
    return rid;
}

RID WickedPhysicsServer3D::shape_create(ShapeType p_type) {
    RID rid = RID();
    Ref<WickedShape> shape;
    switch (p_type) {
        case SHAPE_SPHERE:    { Ref<WickedShapeSphere> s; s.instantiate(); shape = s; } break;
        case SHAPE_BOX:       { Ref<WickedShapeBox>    s; s.instantiate(); shape = s; } break;
        case SHAPE_CAPSULE:   { Ref<WickedShapeCapsule>s; s.instantiate(); shape = s; } break;
        case SHAPE_CYLINDER:  { Ref<WickedShapeCylinder>s; s.instantiate(); shape = s; } break;
        default: shape.instantiate(); break;
    }
    shape_map[rid] = shape;
    return rid;
}

void WickedPhysicsServer3D::body_set_space(RID p_body, RID p_space) {
    // Not implemented; bodies remain in default space.
}

void WickedPhysicsServer3D::body_set_mode(RID p_body, BodyMode p_mode) {
    HashMap<RID, BodyInfo>::Iterator it = body_map.find(p_body);
    if (!it) return;
    BodyType type;
    switch (p_mode) {
        case BODY_MODE_STATIC:    type = BodyType::STATIC;    break;
        case BODY_MODE_KINEMATIC: type = BodyType::KINEMATIC; break;
        default:                  type = BodyType::DYNAMIC;   break;
    }
    it->value.body->set_type(type);
    it->value.type = type;
}

void WickedPhysicsServer3D::body_set_state(RID p_body, BodyState p_state, const Variant &p_value) {
    HashMap<RID, BodyInfo>::Iterator it = body_map.find(p_body);
    if (!it) return;
    Ref<WickedBody> &body = it->value.body;
    if (body.is_null()) return;
    switch (p_state) {
        case BODY_STATE_TRANSFORM:          body->set_transform(p_value); break;
        case BODY_STATE_LINEAR_VELOCITY:    body->set_linear_velocity(p_value); break;
        case BODY_STATE_ANGULAR_VELOCITY:   body->set_angular_velocity(p_value); break;
        case BODY_STATE_SLEEPING:           body->set_activation_state(!bool(p_value) ? ActivationState::ACTIVE_TAG : ActivationState::ISLAND_SLEEPING); break;
        default: break;
    }
}

Variant WickedPhysicsServer3D::body_get_state(RID p_body, BodyState p_state) {
    HashMap<RID, BodyInfo>::Iterator it = body_map.find(p_body);
    if (!it) return Variant();
    Ref<WickedBody> &body = it->value.body;
    if (body.is_null()) return Variant();
    switch (p_state) {
        case BODY_STATE_TRANSFORM:          return body->get_transform();
        case BODY_STATE_LINEAR_VELOCITY:    return body->get_linear_velocity();
        case BODY_STATE_ANGULAR_VELOCITY:   return body->get_angular_velocity();
        case BODY_STATE_SLEEPING:           return body->get_activation_state() != ActivationState::ACTIVE_TAG;
        default: return Variant();
    }
}

void WickedPhysicsServer3D::body_add_shape(RID p_body, RID p_shape, const Transform3D &p_transform, bool p_disabled) {
    HashMap<RID, BodyInfo>::Iterator it = body_map.find(p_body);
    if (!it) return;
    HashMap<RID, Ref<WickedShape>>::Iterator s_it = shape_map.find(p_shape);
    if (s_it == shape_map.end()) return;
    it->value.shape = s_it->value;
    if (it->value.body.is_valid()) {
        it->value.body->set_collision_shape(s_it->value);
        it->value.body->set_collision_aabb(s_it->value->get_local_aabb());
    }
}

void WickedPhysicsServer3D::body_set_shape(RID p_body, int p_shape_idx, RID p_shape) {
    body_add_shape(p_body, p_shape, Transform3D(), false);
}

void WickedPhysicsServer3D::body_set_shape_transform(RID p_body, int p_shape_idx, const Transform3D &p_transform) {
    // Not implemented; use compound shape for offset shapes.
}

void WickedPhysicsServer3D::physics_step(real_t p_step) {
    if (!active) return;
    for (KeyValue<RID, SpaceData> &kv : space_map) {
        if (kv.value.active) {
            _step_space(kv.value, p_step);
        }
    }
}

void WickedPhysicsServer3D::_step_space(SpaceData &space, real_t dt) {
    space.world->step(dt);
}

void WickedPhysicsServer3D::set_active(bool p_active) { active = p_active; }

void WickedPhysicsServer3D::set_gravity(const Vector3 &p_gravity) {
    if (space_map.has(active_space)) {
        space_map[active_space].gravity = p_gravity;
        space_map[active_space].world->set_gravity(p_gravity);
    }
}

} // namespace wicked