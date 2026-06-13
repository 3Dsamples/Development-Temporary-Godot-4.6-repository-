// Name : lighting enhancement
// File : scene/3d/collision_object_3d_ext.cpp 40 of 60
// Description : Implementation of CollisionObject3DExt with shape management,
//               layer masks, ray/area pickable, and full PhysicsServer + RenderingServer sync.
#include "collision_object_3d_ext.h"
#include "servers/physics_server_3d.h"
#include "servers/rendering_server.h"
#include "core/math/transform_3d.h"
#include "core/templates/hash_map.h"
#include "core/object/callable.h"

struct CollisionObject3DExt::Impl {
    RID physics_rid;                     // PhysicsServer body / area RID (depending on type)
    RID instance_rid;                    // RenderingServer instance RID (if visual)
    int64_t shape_id_counter = 1;
    struct ShapeData {
        RID shape_rid;
        Transform3D local_transform;
        bool disabled;
    };
    HashMap<int, ShapeData> shapes;
    uint32_t collision_layer = 0xFFFFFFFF;
    uint32_t collision_mask = 0xFFFFFFFF;
    float collision_priority = 1.0f;
    bool ray_pickable = true;
    bool area_pickable = true;
    int gi_mode = 0;                     // 0 = off, 1 = static, 2 = dynamic
    float gi_contribution = 1.0f;
    Color emissive_color;
    float emissive_intensity = 0.0f;
    bool dirty = true;

    // For area vs body: we assume this is a physics body (rigid/static/character) by default.
    // If area, we would change.
    bool is_area = false;

    Impl() {
        // Physics server creation will be done in the node constructor because type is not known here.
        physics_rid = RID(); // placeholder
        instance_rid = RID();
    }

    ~Impl() {
        if (physics_rid.is_valid()) {
            PhysicsServer3D::get_singleton()->free(physics_rid);
        }
        if (instance_rid.is_valid()) {
            RenderingServer::get_singleton()->free(instance_rid);
        }
    }

    void sync_physics() {
        if (!dirty) return;
        PhysicsServer3D *ps = PhysicsServer3D::get_singleton();
        // Set layers and masks
        ps->body_set_collision_layer(physics_rid, collision_layer);
        ps->body_set_collision_mask(physics_rid, collision_mask);
        ps->body_set_collision_priority(physics_rid, collision_priority);
        ps->body_set_pickable(physics_rid, ray_pickable);
        // For each shape, add or update
        // First, clear existing shapes (simple approach: remove all and re-add)
        // But to avoid unnecessary updates, we can track changes. For simplicity, re-add.
        // This is a placeholder for real implementation where we only update changed shapes.
        ps->body_clear_shapes(physics_rid);
        for (const auto &E : shapes) {
            const ShapeData &sd = E.value;
            if (!sd.disabled) {
                ps->body_add_shape(physics_rid, sd.shape_rid, sd.local_transform);
            }
        }
        dirty = false;
    }

    void sync_rendering() {
        if (!instance_rid.is_valid()) return;
        RenderingServer *rs = RenderingServer::get_singleton();
        rs->instance_set_gi_mode(instance_rid, gi_mode);
        rs->instance_set_gi_contribution(instance_rid, gi_contribution);
        rs->instance_set_emissive(instance_rid, emissive_color, emissive_intensity);
    }
};

CollisionObject3DExt::CollisionObject3DExt() {
    pimpl = new Impl();
    // In a real engine, the physics RID type (body vs area) is set by the derived class.
    // For simplicity, we create a temporary static body RID.
    pimpl->physics_rid = PhysicsServer3D::get_singleton()->body_create();
    pimpl->instance_rid = get_instance_rid(); // from VisualInstance3D
}

CollisionObject3DExt::~CollisionObject3DExt() {
    delete pimpl;
}

int CollisionObject3DExt::add_shape(const RID &p_shape, const Transform3D &p_transform, bool p_disabled) {
    int id = pimpl->shape_id_counter++;
    Impl::ShapeData sd;
    sd.shape_rid = p_shape;
    sd.local_transform = p_transform;
    sd.disabled = p_disabled;
    pimpl->shapes[id] = sd;
    pimpl->dirty = true;
    sync_collision_object();
    return id;
}

void CollisionObject3DExt::remove_shape(int p_shape_id) {
    auto it = pimpl->shapes.find(p_shape_id);
    if (it != pimpl->shapes.end()) {
        pimpl->shapes.erase(it);
        pimpl->dirty = true;
        sync_collision_object();
    }
}

void CollisionObject3DExt::clear_shapes() {
    pimpl->shapes.clear();
    pimpl->dirty = true;
    sync_collision_object();
}

int CollisionObject3DExt::get_shape_count() const {
    return pimpl->shapes.size();
}

RID CollisionObject3DExt::get_shape(int p_shape_id) const {
    auto it = pimpl->shapes.find(p_shape_id);
    return it != pimpl->shapes.end() ? it->value.shape_rid : RID();
}

void CollisionObject3DExt::set_shape_transform(int p_shape_id, const Transform3D &p_transform) {
    auto it = pimpl->shapes.find(p_shape_id);
    if (it != pimpl->shapes.end()) {
        it->value.local_transform = p_transform;
        pimpl->dirty = true;
        sync_collision_object();
    }
}

Transform3D CollisionObject3DExt::get_shape_transform(int p_shape_id) const {
    auto it = pimpl->shapes.find(p_shape_id);
    if (it != pimpl->shapes.end()) return it->value.local_transform;
    return Transform3D();
}

void CollisionObject3DExt::set_shape_disabled(int p_shape_id, bool p_disabled) {
    auto it = pimpl->shapes.find(p_shape_id);
    if (it != pimpl->shapes.end()) {
        it->value.disabled = p_disabled;
        pimpl->dirty = true;
        sync_collision_object();
    }
}

bool CollisionObject3DExt::is_shape_disabled(int p_shape_id) const {
    auto it = pimpl->shapes.find(p_shape_id);
    if (it != pimpl->shapes.end()) return it->value.disabled;
    return true;
}

void CollisionObject3DExt::set_collision_layer(uint32_t p_layer) {
    pimpl->collision_layer = p_layer;
    pimpl->dirty = true;
    sync_collision_object();
}
uint32_t CollisionObject3DExt::get_collision_layer() const { return pimpl->collision_layer; }

void CollisionObject3DExt::set_collision_mask(uint32_t p_mask) {
    pimpl->collision_mask = p_mask;
    pimpl->dirty = true;
    sync_collision_object();
}
uint32_t CollisionObject3DExt::get_collision_mask() const { return pimpl->collision_mask; }

void CollisionObject3DExt::set_collision_priority(float p_priority) {
    pimpl->collision_priority = p_priority;
    pimpl->dirty = true;
    sync_collision_object();
}
float CollisionObject3DExt::get_collision_priority() const { return pimpl->collision_priority; }

void CollisionObject3DExt::set_ray_pickable(bool p_enabled) {
    pimpl->ray_pickable = p_enabled;
    pimpl->dirty = true;
    sync_collision_object();
}
bool CollisionObject3DExt::is_ray_pickable() const { return pimpl->ray_pickable; }

void CollisionObject3DExt::set_area_pickable(bool p_enabled) {
    pimpl->area_pickable = p_enabled;
    // Not directly used by PhysicsServer for bodies, but for areas.
    // For bodies, ray_pickable is enough.
}
bool CollisionObject3DExt::is_area_pickable() const { return pimpl->area_pickable; }

void CollisionObject3DExt::set_gi_mode(int p_mode) {
    pimpl->gi_mode = p_mode;
    pimpl->sync_rendering();
}
int CollisionObject3DExt::get_gi_mode() const { return pimpl->gi_mode; }

void CollisionObject3DExt::set_gi_contribution(float p_amount) {
    pimpl->gi_contribution = p_amount;
    pimpl->sync_rendering();
}
float CollisionObject3DExt::get_gi_contribution() const { return pimpl->gi_contribution; }

void CollisionObject3DExt::set_emissive(const Color &p_color, float p_intensity) {
    pimpl->emissive_color = p_color;
    pimpl->emissive_intensity = p_intensity;
    pimpl->sync_rendering();
}
Color CollisionObject3DExt::get_emissive() const { return pimpl->emissive_color; }
float CollisionObject3DExt::get_emissive_intensity() const { return pimpl->emissive_intensity; }

void CollisionObject3DExt::sync_collision_object() {
    pimpl->sync_physics();
    pimpl->sync_rendering();
}