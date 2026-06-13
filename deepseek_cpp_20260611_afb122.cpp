// Name : lighting enhancement
// File : scene/3d/mesh_instance_3d_ext.cpp 12 of 60
// Description : Implementation of MeshInstance3DExt with mesh resource, skeleton,
//               blend shapes, LOD, shadow, and full RenderingServer synchronization.
#include "mesh_instance_3d_ext.h"
#include "servers/rendering_server.h"
#include "core/templates/hash_map.h"
#include <cstring>

struct MeshInstance3DExt::Impl {
    RID instance_rid;
    RID mesh_rid;
    RID skeleton_rid;
    RID skin_rid;
    RID material_override_rid;
    HashMap<int, RID> surface_materials;
    Vector<float> blend_shape_values;
    float lod_distance = 0.0f;
    int cast_shadow_setting = 1;
    bool receive_shadow = true;
    int gi_mode = 1;
    float gi_contribution = 1.0f;
    bool mesh_dirty = true;
    bool skeleton_dirty = true;
    bool skin_dirty = true;
    bool blend_dirty = true;
    bool material_override_dirty = true;
    bool surface_materials_dirty = true;
    bool lod_dirty = true;
    bool shadow_dirty = true;
    bool gi_dirty = true;

    Impl(RID p_instance_rid) : instance_rid(p_instance_rid) {}

    void sync_mesh() {
        if (!mesh_dirty) return;
        RenderingServer::get_singleton()->instance_set_base(instance_rid, mesh_rid);
        mesh_dirty = false;
    }

    void sync_skeleton() {
        if (!skeleton_dirty && !skin_dirty) return;
        RenderingServer *rs = RenderingServer::get_singleton();
        if (skeleton_dirty) {
            rs->instance_set_skeleton(instance_rid, skeleton_rid);
            skeleton_dirty = false;
        }
        if (skin_dirty) {
            rs->instance_set_skin(instance_rid, skin_rid);
            skin_dirty = false;
        }
    }

    void sync_blend_shapes() {
        if (!blend_dirty) return;
        RenderingServer::get_singleton()->instance_set_blend_shape_values(instance_rid, blend_shape_values);
        blend_dirty = false;
    }

    void sync_materials() {
        if (material_override_dirty) {
            RenderingServer::get_singleton()->instance_set_material_override(instance_rid, material_override_rid);
            material_override_dirty = false;
        }
        if (surface_materials_dirty) {
            RenderingServer *rs = RenderingServer::get_singleton();
            for (const auto &E : surface_materials) {
                rs->instance_set_surface_material(instance_rid, E.key, E.value);
            }
            surface_materials_dirty = false;
        }
    }

    void sync_lod() {
        if (!lod_dirty) return;
        RenderingServer::get_singleton()->instance_set_lod_distance(instance_rid, lod_distance);
        lod_dirty = false;
    }

    void sync_shadow() {
        if (!shadow_dirty) return;
        RenderingServer *rs = RenderingServer::get_singleton();
        rs->instance_set_cast_shadows_setting(instance_rid, (RS::ShadowCastingSetting)cast_shadow_setting);
        rs->instance_set_receive_shadows(instance_rid, receive_shadow);
        shadow_dirty = false;
    }

    void sync_gi() {
        if (!gi_dirty) return;
        RenderingServer *rs = RenderingServer::get_singleton();
        rs->instance_set_gi_mode(instance_rid, gi_mode);
        rs->instance_set_gi_contribution(instance_rid, gi_contribution);
        gi_dirty = false;
    }

    void sync_all() {
        sync_mesh();
        sync_skeleton();
        sync_blend_shapes();
        sync_materials();
        sync_lod();
        sync_shadow();
        sync_gi();
    }
};

MeshInstance3DExt::MeshInstance3DExt() {
    RID inst_rid = get_instance_rid(); // from VisualInstance3DExt base
    pimpl = new Impl(inst_rid);
}

MeshInstance3DExt::~MeshInstance3DExt() {
    delete pimpl;
}

void MeshInstance3DExt::set_mesh(const RID &p_mesh) {
    pimpl->mesh_rid = p_mesh;
    pimpl->mesh_dirty = true;
    sync_mesh();
}

RID MeshInstance3DExt::get_mesh() const {
    return pimpl->mesh_rid;
}

void MeshInstance3DExt::set_material_override(const RID &p_material) {
    pimpl->material_override_rid = p_material;
    pimpl->material_override_dirty = true;
    sync_materials();
}

void MeshInstance3DExt::set_surface_material(int p_surface, const RID &p_material) {
    pimpl->surface_materials[p_surface] = p_material;
    pimpl->surface_materials_dirty = true;
    sync_materials();
}

void MeshInstance3DExt::set_skeleton(const RID &p_skeleton) {
    pimpl->skeleton_rid = p_skeleton;
    pimpl->skeleton_dirty = true;
    sync_skeleton();
}

RID MeshInstance3DExt::get_skeleton() const {
    return pimpl->skeleton_rid;
}

void MeshInstance3DExt::set_skin(const RID &p_skin) {
    pimpl->skin_rid = p_skin;
    pimpl->skin_dirty = true;
    sync_skeleton();
}

RID MeshInstance3DExt::get_skin() const {
    return pimpl->skin_rid;
}

void MeshInstance3DExt::set_blend_shape_value(int p_shape, float p_value) {
    if (pimpl->blend_shape_values.size() <= p_shape) {
        pimpl->blend_shape_values.resize(p_shape + 1, 0.0f);
    }
    pimpl->blend_shape_values[p_shape] = p_value;
    pimpl->blend_dirty = true;
    sync_blend_shapes();
}

float MeshInstance3DExt::get_blend_shape_value(int p_shape) const {
    return (p_shape >= 0 && p_shape < pimpl->blend_shape_values.size()) ? pimpl->blend_shape_values[p_shape] : 0.0f;
}

void MeshInstance3DExt::set_lod_distance(float p_distance) {
    pimpl->lod_distance = p_distance;
    pimpl->lod_dirty = true;
    sync_lod();
}

float MeshInstance3DExt::get_lod_distance() const {
    return pimpl->lod_distance;
}

void MeshInstance3DExt::set_cast_shadow(int p_cast_shadow) {
    pimpl->cast_shadow_setting = p_cast_shadow;
    pimpl->shadow_dirty = true;
    sync_shadow();
}

void MeshInstance3DExt::set_receive_shadow(bool p_receive) {
    pimpl->receive_shadow = p_receive;
    pimpl->shadow_dirty = true;
    sync_shadow();
}

void MeshInstance3DExt::set_gi_mode(int p_mode) {
    pimpl->gi_mode = p_mode;
    pimpl->gi_dirty = true;
    sync_gi();
}

void MeshInstance3DExt::set_gi_contribution(float p_amount) {
    pimpl->gi_contribution = p_amount;
    pimpl->gi_dirty = true;
    sync_gi();
}

void MeshInstance3DExt::sync_mesh() {
    pimpl->sync_mesh();
}

void MeshInstance3DExt::sync_skeleton() {
    pimpl->sync_skeleton();
}

void MeshInstance3DExt::sync_blend_shapes() {
    pimpl->sync_blend_shapes();
}

void MeshInstance3DExt::sync_all() {
    pimpl->sync_all();
}