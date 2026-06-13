// Name : lighting enhancement
// File : scene/3d/geometry_instance_3d_ext.cpp 10 of 60
// Description : Implementation of GeometryInstance3DExt with material overrides,
//               shadow settings, distance fade, LOD, and full RenderingServer sync.
#include "geometry_instance_3d_ext.h"
#include "servers/rendering_server.h"
#include "core/templates/hash_map.h"

struct GeometryInstance3DExt::Impl {
    RID instance_rid;
    RID material_override_rid;
    HashMap<int, RID> surface_materials;
    int cast_shadow_setting = 1; // 0=off,1=on,2=double-sided,3=shadows_only
    bool receive_shadow = true;
    bool distance_fade_enabled = false;
    float distance_fade_min = 0.0f;
    float distance_fade_max = 100.0f;
    float distance_fade_length = 0.0f;
    float lod_distance = 0.0f; // 0 = use default LOD from mesh
    bool material_dirty = true;
    bool shadow_dirty = true;
    bool fade_dirty = true;
    bool lod_dirty = true;

    Impl(RID p_instance_rid) : instance_rid(p_instance_rid) {}

    void sync_materials() {
        if (!material_dirty) return;
        RenderingServer *rs = RenderingServer::get_singleton();
        rs->instance_set_material_override(instance_rid, material_override_rid);
        for (const auto &E : surface_materials) {
            rs->instance_set_surface_material(instance_rid, E.key, E.value);
        }
        material_dirty = false;
    }

    void sync_shadow_params() {
        if (!shadow_dirty) return;
        RenderingServer *rs = RenderingServer::get_singleton();
        rs->instance_set_cast_shadows_setting(instance_rid, (RS::ShadowCastingSetting)cast_shadow_setting);
        rs->instance_set_receive_shadows(instance_rid, receive_shadow);
        shadow_dirty = false;
    }

    void sync_fade() {
        if (!fade_dirty) return;
        RenderingServer *rs = RenderingServer::get_singleton();
        rs->instance_set_distance_fade(instance_rid, distance_fade_enabled, distance_fade_min, distance_fade_max, distance_fade_length);
        fade_dirty = false;
    }

    void sync_lod() {
        if (!lod_dirty) return;
        RenderingServer *rs = RenderingServer::get_singleton();
        rs->instance_set_lod_distance(instance_rid, lod_distance);
        lod_dirty = false;
    }
};

GeometryInstance3DExt::GeometryInstance3DExt() {
    RID inst_rid = get_instance_rid(); // from VisualInstance3DExt base
    pimpl = new Impl(inst_rid);
}

GeometryInstance3DExt::~GeometryInstance3DExt() {
    delete pimpl;
}

void GeometryInstance3DExt::set_material_override(const RID &p_material) {
    pimpl->material_override_rid = p_material;
    pimpl->material_dirty = true;
    sync_materials();
}

RID GeometryInstance3DExt::get_material_override() const {
    return pimpl->material_override_rid;
}

void GeometryInstance3DExt::set_surface_material(int p_surface, const RID &p_material) {
    pimpl->surface_materials[p_surface] = p_material;
    pimpl->material_dirty = true;
    sync_materials();
}

RID GeometryInstance3DExt::get_surface_material(int p_surface) const {
    auto it = pimpl->surface_materials.find(p_surface);
    return it != pimpl->surface_materials.end() ? it->value : RID();
}

void GeometryInstance3DExt::clear_surface_material(int p_surface) {
    pimpl->surface_materials.erase(p_surface);
    pimpl->material_dirty = true;
    sync_materials();
}

void GeometryInstance3DExt::set_cast_shadow(int p_cast_shadow) {
    pimpl->cast_shadow_setting = p_cast_shadow;
    pimpl->shadow_dirty = true;
    sync_shadow_params();
}

int GeometryInstance3DExt::get_cast_shadow() const {
    return pimpl->cast_shadow_setting;
}

void GeometryInstance3DExt::set_receive_shadow(bool p_receive) {
    pimpl->receive_shadow = p_receive;
    pimpl->shadow_dirty = true;
    sync_shadow_params();
}

bool GeometryInstance3DExt::get_receive_shadow() const {
    return pimpl->receive_shadow;
}

void GeometryInstance3DExt::set_distance_fade_enabled(bool p_enabled) {
    pimpl->distance_fade_enabled = p_enabled;
    pimpl->fade_dirty = true;
    sync_fade();
}

bool GeometryInstance3DExt::is_distance_fade_enabled() const {
    return pimpl->distance_fade_enabled;
}

void GeometryInstance3DExt::set_distance_fade_range(float p_min, float p_max, float p_length) {
    pimpl->distance_fade_min = p_min;
    pimpl->distance_fade_max = p_max;
    pimpl->distance_fade_length = p_length;
    pimpl->fade_dirty = true;
    sync_fade();
}

void GeometryInstance3DExt::get_distance_fade_range(float &p_min, float &p_max, float &p_length) const {
    p_min = pimpl->distance_fade_min;
    p_max = pimpl->distance_fade_max;
    p_length = pimpl->distance_fade_length;
}

void GeometryInstance3DExt::set_lod_distance(float p_distance) {
    pimpl->lod_distance = p_distance;
    pimpl->lod_dirty = true;
    sync_lod();
}

float GeometryInstance3DExt::get_lod_distance() const {
    return pimpl->lod_distance;
}

void GeometryInstance3DExt::sync_geometry() {
    // Nothing to sync here; geometry is per mesh instance.
}

void GeometryInstance3DExt::sync_materials() {
    pimpl->sync_materials();
}

void GeometryInstance3DExt::sync_shadow_params() {
    pimpl->sync_shadow_params();
}

void GeometryInstance3DExt::sync_fade() {
    pimpl->sync_fade();
}