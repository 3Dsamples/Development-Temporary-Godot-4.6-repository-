// Name : lighting enhancement
// File : scene/3d/decal_ext.cpp 20 of 60
// Description : Implementation of DecalExt with size, textures (albedo, normal, ORM, emissive),
//               material properties, distance/angle fade, culling, GI, and RenderingServer sync.
#include "decal_ext.h"
#include "servers/rendering_server.h"
#include "core/math/vector3.h"
#include "core/math/color.h"
#include "core/math/math_funcs.h"
#include <cmath>

struct DecalExt::Impl {
    RID decal_rid;
    Vector3 size = Vector3(1, 1, 1);
    RID albedo_texture;
    RID normal_texture;
    RID orm_texture;
    RID emissive_texture;
    Color albedo = Color(1, 1, 1, 1);
    Color emissive_color = Color(0, 0, 0);
    float emissive_intensity = 0.0f;
    float roughness = 0.5f;
    float metalness = 0.0f;
    float occlusion = 1.0f;
    float normal_strength = 1.0f;
    bool distance_fade_enabled = false;
    float distance_fade_begin = 10.0f;
    float distance_fade_end = 20.0f;
    bool angle_fade_enabled = false;
    float angle_fade_threshold = 60.0f; // degrees
    uint32_t cull_mask = 0xFFFFFFFF;
    int gi_mode = 1; // 0=off,1=static,2=dynamic
    float gi_contribution = 1.0f;
    bool dirty = true;

    Impl() {
        decal_rid = RenderingServer::get_singleton()->decal_create();
    }

    ~Impl() {
        if (decal_rid.is_valid()) {
            RenderingServer::get_singleton()->free(decal_rid);
        }
    }

    void sync() {
        if (!dirty) return;
        RenderingServer *rs = RenderingServer::get_singleton();
        rs->decal_set_size(decal_rid, size);
        if (albedo_texture.is_valid()) rs->decal_set_texture(decal_rid, RS::DECAL_TEXTURE_ALBEDO, albedo_texture);
        if (normal_texture.is_valid()) rs->decal_set_texture(decal_rid, RS::DECAL_TEXTURE_NORMAL, normal_texture);
        if (orm_texture.is_valid()) rs->decal_set_texture(decal_rid, RS::DECAL_TEXTURE_ORM, orm_texture);
        if (emissive_texture.is_valid()) rs->decal_set_texture(decal_rid, RS::DECAL_TEXTURE_EMISSIVE, emissive_texture);
        rs->decal_set_albedo(decal_rid, albedo);
        rs->decal_set_emissive(decal_rid, emissive_color, emissive_intensity);
        rs->decal_set_roughness(decal_rid, roughness);
        rs->decal_set_metalness(decal_rid, metalness);
        rs->decal_set_occlusion(decal_rid, occlusion);
        rs->decal_set_normal_strength(decal_rid, normal_strength);
        if (distance_fade_enabled) {
            rs->decal_set_distance_fade(decal_rid, distance_fade_begin, distance_fade_end);
        } else {
            rs->decal_set_distance_fade(decal_rid, -1.0f, -1.0f);
        }
        if (angle_fade_enabled) {
            float rad = Math::deg2rad(angle_fade_threshold);
            rs->decal_set_angle_fade(decal_rid, rad);
        } else {
            rs->decal_set_angle_fade(decal_rid, -1.0f);
        }
        rs->decal_set_cull_mask(decal_rid, cull_mask);
        rs->decal_set_gi_mode(decal_rid, gi_mode);
        rs->decal_set_gi_contribution(decal_rid, gi_contribution);
        dirty = false;
    }
};

DecalExt::DecalExt() {
    pimpl = new Impl();
}

DecalExt::~DecalExt() {
    delete pimpl;
}

void DecalExt::set_size(const Vector3 &p_size) {
    pimpl->size = p_size;
    pimpl->dirty = true;
    sync_decal();
}
Vector3 DecalExt::get_size() const { return pimpl->size; }

void DecalExt::set_albedo_texture(const RID &p_texture) {
    pimpl->albedo_texture = p_texture;
    pimpl->dirty = true;
    sync_decal();
}
RID DecalExt::get_albedo_texture() const { return pimpl->albedo_texture; }

void DecalExt::set_normal_texture(const RID &p_texture) {
    pimpl->normal_texture = p_texture;
    pimpl->dirty = true;
    sync_decal();
}
RID DecalExt::get_normal_texture() const { return pimpl->normal_texture; }

void DecalExt::set_orm_texture(const RID &p_texture) {
    pimpl->orm_texture = p_texture;
    pimpl->dirty = true;
    sync_decal();
}
RID DecalExt::get_orm_texture() const { return pimpl->orm_texture; }

void DecalExt::set_emissive_texture(const RID &p_texture) {
    pimpl->emissive_texture = p_texture;
    pimpl->dirty = true;
    sync_decal();
}
RID DecalExt::get_emissive_texture() const { return pimpl->emissive_texture; }

void DecalExt::set_albedo(const Color &p_color) {
    pimpl->albedo = p_color;
    pimpl->dirty = true;
    sync_decal();
}
Color DecalExt::get_albedo() const { return pimpl->albedo; }

void DecalExt::set_emissive(const Color &p_color, float p_intensity) {
    pimpl->emissive_color = p_color;
    pimpl->emissive_intensity = p_intensity;
    pimpl->dirty = true;
    sync_decal();
}
Color DecalExt::get_emissive() const { return pimpl->emissive_color; }
float DecalExt::get_emissive_intensity() const { return pimpl->emissive_intensity; }

void DecalExt::set_roughness(float p_roughness) {
    pimpl->roughness = p_roughness;
    pimpl->dirty = true;
    sync_decal();
}
float DecalExt::get_roughness() const { return pimpl->roughness; }

void DecalExt::set_metalness(float p_metalness) {
    pimpl->metalness = p_metalness;
    pimpl->dirty = true;
    sync_decal();
}
float DecalExt::get_metalness() const { return pimpl->metalness; }

void DecalExt::set_occlusion(float p_occlusion) {
    pimpl->occlusion = p_occlusion;
    pimpl->dirty = true;
    sync_decal();
}
float DecalExt::get_occlusion() const { return pimpl->occlusion; }

void DecalExt::set_normal_strength(float p_strength) {
    pimpl->normal_strength = p_strength;
    pimpl->dirty = true;
    sync_decal();
}
float DecalExt::get_normal_strength() const { return pimpl->normal_strength; }

void DecalExt::set_distance_fade_enabled(bool p_enabled) {
    pimpl->distance_fade_enabled = p_enabled;
    pimpl->dirty = true;
    sync_decal();
}
bool DecalExt::is_distance_fade_enabled() const { return pimpl->distance_fade_enabled; }

void DecalExt::set_distance_fade_range(float p_begin, float p_end) {
    pimpl->distance_fade_begin = p_begin;
    pimpl->distance_fade_end = p_end;
    pimpl->dirty = true;
    sync_decal();
}
void DecalExt::get_distance_fade_range(float &p_begin, float &p_end) const {
    p_begin = pimpl->distance_fade_begin;
    p_end = pimpl->distance_fade_end;
}

void DecalExt::set_angle_fade_enabled(bool p_enabled) {
    pimpl->angle_fade_enabled = p_enabled;
    pimpl->dirty = true;
    sync_decal();
}
bool DecalExt::is_angle_fade_enabled() const { return pimpl->angle_fade_enabled; }

void DecalExt::set_angle_fade_threshold(float p_angle_deg) {
    pimpl->angle_fade_threshold = p_angle_deg;
    pimpl->dirty = true;
    sync_decal();
}
float DecalExt::get_angle_fade_threshold() const { return pimpl->angle_fade_threshold; }

void DecalExt::set_cull_mask(uint32_t p_mask) {
    pimpl->cull_mask = p_mask;
    pimpl->dirty = true;
    sync_decal();
}
uint32_t DecalExt::get_cull_mask() const { return pimpl->cull_mask; }

void DecalExt::set_gi_mode(int p_mode) {
    pimpl->gi_mode = p_mode;
    pimpl->dirty = true;
    sync_decal();
}
int DecalExt::get_gi_mode() const { return pimpl->gi_mode; }

void DecalExt::set_gi_contribution(float p_amount) {
    pimpl->gi_contribution = p_amount;
    pimpl->dirty = true;
    sync_decal();
}
float DecalExt::get_gi_contribution() const { return pimpl->gi_contribution; }

void DecalExt::sync_decal() {
    pimpl->sync();
}