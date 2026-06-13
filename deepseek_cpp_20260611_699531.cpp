// decal_3d.cpp
#include "decal_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numbers>

namespace lighting {

struct Decal3D::Impl {
    // Textures
    int64_t albedo_tex = -1;
    int64_t normal_tex = -1;
    int64_t orm_tex = -1;
    int64_t emissive_tex = -1;

    // Size half extents
    double size[3] = {1.0, 1.0, 1.0};

    // Fade
    bool distance_fade_enabled = false;
    double distance_fade_begin = 10.0;
    double distance_fade_end = 20.0;
    bool angle_fade_enabled = false;
    float angle_fade_threshold = 60.0f; // degrees

    // Material constants
    float albedo_color[3] = {1.0f, 1.0f, 1.0f};
    float emissive_color[3] = {0.0f, 0.0f, 0.0f};
    float emissive_intensity = 0.0f;
    float roughness = 0.5f;
    float metalness = 0.0f;
    float occlusion = 1.0f;
    float normal_strength = 1.0f;

    // Culling
    uint32_t cull_mask = 0xFFFFFFFF;
    float cull_angle = 180.0f; // no culling by default

    // GI
    int gi_mode = 1;    // static by default
    float gi_contribution = 1.0f;

    // Render server handles
    int64_t decal_rid = -1;
    bool dirty = true;
};

Decal3D::Decal3D() : pimpl(std::make_unique<Impl>()) {}
Decal3D::~Decal3D() = default;

void Decal3D::set_albedo_texture(int64_t texture_rid) {
    pimpl->albedo_tex = texture_rid;
    pimpl->dirty = true;
}
int64_t Decal3D::get_albedo_texture() const { return pimpl->albedo_tex; }
void Decal3D::set_normal_texture(int64_t texture_rid) { pimpl->normal_tex = texture_rid; pimpl->dirty = true; }
int64_t Decal3D::get_normal_texture() const { return pimpl->normal_tex; }
void Decal3D::set_orm_texture(int64_t texture_rid) { pimpl->orm_tex = texture_rid; pimpl->dirty = true; }
int64_t Decal3D::get_orm_texture() const { return pimpl->orm_tex; }
void Decal3D::set_emissive_texture(int64_t texture_rid) { pimpl->emissive_tex = texture_rid; pimpl->dirty = true; }
int64_t Decal3D::get_emissive_texture() const { return pimpl->emissive_tex; }

void Decal3D::set_size(const double* extents) {
    memcpy(pimpl->size, extents, 3*sizeof(double));
    pimpl->dirty = true;
}
void Decal3D::get_size(double* out_extents) const { memcpy(out_extents, pimpl->size, 3*sizeof(double)); }

void Decal3D::set_distance_fade_enabled(bool enabled) { pimpl->distance_fade_enabled = enabled; }
bool Decal3D::is_distance_fade_enabled() const { return pimpl->distance_fade_enabled; }
void Decal3D::set_distance_fade_range(double begin, double end) {
    pimpl->distance_fade_begin = begin;
    pimpl->distance_fade_end = end;
}
void Decal3D::get_distance_fade_range(double& begin, double& end) const {
    begin = pimpl->distance_fade_begin;
    end = pimpl->distance_fade_end;
}
void Decal3D::set_angle_fade_enabled(bool enabled) { pimpl->angle_fade_enabled = enabled; }
bool Decal3D::is_angle_fade_enabled() const { return pimpl->angle_fade_enabled; }
void Decal3D::set_angle_fade_threshold(float angle_deg) { pimpl->angle_fade_threshold = angle_deg; }
float Decal3D::get_angle_fade_threshold() const { return pimpl->angle_fade_threshold; }

void Decal3D::set_albedo(const float* color) { memcpy(pimpl->albedo_color, color, 3*sizeof(float)); }
void Decal3D::get_albedo(float* out_color) const { memcpy(out_color, pimpl->albedo_color, 3*sizeof(float)); }
void Decal3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void Decal3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}
void Decal3D::set_roughness(float roughness) { pimpl->roughness = roughness; }
float Decal3D::get_roughness() const { return pimpl->roughness; }
void Decal3D::set_metalness(float metalness) { pimpl->metalness = metalness; }
float Decal3D::get_metalness() const { return pimpl->metalness; }
void Decal3D::set_occlusion(float occlusion) { pimpl->occlusion = occlusion; }
float Decal3D::get_occlusion() const { return pimpl->occlusion; }
void Decal3D::set_normal_strength(float strength) { pimpl->normal_strength = strength; }
float Decal3D::get_normal_strength() const { return pimpl->normal_strength; }

void Decal3D::set_cull_mask(uint32_t mask) { pimpl->cull_mask = mask; }
uint32_t Decal3D::get_cull_mask() const { return pimpl->cull_mask; }
void Decal3D::set_cull_angle(float angle_deg) { pimpl->cull_angle = angle_deg; }
float Decal3D::get_cull_angle() const { return pimpl->cull_angle; }

void Decal3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; }
int Decal3D::get_gi_mode() const { return pimpl->gi_mode; }
void Decal3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
float Decal3D::get_gi_contribution() const { return pimpl->gi_contribution; }

void Decal3D::ready() {
    Node3D::ready();
    // Create decal in rendering server
    // In real engine: RenderingServer::decal_create()
    pimpl->decal_rid = 1234; // dummy
}

void Decal3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    if (!pimpl->dirty) return;
    // Update decal parameters in rendering server
    // This includes textures, size, fade ranges, material parameters.
    // Also compute world AABB for culling.
    Transform3D global = get_global_transform();
    // AABB in world space = transformed local half extents
    double min_x = -pimpl->size[0], max_x = pimpl->size[0];
    double min_y = -pimpl->size[1], max_y = pimpl->size[1];
    double min_z = -pimpl->size[2], max_z = pimpl->size[2];
    // (Transform would rotate and translate – but for AABB we need to transform all corners)
    // For simplicity, we compute AABB of the rotated box.
    double corners[8][3] = {
        {min_x, min_y, min_z}, {max_x, min_y, min_z},
        {min_x, max_y, min_z}, {max_x, max_y, min_z},
        {min_x, min_y, max_z}, {max_x, min_y, max_z},
        {min_x, max_y, max_z}, {max_x, max_y, max_z}
    };
    double world_min[3] = {1e30,1e30,1e30};
    double world_max[3] = {-1e30,-1e30,-1e30};
    for (int i = 0; i < 8; ++i) {
        Vector3 local(corners[i][0], corners[i][1], corners[i][2]);
        Vector3 world = global * local;
        world_min[0] = std::min(world_min[0], world.x);
        world_min[1] = std::min(world_min[1], world.y);
        world_min[2] = std::min(world_min[2], world.z);
        world_max[0] = std::max(world_max[0], world.x);
        world_max[1] = std::max(world_max[1], world.y);
        world_max[2] = std::max(world_max[2], world.z);
    }
    set_aabb(world_min, world_max);
    double dx = world_max[0]-world_min[0], dy = world_max[1]-world_min[1], dz = world_max[2]-world_min[2];
    set_bounding_sphere_radius(std::sqrt(dx*dx+dy*dy+dz*dz) * 0.5);

    // Set distance fade and angle fade in render server
    // Set textures
    // Set GI mode: if dynamic, may need to update every frame (but usually static)
    pimpl->dirty = false;
}

} // namespace lighting