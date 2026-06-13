// mesh_instance_3d.cpp
#include "mesh_instance_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

namespace lighting {

struct MeshInstance3D::Impl {
    // Mesh
    int64_t mesh_rid = -1;
    char mesh_path[256] = {0};

    // Skinning
    int64_t skeleton_rid = -1;
    int64_t skin_rid = -1;

    // Blend shapes
    std::vector<float> blend_shape_weights;
    std::vector<std::string> blend_shape_names;

    // LOD levels (distance -> mesh RID)
    struct LodEntry { float distance; int64_t mesh_rid; };
    std::vector<LodEntry> lod_meshes;

    // Shadow mesh (simplified)
    int64_t shadow_mesh_rid = -1;

    // Lightmap data
    std::vector<float> lightmap_uvs;     // 2 floats per vertex
    int64_t lightmap_texture_rid = -1;

    // Instancing
    int instance_count = 0;
    std::vector<Transform3D> instance_transforms;

    // AABB (cached)
    double aabb_min[3] = {0,0,0};
    double aabb_max[3] = {0,0,0};
    bool aabb_valid = false;

    // Culling
    bool ignore_frustum_culling = false;
};

MeshInstance3D::MeshInstance3D() : pimpl(std::make_unique<Impl>()) {}
MeshInstance3D::~MeshInstance3D() = default;

void MeshInstance3D::set_mesh(int64_t mesh_rid) {
    pimpl->mesh_rid = mesh_rid;
    pimpl->mesh_path[0] = 0;
    _update_bounding_volume();
    // Notify render server
}
int64_t MeshInstance3D::get_mesh_rid() const { return pimpl->mesh_rid; }

void MeshInstance3D::set_mesh_path(const char* path) {
    strncpy(pimpl->mesh_path, path, 255);
    pimpl->mesh_path[255] = 0;
    // In real engine: load mesh resource, assign mesh_rid
}
const char* MeshInstance3D::get_mesh_path() const { return pimpl->mesh_path; }

void MeshInstance3D::set_skeleton(int64_t skeleton_rid) { pimpl->skeleton_rid = skeleton_rid; }
int64_t MeshInstance3D::get_skeleton_rid() const { return pimpl->skeleton_rid; }
void MeshInstance3D::set_skin(int64_t skin_rid) { pimpl->skin_rid = skin_rid; }
int64_t MeshInstance3D::get_skin_rid() const { return pimpl->skin_rid; }

void MeshInstance3D::set_blend_shape_count(int count) {
    pimpl->blend_shape_weights.resize(count, 0.0f);
    pimpl->blend_shape_names.resize(count);
}
int MeshInstance3D::get_blend_shape_count() const { return (int)pimpl->blend_shape_weights.size(); }
void MeshInstance3D::set_blend_shape_value(int index, float weight) {
    if (index >= 0 && index < (int)pimpl->blend_shape_weights.size())
        pimpl->blend_shape_weights[index] = weight;
}
float MeshInstance3D::get_blend_shape_value(int index) const {
    if (index >= 0 && index < (int)pimpl->blend_shape_weights.size())
        return pimpl->blend_shape_weights[index];
    return 0.0f;
}
void MeshInstance3D::set_blend_shape_names(const char** names, int count) {
    pimpl->blend_shape_names.clear();
    for (int i = 0; i < count && names[i]; ++i)
        pimpl->blend_shape_names.emplace_back(names[i]);
}
const char* MeshInstance3D::get_blend_shape_name(int index) const {
    if (index >= 0 && index < (int)pimpl->blend_shape_names.size())
        return pimpl->blend_shape_names[index].c_str();
    return "";
}

void MeshInstance3D::set_lod_mesh_rid(float distance, int64_t mesh_rid) {
    pimpl->lod_meshes.push_back({distance, mesh_rid});
    std::sort(pimpl->lod_meshes.begin(), pimpl->lod_meshes.end(),
        [](const auto& a, const auto& b) { return a.distance < b.distance; });
}
void MeshInstance3D::clear_lod_meshes() { pimpl->lod_meshes.clear(); }
int64_t MeshInstance3D::get_active_lod_mesh(double camera_distance) const {
    int64_t active = pimpl->mesh_rid;
    for (const auto& lod : pimpl->lod_meshes) {
        if (camera_distance >= lod.distance)
            active = lod.mesh_rid;
        else
            break;
    }
    return active;
}

void MeshInstance3D::set_shadow_mesh_rid(int64_t mesh_rid) { pimpl->shadow_mesh_rid = mesh_rid; }
int64_t MeshInstance3D::get_shadow_mesh_rid() const { return pimpl->shadow_mesh_rid; }

void MeshInstance3D::set_lightmap_uvs(const std::vector<float>& uvs) {
    pimpl->lightmap_uvs = uvs;
}
void MeshInstance3D::set_lightmap_texture(int64_t texture_rid) { pimpl->lightmap_texture_rid = texture_rid; }
int64_t MeshInstance3D::get_lightmap_texture() const { return pimpl->lightmap_texture_rid; }

void MeshInstance3D::set_instance_count(int count) {
    pimpl->instance_count = count;
    pimpl->instance_transforms.resize(count);
}
int MeshInstance3D::get_instance_count() const { return pimpl->instance_count; }
void MeshInstance3D::set_instance_transform(int idx, const Transform3D& transform) {
    if (idx >= 0 && idx < pimpl->instance_count)
        pimpl->instance_transforms[idx] = transform;
}
Transform3D MeshInstance3D::get_instance_transform(int idx) const {
    if (idx >= 0 && idx < pimpl->instance_count)
        return pimpl->instance_transforms[idx];
    return Transform3D();
}

void MeshInstance3D::set_ignore_frustum_culling(bool ignore) {
    pimpl->ignore_frustum_culling = ignore;
    GeometryInstance3D::set_ignore_frustum_culling(ignore);
}
bool MeshInstance3D::get_ignore_frustum_culling() const { return pimpl->ignore_frustum_culling; }

void MeshInstance3D::synchronize_render_server(double delta) {
    GeometryInstance3D::synchronize_render_server(delta);
    // Update transformations for all instances if instanced
    if (pimpl->instance_count > 0) {
        // In real engine: upload instance transforms to GPU buffer
    }
    // Update blend shape weights on server
    // Update skeleton if changed
    // Update lightmap texture
}

void MeshInstance3D::_update_render_instance_transform() {
    GeometryInstance3D::_update_render_instance_transform();
    // For non-instanced, upload single transform
}

void MeshInstance3D::_update_bounding_volume() {
    // Compute AABB from mesh (simplified: could ask rendering server)
    if (pimpl->mesh_rid != -1) {
        // Placeholder – real code would get AABB from mesh data
        pimpl->aabb_min[0] = pimpl->aabb_min[1] = pimpl->aabb_min[2] = -1.0;
        pimpl->aabb_max[0] = pimpl->aabb_max[1] = pimpl->aabb_max[2] = 1.0;
        pimpl->aabb_valid = true;
    } else {
        pimpl->aabb_valid = false;
    }

    if (pimpl->aabb_valid) {
        set_aabb(pimpl->aabb_min, pimpl->aabb_max);
        // Compute sphere radius from AABB
        double dx = pimpl->aabb_max[0] - pimpl->aabb_min[0];
        double dy = pimpl->aabb_max[1] - pimpl->aabb_min[1];
        double dz = pimpl->aabb_max[2] - pimpl->aabb_min[2];
        double radius = sqrt(dx*dx + dy*dy + dz*dz) * 0.5;
        set_bounding_sphere_radius(radius);
    }
}

} // namespace lighting