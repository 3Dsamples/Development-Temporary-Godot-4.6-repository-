// multi_mesh_instance_3d.cpp
#include "multi_mesh_instance_3d.h"
#include <cstring>
#include <algorithm>
#include <limits>

namespace lighting {

// ============================================================================
// MultiMesh implementation
// ============================================================================
struct MultiMesh::Impl {
    int64_t mesh_rid = -1;
    int instance_count = 0;
    std::vector<Transform3D> transforms;
    std::vector<float> colors;       // RGBA per instance, 4 floats each
    std::vector<float> custom_data;  // up to 4 floats per instance
    int visible_start = 0;
    int visible_end = 0;             // 0 means all

    int buffer_hint = 0;             // 0 = static
    int64_t multi_mesh_rid = -1;     // RenderingServer handle
    bool buffer_dirty = true;
};

MultiMesh::MultiMesh() : pimpl(std::make_unique<Impl>()) {}
MultiMesh::~MultiMesh() = default;

void MultiMesh::set_mesh(int64_t mesh_rid) {
    pimpl->mesh_rid = mesh_rid;
    pimpl->buffer_dirty = true;
}
int64_t MultiMesh::get_mesh_rid() const { return pimpl->mesh_rid; }

void MultiMesh::set_instance_count(int count) {
    if (count < 0) return;
    pimpl->instance_count = count;
    pimpl->transforms.resize(count);
    pimpl->colors.assign(count * 4, 1.0f);
    pimpl->custom_data.assign(count * 4, 0.0f);
    pimpl->buffer_dirty = true;
}
int MultiMesh::get_instance_count() const { return pimpl->instance_count; }

void MultiMesh::set_instance_transform(int index, const Transform3D& transform) {
    if (index < 0 || index >= pimpl->instance_count) return;
    pimpl->transforms[index] = transform;
    pimpl->buffer_dirty = true;
}
Transform3D MultiMesh::get_instance_transform(int index) const {
    if (index < 0 || index >= pimpl->instance_count) return Transform3D();
    return pimpl->transforms[index];
}

void MultiMesh::set_instance_color(int index, const float* color) {
    if (index < 0 || index >= pimpl->instance_count) return;
    memcpy(&pimpl->colors[index * 4], color, 4 * sizeof(float));
    pimpl->buffer_dirty = true;
}
void MultiMesh::get_instance_color(int index, float* out_color) const {
    if (index < 0 || index >= pimpl->instance_count) return;
    memcpy(out_color, &pimpl->colors[index * 4], 4 * sizeof(float));
}

void MultiMesh::set_instance_custom_data(int index, const float* data, int data_size) {
    if (index < 0 || index >= pimpl->instance_count) return;
    int copy_size = std::min(data_size, 4);
    memcpy(&pimpl->custom_data[index * 4], data, copy_size * sizeof(float));
    for (int i = copy_size; i < 4; ++i) pimpl->custom_data[index * 4 + i] = 0.0f;
    pimpl->buffer_dirty = true;
}
void MultiMesh::get_instance_custom_data(int index, float* out_data, int data_size) const {
    if (index < 0 || index >= pimpl->instance_count) return;
    int copy_size = std::min(data_size, 4);
    memcpy(out_data, &pimpl->custom_data[index * 4], copy_size * sizeof(float));
    for (int i = copy_size; i < data_size; ++i) out_data[i] = 0.0f;
}

void MultiMesh::set_transform_array(const std::vector<Transform3D>& transforms) {
    if (transforms.size() != (size_t)pimpl->instance_count) return;
    pimpl->transforms = transforms;
    pimpl->buffer_dirty = true;
}
void MultiMesh::set_color_array(const std::vector<float>& colors) {
    if (colors.size() != (size_t)pimpl->instance_count * 4) return;
    pimpl->colors = colors;
    pimpl->buffer_dirty = true;
}

void MultiMesh::set_visible_instance_range(int start, int end) {
    pimpl->visible_start = start;
    pimpl->visible_end = end;
    pimpl->buffer_dirty = true;
}
void MultiMesh::get_visible_instance_range(int& start, int& end) const {
    start = pimpl->visible_start;
    end = pimpl->visible_end;
}

void MultiMesh::set_buffer_dirty() { pimpl->buffer_dirty = true; }
void MultiMesh::update_buffers() {
    if (!pimpl->buffer_dirty) return;
    if (pimpl->multi_mesh_rid == -1) {
        // In real engine: pimpl->multi_mesh_rid = RenderingServer::multi_mesh_create();
    }
    // Set mesh, instance count, buffer data (transforms, colors, custom data)
    // RenderingServer::multi_mesh_set_mesh(pimpl->multi_mesh_rid, pimpl->mesh_rid);
    // RenderingServer::multi_mesh_set_instance_count(pimpl->multi_mesh_rid, pimpl->instance_count);
    // RenderServer::multi_mesh_set_buffer(pimpl->multi_mesh_rid, transforms.data(), colors.data(), custom_data.data());
    // Set visible range
    pimpl->buffer_dirty = false;
}
void MultiMesh::set_buffer_usage_hint(int hint) { pimpl->buffer_hint = hint; }
int64_t MultiMesh::get_multi_mesh_rid() const { return pimpl->multi_mesh_rid; }

// ============================================================================
// MultiMeshInstance3D implementation
// ============================================================================
struct MultiMeshInstance3D::Impl {
    std::shared_ptr<MultiMesh> multi_mesh;
    bool frustum_culling_enabled = true;
    // Per‑instance AABBs for culling (if frustum culling enabled)
    struct InstanceAABB {
        double min[3], max[3];
        bool valid = false;
    };
    std::vector<InstanceAABB> instance_aabbs;

    bool cast_shadow = true;
    bool receive_shadow = true;
    int gi_mode = 1;          // static by default
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    bool dirty = true;
    int64_t instance_rid = -1; // RenderingServer instance for the multi mesh
};

MultiMeshInstance3D::MultiMeshInstance3D() : pimpl(std::make_unique<Impl>()) {}
MultiMeshInstance3D::~MultiMeshInstance3D() = default;

void MultiMeshInstance3D::set_multi_mesh(const std::shared_ptr<MultiMesh>& multi_mesh) {
    pimpl->multi_mesh = multi_mesh;
    pimpl->dirty = true;
}
std::shared_ptr<MultiMesh> MultiMeshInstance3D::get_multi_mesh() const { return pimpl->multi_mesh; }

void MultiMeshInstance3D::set_frustum_culling_enabled(bool enabled) {
    pimpl->frustum_culling_enabled = enabled;
    if (enabled && pimpl->multi_mesh) {
        int count = pimpl->multi_mesh->get_instance_count();
        pimpl->instance_aabbs.resize(count);
    }
}
bool MultiMeshInstance3D::is_frustum_culling_enabled() const { return pimpl->frustum_culling_enabled; }

void MultiMeshInstance3D::set_instance_aabb(int index, const double* min, const double* max) {
    if (!pimpl->multi_mesh || index < 0 || index >= pimpl->multi_mesh->get_instance_count()) return;
    if (!pimpl->frustum_culling_enabled) set_frustum_culling_enabled(true);
    if (index >= (int)pimpl->instance_aabbs.size()) pimpl->instance_aabbs.resize(index+1);
    pimpl->instance_aabbs[index].valid = true;
    memcpy(pimpl->instance_aabbs[index].min, min, 3*sizeof(double));
    memcpy(pimpl->instance_aabbs[index].max, max, 3*sizeof(double));
}
void MultiMeshInstance3D::clear_instance_aabb(int index) {
    if (!pimpl->multi_mesh || index < 0 || index >= (int)pimpl->instance_aabbs.size()) return;
    pimpl->instance_aabbs[index].valid = false;
}

void MultiMeshInstance3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; GeometryInstance3D::set_cast_shadow(cast); }
void MultiMeshInstance3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
void MultiMeshInstance3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; GeometryInstance3D::set_gi_mode(mode); }
void MultiMeshInstance3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
void MultiMeshInstance3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void MultiMeshInstance3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}

void MultiMeshInstance3D::update_multi_mesh() {
    if (!pimpl->multi_mesh) return;
    pimpl->multi_mesh->update_buffers();
    pimpl->dirty = true;
}

void MultiMeshInstance3D::synchronize_render_server(double delta) {
    GeometryInstance3D::synchronize_render_server(delta);
    if (!pimpl->multi_mesh) return;
    if (pimpl->dirty || pimpl->multi_mesh->get_multi_mesh_rid() == -1) {
        if (pimpl->instance_rid == -1) {
            // In real engine: pimpl->instance_rid = RenderingServer::instance_create();
        }
        // RenderingServer::instance_set_base(pimpl->instance_rid, pimpl->multi_mesh->get_multi_mesh_rid());
        // Set transform, visibility, shadow, GI, etc.
        Transform3D global = get_global_transform();
        // RenderingServer::instance_set_transform(pimpl->instance_rid, global);
        // Set cast_shadow, gi_mode, emissive
        pimpl->dirty = false;
    }
    // If frustum culling enabled, we would compute per‑instance visibility and set visible instances
    if (pimpl->frustum_culling_enabled && pimpl->multi_mesh) {
        // For each instance, test AABB against camera frustum, build list of visible indices,
        // then set visible instance range or update instance buffer.
    }
}

void MultiMeshInstance3D::process(double delta) {
    GeometryInstance3D::process(delta);
}

} // namespace lighting