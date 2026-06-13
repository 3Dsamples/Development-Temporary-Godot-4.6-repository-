// multi_mesh_instance_3d.cpp
#include "multi_mesh_instance_3d.h"
#include <cstring>
#include <algorithm>
#include <vector>
#include <limits>

namespace lighting {

// ============================================================================
// Per‑instance data (CPU copy for dynamic updates)
// ============================================================================
struct PerInstanceData {
    Transform3D transform;
    float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> custom_data;
    bool cast_shadow = true;
    int gi_mode = -1;           // -1 = use global default
    float gi_contribution = -1.0f;
    float emissive_color[3] = {0.0f, 0.0f, 0.0f};
    float emissive_intensity = 0.0f;
};

// ============================================================================
// Implementation
// ============================================================================
struct MultiMeshInstance3D::Impl {
    int64_t base_mesh_rid = -1;
    int instance_count = 0;
    std::vector<PerInstanceData> instances;
    std::vector<bool> instance_dirty;      // per‑instance dirty flags
    bool global_dirty = false;

    // Global lighting defaults
    bool global_cast_shadow = true;
    bool global_receive_shadow = true;
    int global_gi_mode = 1;               // static by default
    float global_gi_contribution = 1.0f;
    float global_emissive_color[3] = {0,0,0};
    float global_emissive_intensity = 0.0f;

    // GPU handles
    int64_t multi_mesh_rid = -1;          // handle in RenderingServer (multi‑mesh object)
    int64_t instance_rid = -1;            // inherited visual instance
    bool multi_mesh_dirty = true;
    bool data_upload_needed = false;

    void upload_to_gpu();
};

MultiMeshInstance3D::MultiMeshInstance3D() : pimpl(std::make_unique<Impl>()) {}
MultiMeshInstance3D::~MultiMeshInstance3D() = default;

void MultiMeshInstance3D::set_mesh(int64_t mesh_rid) {
    pimpl->base_mesh_rid = mesh_rid;
    pimpl->global_dirty = true;
    pimpl->multi_mesh_dirty = true;
}
int64_t MultiMeshInstance3D::get_mesh() const { return pimpl->base_mesh_rid; }

void MultiMeshInstance3D::set_instance_count(int count) {
    if (count == pimpl->instance_count) return;
    pimpl->instances.resize(count);
    pimpl->instance_dirty.resize(count, true);
    pimpl->instance_count = count;
    pimpl->global_dirty = true;
    pimpl->multi_mesh_dirty = true;
}
int MultiMeshInstance3D::get_instance_count() const { return pimpl->instance_count; }

void MultiMeshInstance3D::set_instance_transform(int idx, const Transform3D& transform) {
    if (idx < 0 || idx >= pimpl->instance_count) return;
    pimpl->instances[idx].transform = transform;
    pimpl->instance_dirty[idx] = true;
    pimpl->global_dirty = true;
}
Transform3D MultiMeshInstance3D::get_instance_transform(int idx) const {
    if (idx >= 0 && idx < pimpl->instance_count) return pimpl->instances[idx].transform;
    return Transform3D();
}
void MultiMeshInstance3D::set_instance_color(int idx, const float* rgba) {
    if (idx < 0 || idx >= pimpl->instance_count) return;
    memcpy(pimpl->instances[idx].color, rgba, 4*sizeof(float));
    pimpl->instance_dirty[idx] = true;
    pimpl->global_dirty = true;
}
void MultiMeshInstance3D::get_instance_color(int idx, float* out_rgba) const {
    if (idx >= 0 && idx < pimpl->instance_count) {
        memcpy(out_rgba, pimpl->instances[idx].color, 4*sizeof(float));
    } else {
        out_rgba[0]=out_rgba[1]=out_rgba[2]=1.0f; out_rgba[3]=1.0f;
    }
}
void MultiMeshInstance3D::set_instance_custom_data(int idx, const float* data, int size) {
    if (idx < 0 || idx >= pimpl->instance_count) return;
    pimpl->instances[idx].custom_data.assign(data, data + size);
    pimpl->instance_dirty[idx] = true;
    pimpl->global_dirty = true;
}
void MultiMeshInstance3D::get_instance_custom_data(int idx, float* out_data, int max_size) const {
    if (idx >= 0 && idx < pimpl->instance_count) {
        int copy_sz = std::min(max_size, (int)pimpl->instances[idx].custom_data.size());
        memcpy(out_data, pimpl->instances[idx].custom_data.data(), copy_sz*sizeof(float));
    }
}

void MultiMeshInstance3D::set_all_transforms(const std::vector<Transform3D>& transforms) {
    int new_count = (int)transforms.size();
    set_instance_count(new_count);
    for (int i = 0; i < new_count; ++i) {
        pimpl->instances[i].transform = transforms[i];
        pimpl->instance_dirty[i] = true;
    }
    pimpl->global_dirty = true;
}
void MultiMeshInstance3D::set_all_colors(const std::vector<float>& colors_rgba) {
    int count = (int)colors_rgba.size() / 4;
    if (count > pimpl->instance_count) set_instance_count(count);
    for (int i = 0; i < count; ++i) {
        memcpy(pimpl->instances[i].color, &colors_rgba[i*4], 4*sizeof(float));
        pimpl->instance_dirty[i] = true;
    }
    pimpl->global_dirty = true;
}
void MultiMeshInstance3D::flush_changes() {
    if (pimpl->global_dirty) {
        pimpl->upload_to_gpu();
        pimpl->global_dirty = false;
        for (int i = 0; i < pimpl->instance_count; ++i) pimpl->instance_dirty[i] = false;
    }
}

void MultiMeshInstance3D::set_instance_cast_shadow(int idx, bool cast) {
    if (idx < 0 || idx >= pimpl->instance_count) return;
    pimpl->instances[idx].cast_shadow = cast;
    pimpl->instance_dirty[idx] = true;
}
bool MultiMeshInstance3D::get_instance_cast_shadow(int idx) const {
    if (idx >=0 && idx < pimpl->instance_count) return pimpl->instances[idx].cast_shadow;
    return pimpl->global_cast_shadow;
}
void MultiMeshInstance3D::set_instance_gi_mode(int idx, int mode) {
    if (idx < 0 || idx >= pimpl->instance_count) return;
    pimpl->instances[idx].gi_mode = mode;
    pimpl->instance_dirty[idx] = true;
}
int MultiMeshInstance3D::get_instance_gi_mode(int idx) const {
    if (idx >=0 && idx < pimpl->instance_count && pimpl->instances[idx].gi_mode != -1)
        return pimpl->instances[idx].gi_mode;
    return pimpl->global_gi_mode;
}
void MultiMeshInstance3D::set_instance_gi_contribution(int idx, float amount) {
    if (idx < 0 || idx >= pimpl->instance_count) return;
    pimpl->instances[idx].gi_contribution = amount;
    pimpl->instance_dirty[idx] = true;
}
float MultiMeshInstance3D::get_instance_gi_contribution(int idx) const {
    if (idx >=0 && idx < pimpl->instance_count && pimpl->instances[idx].gi_contribution >= 0)
        return pimpl->instances[idx].gi_contribution;
    return pimpl->global_gi_contribution;
}
void MultiMeshInstance3D::set_instance_emissive(int idx, const float* color, float intensity) {
    if (idx < 0 || idx >= pimpl->instance_count) return;
    memcpy(pimpl->instances[idx].emissive_color, color, 3*sizeof(float));
    pimpl->instances[idx].emissive_intensity = intensity;
    pimpl->instance_dirty[idx] = true;
}
void MultiMeshInstance3D::get_instance_emissive(int idx, float* out_color, float& out_intensity) const {
    if (idx >=0 && idx < pimpl->instance_count) {
        memcpy(out_color, pimpl->instances[idx].emissive_color, 3*sizeof(float));
        out_intensity = pimpl->instances[idx].emissive_intensity;
    } else {
        memcpy(out_color, pimpl->global_emissive_color, 3*sizeof(float));
        out_intensity = pimpl->global_emissive_intensity;
    }
}

void MultiMeshInstance3D::set_cast_shadow(bool cast) {
    pimpl->global_cast_shadow = cast;
    GeometryInstance3D::set_cast_shadow(cast);
}
void MultiMeshInstance3D::set_receive_shadow(bool receive) {
    pimpl->global_receive_shadow = receive;
}
void MultiMeshInstance3D::set_gi_mode(int mode) {
    pimpl->global_gi_mode = mode;
    GeometryInstance3D::set_gi_mode(mode);
}
void MultiMeshInstance3D::set_gi_contribution(float amount) {
    pimpl->global_gi_contribution = amount;
}
void MultiMeshInstance3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->global_emissive_color, color, 3*sizeof(float));
    pimpl->global_emissive_intensity = intensity;
}

void MultiMeshInstance3D::Impl::upload_to_gpu() {
    if (instance_count == 0 || base_mesh_rid == -1) return;
    if (multi_mesh_rid == -1) {
        // multi_mesh_rid = RenderingServer::multi_mesh_create();
        // RenderingServer::multi_mesh_set_mesh(multi_mesh_rid, base_mesh_rid);
    }
    // Prepare instance data buffers: transforms (4x4 matrix per instance), colors, custom data, and per‑instance flags.
    // For performance, we send only dirty instances or full update.
    // In real engine, we would use buffer updates.
    std::vector<float> transform_data; // 16 floats per instance (row‑major)
    std::vector<float> color_data;     // 4 floats per instance
    std::vector<int32_t> flag_data;    // bitfield: cast_shadow, gi_mode, etc.
    transform_data.reserve(instance_count * 16);
    color_data.reserve(instance_count * 4);
    flag_data.reserve(instance_count);

    for (int i = 0; i < instance_count; ++i) {
        const auto& inst = instances[i];
        const Transform3D& t = inst.transform;
        // Extract matrix 4x4 (row‑major)
        double mat[16];
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                mat[row*4 + col] = t.basis[row*3 + col];
            }
            mat[row*4 + 3] = t.origin[row];
        }
        mat[12] = mat[13] = mat[14] = 0; mat[15] = 1;
        for (int j = 0; j < 16; ++j) transform_data.push_back((float)mat[j]);

        color_data.push_back(inst.color[0]);
        color_data.push_back(inst.color[1]);
        color_data.push_back(inst.color[2]);
        color_data.push_back(inst.color[3]);

        uint32_t flags = 0;
        if (inst.cast_shadow) flags |= 1;
        int gi = (inst.gi_mode != -1) ? inst.gi_mode : global_gi_mode;
        flags |= (gi & 3) << 1;
        flag_data.push_back((int32_t)flags);
    }
    // RenderingServer::multi_mesh_set_instance_transforms(multi_mesh_rid, transform_data.data(), instance_count);
    // RenderingServer::multi_mesh_set_instance_colors(multi_mesh_rid, color_data.data(), instance_count);
    // RenderingServer::multi_mesh_set_instance_flags(multi_mesh_rid, flag_data.data(), instance_count);
    // Also custom data buffer if provided.
    multi_mesh_dirty = false;
}

void MultiMeshInstance3D::synchronize_render_server(double delta) {
    GeometryInstance3D::synchronize_render_server(delta);
    if (pimpl->multi_mesh_dirty || pimpl->global_dirty) {
        flush_changes();
    }
    // Ensure that the visual instance uses the multi‑mesh resource.
    if (pimpl->multi_mesh_rid != -1 && get_render_instance_id() != -1) {
        // RenderingServer::instance_set_multimesh(get_render_instance_id(), pimpl->multi_mesh_rid);
    }
    // Global lighting overrides also affect the multi‑mesh instance.
    if (pimpl->global_emissive_intensity > 0.0f && pimpl->global_gi_mode > 0) {
        // Register multi‑mesh as emissive (average across instances or use per‑instance injection)
    }
}

void MultiMeshInstance3D::process(double delta) {
    GeometryInstance3D::process(delta);
    // Could handle dynamic updates here (e.g., animated transforms).
}

} // namespace lighting