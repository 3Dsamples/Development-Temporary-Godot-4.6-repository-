// multi_mesh_instance_3d.cpp
#include "multi_mesh_instance_3d.h"
#include <cstring>
#include <algorithm>
#include <vector>
#include <limits>

namespace lighting {

// ============================================================================
// Per‑instance data storage (CPU side)
// ============================================================================
struct PerInstanceData {
    Transform3D transform;
    float color[4] = {1.0f,1.0f,1.0f,1.0f};
    float custom_data[4] = {0.0f,0.0f,0.0f,0.0f};
    bool cast_shadow = true;
    int gi_mode = 1;          // static by default
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;
    bool active = true;
};

// ============================================================================
// Implementation
// ============================================================================
struct MultiMeshInstance3D::Impl {
    int64_t multimesh_rid = -1;   // handle to RenderingServer multi‑mesh resource
    std::vector<PerInstanceData> instances;
    bool use_color = false;
    bool use_custom_data = false;
    int color_format = 0;         // RGBA

    // Global defaults
    bool global_cast_shadow = true;
    bool global_receive_shadow = true;
    int global_gi_mode = 1;
    float global_gi_contribution = 1.0f;
    float global_emissive_color[3] = {0,0,0};
    float global_emissive_intensity = 0.0f;

    bool dirty = true;            // mark that GPU buffers need update
    bool transform_dirty = true;
    int64_t mesh_rid = -1;        // underlying mesh (if multimesh not set)

    void update_gpu_buffers();
};

MultiMeshInstance3D::MultiMeshInstance3D() : pimpl(std::make_unique<Impl>()) {}
MultiMeshInstance3D::~MultiMeshInstance3D() = default;

void MultiMeshInstance3D::set_multimesh(int64_t multimesh_rid) {
    pimpl->multimesh_rid = multimesh_rid;
    pimpl->dirty = true;
    // If using external multimesh, we should not manage instances locally
    // For simplicity, we keep both, but in production use one or the other.
}
int64_t MultiMeshInstance3D::get_multimesh_rid() const { return pimpl->multimesh_rid; }

void MultiMeshInstance3D::set_instance_count(int count) {
    pimpl->instances.resize(count);
    pimpl->dirty = true;
}
int MultiMeshInstance3D::get_instance_count() const { return (int)pimpl->instances.size(); }

void MultiMeshInstance3D::set_instance_transform(int idx, const Transform3D& transform) {
    if (idx >= 0 && idx < (int)pimpl->instances.size()) {
        pimpl->instances[idx].transform = transform;
        pimpl->transform_dirty = true;
    }
}
Transform3D MultiMeshInstance3D::get_instance_transform(int idx) const {
    if (idx >= 0 && idx < (int)pimpl->instances.size())
        return pimpl->instances[idx].transform;
    return Transform3D();
}

void MultiMeshInstance3D::set_instance_color(int idx, float r, float g, float b, float a) {
    if (idx >= 0 && idx < (int)pimpl->instances.size()) {
        pimpl->instances[idx].color[0] = r;
        pimpl->instances[idx].color[1] = g;
        pimpl->instances[idx].color[2] = b;
        pimpl->instances[idx].color[3] = a;
        pimpl->dirty = true;
    }
}
void MultiMeshInstance3D::get_instance_color(int idx, float* out_rgba) const {
    if (idx >= 0 && idx < (int)pimpl->instances.size())
        memcpy(out_rgba, pimpl->instances[idx].color, 4*sizeof(float));
    else
        out_rgba[0]=out_rgba[1]=out_rgba[2]=1.0f; out_rgba[3]=1.0f;
}

void MultiMeshInstance3D::set_instance_custom_data(int idx, const float* data, int size) {
    if (idx >= 0 && idx < (int)pimpl->instances.size()) {
        int copy = std::min(size, 4);
        memcpy(pimpl->instances[idx].custom_data, data, copy*sizeof(float));
        pimpl->dirty = true;
    }
}
void MultiMeshInstance3D::get_instance_custom_data(int idx, float* out_data, int size) const {
    if (idx >= 0 && idx < (int)pimpl->instances.size()) {
        int copy = std::min(size, 4);
        memcpy(out_data, pimpl->instances[idx].custom_data, copy*sizeof(float));
    } else {
        for (int i=0;i<size;++i) out_data[i]=0.0f;
    }
}

void MultiMeshInstance3D::set_use_color(bool use) { pimpl->use_color = use; pimpl->dirty = true; }
bool MultiMeshInstance3D::is_using_color() const { return pimpl->use_color; }
void MultiMeshInstance3D::set_use_custom_data(bool use) { pimpl->use_custom_data = use; pimpl->dirty = true; }
bool MultiMeshInstance3D::is_using_custom_data() const { return pimpl->use_custom_data; }
void MultiMeshInstance3D::set_color_format(int format) { pimpl->color_format = format; pimpl->dirty = true; }
int MultiMeshInstance3D::get_color_format() const { return pimpl->color_format; }

void MultiMeshInstance3D::set_instance_cast_shadow(int idx, bool cast) {
    if (idx >= 0 && idx < (int)pimpl->instances.size())
        pimpl->instances[idx].cast_shadow = cast;
}
bool MultiMeshInstance3D::get_instance_cast_shadow(int idx) const {
    if (idx >= 0 && idx < (int)pimpl->instances.size())
        return pimpl->instances[idx].cast_shadow;
    return pimpl->global_cast_shadow;
}

void MultiMeshInstance3D::set_instance_gi_mode(int idx, int mode) {
    if (idx >= 0 && idx < (int)pimpl->instances.size())
        pimpl->instances[idx].gi_mode = mode;
}
int MultiMeshInstance3D::get_instance_gi_mode(int idx) const {
    if (idx >= 0 && idx < (int)pimpl->instances.size())
        return pimpl->instances[idx].gi_mode;
    return pimpl->global_gi_mode;
}

void MultiMeshInstance3D::set_instance_emissive(int idx, const float* color, float intensity) {
    if (idx >= 0 && idx < (int)pimpl->instances.size()) {
        memcpy(pimpl->instances[idx].emissive_color, color, 3*sizeof(float));
        pimpl->instances[idx].emissive_intensity = intensity;
        pimpl->dirty = true;
    }
}
void MultiMeshInstance3D::get_instance_emissive(int idx, float* out_color, float& out_intensity) const {
    if (idx >= 0 && idx < (int)pimpl->instances.size()) {
        memcpy(out_color, pimpl->instances[idx].emissive_color, 3*sizeof(float));
        out_intensity = pimpl->instances[idx].emissive_intensity;
    } else {
        out_color[0]=out_color[1]=out_color[2]=0.0f; out_intensity=0.0f;
    }
}

void MultiMeshInstance3D::set_cast_shadow(bool cast) { pimpl->global_cast_shadow = cast; }
void MultiMeshInstance3D::set_receive_shadow(bool receive) { pimpl->global_receive_shadow = receive; }
void MultiMeshInstance3D::set_gi_mode(int mode) { pimpl->global_gi_mode = mode; }
void MultiMeshInstance3D::set_gi_contribution(float amount) { pimpl->global_gi_contribution = amount; }
void MultiMeshInstance3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->global_emissive_color, color, 3*sizeof(float));
    pimpl->global_emissive_intensity = intensity;
}

void MultiMeshInstance3D::Impl::update_gpu_buffers() {
    if (multimesh_rid == -1) {
        // create a new multi‑mesh resource
        // multimesh_rid = RenderingServer::multimesh_create();
    }
    // Set instance count, format, etc.
    // For each instance, set transform matrix, color, custom data, shadow flag, GI mode.
    // In production, we would use a single buffer upload for all instances.
    // For now, mark as ready.
    dirty = false;
    transform_dirty = false;
}

void MultiMeshInstance3D::update_instances() {
    if (!pimpl->dirty) return;
    pimpl->update_gpu_buffers();
}

void MultiMeshInstance3D::synchronize_render_server(double delta) {
    GeometryInstance3D::synchronize_render_server(delta);
    if (pimpl->dirty || pimpl->transform_dirty) {
        update_instances();
    }
    // Also update the transform of the whole MultiMeshInstance if needed.
    // Compute bounding box from all instance transforms (approximate)
    if (!pimpl->instances.empty()) {
        double min_x = 1e30, max_x = -1e30, min_y = 1e30, max_y = -1e30, min_z = 1e30, max_z = -1e30;
        for (const auto& inst : pimpl->instances) {
            const double* origin = inst.transform.origin;
            // approximate bounding sphere radius from mesh (if known) - for now assume 1 unit radius.
            double rad = 1.0;
            min_x = std::min(min_x, origin[0] - rad);
            max_x = std::max(max_x, origin[0] + rad);
            min_y = std::min(min_y, origin[1] - rad);
            max_y = std::max(max_y, origin[1] + rad);
            min_z = std::min(min_z, origin[2] - rad);
            max_z = std::max(max_z, origin[2] + rad);
        }
        set_aabb(&min_x, &max_x);
        double dx = max_x-min_x, dy = max_y-min_y, dz = max_z-min_z;
        set_bounding_sphere_radius(std::sqrt(dx*dx+dy*dy+dz*dz)*0.5);
    }
    // If any instance is emissive, contribute to GI.
    if (pimpl->global_emissive_intensity > 0.0f && pimpl->global_gi_mode > 0) {
        // Inject emissive.
    } else {
        for (const auto& inst : pimpl->instances) {
            if (inst.emissive_intensity > 0.0f && inst.gi_mode > 0) {
                // inject
                break;
            }
        }
    }
}

void MultiMeshInstance3D::process(double delta) {
    GeometryInstance3D::process(delta);
    // Nothing per‑frame.
}

} // namespace lighting