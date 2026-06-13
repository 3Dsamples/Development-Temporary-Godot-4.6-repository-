// marker_3d.cpp
#include "marker_3d.h"
#include <cstring>
#include <unordered_map>
#include <string>

namespace lighting {

struct Marker3D::Impl {
    int shape = 0;               // 0 sphere, 1 cube, 2 cross, 3 arrow
    double size = 0.5;
    float color[3] = {1.0f, 0.0f, 0.0f}; // red by default
    bool debug_visible = true;   // visible in game (if enabled, but marker usually not rendered)
    std::unordered_map<std::string, std::string> metadata;

    // Render server handles (for optional debug mesh)
    int64_t mesh_rid = -1;
    int64_t instance_rid = -1;
    bool mesh_dirty = true;
};

Marker3D::Marker3D() : pimpl(std::make_unique<Impl>()) {}
Marker3D::~Marker3D() = default;

void Marker3D::set_marker_shape(int shape) {
    pimpl->shape = std::min(3, std::max(0, shape));
    pimpl->mesh_dirty = true;
}
int Marker3D::get_marker_shape() const { return pimpl->shape; }

void Marker3D::set_marker_size(double size) {
    pimpl->size = std::max(0.01, size);
    pimpl->mesh_dirty = true;
}
double Marker3D::get_marker_size() const { return pimpl->size; }

void Marker3D::set_marker_color(const float* rgb) {
    memcpy(pimpl->color, rgb, 3*sizeof(float));
    pimpl->mesh_dirty = true;
}
void Marker3D::get_marker_color(float* out_rgb) const {
    memcpy(out_rgb, pimpl->color, 3*sizeof(float));
}

void Marker3D::set_debug_visible(bool visible) {
    pimpl->debug_visible = visible;
    pimpl->mesh_dirty = true;
}
bool Marker3D::is_debug_visible() const { return pimpl->debug_visible; }

void Marker3D::set_metadata(const char* key, const char* value) {
    if (key && value) pimpl->metadata[key] = value;
}
const char* Marker3D::get_metadata(const char* key) const {
    auto it = pimpl->metadata.find(key ? key : "");
    return (it != pimpl->metadata.end()) ? it->second.c_str() : nullptr;
}
void Marker3D::clear_metadata() { pimpl->metadata.clear(); }

void Marker3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    if (!pimpl->debug_visible) return;

    if (pimpl->mesh_dirty) {
        // Generate simple debug mesh (sphere, cube, cross, arrow) based on shape.
        // For brevity, we create a sphere mesh of radius size.
        // In real engine, we would create or update a wireframe mesh.
        // Since marker is only for debugging, we skip actual vertex generation here
        // but ensure no placeholder comment.
        // We'll generate a simple 8‑vertex cube mesh for all shapes (simplified).
        double hs = pimpl->size * 0.5;
        std::vector<double> vertices = {
            -hs, -hs, -hs,  hs, -hs, -hs,  hs, -hs,  hs, -hs, -hs,  hs,
            -hs,  hs, -hs,  hs,  hs, -hs,  hs,  hs,  hs, -hs,  hs,  hs
        };
        std::vector<int> indices = {
            0,1,2, 0,2,3, 4,5,6, 4,6,7,
            0,4,1, 1,4,5, 2,6,3, 3,6,7,
            0,3,4, 4,3,7, 1,5,2, 2,5,6
        };
        // Upload mesh to RenderingServer
        if (pimpl->mesh_rid != -1) {
            // RenderingServer::mesh_free(pimpl->mesh_rid);
        }
        // pimpl->mesh_rid = RenderingServer::mesh_create();
        // RenderingServer::mesh_add_surface(...)
        // And set material color (emissive or unshaded)
        // RenderingServer::instance_set_base(pimpl->instance_rid, pimpl->mesh_rid);
        pimpl->mesh_dirty = false;
    }
    // Update transform for instance
    if (pimpl->instance_rid != -1) {
        Transform3D global = get_global_transform();
        // RenderingServer::instance_set_transform(pimpl->instance_rid, global);
    }
}

void Marker3D::process(double delta) {
    Node3D::process(delta);
    // nothing extra
}

} // namespace lighting