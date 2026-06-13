// position_3d.cpp
#include "position_3d.h"
#include <cstring>
#include <unordered_map>
#include <string>

namespace lighting {

struct Position3D::Impl {
    bool debug_visible = false;
    float debug_color[3] = {0.8f, 0.8f, 0.0f}; // yellowish
    double debug_size = 0.2;

    std::unordered_map<std::string, std::string> metadata;

    // Debug mesh (simple cross or sphere)
    int64_t debug_mesh_rid = -1;
    int64_t debug_instance_rid = -1;
    bool dirty = true;

    ~Impl() {
        // free render resources if created
    }

    void update_debug_mesh();
};

Position3D::Position3D() : pimpl(std::make_unique<Impl>()) {}
Position3D::~Position3D() = default;

void Position3D::set_debug_icon_visible(bool visible) {
    pimpl->debug_visible = visible;
    pimpl->dirty = true;
}
bool Position3D::is_debug_icon_visible() const { return pimpl->debug_visible; }

void Position3D::set_debug_icon_color(const float* rgb) {
    memcpy(pimpl->debug_color, rgb, 3*sizeof(float));
    pimpl->dirty = true;
}
void Position3D::get_debug_icon_color(float* out_rgb) const {
    memcpy(out_rgb, pimpl->debug_color, 3*sizeof(float));
}

void Position3D::set_debug_icon_size(double size) {
    pimpl->debug_size = std::max(0.01, size);
    pimpl->dirty = true;
}
double Position3D::get_debug_icon_size() const { return pimpl->debug_size; }

void Position3D::set_metadata(const char* key, const char* value) {
    if (key && value) pimpl->metadata[key] = value;
}
const char* Position3D::get_metadata(const char* key) const {
    auto it = pimpl->metadata.find(key ? key : "");
    return (it != pimpl->metadata.end()) ? it->second.c_str() : nullptr;
}
void Position3D::clear_metadata() { pimpl->metadata.clear(); }

void Position3D::Impl::update_debug_mesh() {
    if (!debug_visible) {
        // hide instance if exists
        if (debug_instance_rid != -1) {
            // RenderingServer::instance_set_visible(debug_instance_rid, false);
        }
        return;
    }

    // Create a simple cross (two lines: X and Z axis) or a small sphere.
    // For simplicity, generate a 3D cross made of 6 lines (3 axis).
    std::vector<double> vertices;
    std::vector<int> indices;
    double s = debug_size;
    // X axis line
    vertices.push_back(-s); vertices.push_back(0); vertices.push_back(0);
    vertices.push_back( s); vertices.push_back(0); vertices.push_back(0);
    // Y axis line
    vertices.push_back(0); vertices.push_back(-s); vertices.push_back(0);
    vertices.push_back(0); vertices.push_back( s); vertices.push_back(0);
    // Z axis line
    vertices.push_back(0); vertices.push_back(0); vertices.push_back(-s);
    vertices.push_back(0); vertices.push_back(0); vertices.push_back( s);
    // Indices: each line is two vertices (0-1, 2-3, 4-5)
    indices.push_back(0); indices.push_back(1);
    indices.push_back(2); indices.push_back(3);
    indices.push_back(4); indices.push_back(5);

    // Generate mesh (line primitive)
    if (debug_mesh_rid == -1) {
        // debug_mesh_rid = RenderingServer::mesh_create();
    }
    // RenderingServer::mesh_add_surface(debug_mesh_rid, RenderingServer::PRIMITIVE_LINES, vertices, indices);
    // Also add vertex colors (optional)
    // Set material for lines: unlit with debug_color (or use emissive)
    // Instance
    if (debug_instance_rid == -1) {
        // debug_instance_rid = RenderingServer::instance_create();
    }
    // RenderingServer::instance_set_base(debug_instance_rid, debug_mesh_rid);
    // RenderingServer::instance_set_transform(debug_instance_rid, get_global_transform());
    // RenderingServer::instance_set_visible(debug_instance_rid, true);
}

void Position3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    if (pimpl->dirty) {
        pimpl->update_debug_mesh();
        pimpl->dirty = false;
    }
    // If transform changed and debug visible, update instance transform.
    if (pimpl->debug_visible && pimpl->debug_instance_rid != -1 && is_transform_dirty()) {
        Transform3D global = get_global_transform();
        // RenderingServer::instance_set_transform(pimpl->debug_instance_rid, global);
    }
}

void Position3D::process(double delta) {
    Node3D::process(delta);
    // nothing extra
}

} // namespace lighting