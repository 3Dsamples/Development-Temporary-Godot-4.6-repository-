// occluder_3d.cpp
#include "occluder_3d.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

namespace lighting {

struct Occluder3D::Impl {
    OccluderShape shape = OccluderShape::NONE;
    double size[3] = {1.0, 1.0, 1.0};          // half extents (box) or radius (sphere)
    int64_t mesh_rid = -1;
    std::vector<double> custom_vertices;       // if mesh provided via vertices
    std::vector<int> custom_indices;

    bool enabled = true;
    bool force_occluder = false;
    uint32_t occlusion_layer = 0xFFFFFFFF;

    double bounds_min[3] = {0,0,0};
    double bounds_max[3] = {0,0,0};
    bool has_bounds_override = false;

    bool debug_visible = false;
    float debug_color[3] = {0.5f, 0.5f, 0.5f};
    bool emissive_debug = false;
    float emissive_intensity = 0.2f;
    float gi_contribution = 0.5f;
    bool cast_shadow = true;

    // Rendering server handles for debug mesh
    int64_t debug_mesh_rid = -1;
    int64_t debug_instance_rid = -1;
    bool dirty = true;

    // Occluder data (would be consumed by rendering server)
    int64_t occluder_rid = -1;

    void update_debug_mesh();
    void update_occluder_data();
};

Occluder3D::Occluder3D() : pimpl(std::make_unique<Impl>()) {}
Occluder3D::~Occluder3D() = default;

void Occluder3D::set_shape(OccluderShape shape) {
    pimpl->shape = shape;
    pimpl->dirty = true;
}
OccluderShape Occluder3D::get_shape() const { return pimpl->shape; }

void Occluder3D::set_size(const double* size) {
    memcpy(pimpl->size, size, 3*sizeof(double));
    if (pimpl->shape == OccluderShape::SPHERE) {
        // ensure sphere radius is uniform
        double r = size[0];
        pimpl->size[0] = pimpl->size[1] = pimpl->size[2] = r;
    }
    pimpl->dirty = true;
}
void Occluder3D::get_size(double* out_size) const { memcpy(out_size, pimpl->size, 3*sizeof(double)); }

void Occluder3D::set_mesh(int64_t mesh_rid) {
    pimpl->mesh_rid = mesh_rid;
    pimpl->custom_vertices.clear();
    pimpl->custom_indices.clear();
    pimpl->shape = OccluderShape::MESH;
    pimpl->dirty = true;
}
int64_t Occluder3D::get_mesh_rid() const { return pimpl->mesh_rid; }

void Occluder3D::set_vertices(const std::vector<double>& vertices, const std::vector<int>& indices) {
    pimpl->custom_vertices = vertices;
    pimpl->custom_indices = indices;
    pimpl->mesh_rid = -1;
    pimpl->shape = OccluderShape::MESH;
    pimpl->dirty = true;
}
void Occluder3D::clear_mesh() {
    pimpl->custom_vertices.clear();
    pimpl->custom_indices.clear();
    pimpl->mesh_rid = -1;
    pimpl->shape = OccluderShape::NONE;
    pimpl->dirty = true;
}

void Occluder3D::set_enabled(bool enabled) { pimpl->enabled = enabled; pimpl->dirty = true; }
bool Occluder3D::is_enabled() const { return pimpl->enabled; }
void Occluder3D::set_force_occluder(bool force) { pimpl->force_occluder = force; }
bool Occluder3D::is_force_occluder() const { return pimpl->force_occluder; }
void Occluder3D::set_occlusion_layer(uint32_t layer) { pimpl->occlusion_layer = layer; }
uint32_t Occluder3D::get_occlusion_layer() const { return pimpl->occlusion_layer; }

void Occluder3D::set_bounds_override(const double* min, const double* max) {
    memcpy(pimpl->bounds_min, min, 3*sizeof(double));
    memcpy(pimpl->bounds_max, max, 3*sizeof(double));
    pimpl->has_bounds_override = true;
    pimpl->dirty = true;
}
void Occluder3D::clear_bounds_override() {
    pimpl->has_bounds_override = false;
    pimpl->dirty = true;
}

void Occluder3D::set_debug_visible(bool visible) { pimpl->debug_visible = visible; pimpl->dirty = true; }
bool Occluder3D::is_debug_visible() const { return pimpl->debug_visible; }
void Occluder3D::set_debug_color(const float* rgb) { memcpy(pimpl->debug_color, rgb, 3*sizeof(float)); pimpl->dirty = true; }
void Occluder3D::get_debug_color(float* out_rgb) const { memcpy(out_rgb, pimpl->debug_color, 3*sizeof(float)); }
void Occluder3D::set_emissive_debug(bool enable, float intensity) {
    pimpl->emissive_debug = enable;
    pimpl->emissive_intensity = intensity;
    pimpl->dirty = true;
}
bool Occluder3D::is_emissive_debug() const { return pimpl->emissive_debug; }
void Occluder3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
float Occluder3D::get_gi_contribution() const { return pimpl->gi_contribution; }
void Occluder3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; }
bool Occluder3D::get_cast_shadow() const { return pimpl->cast_shadow; }

void Occluder3D::Impl::update_debug_mesh() {
    if (!debug_visible) return;
    // Generate a simple debug mesh based on shape and size.
    // For box: generate 12 triangles (6 faces) with correct normals.
    // For sphere: generate a low‑poly sphere (icosahedron or UV sphere).
    // For mesh: use provided custom vertices/indices.
    // Then assign material (unlit or emissive if emissive_debug).
    // For brevity, we create a placeholder cube if shape is BOX.
    if (shape == OccluderShape::BOX) {
        double hx = size[0], hy = size[1], hz = size[2];
        std::vector<double> verts = {
            -hx,-hy,-hz,  hx,-hy,-hz,  hx,-hy, hz, -hx,-hy, hz,
            -hx, hy,-hz,  hx, hy,-hz,  hx, hy, hz, -hx, hy, hz
        };
        std::vector<int> idxs = {
            0,1,2, 0,2,3, 4,5,6, 4,6,7,
            0,4,1, 1,4,5, 2,6,3, 3,6,7,
            0,3,4, 4,3,7, 1,5,2, 2,5,6
        };
        // Upload to RenderingServer
        if (debug_mesh_rid == -1) {
            // debug_mesh_rid = RenderingServer::mesh_create();
        }
        // RenderingServer::mesh_add_surface_from_arrays(...)
    } else if (shape == OccluderShape::SPHERE) {
        // simplified sphere (icosahedron)
    } else if (shape == OccluderShape::MESH && !custom_vertices.empty()) {
        // use custom vertices
    }
    // Set material color and emissive
    // Instance transform
    Transform3D global = get_global_transform();
    if (debug_instance_rid == -1) {
        // debug_instance_rid = RenderingServer::instance_create();
    }
    // RenderingServer::instance_set_base(debug_instance_rid, debug_mesh_rid);
    // RenderingServer::instance_set_transform(debug_instance_rid, global);
    // RenderingServer::instance_set_visible(debug_instance_rid, debug_visible);
}

void Occluder3D::Impl::update_occluder_data() {
    // Create or update the occluder in the rendering server / occlusion system.
    if (occluder_rid == -1) {
        // occluder_rid = RenderingServer::occluder_create();
    }
    // Set shape, size, mesh, and transform.
    Transform3D global = get_global_transform();
    // RenderingServer::occluder_set_transform(occluder_rid, global);
    // RenderingServer::occluder_set_shape(occluder_rid, shape, size);
    // if mesh: RenderingServer::occluder_set_mesh(occluder_rid, mesh_rid or custom vertices);
    // Set enabled, layer, force flag.
    // Also set bounds override if provided.
}

void Occluder3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    if (pimpl->dirty) {
        pimpl->update_occluder_data();
        if (pimpl->debug_visible) pimpl->update_debug_mesh();
        pimpl->dirty = false;
    }
}

void Occluder3D::process(double delta) {
    Node3D::process(delta);
    // If transform changed, mark dirty.
    if (is_transform_dirty()) {
        pimpl->dirty = true;
    }
}

} // namespace lighting