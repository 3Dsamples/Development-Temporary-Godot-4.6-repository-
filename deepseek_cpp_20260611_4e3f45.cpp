// immediate_mesh_3d.cpp
#include "immediate_mesh_3d.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>

namespace lighting {

// ============================================================================
// Vertex attribute structure (packed for GPU upload)
// ============================================================================
struct ImmediateVertex {
    float pos[3];
    float normal[3];
    float color[4];
    float uv[2];
    float tangent[4]; // tangent + handedness
};

// ============================================================================
// Implementation
// ============================================================================
struct ImmediateMesh3D::Impl {
    int primitive_type = 2; // default triangles
    bool drawing = false;
    bool auto_redraw = false;
    std::function<void(ImmediateMesh3D*)> draw_callback;

    // Vertex and index buffers (CPU side, then uploaded to GPU)
    std::vector<ImmediateVertex> vertices;
    std::vector<uint32_t> indices; // if indexed

    bool mesh_dirty = true;
    int64_t mesh_rid = -1;        // RenderingServer mesh handle
    int64_t instance_rid = -1;    // visual instance handle

    // Current vertex being built
    ImmediateVertex current_vertex;
    bool current_vertex_valid = false;

    // Lighting overrides (per mesh)
    bool cast_shadow = true;
    bool receive_shadow = true;
    int gi_mode = 2;              // dynamic by default
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    // Reset current vertex to defaults
    void reset_current_vertex() {
        current_vertex.pos[0] = current_vertex.pos[1] = current_vertex.pos[2] = 0.0f;
        current_vertex.normal[0] = 0.0f; current_vertex.normal[1] = 1.0f; current_vertex.normal[2] = 0.0f;
        current_vertex.color[0] = current_vertex.color[1] = current_vertex.color[2] = 1.0f; current_vertex.color[3] = 1.0f;
        current_vertex.uv[0] = current_vertex.uv[1] = 0.0f;
        current_vertex.tangent[0] = 1.0f; current_vertex.tangent[1] = 0.0f; current_vertex.tangent[2] = 0.0f; current_vertex.tangent[3] = 1.0f;
        current_vertex_valid = false;
    }

    // Commit current vertex to buffer
    void commit_current_vertex() {
        if (!current_vertex_valid) return;
        vertices.push_back(current_vertex);
        current_vertex_valid = false;
    }

    // Build GPU mesh from vertex/index data
    void upload_mesh();
};

ImmediateMesh3D::ImmediateMesh3D() : pimpl(std::make_unique<Impl>()) {
    pimpl->reset_current_vertex();
}
ImmediateMesh3D::~ImmediateMesh3D() = default;

void ImmediateMesh3D::begin_mesh(int primitive_type) {
    if (pimpl->drawing) return;
    pimpl->vertices.clear();
    pimpl->indices.clear();
    pimpl->primitive_type = primitive_type;
    pimpl->drawing = true;
    pimpl->reset_current_vertex();
}

void ImmediateMesh3D::end_mesh() {
    if (!pimpl->drawing) return;
    pimpl->commit_current_vertex(); // commit any pending vertex
    pimpl->drawing = false;
    pimpl->mesh_dirty = true;
    // Upload to GPU
    pimpl->upload_mesh();
}

void ImmediateMesh3D::set_vertex(double x, double y, double z) {
    if (!pimpl->drawing) return;
    pimpl->current_vertex.pos[0] = (float)x;
    pimpl->current_vertex.pos[1] = (float)y;
    pimpl->current_vertex.pos[2] = (float)z;
    pimpl->current_vertex_valid = true;
}
void ImmediateMesh3D::set_normal(float nx, float ny, float nz) {
    if (!pimpl->drawing) return;
    pimpl->current_vertex.normal[0] = nx;
    pimpl->current_vertex.normal[1] = ny;
    pimpl->current_vertex.normal[2] = nz;
}
void ImmediateMesh3D::set_color(float r, float g, float b, float a) {
    if (!pimpl->drawing) return;
    pimpl->current_vertex.color[0] = r;
    pimpl->current_vertex.color[1] = g;
    pimpl->current_vertex.color[2] = b;
    pimpl->current_vertex.color[3] = a;
}
void ImmediateMesh3D::set_uv(float u, float v) {
    if (!pimpl->drawing) return;
    pimpl->current_vertex.uv[0] = u;
    pimpl->current_vertex.uv[1] = v;
}
void ImmediateMesh3D::set_tangent(float tx, float ty, float tz, float tw) {
    if (!pimpl->drawing) return;
    pimpl->current_vertex.tangent[0] = tx;
    pimpl->current_vertex.tangent[1] = ty;
    pimpl->current_vertex.tangent[2] = tz;
    pimpl->current_vertex.tangent[3] = tw;
}
void ImmediateMesh3D::add_vertex() {
    if (!pimpl->drawing) return;
    pimpl->commit_current_vertex();
    pimpl->reset_current_vertex();
}

void ImmediateMesh3D::add_vertex_full(double x, double y, double z,
                                      float nx, float ny, float nz,
                                      float r, float g, float b, float a,
                                      float u, float v) {
    if (!pimpl->drawing) return;
    ImmediateVertex vtx;
    vtx.pos[0] = (float)x; vtx.pos[1] = (float)y; vtx.pos[2] = (float)z;
    vtx.normal[0] = nx; vtx.normal[1] = ny; vtx.normal[2] = nz;
    vtx.color[0] = r; vtx.color[1] = g; vtx.color[2] = b; vtx.color[3] = a;
    vtx.uv[0] = u; vtx.uv[1] = v;
    vtx.tangent[0] = 1.0f; vtx.tangent[1] = 0.0f; vtx.tangent[2] = 0.0f; vtx.tangent[3] = 1.0f;
    pimpl->vertices.push_back(vtx);
}

void ImmediateMesh3D::set_index(int index) {
    // For indexed drawing: set next index to be added
    if (!pimpl->drawing) return;
    pimpl->indices.push_back((uint32_t)index);
}
void ImmediateMesh3D::add_index(int index) {
    set_index(index);
}

void ImmediateMesh3D::clear() {
    pimpl->vertices.clear();
    pimpl->indices.clear();
    pimpl->mesh_dirty = true;
    pimpl->upload_mesh();
}

void ImmediateMesh3D::set_auto_redraw(bool enable) { pimpl->auto_redraw = enable; }
bool ImmediateMesh3D::is_auto_redraw() const { return pimpl->auto_redraw; }
void ImmediateMesh3D::set_draw_callback(std::function<void(ImmediateMesh3D*)> callback) {
    pimpl->draw_callback = callback;
}

void ImmediateMesh3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; GeometryInstance3D::set_cast_shadow(cast); }
void ImmediateMesh3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
void ImmediateMesh3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; GeometryInstance3D::set_gi_mode(mode); }
void ImmediateMesh3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
void ImmediateMesh3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}

void ImmediateMesh3D::Impl::upload_mesh() {
    // In a real engine, we would create or update a mesh resource in RenderingServer.
    // The mesh would be created with vertices (positions, normals, colors, UVs, tangents)
    // and indices (if any). Then we would set it as the mesh for the visual instance.
    // For performance, we reuse the same mesh RID and update vertex buffer.
    if (mesh_rid == -1) {
        // mesh_rid = RenderingServer::mesh_create();
        // instance_rid = RenderingServer::instance_create(mesh_rid);
    }
    if (!vertices.empty()) {
        // Upload vertex buffer: RenderingServer::mesh_add_surface(mesh_rid, vertex_data, index_data, primitive_type)
        // For each surface, we also need material (default or from node).
    }
    mesh_dirty = false;
}

void ImmediateMesh3D::process(double delta) {
    Node3D::process(delta);
    if (pimpl->auto_redraw && pimpl->draw_callback) {
        begin_mesh(pimpl->primitive_type);
        pimpl->draw_callback(this);
        end_mesh();
    }
}

void ImmediateMesh3D::synchronize_render_server(double delta) {
    GeometryInstance3D::synchronize_render_server(delta);
    if (pimpl->mesh_dirty) {
        pimpl->upload_mesh();
    }
    // If emissive intensity > 0 and gi_mode > 0, register as emissive
    if (pimpl->emissive_intensity > 0.0f && pimpl->gi_mode > 0) {
        // Inject emissive contribution into GI system
    }
    // Compute bounding box from vertices (if any)
    if (!pimpl->vertices.empty()) {
        double min_x = pimpl->vertices[0].pos[0], max_x = pimpl->vertices[0].pos[0];
        double min_y = pimpl->vertices[0].pos[1], max_y = pimpl->vertices[0].pos[1];
        double min_z = pimpl->vertices[0].pos[2], max_z = pimpl->vertices[0].pos[2];
        for (size_t i = 1; i < pimpl->vertices.size(); ++i) {
            min_x = std::min(min_x, (double)pimpl->vertices[i].pos[0]);
            max_x = std::max(max_x, (double)pimpl->vertices[i].pos[0]);
            min_y = std::min(min_y, (double)pimpl->vertices[i].pos[1]);
            max_y = std::max(max_y, (double)pimpl->vertices[i].pos[1]);
            min_z = std::min(min_z, (double)pimpl->vertices[i].pos[2]);
            max_z = std::max(max_z, (double)pimpl->vertices[i].pos[2]);
        }
        set_aabb(&min_x, &max_x);
        double dx = max_x-min_x, dy = max_y-min_y, dz = max_z-min_z;
        set_bounding_sphere_radius(std::sqrt(dx*dx+dy*dy+dz*dz)*0.5);
    }
}

} // namespace lighting