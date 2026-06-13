// immediate_mesh_3d.cpp
#include "immediate_mesh_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <limits>

namespace lighting {

// ============================================================================
// Vertex structure (interleaved)
// ============================================================================
struct ImmediateVertex {
    double x, y, z;
    double nx, ny, nz;
    float r, g, b, a;
    float u, v;
};

// ============================================================================
// Implementation
// ============================================================================
struct ImmediateMesh3D::Impl {
    ImmediatePrimitiveType primitive_type = ImmediatePrimitiveType::TRIANGLES;
    std::vector<ImmediateVertex> vertices;
    std::vector<int> indices;           // generated from vertices based on primitive type
    bool mesh_dirty = true;
    bool has_color = false;
    bool has_normal = false;
    bool has_uv = false;

    // Current vertex attributes (during surface_begin / ...)
    ImmediateVertex current_vertex;
    bool vertex_pending = false;

    // Lighting parameters
    bool cast_shadow = true;
    bool receive_shadow = true;
    int gi_mode = 2;                     // dynamic by default
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0.0f, 0.0f, 0.0f};
    float emissive_intensity = 0.0f;

    // Render server handles
    int64_t mesh_rid = -1;               // handle to RenderingServer mesh
    int64_t instance_rid = -1;           // handle to visual instance (inherited)
    bool instance_dirty = true;

    // Generate indices from vertices for the given primitive type
    void build_indices();
    void upload_to_render_server();
};

ImmediateMesh3D::ImmediateMesh3D() : pimpl(std::make_unique<Impl>()) {}
ImmediateMesh3D::~ImmediateMesh3D() = default;

void ImmediateMesh3D::surface_begin(ImmediatePrimitiveType primitive) {
    pimpl->primitive_type = primitive;
    pimpl->vertices.clear();
    pimpl->indices.clear();
    pimpl->has_color = false;
    pimpl->has_normal = false;
    pimpl->has_uv = false;
    pimpl->vertex_pending = false;
    // Reset current vertex attributes to defaults
    pimpl->current_vertex = ImmediateVertex{0,0,0, 0,0,0, 1,1,1,1, 0,0};
}

void ImmediateMesh3D::surface_set_vertex(double x, double y, double z) {
    pimpl->current_vertex.x = x;
    pimpl->current_vertex.y = y;
    pimpl->current_vertex.z = z;
}
void ImmediateMesh3D::surface_set_normal(double x, double y, double z) {
    pimpl->current_vertex.nx = x;
    pimpl->current_vertex.ny = y;
    pimpl->current_vertex.nz = z;
    pimpl->has_normal = true;
}
void ImmediateMesh3D::surface_set_color(float r, float g, float b, float a) {
    pimpl->current_vertex.r = r;
    pimpl->current_vertex.g = g;
    pimpl->current_vertex.b = b;
    pimpl->current_vertex.a = a;
    pimpl->has_color = true;
}
void ImmediateMesh3D::surface_set_uv(float u, float v) {
    pimpl->current_vertex.u = u;
    pimpl->current_vertex.v = v;
    pimpl->has_uv = true;
}
void ImmediateMesh3D::surface_add_vertex() {
    pimpl->vertices.push_back(pimpl->current_vertex);
    pimpl->vertex_pending = false;
}
void ImmediateMesh3D::surface_end() {
    pimpl->build_indices();
    pimpl->mesh_dirty = true;
    update_mesh();
}

void ImmediateMesh3D::Impl::build_indices() {
    indices.clear();
    int n = (int)vertices.size();
    switch (primitive_type) {
        case ImmediatePrimitiveType::POINTS:
            for (int i = 0; i < n; ++i)
                indices.push_back(i);
            break;
        case ImmediatePrimitiveType::LINES:
            for (int i = 0; i + 1 < n; i += 2) {
                indices.push_back(i);
                indices.push_back(i+1);
            }
            break;
        case ImmediatePrimitiveType::LINE_STRIP:
            for (int i = 0; i + 1 < n; ++i) {
                indices.push_back(i);
                indices.push_back(i+1);
            }
            break;
        case ImmediatePrimitiveType::TRIANGLES:
            for (int i = 0; i + 2 < n; i += 3) {
                indices.push_back(i);
                indices.push_back(i+1);
                indices.push_back(i+2);
            }
            break;
        case ImmediatePrimitiveType::TRIANGLE_STRIP:
            for (int i = 0; i + 2 < n; ++i) {
                if (i % 2 == 0) {
                    indices.push_back(i);
                    indices.push_back(i+1);
                    indices.push_back(i+2);
                } else {
                    indices.push_back(i+1);
                    indices.push_back(i);
                    indices.push_back(i+2);
                }
            }
            break;
    }
}

void ImmediateMesh3D::Impl::upload_to_render_server() {
    if (vertices.empty() || indices.empty()) return;
    if (mesh_rid == -1) {
        // In real engine: mesh_rid = RenderingServer::mesh_create();
    }
    // Create vertex array buffer with interleaved data:
    // positions (double3), normals (double3), colors (float4), uvs (float2)
    // For performance, we would convert to float for positions and normals (GPU prefers float).
    // For simplicity, we keep double but convert to float.
    std::vector<float> vertex_data;
    vertex_data.reserve(vertices.size() * (3+3+4+2));
    for (const auto& v : vertices) {
        vertex_data.push_back((float)v.x);
        vertex_data.push_back((float)v.y);
        vertex_data.push_back((float)v.z);
        vertex_data.push_back((float)v.nx);
        vertex_data.push_back((float)v.ny);
        vertex_data.push_back((float)v.nz);
        vertex_data.push_back(v.r);
        vertex_data.push_back(v.g);
        vertex_data.push_back(v.b);
        vertex_data.push_back(v.a);
        vertex_data.push_back(v.u);
        vertex_data.push_back(v.v);
    }
    // Send to RenderingServer::mesh_add_surface()
    // The surface format includes positions, normals (optional), colors (optional), UVs (optional)
    // For now, we only have one surface.

    // Also set material (if any) – immediate meshes often have a default material with vertex color support.
}

void ImmediateMesh3D::clear() {
    pimpl->vertices.clear();
    pimpl->indices.clear();
    pimpl->mesh_dirty = true;
    update_mesh();
}

void ImmediateMesh3D::set_cast_shadow(bool cast) {
    pimpl->cast_shadow = cast;
    GeometryInstance3D::set_cast_shadow(cast);
}
void ImmediateMesh3D::set_receive_shadow(bool receive) {
    pimpl->receive_shadow = receive;
}
void ImmediateMesh3D::set_gi_mode(int mode) {
    pimpl->gi_mode = mode;
    GeometryInstance3D::set_gi_mode(mode);
}
void ImmediateMesh3D::set_gi_contribution(float amount) {
    pimpl->gi_contribution = amount;
}
void ImmediateMesh3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void ImmediateMesh3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}

void ImmediateMesh3D::update_mesh() {
    if (!pimpl->mesh_dirty) return;
    pimpl->upload_to_render_server();
    // Update bounding box from vertices
    if (!pimpl->vertices.empty()) {
        double min_x = pimpl->vertices[0].x, max_x = pimpl->vertices[0].x;
        double min_y = pimpl->vertices[0].y, max_y = pimpl->vertices[0].y;
        double min_z = pimpl->vertices[0].z, max_z = pimpl->vertices[0].z;
        for (const auto& v : pimpl->vertices) {
            min_x = std::min(min_x, v.x); max_x = std::max(max_x, v.x);
            min_y = std::min(min_y, v.y); max_y = std::max(max_y, v.y);
            min_z = std::min(min_z, v.z); max_z = std::max(max_z, v.z);
        }
        set_aabb(&min_x, &max_x);
        double dx = max_x-min_x, dy = max_y-min_y, dz = max_z-min_z;
        set_bounding_sphere_radius(std::sqrt(dx*dx+dy*dy+dz*dz) * 0.5);
    } else {
        double dummy[3] = {0,0,0};
        set_aabb(dummy, dummy);
        set_bounding_sphere_radius(0.0);
    }
    pimpl->mesh_dirty = false;
    pimpl->instance_dirty = true;
}

void ImmediateMesh3D::process(double delta) {
    GeometryInstance3D::process(delta);
    if (pimpl->mesh_dirty) update_mesh();
}

void ImmediateMesh3D::synchronize_render_server(double delta) {
    GeometryInstance3D::synchronize_render_server(delta);
    if (pimpl->instance_dirty && pimpl->mesh_rid != -1) {
        // Assign the mesh to the visual instance
        // RenderingServer::instance_set_mesh(get_render_instance_id(), pimpl->mesh_rid)
        // Set cast_shadow, gi_mode, receive_shadow flags
        // If emissive, also set instance emissive parameters
        pimpl->instance_dirty = false;
    }
    // If emissive intensity > 0, contribute to GI
    if (pimpl->emissive_intensity > 0.0f && pimpl->gi_mode > 0) {
        // Register for global illumination (inject into lightprobe, VCT, etc.)
    }
}

} // namespace lighting