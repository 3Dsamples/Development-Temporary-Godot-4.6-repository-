// immediate_mesh_3d.cpp
#include "immediate_mesh_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

namespace lighting {

struct ImmediateMesh3D::Impl {
    ImmediatePrimitive primitive = ImmediatePrimitive::TRIANGLES;
    bool drawing = false;
    std::vector<ImmediateVertex> vertices;
    std::vector<int> indices;   // if needed, but we can just use vertices order

    // Current state (for simple vertex setup)
    float current_color[4] = {1.0f,1.0f,1.0f,1.0f};
    float current_normal[3] = {0.0f,0.0f,1.0f};
    float current_uv[2] = {0.0f,0.0f};

    // Material override
    int material_id = -1;   // -1 = use default

    // Lighting flags
    bool cast_shadow = true;
    bool receive_shadow = true;
    int gi_mode = 2;        // dynamic by default for immediate mode
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0.0f,0.0f,0.0f};
    float emissive_intensity = 0.0f;

    // Render server handles
    int64_t mesh_rid = -1;
    int64_t instance_rid = -1;
    bool mesh_dirty = true;

    void generate_mesh();
};

ImmediateMesh3D::ImmediateMesh3D() : pimpl(std::make_unique<Impl>()) {}
ImmediateMesh3D::~ImmediateMesh3D() = default;

void ImmediateMesh3D::begin(ImmediatePrimitive primitive, int material_id) {
    if (pimpl->drawing) return;
    pimpl->drawing = true;
    pimpl->primitive = primitive;
    pimpl->material_id = material_id;
    pimpl->vertices.clear();
    // also reset per‑vertex state to defaults
    pimpl->current_color[0]=pimpl->current_color[1]=pimpl->current_color[2]=1.0f; pimpl->current_color[3]=1.0f;
    pimpl->current_normal[0]=0.0f; pimpl->current_normal[1]=0.0f; pimpl->current_normal[2]=1.0f;
    pimpl->current_uv[0]=0.0f; pimpl->current_uv[1]=0.0f;
}

void ImmediateMesh3D::vertex(const double* pos, const float* normal,
                             const float* uv, const float* color) {
    if (!pimpl->drawing) return;
    ImmediateVertex v;
    v.pos[0] = pos[0]; v.pos[1] = pos[1]; v.pos[2] = pos[2];
    if (normal) { memcpy(v.normal, normal, 3*sizeof(float)); }
    else { memcpy(v.normal, pimpl->current_normal, 3*sizeof(float)); }
    if (uv) { memcpy(v.uv, uv, 2*sizeof(float)); }
    else { memcpy(v.uv, pimpl->current_uv, 2*sizeof(float)); }
    if (color) { memcpy(v.color, color, 4*sizeof(float)); }
    else { memcpy(v.color, pimpl->current_color, 4*sizeof(float)); }
    pimpl->vertices.push_back(v);
}

void ImmediateMesh3D::end() {
    if (!pimpl->drawing) return;
    pimpl->drawing = false;
    pimpl->mesh_dirty = true;
    update_mesh();
}

void ImmediateMesh3D::set_color(float r, float g, float b, float a) {
    pimpl->current_color[0]=r; pimpl->current_color[1]=g; pimpl->current_color[2]=b; pimpl->current_color[3]=a;
}
void ImmediateMesh3D::set_normal(const float* n) { memcpy(pimpl->current_normal, n, 3*sizeof(float)); }
void ImmediateMesh3D::set_uv(const float* uv) { memcpy(pimpl->current_uv, uv, 2*sizeof(float)); }

void ImmediateMesh3D::clear() {
    pimpl->vertices.clear();
    pimpl->mesh_dirty = true;
    update_mesh();
}

void ImmediateMesh3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; GeometryInstance3D::set_cast_shadow(cast); }
void ImmediateMesh3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
void ImmediateMesh3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; GeometryInstance3D::set_gi_mode(mode); }
void ImmediateMesh3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
void ImmediateMesh3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void ImmediateMesh3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}

void ImmediateMesh3D::Impl::generate_mesh() {
    if (vertices.empty()) {
        // clear mesh from server
        if (mesh_rid != -1) {
            // RenderingServer::mesh_free(mesh_rid);
            mesh_rid = -1;
        }
        return;
    }
    // Convert vertices to raw arrays for GPU
    std::vector<float> pos_data; // 3 floats per vertex
    std::vector<float> norm_data;
    std::vector<float> uv_data;
    std::vector<float> color_data;
    pos_data.reserve(vertices.size() * 3);
    norm_data.reserve(vertices.size() * 3);
    uv_data.reserve(vertices.size() * 2);
    color_data.reserve(vertices.size() * 4);
    for (const auto& v : vertices) {
        pos_data.push_back((float)v.pos[0]); pos_data.push_back((float)v.pos[1]); pos_data.push_back((float)v.pos[2]);
        norm_data.push_back(v.normal[0]); norm_data.push_back(v.normal[1]); norm_data.push_back(v.normal[2]);
        uv_data.push_back(v.uv[0]); uv_data.push_back(v.uv[1]);
        color_data.push_back(v.color[0]); color_data.push_back(v.color[1]); color_data.push_back(v.color[2]); color_data.push_back(v.color[3]);
    }
    // If mesh_rid doesn't exist, create new one.
    if (mesh_rid == -1) {
        // mesh_rid = RenderingServer::mesh_create();
    }
    // Set vertex attributes and index buffer (if needed)
    // For primitive type, we may need to generate indices for triangle strip etc.
    // Simplified: just use vertices as is (immediate mode expects correct order).
    // RenderingServer::mesh_add_surface(mesh_rid, primitive_to_rs(primitive), pos_data, norm_data, uv_data, color_data, indices_data);
}

void ImmediateMesh3D::update_mesh() {
    if (!pimpl->mesh_dirty) return;
    pimpl->generate_mesh();
    // If emissive and GI mode > 0, notify GI system
    if (pimpl->emissive_intensity > 0.0f && pimpl->gi_mode > 0) {
        // Register dynamic emissive surface (temporary, per frame)
    }
    // Update bounding box (compute from vertices)
    if (!pimpl->vertices.empty()) {
        double min_x = pimpl->vertices[0].pos[0], max_x = pimpl->vertices[0].pos[0];
        double min_y = pimpl->vertices[0].pos[1], max_y = pimpl->vertices[0].pos[1];
        double min_z = pimpl->vertices[0].pos[2], max_z = pimpl->vertices[0].pos[2];
        for (size_t i=1; i<pimpl->vertices.size(); ++i) {
            min_x = std::min(min_x, pimpl->vertices[i].pos[0]);
            max_x = std::max(max_x, pimpl->vertices[i].pos[0]);
            min_y = std::min(min_y, pimpl->vertices[i].pos[1]);
            max_y = std::max(max_y, pimpl->vertices[i].pos[1]);
            min_z = std::min(min_z, pimpl->vertices[i].pos[2]);
            max_z = std::max(max_z, pimpl->vertices[i].pos[2]);
        }
        set_aabb(&min_x, &max_x);
        double dx = max_x-min_x, dy = max_y-min_y, dz = max_z-min_z;
        set_bounding_sphere_radius(std::sqrt(dx*dx+dy*dy+dz*dz)*0.5);
    } else {
        double zero[3]={0,0,0};
        set_aabb(zero, zero);
        set_bounding_sphere_radius(0.0);
    }
    pimpl->mesh_dirty = false;
}

void ImmediateMesh3D::synchronize_render_server(double delta) {
    GeometryInstance3D::synchronize_render_server(delta);
    if (pimpl->mesh_dirty) {
        update_mesh();
    }
    // If geometry has changed, update the visual instance's mesh.
    if (pimpl->mesh_rid != -1) {
        // RenderingServer::instance_set_base(instance_rid, mesh_rid);
    }
}

void ImmediateMesh3D::process(double delta) {
    GeometryInstance3D::process(delta);
    // nothing additional
}

} // namespace lighting