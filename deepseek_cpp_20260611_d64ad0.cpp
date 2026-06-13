// texture_rect_3d.cpp
#include "texture_rect_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

namespace lighting {

// ============================================================================
// Internal: generate plane mesh (two triangles) with UVs, optionally tiled
// ============================================================================
static void generate_plane_mesh(double width, double height, bool flip_h, bool flip_v,
                                TextureRectMode mode, TextureRectBillboard billboard,
                                std::vector<double>& out_vertices,
                                std::vector<int>& out_indices,
                                std::vector<float>& out_normals,
                                std::vector<float>& out_uvs) {
    out_vertices.clear();
    out_indices.clear();
    out_normals.clear();
    out_uvs.clear();

    double hw = width * 0.5;
    double hh = height * 0.5;
    // For billboard, we keep plane at origin (0,0,0) and let rendering server handle orientation.
    // For non‑billboard, plane faces local Z axis (forward).
    double vertices[4][3] = {
        {-hw, -hh, 0.0},
        { hw, -hh, 0.0},
        { hw,  hh, 0.0},
        {-hw,  hh, 0.0}
    };
    // UVs (flip control)
    float u0 = flip_h ? 1.0f : 0.0f;
    float u1 = flip_h ? 0.0f : 1.0f;
    float v0 = flip_v ? 1.0f : 0.0f;
    float v1 = flip_v ? 0.0f : 1.0f;

    float uvs[4][2] = {
        {u0, v0},
        {u1, v0},
        {u1, v1},
        {u0, v1}
    };
    // Indices (two triangles)
    int indices[6] = {0,1,2, 0,2,3};
    // Normals (facing +Z)
    float normal[3] = {0.0f, 0.0f, 1.0f};

    for (int i = 0; i < 4; ++i) {
        out_vertices.push_back(vertices[i][0]);
        out_vertices.push_back(vertices[i][1]);
        out_vertices.push_back(vertices[i][2]);
        out_normals.push_back(normal[0]);
        out_normals.push_back(normal[1]);
        out_normals.push_back(normal[2]);
        out_uvs.push_back(uvs[i][0]);
        out_uvs.push_back(uvs[i][1]);
    }
    for (int i = 0; i < 6; ++i) {
        out_indices.push_back(indices[i]);
    }
}

// ============================================================================
// TextureRect3D implementation
// ============================================================================
struct TextureRect3D::Impl {
    int64_t texture_rid = -1;
    int64_t material_rid = -1;
    int texture_filter = 1;          // linear
    bool repeat_u = false;
    bool repeat_v = false;

    double width = 1.0;
    double height = 1.0;
    double offset[3] = {0,0,0};
    bool flip_h = false;
    bool flip_v = false;
    TextureRectMode mode = TextureRectMode::STRETCH;

    TextureRectBillboard billboard = TextureRectBillboard::DISABLED;
    double pixel_offset[2] = {0,0};

    float modulate[4] = {1.0f,1.0f,1.0f,1.0f};
    float opacity = 1.0f;
    bool transparent = true;

    bool cast_shadow = false;   // textures usually don't cast shadow
    bool receive_shadow = true;
    int gi_mode = 1;            // static by default
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    bool dirty = true;
    int64_t mesh_rid = -1;
    int64_t instance_rid = -1;

    // Mesh data
    std::vector<double> vertices;
    std::vector<int> indices;
    std::vector<float> normals;
    std::vector<float> uvs;

    void regenerate_mesh();
    void update_render_server();
};

TextureRect3D::TextureRect3D() : pimpl(std::make_unique<Impl>()) {}
TextureRect3D::~TextureRect3D() = default;

void TextureRect3D::set_texture(int64_t texture_rid) {
    pimpl->texture_rid = texture_rid;
    pimpl->dirty = true;
}
int64_t TextureRect3D::get_texture_rid() const { return pimpl->texture_rid; }
void TextureRect3D::set_material(int64_t material_rid) { pimpl->material_rid = material_rid; pimpl->dirty = true; }
int64_t TextureRect3D::get_material_rid() const { return pimpl->material_rid; }
void TextureRect3D::set_texture_filter(int filter) { pimpl->texture_filter = filter; pimpl->dirty = true; }
int TextureRect3D::get_texture_filter() const { return pimpl->texture_filter; }
void TextureRect3D::set_texture_repeat(bool repeat_u, bool repeat_v) {
    pimpl->repeat_u = repeat_u;
    pimpl->repeat_v = repeat_v;
    pimpl->dirty = true;
}
void TextureRect3D::get_texture_repeat(bool& repeat_u, bool& repeat_v) const {
    repeat_u = pimpl->repeat_u;
    repeat_v = pimpl->repeat_v;
}

void TextureRect3D::set_size(double width, double height) {
    pimpl->width = std::max(0.0, width);
    pimpl->height = std::max(0.0, height);
    pimpl->dirty = true;
}
void TextureRect3D::get_size(double& width, double& height) const {
    width = pimpl->width;
    height = pimpl->height;
}
void TextureRect3D::set_offset(const double* offset) { memcpy(pimpl->offset, offset, 3*sizeof(double)); pimpl->dirty = true; }
void TextureRect3D::get_offset(double* out_offset) const { memcpy(out_offset, pimpl->offset, 3*sizeof(double)); }
void TextureRect3D::set_flip(bool flip_h, bool flip_v) { pimpl->flip_h = flip_h; pimpl->flip_v = flip_v; pimpl->dirty = true; }
void TextureRect3D::get_flip(bool& flip_h, bool& flip_v) const { flip_h = pimpl->flip_h; flip_v = pimpl->flip_v; }

void TextureRect3D::set_billboard_mode(TextureRectBillboard mode) { pimpl->billboard = mode; pimpl->dirty = true; }
TextureRectBillboard TextureRect3D::get_billboard_mode() const { return pimpl->billboard; }
void TextureRect3D::set_pixel_offset(const double* offset) { memcpy(pimpl->pixel_offset, offset, 2*sizeof(double)); }
void TextureRect3D::get_pixel_offset(double* out_offset) const { memcpy(out_offset, pimpl->pixel_offset, 2*sizeof(double)); }

void TextureRect3D::set_modulate(const float* rgba) { memcpy(pimpl->modulate, rgba, 4*sizeof(float)); pimpl->dirty = true; }
void TextureRect3D::get_modulate(float* out_rgba) const { memcpy(out_rgba, pimpl->modulate, 4*sizeof(float)); }
void TextureRect3D::set_opacity(float opacity) { pimpl->opacity = std::clamp(opacity, 0.0f, 1.0f); pimpl->dirty = true; }
float TextureRect3D::get_opacity() const { return pimpl->opacity; }
void TextureRect3D::set_transparent(bool transparent) { pimpl->transparent = transparent; pimpl->dirty = true; }
bool TextureRect3D::is_transparent() const { return pimpl->transparent; }

void TextureRect3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; GeometryInstance3D::set_cast_shadow(cast); }
void TextureRect3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
void TextureRect3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; GeometryInstance3D::set_gi_mode(mode); }
void TextureRect3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
void TextureRect3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void TextureRect3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}

void TextureRect3D::Impl::regenerate_mesh() {
    generate_plane_mesh(width, height, flip_h, flip_v, TextureRectMode::STRETCH,
                        billboard, vertices, indices, normals, uvs);
    // Apply offset to vertices (translate)
    if (offset[0] != 0.0 || offset[1] != 0.0 || offset[2] != 0.0) {
        for (size_t i = 0; i < vertices.size(); i += 3) {
            vertices[i]   += offset[0];
            vertices[i+1] += offset[1];
            vertices[i+2] += offset[2];
        }
    }
    // Recompute AABB
    double min_x = vertices[0], max_x = vertices[0];
    double min_y = vertices[1], max_y = vertices[1];
    double min_z = vertices[2], max_z = vertices[2];
    for (size_t i = 3; i < vertices.size(); i += 3) {
        min_x = std::min(min_x, vertices[i]);
        max_x = std::max(max_x, vertices[i]);
        min_y = std::min(min_y, vertices[i+1]);
        max_y = std::max(max_y, vertices[i+1]);
        min_z = std::min(min_z, vertices[i+2]);
        max_z = std::max(max_z, vertices[i+2]);
    }
    set_aabb(&min_x, &max_x);
    double dx = max_x-min_x, dy = max_y-min_y, dz = max_z-min_z;
    set_bounding_sphere_radius(std::sqrt(dx*dx+dy*dy+dz*dz)*0.5);
}

void TextureRect3D::Impl::update_render_server() {
    if (mesh_rid == -1) {
        // mesh_rid = RenderingServer::mesh_create();
    }
    // Add surface if vertices changed
    // RenderingServer::mesh_clear(mesh_rid);
    // RenderingServer::mesh_add_surface_from_arrays(mesh_rid, PRIMITIVE_TRIANGLES, vertices, normals, uvs, indices);
    if (instance_rid == -1) {
        // instance_rid = RenderingServer::instance_create();
    }
    // RenderingServer::instance_set_base(instance_rid, mesh_rid);
    // Set transform (billboard handling is done via transform update each frame)
    // For billboard, we need to compute transform based on camera.
    // That is usually done by the rendering server, but we can also update instance transform each frame.
}

void TextureRect3D::update_rect() {
    if (!pimpl->dirty) return;
    pimpl->regenerate_mesh();
    pimpl->update_render_server();
    pimpl->dirty = false;
}

void TextureRect3D::ready() {
    GeometryInstance3D::ready();
    update_rect();
}

void TextureRect3D::process(double delta) {
    GeometryInstance3D::process(delta);
    // For billboard mode, we could update transform each frame (but Godot does it automatically if instance flag is set).
    // We'll rely on render server's billboard mode via material or instance flag.
    // For completeness, we mark dirty if billboard is enabled and camera moved? Not needed.
}

void TextureRect3D::synchronize_render_server(double delta) {
    GeometryInstance3D::synchronize_render_server(delta);
    if (pimpl->dirty) {
        update_rect();
    }
    // Update instance transform (if not billboard, use node's global transform; if billboard, identity + pixel offset)
    if (pimpl->instance_rid != -1) {
        if (pimpl->billboard != TextureRectBillboard::DISABLED) {
            // For billboard, we need to set instance transform to identity and let shader handle orientation.
            // But typical Godot uses a shader parameter; we skip for brevity.
            // We'll just use node's transform as is.
        }
        Transform3D global = get_global_transform();
        // RenderingServer::instance_set_transform(pimpl->instance_rid, global);
    }
    // If emissive, add to GI system
    if (pimpl->emissive_intensity > 0.0f && pimpl->gi_mode > 0) {
        // Register as emissive surface for GI (placeholder)
    }
}

} // namespace lighting