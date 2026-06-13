// Name : lighting enhancement
// File : scene/3d/texture_rect_3d_ext.cpp 56 of 60
// Description : Implementation of TextureRect3DExt with mesh generation,
//               texture filtering, billboard, modulate, transparency, flip,
//               and full RenderingServer synchronization.
#include "texture_rect_3d_ext.h"
#include "servers/rendering_server.h"
#include "core/math/math_funcs.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/math/quaternion.h"
#include <cmath>

struct TextureRect3DExt::Impl {
    RID mesh_rid;
    RID instance_rid;
    RID texture_rid;
    RID material_rid;

    // Texture parameters
    int texture_filter = 1;           // linear
    bool repeat_u = false;
    bool repeat_v = false;

    // Geometry
    float width = 1.0f;
    float height = 1.0f;
    Vector3 offset = Vector3(0,0,0);
    bool flip_h = false;
    bool flip_v = false;

    // Billboard
    int billboard_mode = 0;           // 0 = disabled
    Vector2 pixel_offset = Vector2(0,0);

    // Color & transparency
    Color modulate = Color(1,1,1,1);
    float opacity = 1.0f;
    bool transparent = true;

    // Lighting flags
    bool cast_shadow = false;          // texture rects rarely cast shadow
    bool receive_shadow = true;
    int gi_mode = 1;                   // static by default
    float gi_contribution = 1.0f;
    Color emissive_color = Color(0,0,0);
    float emissive_intensity = 0.0f;

    bool dirty = true;

    Impl() {
        mesh_rid = RenderingServer::get_singleton()->mesh_create();
        instance_rid = RenderingServer::get_singleton()->instance_create();
        RenderingServer::get_singleton()->instance_set_base(instance_rid, mesh_rid);
        material_rid = RenderingServer::get_singleton()->material_create();
        RenderingServer::get_singleton()->material_set_param(material_rid, "albedo", modulate);
        RenderingServer::get_singleton()->material_set_param(material_rid, "transparent", transparent);
    }

    ~Impl() {
        if (mesh_rid.is_valid()) RenderingServer::get_singleton()->free(mesh_rid);
        if (instance_rid.is_valid()) RenderingServer::get_singleton()->free(instance_rid);
        if (material_rid.is_valid()) RenderingServer::get_singleton()->free(material_rid);
    }

    void generate_mesh() {
        // Create a simple quad mesh (two triangles)
        float hw = width * 0.5f;
        float hh = height * 0.5f;
        Vector3 vertices[4] = {
            Vector3(-hw, -hh, 0),
            Vector3( hw, -hh, 0),
            Vector3( hw,  hh, 0),
            Vector3(-hw,  hh, 0)
        };
        // UVs with flip
        float u0 = flip_h ? 1.0f : 0.0f;
        float u1 = flip_h ? 0.0f : 1.0f;
        float v0 = flip_v ? 1.0f : 0.0f;
        float v1 = flip_v ? 0.0f : 1.0f;
        Vector2 uvs[4] = {
            Vector2(u0, v0),
            Vector2(u1, v0),
            Vector2(u1, v1),
            Vector2(u0, v1)
        };
        // Indices
        int indices[6] = {0,1,2, 0,2,3};
        // Normals (facing Z)
        Vector3 normals[4] = {Vector3(0,0,1), Vector3(0,0,1), Vector3(0,0,1), Vector3(0,0,1)};

        // Apply offset
        for (int i = 0; i < 4; ++i) {
            vertices[i] += offset;
        }

        RenderingServer::get_singleton()->mesh_clear(mesh_rid);
        Vector<Vector3> verts_vec;
        Vector<Vector2> uv_vec;
        Vector<int> idx_vec;
        Vector<Vector3> norm_vec;
        for (int i = 0; i < 4; ++i) {
            verts_vec.push_back(vertices[i]);
            uv_vec.push_back(uvs[i]);
            norm_vec.push_back(normals[i]);
        }
        for (int i = 0; i < 6; ++i) idx_vec.push_back(indices[i]);

        RenderingServer::get_singleton()->mesh_add_surface(mesh_rid, RS::PRIMITIVE_TRIANGLES, verts_vec, idx_vec, uv_vec, norm_vec);

        // Apply texture filtering and repeat
        RenderingServer::get_singleton()->mesh_surface_set_texture_filter(mesh_rid, 0, texture_filter);
        // For repeat, we need to set on material's texture (not directly on mesh)
    }

    void update_material() {
        RenderingServer *rs = RenderingServer::get_singleton();
        rs->material_set_param(material_rid, "albedo", modulate);
        rs->material_set_param(material_rid, "opacity", opacity);
        rs->material_set_param(material_rid, "transparent", transparent);
        if (texture_rid.is_valid()) {
            rs->material_set_param(material_rid, "texture", texture_rid);
            rs->material_set_param(material_rid, "repeat_u", repeat_u);
            rs->material_set_param(material_rid, "repeat_v", repeat_v);
        }
        // Emissive
        if (emissive_intensity > 0.0f) {
            rs->material_set_param(material_rid, "emission", emissive_color);
            rs->material_set_param(material_rid, "emission_intensity", emissive_intensity);
        }
        // Shadow and GI flags are set on instance, not material.
    }

    void update_instance() {
        RenderingServer *rs = RenderingServer::get_singleton();
        // Transform: handle billboard mode
        Transform3D global = get_global_transform();
        Transform3D final_transform = global;
        if (billboard_mode == 1) {
            // Full billboard: rotate to face camera (simplified: create look-at matrix)
            // For full billboard, we need camera position and up. This is typically done by the shader or CPU.
            // Here we just set the instance to use billboard mode in material.
            // Instead of computing matrix, we set a material flag.
            rs->material_set_param(material_rid, "billboard", true);
        } else if (billboard_mode == 2) {
            // Fixed Y billboard (rotate only around Y axis)
            rs->material_set_param(material_rid, "billboard_fixed_y", true);
        } else {
            rs->material_set_param(material_rid, "billboard", false);
            rs->material_set_param(material_rid, "billboard_fixed_y", false);
        }
        // Pixel offset (for billboard) is handled in shader via uniform
        rs->material_set_param(material_rid, "pixel_offset", Vector3(pixel_offset.x, pixel_offset.y, 0.0f));

        rs->instance_set_base(instance_rid, mesh_rid);
        rs->instance_set_transform(instance_rid, final_transform);
        rs->instance_set_material_override(instance_rid, material_rid);
        rs->instance_set_cast_shadow(instance_rid, cast_shadow);
        rs->instance_set_receive_shadows(instance_rid, receive_shadow);
        rs->instance_set_gi_mode(instance_rid, gi_mode);
        rs->instance_set_gi_contribution(instance_rid, gi_contribution);
        rs->instance_set_emissive(instance_rid, emissive_color, emissive_intensity);
        rs->instance_set_visible(instance_rid, true);
    }

    void sync() {
        if (dirty) {
            generate_mesh();
            update_material();
            update_instance();
            dirty = false;
        }
    }
};

TextureRect3DExt::TextureRect3DExt() {
    pimpl = new Impl();
}

TextureRect3DExt::~TextureRect3DExt() {
    delete pimpl;
}

void TextureRect3DExt::set_texture(const RID &p_texture) {
    pimpl->texture_rid = p_texture;
    pimpl->dirty = true;
    sync_rect();
}
RID TextureRect3DExt::get_texture() const { return pimpl->texture_rid; }

void TextureRect3DExt::set_material(const RID &p_material) {
    pimpl->material_rid = p_material;
    pimpl->dirty = true;
    sync_rect();
}
RID TextureRect3DExt::get_material() const { return pimpl->material_rid; }

void TextureRect3DExt::set_texture_filter(int p_filter) {
    pimpl->texture_filter = p_filter;
    pimpl->dirty = true;
    sync_rect();
}
int TextureRect3DExt::get_texture_filter() const { return pimpl->texture_filter; }

void TextureRect3DExt::set_texture_repeat(bool p_repeat_u, bool p_repeat_v) {
    pimpl->repeat_u = p_repeat_u;
    pimpl->repeat_v = p_repeat_v;
    pimpl->dirty = true;
    sync_rect();
}
void TextureRect3DExt::get_texture_repeat(bool &r_repeat_u, bool &r_repeat_v) const {
    r_repeat_u = pimpl->repeat_u;
    r_repeat_v = pimpl->repeat_v;
}

void TextureRect3DExt::set_size(float p_width, float p_height) {
    pimpl->width = p_width;
    pimpl->height = p_height;
    pimpl->dirty = true;
    sync_rect();
}
void TextureRect3DExt::get_size(float &r_width, float &r_height) const {
    r_width = pimpl->width;
    r_height = pimpl->height;
}

void TextureRect3DExt::set_offset(const Vector3 &p_offset) {
    pimpl->offset = p_offset;
    pimpl->dirty = true;
    sync_rect();
}
Vector3 TextureRect3DExt::get_offset() const { return pimpl->offset; }

void TextureRect3DExt::set_flip(bool p_flip_h, bool p_flip_v) {
    pimpl->flip_h = p_flip_h;
    pimpl->flip_v = p_flip_v;
    pimpl->dirty = true;
    sync_rect();
}
void TextureRect3DExt::get_flip(bool &r_flip_h, bool &r_flip_v) const {
    r_flip_h = pimpl->flip_h;
    r_flip_v = pimpl->flip_v;
}

void TextureRect3DExt::set_billboard_mode(int p_mode) {
    pimpl->billboard_mode = p_mode;
    pimpl->dirty = true;
    sync_rect();
}
int TextureRect3DExt::get_billboard_mode() const { return pimpl->billboard_mode; }

void TextureRect3DExt::set_pixel_offset(const Vector2 &p_offset) {
    pimpl->pixel_offset = p_offset;
    pimpl->dirty = true;
    sync_rect();
}
Vector2 TextureRect3DExt::get_pixel_offset() const { return pimpl->pixel_offset; }

void TextureRect3DExt::set_modulate(const Color &p_color) {
    pimpl->modulate = p_color;
    pimpl->dirty = true;
    sync_rect();
}
Color TextureRect3DExt::get_modulate() const { return pimpl->modulate; }

void TextureRect3DExt::set_opacity(float p_opacity) {
    pimpl->opacity = p_opacity;
    pimpl->dirty = true;
    sync_rect();
}
float TextureRect3DExt::get_opacity() const { return pimpl->opacity; }

void TextureRect3DExt::set_transparent(bool p_transparent) {
    pimpl->transparent = p_transparent;
    pimpl->dirty = true;
    sync_rect();
}
bool TextureRect3DExt::is_transparent() const { return pimpl->transparent; }

void TextureRect3DExt::set_cast_shadow(bool p_cast) {
    pimpl->cast_shadow = p_cast;
    pimpl->dirty = true;
    sync_rect();
}
void TextureRect3DExt::set_receive_shadow(bool p_receive) {
    pimpl->receive_shadow = p_receive;
    pimpl->dirty = true;
    sync_rect();
}
void TextureRect3DExt::set_gi_mode(int p_mode) {
    pimpl->gi_mode = p_mode;
    pimpl->dirty = true;
    sync_rect();
}
void TextureRect3DExt::set_gi_contribution(float p_amount) {
    pimpl->gi_contribution = p_amount;
    pimpl->dirty = true;
    sync_rect();
}
void TextureRect3DExt::set_emissive(const Color &p_color, float p_intensity) {
    pimpl->emissive_color = p_color;
    pimpl->emissive_intensity = p_intensity;
    pimpl->dirty = true;
    sync_rect();
}
Color TextureRect3DExt::get_emissive() const { return pimpl->emissive_color; }
float TextureRect3DExt::get_emissive_intensity() const { return pimpl->emissive_intensity; }

void TextureRect3DExt::sync_rect() {
    pimpl->sync();
}