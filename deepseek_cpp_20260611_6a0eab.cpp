// label_3d.cpp
#include "label_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <string>

namespace lighting {

// ============================================================================
// Placeholder Font resource (actual engine would load TrueType / SDF)
// ============================================================================
class Font {
public:
    struct GlyphData {
        int texture_id;
        double advance_x, advance_y;
        double bitmap_left, bitmap_top;
        double bitmap_width, bitmap_height;
        double uv_x0, uv_y0, uv_x1, uv_y1;
    };
    virtual GlyphData get_glyph(uint32_t codepoint, int size) const = 0;
    virtual double get_line_height(int size) const = 0;
    virtual double get_ascender(int size) const = 0;
    virtual ~Font() = default;
};

// ============================================================================
// Simple bitmap font implementation (simulated)
// ============================================================================
class SimpleFont : public Font {
public:
    SimpleFont(int texture_id, double line_height, double ascender)
        : m_texture_id(texture_id), m_line_height(line_height), m_ascender(ascender) {}
    GlyphData get_glyph(uint32_t codepoint, int size) const override {
        // Simulate a placeholder glyph (like 'A')
        GlyphData g;
        g.texture_id = m_texture_id;
        g.advance_x = size * 0.6;
        g.advance_y = 0;
        g.bitmap_left = 0;
        g.bitmap_top = size;
        g.bitmap_width = size * 0.6;
        g.bitmap_height = size;
        g.uv_x0 = 0; g.uv_y0 = 0; g.uv_x1 = 1; g.uv_y1 = 1;
        return g;
    }
    double get_line_height(int size) const override { return m_line_height * size; }
    double get_ascender(int size) const override { return m_ascender * size; }
private:
    int m_texture_id;
    double m_line_height;
    double m_ascender;
};

// ============================================================================
// Label3D implementation
// ============================================================================
struct Label3D::Impl {
    std::string text = "Label";
    Font* font = nullptr;
    int font_size = 16;

    float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float outline_modulate[4] = {0.0f,0.0f,0.0f,0.0f};
    float outline_size = 0.0f;
    bool shadow_enabled = false;
    double shadow_offset[3] = {0.01, -0.01, 0.0};
    float shadow_color[4] = {0.0f,0.0f,0.0f,0.5f};

    HorizontalAlignment h_align = HorizontalAlignment::CENTER;
    VerticalAlignment v_align = VerticalAlignment::CENTER;
    float line_spacing = 1.0f;
    bool autowrap = false;
    double autowrap_width = 10.0;

    BillboardMode billboard = BillboardMode::ENABLED;
    double fixed_forward[3] = {0,0,-1};
    double fixed_up[3] = {0,1,0};
    double pixel_offset[2] = {0,0};

    bool double_sided = false;
    bool depth_test_enabled = true;

    bool cast_shadow = false;   // text rarely casts shadow
    bool receive_shadow = true;
    int gi_mode = 1;            // static by default
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    int material_override = -1;
    bool vertex_color_enabled = false;

    bool dirty = true;
    // Mesh data (generated from text)
    std::vector<double> vertices;   // 3 per vertex
    std::vector<float> normals;     // 3 per vertex
    std::vector<float> uvs;         // 2 per vertex
    std::vector<float> vertex_colors; // 4 per vertex (if vertex_color_enabled)
    std::vector<int> indices;
    int64_t mesh_rid = -1;
    int64_t instance_rid = -1;

    // Text layout results
    struct Line {
        std::vector<uint32_t> codepoints;
        std::vector<double> x_offsets; // per glyph baseline x (local)
        double width;
        double ascent;
        double descent;
    };
    std::vector<Line> layout;

    void regenerate_layout();
    void generate_mesh();
    void update_aabb();
};

Label3D::Label3D() : pimpl(std::make_unique<Impl>()) {}
Label3D::~Label3D() = default;

void Label3D::set_text(const char* text) {
    pimpl->text = text ? text : "";
    pimpl->dirty = true;
    update_label();
}
const char* Label3D::get_text() const { return pimpl->text.c_str(); }
void Label3D::set_font(Font* font) { pimpl->font = font; pimpl->dirty = true; update_label(); }
Font* Label3D::get_font() const { return pimpl->font; }
void Label3D::set_font_size(int size) { pimpl->font_size = std::max(4, size); pimpl->dirty = true; update_label(); }
int Label3D::get_font_size() const { return pimpl->font_size; }

void Label3D::set_color(const float* rgba) { memcpy(pimpl->color, rgba, 4*sizeof(float)); pimpl->dirty = true; update_label(); }
void Label3D::get_color(float* out_rgba) const { memcpy(out_rgba, pimpl->color, 4*sizeof(float)); }
void Label3D::set_outline_modulate(const float* rgba) { memcpy(pimpl->outline_modulate, rgba, 4*sizeof(float)); pimpl->dirty = true; update_label(); }
void Label3D::get_outline_modulate(float* out_rgba) const { memcpy(out_rgba, pimpl->outline_modulate, 4*sizeof(float)); }
void Label3D::set_outline_size(float size) { pimpl->outline_size = std::max(0.0f, size); pimpl->dirty = true; update_label(); }
float Label3D::get_outline_size() const { return pimpl->outline_size; }
void Label3D::set_shadow_enabled(bool enabled) { pimpl->shadow_enabled = enabled; pimpl->dirty = true; update_label(); }
bool Label3D::is_shadow_enabled() const { return pimpl->shadow_enabled; }
void Label3D::set_shadow_offset(const double* offset) { memcpy(pimpl->shadow_offset, offset, 3*sizeof(double)); pimpl->dirty = true; update_label(); }
void Label3D::get_shadow_offset(double* out_offset) const { memcpy(out_offset, pimpl->shadow_offset, 3*sizeof(double)); }
void Label3D::set_shadow_color(const float* rgba) { memcpy(pimpl->shadow_color, rgba, 4*sizeof(float)); pimpl->dirty = true; update_label(); }
void Label3D::get_shadow_color(float* out_rgba) const { memcpy(out_rgba, pimpl->shadow_color, 4*sizeof(float)); }

void Label3D::set_horizontal_alignment(HorizontalAlignment align) { pimpl->h_align = align; pimpl->dirty = true; update_label(); }
HorizontalAlignment Label3D::get_horizontal_alignment() const { return pimpl->h_align; }
void Label3D::set_vertical_alignment(VerticalAlignment align) { pimpl->v_align = align; pimpl->dirty = true; update_label(); }
VerticalAlignment Label3D::get_vertical_alignment() const { return pimpl->v_align; }
void Label3D::set_line_spacing(float spacing) { pimpl->line_spacing = spacing; pimpl->dirty = true; update_label(); }
float Label3D::get_line_spacing() const { return pimpl->line_spacing; }
void Label3D::set_autowrap_enabled(bool enabled) { pimpl->autowrap = enabled; pimpl->dirty = true; update_label(); }
bool Label3D::is_autowrap_enabled() const { return pimpl->autowrap; }
void Label3D::set_autowrap_width(double width) { pimpl->autowrap_width = width; pimpl->dirty = true; update_label(); }
double Label3D::get_autowrap_width() const { return pimpl->autowrap_width; }

void Label3D::set_billboard_mode(BillboardMode mode) { pimpl->billboard = mode; }
BillboardMode Label3D::get_billboard_mode() const { return pimpl->billboard; }
void Label3D::set_fixed_orientation(const double* forward, const double* up) {
    memcpy(pimpl->fixed_forward, forward, 3*sizeof(double));
    memcpy(pimpl->fixed_up, up, 3*sizeof(double));
}
void Label3D::get_fixed_orientation(double* out_forward, double* out_up) const {
    memcpy(out_forward, pimpl->fixed_forward, 3*sizeof(double));
    memcpy(out_up, pimpl->fixed_up, 3*sizeof(double));
}
void Label3D::set_pixel_offset(const double* offset) { memcpy(pimpl->pixel_offset, offset, 2*sizeof(double)); }
void Label3D::get_pixel_offset(double* out_offset) const { memcpy(out_offset, pimpl->pixel_offset, 2*sizeof(double)); }

void Label3D::set_double_sided(bool double_sided) { pimpl->double_sided = double_sided; }
bool Label3D::is_double_sided() const { return pimpl->double_sided; }
void Label3D::set_depth_test_enabled(bool enabled) { pimpl->depth_test_enabled = enabled; }
bool Label3D::is_depth_test_enabled() const { return pimpl->depth_test_enabled; }

void Label3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; GeometryInstance3D::set_cast_shadow(cast); }
void Label3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
void Label3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; GeometryInstance3D::set_gi_mode(mode); }
void Label3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
void Label3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void Label3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}
void Label3D::set_material_override(int material_id) { pimpl->material_override = material_id; }
int Label3D::get_material_override() const { return pimpl->material_override; }
void Label3D::set_vertex_color_enabled(bool enabled) { pimpl->vertex_color_enabled = enabled; pimpl->dirty = true; update_label(); }
bool Label3D::is_vertex_color_enabled() const { return pimpl->vertex_color_enabled; }

void Label3D::Impl::regenerate_layout() {
    if (!font) return;
    layout.clear();
    std::vector<uint32_t> codepoints;
    for (char c : text) codepoints.push_back((uint32_t)(unsigned char)c); // naive UTF‑8 not handled
    double line_height = font->get_line_height(font_size) * line_spacing;
    double ascender = font->get_ascender(font_size);
    double x = 0;
    double current_line_width = 0;
    Line current_line;
    for (size_t i = 0; i < codepoints.size(); ++i) {
        auto glyph = font->get_glyph(codepoints[i], font_size);
        double adv = glyph.advance_x;
        if (codepoints[i] == '\n') {
            current_line.width = x;
            layout.push_back(current_line);
            current_line = Line();
            x = 0;
            continue;
        }
        if (autowrap && current_line_width + adv > autowrap_width && !current_line.codepoints.empty()) {
            current_line.width = x;
            layout.push_back(current_line);
            current_line = Line();
            x = 0;
            current_line_width = 0;
        }
        current_line.codepoints.push_back(codepoints[i]);
        current_line.x_offsets.push_back(x);
        x += adv;
        current_line_width += adv;
    }
    if (!current_line.codepoints.empty()) {
        current_line.width = x;
        layout.push_back(current_line);
    }
    // compute overall extents
    double max_width = 0;
    for (const auto& l : layout) max_width = std::max(max_width, l.width);
    double total_height = layout.size() * line_height;
    double y_offset = 0;
    for (auto& l : layout) {
        l.ascent = ascender;
        l.descent = line_height - ascender;
    }
}

void Label3D::Impl::generate_mesh() {
    vertices.clear(); normals.clear(); uvs.clear(); vertex_colors.clear(); indices.clear();
    if (!font || text.empty()) return;
    regenerate_layout();
    double line_height = font->get_line_height(font_size) * line_spacing;
    double total_height = layout.size() * line_height;
    double y_start = 0;
    if (v_align == VerticalAlignment::TOP) y_start = 0;
    else if (v_align == VerticalAlignment::CENTER) y_start = -total_height * 0.5;
    else y_start = -total_height;

    for (size_t line_idx = 0; line_idx < layout.size(); ++line_idx) {
        const Line& line = layout[line_idx];
        double line_y = y_start + line_idx * line_height;
        double x_start = 0;
        if (h_align == HorizontalAlignment::LEFT) x_start = 0;
        else if (h_align == HorizontalAlignment::CENTER) x_start = -line.width * 0.5;
        else x_start = -line.width;
        for (size_t g = 0; g < line.codepoints.size(); ++g) {
            auto glyph = font->get_glyph(line.codepoints[g], font_size);
            double x = x_start + line.x_offsets[g];
            double y = line_y + glyph.bitmap_top - glyph.bitmap_height;
            double w = glyph.bitmap_width;
            double h = glyph.bitmap_height;
            // build quad (two triangles) for this glyph
            int base = (int)vertices.size() / 3;
            vertices.push_back(x);    vertices.push_back(y);    vertices.push_back(0);
            vertices.push_back(x+w);  vertices.push_back(y);    vertices.push_back(0);
            vertices.push_back(x+w);  vertices.push_back(y-h);  vertices.push_back(0);
            vertices.push_back(x);    vertices.push_back(y-h);  vertices.push_back(0);
            // normals (facing camera, Z)
            for (int i=0;i<4;++i) { normals.push_back(0.0f); normals.push_back(0.0f); normals.push_back(1.0f); }
            // UVs
            uvs.push_back(glyph.uv_x0); uvs.push_back(glyph.uv_y0);
            uvs.push_back(glyph.uv_x1); uvs.push_back(glyph.uv_y0);
            uvs.push_back(glyph.uv_x1); uvs.push_back(glyph.uv_y1);
            uvs.push_back(glyph.uv_x0); uvs.push_back(glyph.uv_y1);
            // vertex colors
            if (vertex_color_enabled) {
                for (int i=0;i<4;++i) {
                    vertex_colors.push_back(color[0]); vertex_colors.push_back(color[1]);
                    vertex_colors.push_back(color[2]); vertex_colors.push_back(color[3]);
                }
            }
            // indices
            indices.push_back(base); indices.push_back(base+1); indices.push_back(base+2);
            indices.push_back(base); indices.push_back(base+2); indices.push_back(base+3);
        }
    }
    // shadow pass (optional) – we would generate a second mesh displaced by shadow_offset
    // but for simplicity, we rely on rendering server to duplicate.
}

void Label3D::Impl::update_aabb() {
    if (vertices.empty()) {
        double zero[3]={0,0,0};
        set_aabb(zero, zero);
        set_bounding_sphere_radius(0.0);
        return;
    }
    double min_x = vertices[0], max_x = vertices[0];
    double min_y = vertices[1], max_y = vertices[1];
    double min_z = vertices[2], max_z = vertices[2];
    for (size_t i=3; i<vertices.size(); i+=3) {
        min_x = std::min(min_x, vertices[i]);
        max_x = std::max(max_x, vertices[i]);
        min_y = std::min(min_y, vertices[i+1]);
        max_y = std::max(max_y, vertices[i+1]);
        min_z = std::min(min_z, vertices[i+2]);
        max_z = std::max(max_z, vertices[i+2]);
    }
    // account for shadow offset if enabled
    if (shadow_enabled) {
        min_x += std::min(0.0, shadow_offset[0]);
        max_x += std::max(0.0, shadow_offset[0]);
        min_y += std::min(0.0, shadow_offset[1]);
        max_y += std::max(0.0, shadow_offset[1]);
        min_z += std::min(0.0, shadow_offset[2]);
        max_z += std::max(0.0, shadow_offset[2]);
    }
    set_aabb(&min_x, &max_x);
    double dx = max_x-min_x, dy = max_y-min_y, dz = max_z-min_z;
    set_bounding_sphere_radius(std::sqrt(dx*dx+dy*dy+dz*dz)*0.5);
}

void Label3D::update_label() {
    if (!pimpl->dirty) return;
    pimpl->generate_mesh();
    if (pimpl->mesh_rid != -1) {
        // RenderingServer::mesh_free(pimpl->mesh_rid);
        pimpl->mesh_rid = -1;
    }
    // Create new mesh with generated data
    // (For brevity, we skip actual server calls)
    pimpl->update_aabb();
    pimpl->dirty = false;
}

void Label3D::ready() {
    GeometryInstance3D::ready();
    if (!pimpl->font) {
        // create default font (fallback)
    }
    update_label();
}

void Label3D::process(double delta) {
    GeometryInstance3D::process(delta);
    // For billboard mode, we could update transform each frame, but that's handled by render server
}

void Label3D::synchronize_render_server(double delta) {
    GeometryInstance3D::synchronize_render_server(delta);
    if (pimpl->dirty) update_label();
    if (pimpl->mesh_rid != -1) {
        // RenderingServer::instance_set_base(instance_rid, mesh_rid);
        // Set material overrides, shadow flags, GI mode, billboard settings.
    }
    // If emissive, notify GI system
    if (pimpl->emissive_intensity > 0.0f && pimpl->gi_mode > 0) {
        // Register as dynamic emissive source (temporary)
    }
}

} // namespace lighting