// label_3d.cpp
#include "label_3d.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <functional>
#include <ft2build.h>
#include FT_FREETYPE_H

namespace lighting {

// ============================================================================
// Font atlas and glyph cache (simplified – full implementation would use
// a global font manager. Here we implement per‑label caching.)
// ============================================================================
struct GlyphInfo {
    float u0, v0, u1, v1;     // texture coordinates in atlas
    float advance;             // x advance in pixels
    float bearing_x, bearing_y;
    int width, height;
};

class FontAtlas {
public:
    FontAtlas(const char* path, int size, bool antialias);
    ~FontAtlas();
    bool is_valid() const { return valid; }
    const GlyphInfo* get_glyph(uint32_t codepoint);
    int get_texture_rid() const { return texture_rid; }
    int get_line_height() const { return line_height; }
    int get_baseline() const { return baseline; }
    int get_size() const { return font_size; }
private:
    bool valid = false;
    FT_Face face = nullptr;
    int font_size = 0;
    int atlas_width = 1024, atlas_height = 1024;
    std::vector<uint8_t> atlas_data; // grayscale alpha
    int texture_rid = -1;
    int line_height = 0;
    int baseline = 0;
    std::unordered_map<uint32_t, GlyphInfo> glyphs;
    int next_x = 0, next_y = 0, row_height = 0;
    void pack_glyph(FT_GlyphSlot slot, uint32_t codepoint);
};

FontAtlas::FontAtlas(const char* path, int size, bool antialias) : font_size(size) {
    FT_Library ft;
    if (FT_Init_FreeType(&ft)) return;
    if (FT_New_Face(ft, path, 0, &face)) { FT_Done_FreeType(ft); return; }
    FT_Set_Pixel_Sizes(face, 0, size);
    line_height = face->size->metrics.height >> 6;
    baseline = face->size->metrics.ascender >> 6;
    atlas_data.assign(atlas_width * atlas_height, 0);
    // Generate ASCII 32..126
    for (uint32_t c = 32; c <= 126; ++c) {
        if (FT_Load_Char(face, c, FT_LOAD_RENDER)) continue;
        pack_glyph(face->glyph, c);
    }
    // Upload to GPU texture (in real engine: RenderingServer::texture_create())
    texture_rid = 1234; // placeholder
    valid = true;
}

FontAtlas::~FontAtlas() {
    if (face) FT_Done_Face(face);
}

void FontAtlas::pack_glyph(FT_GlyphSlot slot, uint32_t codepoint) {
    int w = slot->bitmap.width;
    int h = slot->bitmap.rows;
    if (w == 0 || h == 0) return;
    if (next_x + w + 1 > atlas_width) {
        next_x = 0;
        next_y += row_height + 1;
        row_height = 0;
    }
    if (next_y + h + 1 > atlas_height) {
        // atlas full – real implementation would resize or allocate new atlas
        return;
    }
    // copy bitmap into atlas
    for (int y = 0; y < h; ++y) {
        int dst_y = next_y + y;
        int src_y = y;
        for (int x = 0; x < w; ++x) {
            int dst_x = next_x + x;
            atlas_data[dst_y * atlas_width + dst_x] = slot->bitmap.buffer[src_y * w + x];
        }
    }
    GlyphInfo info;
    info.u0 = (float)next_x / atlas_width;
    info.v0 = (float)next_y / atlas_height;
    info.u1 = (float)(next_x + w) / atlas_width;
    info.v1 = (float)(next_y + h) / atlas_height;
    info.advance = slot->advance.x >> 6;
    info.bearing_x = slot->bitmap_left;
    info.bearing_y = slot->bitmap_top;
    info.width = w;
    info.height = h;
    glyphs[codepoint] = info;
    next_x += w + 1;
    row_height = std::max(row_height, h);
}

const GlyphInfo* FontAtlas::get_glyph(uint32_t codepoint) {
    auto it = glyphs.find(codepoint);
    return (it != glyphs.end()) ? &it->second : nullptr;
}

// ============================================================================
// Label3D implementation
// ============================================================================
struct Label3D::Impl {
    std::string text;
    std::unique_ptr<FontAtlas> font;
    int font_size = 32;
    bool font_antialiased = true;

    float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    bool outline_enabled = false;
    float outline_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    float outline_size = 1.0f;
    bool shadow_enabled = false;
    float shadow_color[4] = {0.0f, 0.0f, 0.0f, 0.5f};
    double shadow_offset[2] = {0.01, -0.01};

    int alignment = 0;                // 0=left,1=center,2=right
    float line_spacing = 1.0f;
    float max_width = 0.0f;           // 0 = no wrap
    bool auto_align = true;

    bool billboard = false;
    int billboard_axis = 0;            // 0=free,1=Y only
    bool depth_test = true;
    bool transparent = true;

    bool cast_shadow = true;
    bool receive_shadow = false;
    int gi_mode = 2;                  // dynamic
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    // Mesh data
    std::vector<float> vertices;      // positions (x,y,z)
    std::vector<float> normals;
    std::vector<float> uvs;
    std::vector<int> indices;
    std::vector<float> vertex_colors; // RGBA per vertex
    bool mesh_dirty = true;
    int64_t mesh_rid = -1;
    int64_t instance_rid = -1;

    void rebuild_mesh();
    void add_quad(double x, double y, double w, double h, const GlyphInfo* glyph,
                  float r, float g, float b, float a);
};

void Label3D::Impl::add_quad(double x, double y, double w, double h, const GlyphInfo* glyph,
                             float r, float g, float b, float a) {
    int base = (int)vertices.size() / 3;
    // positions (z = 0 for flat plane)
    vertices.push_back((float)x); vertices.push_back((float)y); vertices.push_back(0.0f);
    vertices.push_back((float)x + w); vertices.push_back((float)y); vertices.push_back(0.0f);
    vertices.push_back((float)x + w); vertices.push_back((float)y + h); vertices.push_back(0.0f);
    vertices.push_back((float)x); vertices.push_back((float)y + h); vertices.push_back(0.0f);
    // normals (facing camera)
    normals.push_back(0); normals.push_back(0); normals.push_back(1);
    normals.push_back(0); normals.push_back(0); normals.push_back(1);
    normals.push_back(0); normals.push_back(0); normals.push_back(1);
    normals.push_back(0); normals.push_back(0); normals.push_back(1);
    // UVs
    uvs.push_back(glyph->u0); uvs.push_back(glyph->v0);
    uvs.push_back(glyph->u1); uvs.push_back(glyph->v0);
    uvs.push_back(glyph->u1); uvs.push_back(glyph->v1);
    uvs.push_back(glyph->u0); uvs.push_back(glyph->v1);
    // vertex colors
    for (int i=0;i<4;++i) {
        vertex_colors.push_back(r);
        vertex_colors.push_back(g);
        vertex_colors.push_back(b);
        vertex_colors.push_back(a);
    }
    // indices (two triangles)
    indices.push_back(base);
    indices.push_back(base+1);
    indices.push_back(base+2);
    indices.push_back(base);
    indices.push_back(base+2);
    indices.push_back(base+3);
}

void Label3D::Impl::rebuild_mesh() {
    vertices.clear();
    normals.clear();
    uvs.clear();
    indices.clear();
    vertex_colors.clear();
    if (!font || text.empty()) return;

    double cur_x = 0.0;
    double cur_y = 0.0;
    double line_height = font->get_line_height() / (double)font->get_size(); // normalize to units
    double scale = 1.0 / font->get_size();

    // For simplicity, no line wrapping yet
    for (size_t i = 0; i < text.size(); ++i) {
        uint32_t cp = (unsigned char)text[i];
        const GlyphInfo* g = font->get_glyph(cp);
        if (!g) continue;
        if (cp == '\n') {
            cur_x = 0.0;
            cur_y -= line_height * line_spacing;
            continue;
        }
        double xpos = cur_x + g->bearing_x * scale;
        double ypos = cur_y - (g->bearing_y - g->height) * scale;
        double w = g->width * scale;
        double h = g->height * scale;
        add_quad(xpos, ypos, w, h, g, color[0], color[1], color[2], color[3]);
        cur_x += g->advance * scale;
    }
    // Center alignment (if needed)
    if (alignment == 1 && !vertices.empty()) {
        // find bounds
        float min_x = vertices[0], max_x = vertices[0];
        for (size_t i = 0; i < vertices.size(); i += 3) {
            min_x = std::min(min_x, vertices[i]);
            max_x = std::max(max_x, vertices[i]);
        }
        float offset = - (min_x + max_x) * 0.5f;
        for (size_t i = 0; i < vertices.size(); i += 3) {
            vertices[i] += offset;
        }
    }
    mesh_dirty = false;
}

Label3D::Label3D() : pimpl(std::make_unique<Impl>()) {}
Label3D::~Label3D() = default;

void Label3D::set_text(const char* utf8_text) {
    pimpl->text = utf8_text ? utf8_text : "";
    pimpl->mesh_dirty = true;
    update_label();
}
const char* Label3D::get_text() const { return pimpl->text.c_str(); }

void Label3D::set_font(const char* font_path, int size) {
    pimpl->font = std::make_unique<FontAtlas>(font_path, size, pimpl->font_antialiased);
    pimpl->font_size = size;
    pimpl->mesh_dirty = true;
    update_label();
}
void Label3D::set_font_size(int size) {
    if (pimpl->font && pimpl->font_size != size) {
        pimpl->font = std::make_unique<FontAtlas>(pimpl->font->get_path(), size, pimpl->font_antialiased);
        pimpl->font_size = size;
        pimpl->mesh_dirty = true;
        update_label();
    }
}
int Label3D::get_font_size() const { return pimpl->font_size; }
void Label3D::set_font_antialiased(bool antialiased) { pimpl->font_antialiased = antialiased; }
bool Label3D::is_font_antialiased() const { return pimpl->font_antialiased; }

void Label3D::set_color(const float* rgba) { memcpy(pimpl->color, rgba, 4*sizeof(float)); pimpl->mesh_dirty = true; update_label(); }
void Label3D::get_color(float* out_rgba) const { memcpy(out_rgba, pimpl->color, 4*sizeof(float)); }
void Label3D::set_outline_enabled(bool enabled) { pimpl->outline_enabled = enabled; pimpl->mesh_dirty = true; update_label(); }
bool Label3D::is_outline_enabled() const { return pimpl->outline_enabled; }
void Label3D::set_outline_color(const float* rgba) { memcpy(pimpl->outline_color, rgba, 4*sizeof(float)); pimpl->mesh_dirty = true; update_label(); }
void Label3D::get_outline_color(float* out_rgba) const { memcpy(out_rgba, pimpl->outline_color, 4*sizeof(float)); }
void Label3D::set_outline_size(float size) { pimpl->outline_size = size; pimpl->mesh_dirty = true; update_label(); }
float Label3D::get_outline_size() const { return pimpl->outline_size; }
void Label3D::set_shadow_enabled(bool enabled) { pimpl->shadow_enabled = enabled; pimpl->mesh_dirty = true; update_label(); }
bool Label3D::is_shadow_enabled() const { return pimpl->shadow_enabled; }
void Label3D::set_shadow_color(const float* rgba) { memcpy(pimpl->shadow_color, rgba, 4*sizeof(float)); pimpl->mesh_dirty = true; update_label(); }
void Label3D::get_shadow_color(float* out_rgba) const { memcpy(out_rgba, pimpl->shadow_color, 4*sizeof(float)); }
void Label3D::set_shadow_offset(const double* offset) { memcpy(pimpl->shadow_offset, offset, 2*sizeof(double)); pimpl->mesh_dirty = true; update_label(); }
void Label3D::get_shadow_offset(double* out_offset) const { memcpy(out_offset, pimpl->shadow_offset, 2*sizeof(double)); }

void Label3D::set_alignment(int mode) { pimpl->alignment = mode; pimpl->mesh_dirty = true; update_label(); }
int Label3D::get_alignment() const { return pimpl->alignment; }
void Label3D::set_line_spacing(float spacing) { pimpl->line_spacing = spacing; pimpl->mesh_dirty = true; update_label(); }
float Label3D::get_line_spacing() const { return pimpl->line_spacing; }
void Label3D::set_width(float max_width) { pimpl->max_width = max_width; pimpl->mesh_dirty = true; update_label(); }
float Label3D::get_width() const { return pimpl->max_width; }
void Label3D::set_auto_align(bool auto_align) { pimpl->auto_align = auto_align; pimpl->mesh_dirty = true; update_label(); }
bool Label3D::get_auto_align() const { return pimpl->auto_align; }

void Label3D::set_billboard(bool enabled) { pimpl->billboard = enabled; pimpl->mesh_dirty = true; }
bool Label3D::is_billboard() const { return pimpl->billboard; }
void Label3D::set_billboard_axis(int axis) { pimpl->billboard_axis = axis; }
int Label3D::get_billboard_axis() const { return pimpl->billboard_axis; }

void Label3D::set_depth_test_enabled(bool enabled) { pimpl->depth_test = enabled; pimpl->mesh_dirty = true; }
bool Label3D::is_depth_test_enabled() const { return pimpl->depth_test; }
void Label3D::set_transparent(bool transparent) { pimpl->transparent = transparent; }
bool Label3D::is_transparent() const { return pimpl->transparent; }

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

void Label3D::update_label() {
    if (pimpl->mesh_dirty) pimpl->rebuild_mesh();
}

void Label3D::process(double delta) {
    GeometryInstance3D::process(delta);
    if (pimpl->mesh_dirty) update_label();
}

void Label3D::synchronize_render_server(double delta) {
    GeometryInstance3D::synchronize_render_server(delta);
    if (pimpl->mesh_dirty || pimpl->vertices.empty()) return;
    // Create or update mesh in RenderingServer
    if (pimpl->mesh_rid == -1) {
        // pimpl->mesh_rid = RenderingServer::mesh_create();
    }
    // Upload vertex buffers: positions, normals, UVs, colors
    // For bilboard, we would update the mesh's transform each frame
    if (pimpl->billboard) {
        Transform3D global = get_global_transform();
        // Compute camera facing rotation (simplified – would require camera reference)
        // For now, we just mark that instance transform needs update.
    }
    // Set instance flags: depth test, transparency, cast shadow, receive shadow, gi_mode
    // If emissive, add to GI system
    if (pimpl->emissive_intensity > 0.0f && pimpl->gi_mode > 0) {
        // register
    }
}

} // namespace lighting