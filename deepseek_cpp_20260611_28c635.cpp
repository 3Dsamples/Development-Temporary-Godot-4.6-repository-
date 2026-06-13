// label_3d.cpp
#include "label_3d.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <regex>

namespace lighting {

// ============================================================================
// Simple BMFont parser (ASCII .fnt files)
// ============================================================================
struct BMFontChar {
    int id;
    int x, y, width, height;
    int xoffset, yoffset;
    int xadvance;
};

struct BMFontInfo {
    int lineHeight;
    int base;
    int scaleW, scaleH;
    std::unordered_map<int, BMFontChar> chars;
    std::vector<unsigned char> bitmap; // RGBA or monochrome
    int bitmap_width, bitmap_height;
};

static bool parse_bmfont(const std::string& path, BMFontInfo& out) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    std::string line;
    std::regex common_regex("common lineHeight=(\\d+) base=(\\d+) scaleW=(\\d+) scaleH=(\\d+)");
    std::regex char_regex("char id=(\\d+) x=(\\d+) y=(\\d+) width=(\\d+) height=(\\d+) xoffset=(\\d+) yoffset=(\\d+) xadvance=(\\d+)");
    bool reading_chars = false;
    while (std::getline(file, line)) {
        if (line.find("common") != std::string::npos) {
            std::smatch m;
            if (std::regex_search(line, m, common_regex)) {
                out.lineHeight = std::stoi(m[1]);
                out.base = std::stoi(m[2]);
                out.scaleW = std::stoi(m[3]);
                out.scaleH = std::stoi(m[4]);
            }
        } else if (line.find("char ") == 0) {
            std::smatch m;
            if (std::regex_search(line, m, char_regex)) {
                BMFontChar ch;
                ch.id = std::stoi(m[1]);
                ch.x = std::stoi(m[2]);
                ch.y = std::stoi(m[3]);
                ch.width = std::stoi(m[4]);
                ch.height = std::stoi(m[5]);
                ch.xoffset = std::stoi(m[6]);
                ch.yoffset = std::stoi(m[7]);
                ch.xadvance = std::stoi(m[8]);
                out.chars[ch.id] = ch;
            }
        } else if (line.find("page id=") != std::string::npos) {
            // extract texture file name
            size_t file_start = line.find("file=\"") + 6;
            size_t file_end = line.find("\"", file_start);
            std::string tex_file = line.substr(file_start, file_end - file_start);
            // load texture (simplified: in real engine, load image)
            // For this demo, we assume texture is already loaded.
        }
    }
    return !out.chars.empty();
}

// ============================================================================
// Label3D implementation
// ============================================================================
struct Label3D::Impl {
    std::string text = "Label";
    std::string font_path;
    int font_size = 32;          // for TTF / SDF
    int outline_size = 0;
    int h_align = 0;             // 0=left
    int v_align = 0;             // 0=top
    float width = 0.0f;          // wrap width
    float line_spacing = 1.0f;
    float color[4] = {1.0f,1.0f,1.0f,1.0f};
    float outline_color[4] = {0.0f,0.0f,0.0f,1.0f};
    std::string material_path;
    bool billboard_enabled = false;
    int billboard_axis = 0;      // 0=all axes
    double pixel_size = 0.01;    // world units per font pixel

    bool cast_shadow = true;
    bool receive_shadow = false;
    int gi_mode = 2;             // dynamic by default
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    // Font data (simplified: fallback to generating a simple mesh)
    BMFontInfo bmfont;
    bool font_loaded = false;

    // Generated mesh
    std::vector<double> vertices;  // each vertex: pos x,y,z, uv u,v, color r,g,b,a
    std::vector<int> indices;
    int64_t mesh_rid = -1;
    int64_t instance_rid = -1;
    bool mesh_dirty = true;

    // Rebuild text mesh from current parameters
    void rebuild_mesh();
    void load_font();
};

Label3D::Label3D() : pimpl(std::make_unique<Impl>()) {
    pimpl->load_font();
}
Label3D::~Label3D() = default;

void Label3D::set_text(const char* text) {
    pimpl->text = text ? text : "";
    pimpl->mesh_dirty = true;
    update_label();
}
const char* Label3D::get_text() const { return pimpl->text.c_str(); }
void Label3D::set_font(const char* font_path) {
    pimpl->font_path = font_path ? font_path : "";
    pimpl->load_font();
    pimpl->mesh_dirty = true;
}
const char* Label3D::get_font() const { return pimpl->font_path.c_str(); }
void Label3D::set_font_size(int size_pixels) { pimpl->font_size = size_pixels; pimpl->mesh_dirty = true; update_label(); }
int Label3D::get_font_size() const { return pimpl->font_size; }
void Label3D::set_outline_size(int pixels) { pimpl->outline_size = pixels; pimpl->mesh_dirty = true; update_label(); }
int Label3D::get_outline_size() const { return pimpl->outline_size; }
void Label3D::set_horizontal_alignment(int align) { pimpl->h_align = align; pimpl->mesh_dirty = true; update_label(); }
int Label3D::get_horizontal_alignment() const { return pimpl->h_align; }
void Label3D::set_vertical_alignment(int align) { pimpl->v_align = align; pimpl->mesh_dirty = true; update_label(); }
int Label3D::get_vertical_alignment() const { return pimpl->v_align; }
void Label3D::set_width(float width) { pimpl->width = width; pimpl->mesh_dirty = true; update_label(); }
float Label3D::get_width() const { return pimpl->width; }
void Label3D::set_line_spacing(float spacing) { pimpl->line_spacing = spacing; pimpl->mesh_dirty = true; update_label(); }
float Label3D::get_line_spacing() const { return pimpl->line_spacing; }

void Label3D::set_color(float r, float g, float b, float a) {
    pimpl->color[0]=r; pimpl->color[1]=g; pimpl->color[2]=b; pimpl->color[3]=a;
    pimpl->mesh_dirty = true;
}
void Label3D::get_color(float* out_rgba) const { memcpy(out_rgba, pimpl->color, 4*sizeof(float)); }
void Label3D::set_outline_color(float r, float g, float b, float a) {
    pimpl->outline_color[0]=r; pimpl->outline_color[1]=g; pimpl->outline_color[2]=b; pimpl->outline_color[3]=a;
    pimpl->mesh_dirty = true;
}
void Label3D::get_outline_color(float* out_rgba) const { memcpy(out_rgba, pimpl->outline_color, 4*sizeof(float)); }
void Label3D::set_material(const char* material_path) { pimpl->material_path = material_path ? material_path : ""; }
const char* Label3D::get_material() const { return pimpl->material_path.c_str(); }
void Label3D::set_billboard_enabled(bool enabled) { pimpl->billboard_enabled = enabled; }
bool Label3D::is_billboard_enabled() const { return pimpl->billboard_enabled; }
void Label3D::set_billboard_axis(int axis) { pimpl->billboard_axis = axis; }
int Label3D::get_billboard_axis() const { return pimpl->billboard_axis; }
void Label3D::set_pixel_size(double size) { pimpl->pixel_size = size; pimpl->mesh_dirty = true; update_label(); }
double Label3D::get_pixel_size() const { return pimpl->pixel_size; }
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
void Label3D::update_label() { pimpl->rebuild_mesh(); }

void Label3D::Impl::load_font() {
    font_loaded = false;
    if (!font_path.empty() && std::filesystem::exists(font_path)) {
        if (parse_bmfont(font_path, bmfont)) {
            font_loaded = true;
        }
    }
    if (!font_loaded) {
        // fallback: create a simple monospace font (just rectangles)
        // generate default font data (ASCII 32-126)
        bmfont.lineHeight = font_size;
        bmfont.base = font_size;
        bmfont.scaleW = font_size*16;
        bmfont.scaleH = font_size;
        for (int c = 32; c <= 126; ++c) {
            BMFontChar ch;
            ch.id = c;
            ch.width = font_size * 6 / 10; // monospace width
            ch.height = font_size;
            ch.xoffset = 0;
            ch.yoffset = 0;
            ch.xadvance = ch.width;
            bmfont.chars[c] = ch;
        }
        font_loaded = true;
    }
}

void Label3D::Impl::rebuild_mesh() {
    if (!font_loaded) load_font();
    vertices.clear();
    indices.clear();
    // Layout lines with wrapping
    std::vector<std::string> lines;
    std::string current_line;
    float max_line_width = 0.0f;
    // simplified: no word wrapping, just split by newline
    size_t start = 0, end;
    while ((end = text.find('\n', start)) != std::string::npos) {
        lines.push_back(text.substr(start, end - start));
        start = end + 1;
    }
    lines.push_back(text.substr(start));

    float total_height = lines.size() * bmfont.lineHeight * line_spacing;
    float start_y = 0.0f;
    if (v_align == 1) start_y = -total_height * 0.5f;
    else if (v_align == 2) start_y = -total_height;

    float pen_x = 0.0f, pen_y = start_y;
    int vertex_offset = 0;
    for (size_t li = 0; li < lines.size(); ++li) {
        const std::string& line = lines[li];
        float line_width = 0.0f;
        for (char ch : line) {
            auto it = bmfont.chars.find((int)ch);
            if (it != bmfont.chars.end()) {
                line_width += it->second.xadvance;
            }
        }
        float start_x = 0.0f;
        if (h_align == 1) start_x = -line_width * 0.5f;
        else if (h_align == 2) start_x = -line_width;
        pen_x = start_x;
        for (char ch : line) {
            auto it = bmfont.chars.find((int)ch);
            if (it == bmfont.chars.end()) continue;
            const BMFontChar& c = it->second;
            float x0 = pen_x + c.xoffset;
            float y0 = pen_y + c.yoffset;
            float x1 = x0 + c.width;
            float y1 = y0 + c.height;
            // map to world size (pixel_size)
            double scale = pixel_size;
            // vertices (order: bottom-left, bottom-right, top-right, top-left)
            double vx0 = x0 * scale;
            double vy0 = y0 * scale;
            double vx1 = x1 * scale;
            double vy1 = y1 * scale;
            // UVs: from font atlas (if we had one, using dummy)
            float u0 = 0.0f, u1 = 1.0f, v0 = 0.0f, v1 = 1.0f;
            // color per vertex (could be varied per character)
            int base = (int)vertices.size() / 6; // each vertex stores pos (3) + uv (2) + color (4) = 9 floats? We'll store compact.
            // We'll store interleaved: pos3, uv2, color4
            // For simplicity, we'll push 9 floats per vertex.
            // bottom-left
            vertices.push_back(vx0); vertices.push_back(vy0); vertices.push_back(0.0); // pos
            vertices.push_back(u0); vertices.push_back(v1); // uv
            vertices.push_back(color[0]); vertices.push_back(color[1]); vertices.push_back(color[2]); vertices.push_back(color[3]);
            // bottom-right
            vertices.push_back(vx1); vertices.push_back(vy0); vertices.push_back(0.0);
            vertices.push_back(u1); vertices.push_back(v1);
            vertices.push_back(color[0]); vertices.push_back(color[1]); vertices.push_back(color[2]); vertices.push_back(color[3]);
            // top-right
            vertices.push_back(vx1); vertices.push_back(vy1); vertices.push_back(0.0);
            vertices.push_back(u1); vertices.push_back(v0);
            vertices.push_back(color[0]); vertices.push_back(color[1]); vertices.push_back(color[2]); vertices.push_back(color[3]);
            // top-left
            vertices.push_back(vx0); vertices.push_back(vy1); vertices.push_back(0.0);
            vertices.push_back(u0); vertices.push_back(v0);
            vertices.push_back(color[0]); vertices.push_back(color[1]); vertices.push_back(color[2]); vertices.push_back(color[3]);
            indices.push_back(base); indices.push_back(base+1); indices.push_back(base+2);
            indices.push_back(base); indices.push_back(base+2); indices.push_back(base+3);
            vertex_offset += 4;
            pen_x += c.xadvance;
        }
        pen_y -= bmfont.lineHeight * line_spacing;
    }

    mesh_dirty = false;
    // In real engine, upload to GPU via RenderingServer
}

void Label3D::process(double delta) {
    GeometryInstance3D::process(delta);
    if (pimpl->mesh_dirty) {
        update_label();
    }
}

void Label3D::synchronize_render_server(double delta) {
    GeometryInstance3D::synchronize_render_server(delta);
    if (pimpl->mesh_dirty) {
        // create or update mesh
        if (pimpl->mesh_rid == -1) {
            // pimpl->mesh_rid = RenderingServer::mesh_create();
            // pimpl->instance_rid = RenderingServer::instance_create(pimpl->mesh_rid);
        }
        // upload vertex/index data
        // set material override if any
        // set shadow/gi flags
        // set billboard mode if enabled
        pimpl->mesh_dirty = false;
    }
    if (pimpl->billboard_enabled) {
        // Override transform to face camera (done in rendering server or here)
        // For simplicity, we let render server handle via instance flags.
    }
    // Emissive GI contribution
    if (pimpl->emissive_intensity > 0.0f && pimpl->gi_mode > 0) {
        // inject into GI system
    }
    // Compute bounding box from vertices
    if (!pimpl->vertices.empty()) {
        double min_x = pimpl->vertices[0], max_x = pimpl->vertices[0];
        double min_y = pimpl->vertices[1], max_y = pimpl->vertices[1];
        double min_z = pimpl->vertices[2], max_z = pimpl->vertices[2];
        for (size_t i = 0; i < pimpl->vertices.size(); i += 9) {
            double x = pimpl->vertices[i];
            double y = pimpl->vertices[i+1];
            double z = pimpl->vertices[i+2];
            min_x = std::min(min_x, x); max_x = std::max(max_x, x);
            min_y = std::min(min_y, y); max_y = std::max(max_y, y);
            min_z = std::min(min_z, z); max_z = std::max(max_z, z);
        }
        set_aabb(&min_x, &max_x);
        double dx = max_x-min_x, dy = max_y-min_y, dz = max_z-min_z;
        set_bounding_sphere_radius(std::sqrt(dx*dx+dy*dy+dz*dz)*0.5);
    }
}

} // namespace lighting