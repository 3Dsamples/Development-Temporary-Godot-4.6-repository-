// include/xtu/godot/xtext_server.hpp
// xtensor-unified - Text Server and Font Rendering for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XTEXT_SERVER_HPP
#define XTU_GODOT_XTEXT_SERVER_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xrenderingserver.hpp"

#ifdef XTU_USE_HARFBUZZ
#include <hb.h>
#include <hb-ft.h>
#endif

#ifdef XTU_USE_FREETYPE
#include <ft2build.h>
#include FT_FREETYPE_H
#endif

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class TextServer;
class TextServerAdvanced;
class Font;
class DynamicFont;

// #############################################################################
// Font hinting mode
// #############################################################################
enum class FontHinting : uint8_t {
    HINTING_NONE = 0,
    HINTING_SLIGHT = 1,
    HINTING_NORMAL = 2,
    HINTING_FULL = 3
};

// #############################################################################
// Font subpixel positioning
// #############################################################################
enum class FontSubpixelPositioning : uint8_t {
    SUBPIXEL_DISABLED = 0,
    SUBPIXEL_AUTO = 1,
    SUBPIXEL_HALF = 2,
    SUBPIXEL_ONE_QUARTER = 3
};

// #############################################################################
// Text direction
// #############################################################################
enum class TextDirection : uint8_t {
    DIRECTION_AUTO = 0,
    DIRECTION_LTR = 1,
    DIRECTION_RTL = 2
};

// #############################################################################
// Text alignment
// #############################################################################
enum class TextAlignment : uint8_t {
    ALIGNMENT_LEFT = 0,
    ALIGNMENT_CENTER = 1,
    ALIGNMENT_RIGHT = 2,
    ALIGNMENT_FILL = 3
};

// #############################################################################
// Font features
// #############################################################################
enum class FontFeature : uint32_t {
    FEATURE_NONE = 0,
    FEATURE_KERNING = 1 << 0,
    FEATURE_LIGATURES = 1 << 1,
    FEATURE_ALTERNATES = 1 << 2
};

// #############################################################################
// Glyph info
// #############################################################################
struct GlyphInfo {
    uint32_t index = 0;
    vec2f advance;
    vec2f offset;
    vec2f size;
    vec2f uv_tl;
    vec2f uv_br;
    int atlas_id = 0;
    bool valid = false;
};

// #############################################################################
// Text line
// #############################################################################
struct TextLine {
    std::vector<GlyphInfo> glyphs;
    float width = 0.0f;
    float ascent = 0.0f;
    float descent = 0.0f;
    float leading = 0.0f;
    TextDirection direction = TextDirection::DIRECTION_LTR;
};

// #############################################################################
// TextServer - Base text server
// #############################################################################
class TextServer : public Object {
    XTU_GODOT_REGISTER_CLASS(TextServer, Object)

public:
    static StringName get_class_static() { return StringName("TextServer"); }

    virtual Ref<Font> create_font() = 0;
    virtual std::vector<TextLine> shape_text(const String& text, const Ref<Font>& font, 
                                              int size, TextDirection direction = TextDirection::DIRECTION_AUTO) = 0;
    virtual vec2f get_string_size(const String& text, const Ref<Font>& font, int size) = 0;
    virtual float get_ascent(const Ref<Font>& font, int size) = 0;
    virtual float get_descent(const Ref<Font>& font, int size) = 0;
    virtual void draw_string(RID canvas_item, const vec2f& pos, const String& text,
                             const Ref<Font>& font, int size, const Color& color) = 0;
};

// #############################################################################
// TextServerAdvanced - Advanced text server with HarfBuzz/FreeType
// #############################################################################
class TextServerAdvanced : public TextServer {
    XTU_GODOT_REGISTER_CLASS(TextServerAdvanced, TextServer)

private:
    static TextServerAdvanced* s_singleton;
#ifdef XTU_USE_FREETYPE
    FT_Library m_ft_library = nullptr;
#endif
    std::unordered_map<String, std::vector<uint8_t>> m_font_cache;
    std::mutex m_mutex;

public:
    static TextServerAdvanced* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("TextServerAdvanced"); }

    TextServerAdvanced() {
        s_singleton = this;
#ifdef XTU_USE_FREETYPE
        FT_Init_FreeType(&m_ft_library);
#endif
    }

    ~TextServerAdvanced() {
#ifdef XTU_USE_FREETYPE
        if (m_ft_library) FT_Done_FreeType(m_ft_library);
#endif
        s_singleton = nullptr;
    }

    Ref<Font> create_font() override {
        Ref<DynamicFont> font;
        font.instance();
        font->set_text_server(this);
        return font;
    }

    std::vector<TextLine> shape_text(const String& text, const Ref<Font>& font,
                                      int size, TextDirection direction) override {
        std::vector<TextLine> lines;
#ifdef XTU_USE_HARFBUZZ
        DynamicFont* dyn_font = dynamic_cast<DynamicFont*>(font.ptr());
        if (!dyn_font) return lines;

        hb_font_t* hb_font = dyn_font->get_hb_font(size);
        if (!hb_font) return lines;

        hb_buffer_t* buffer = hb_buffer_create();
        hb_buffer_add_utf8(buffer, text.utf8(), -1, 0, -1);
        hb_buffer_set_direction(buffer, direction == TextDirection::DIRECTION_RTL ? 
                                HB_DIRECTION_RTL : HB_DIRECTION_LTR);
        hb_buffer_set_script(buffer, HB_SCRIPT_COMMON);
        hb_buffer_set_language(buffer, hb_language_get_default());

        hb_shape(hb_font, buffer, nullptr, 0);

        unsigned int glyph_count;
        hb_glyph_info_t* glyph_info = hb_buffer_get_glyph_infos(buffer, &glyph_count);
        hb_glyph_position_t* glyph_pos = hb_buffer_get_glyph_positions(buffer, &glyph_count);

        TextLine line;
        line.direction = direction;
        float x = 0.0f;
        float y = 0.0f;

        for (unsigned int i = 0; i < glyph_count; ++i) {
            GlyphInfo glyph;
            glyph.index = glyph_info[i].codepoint;
            glyph.advance = vec2f(glyph_pos[i].x_advance / 64.0f, glyph_pos[i].y_advance / 64.0f);
            glyph.offset = vec2f(glyph_pos[i].x_offset / 64.0f, -glyph_pos[i].y_offset / 64.0f);
            glyph.valid = true;

            dyn_font->get_glyph_metrics(glyph.index, size, glyph);

            line.glyphs.push_back(glyph);
            line.width += glyph.advance.x();
        }

        line.ascent = dyn_font->get_ascent(size);
        line.descent = dyn_font->get_descent(size);
        line.leading = 0.0f;

        lines.push_back(line);
        hb_buffer_destroy(buffer);
#endif
        return lines;
    }

    vec2f get_string_size(const String& text, const Ref<Font>& font, int size) override {
        auto lines = shape_text(text, font, size);
        float width = 0.0f;
        float height = 0.0f;
        for (const auto& line : lines) {
            width = std::max(width, line.width);
            height += line.ascent + line.descent;
        }
        return vec2f(width, height);
    }

    float get_ascent(const Ref<Font>& font, int size) override {
        if (auto* dyn_font = dynamic_cast<DynamicFont*>(font.ptr())) {
            return dyn_font->get_ascent(size);
        }
        return size * 0.8f;
    }

    float get_descent(const Ref<Font>& font, int size) override {
        if (auto* dyn_font = dynamic_cast<DynamicFont*>(font.ptr())) {
            return dyn_font->get_descent(size);
        }
        return size * 0.2f;
    }

    void draw_string(RID canvas_item, const vec2f& pos, const String& text,
                     const Ref<Font>& font, int size, const Color& color) override {
        auto lines = shape_text(text, font, size);
        vec2f cursor = pos;
        for (const auto& line : lines) {
            cursor.y() += line.ascent;
            for (const auto& glyph : line.glyphs) {
                if (!glyph.valid) continue;
                vec2f glyph_pos = cursor + glyph.offset;
                Rect2 src_rect(glyph.uv_tl, glyph.uv_br - glyph.uv_tl);
                Rect2 dst_rect(glyph_pos, glyph.size);
                RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(
                    canvas_item, dst_rect, font->get_texture_rid(glyph.atlas_id), src_rect, color);
                cursor.x() += glyph.advance.x();
            }
            cursor.x() = pos.x();
            cursor.y() += line.descent;
        }
    }

#ifdef XTU_USE_FREETYPE
    FT_Library get_ft_library() const { return m_ft_library; }
#endif

    std::vector<uint8_t> load_font_data(const String& path) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_font_cache.find(path);
        if (it != m_font_cache.end()) {
            return it->second;
        }
        Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
        if (!file.is_valid()) return {};
        std::vector<uint8_t> data = file->get_buffer(file->get_length());
        m_font_cache[path] = data;
        return data;
    }
};

// #############################################################################
// Font - Base font resource
// #############################################################################
class Font : public Resource {
    XTU_GODOT_REGISTER_CLASS(Font, Resource)

public:
    static StringName get_class_static() { return StringName("Font"); }

    virtual void set_font_path(const String& path) = 0;
    virtual String get_font_path() const = 0;
    virtual void set_size(int size) = 0;
    virtual int get_size() const = 0;
    virtual RID get_texture_rid(int atlas_id) const = 0;
    virtual Ref<Texture2D> get_texture(int atlas_id) const = 0;
    virtual std::vector<TextLine> shape_text(const String& text, TextDirection direction = TextDirection::DIRECTION_AUTO) = 0;
};

// #############################################################################
// DynamicFont - Runtime font with atlas
// #############################################################################
class DynamicFont : public Font {
    XTU_GODOT_REGISTER_CLASS(DynamicFont, Font)

private:
    String m_font_path;
    int m_size = 16;
    FontHinting m_hinting = FontHinting::HINTING_NORMAL;
    FontSubpixelPositioning m_subpixel = FontSubpixelPositioning::SUBPIXEL_AUTO;
    uint32_t m_features = static_cast<uint32_t>(FontFeature::FEATURE_KERNING) |
                         static_cast<uint32_t>(FontFeature::FEATURE_LIGATURES);
    TextServerAdvanced* m_text_server = nullptr;
#ifdef XTU_USE_FREETYPE
    FT_Face m_face = nullptr;
    std::vector<uint8_t> m_font_data;
#endif
#ifdef XTU_USE_HARFBUZZ
    hb_font_t* m_hb_font = nullptr;
    int m_hb_font_size = 0;
#endif
    std::vector<Ref<Texture2D>> m_atlases;
    std::unordered_map<uint32_t, GlyphInfo> m_glyph_cache;
    int m_atlas_size = 512;
    int m_current_atlas = 0;
    vec2i m_atlas_cursor;
    int m_atlas_row_height = 0;
    std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("DynamicFont"); }

    DynamicFont() {
        m_text_server = TextServerAdvanced::get_singleton();
    }

    ~DynamicFont() {
        cleanup();
    }

    void set_text_server(TextServerAdvanced* server) { m_text_server = server; }

    void set_font_path(const String& path) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_font_path == path) return;
        m_font_path = path;
        load_font();
    }

    String get_font_path() const override { return m_font_path; }

    void set_size(int size) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_size = size;
#ifdef XTU_USE_HARFBUZZ
        if (m_hb_font) {
            hb_font_destroy(m_hb_font);
            m_hb_font = nullptr;
        }
#endif
    }

    int get_size() const override { return m_size; }

    void set_hinting(FontHinting hinting) { m_hinting = hinting; }
    FontHinting get_hinting() const { return m_hinting; }

    void set_subpixel_positioning(FontSubpixelPositioning subpixel) { m_subpixel = subpixel; }
    FontSubpixelPositioning get_subpixel_positioning() const { return m_subpixel; }

    void set_font_feature(FontFeature feature, bool enable) {
        if (enable) m_features |= static_cast<uint32_t>(feature);
        else m_features &= ~static_cast<uint32_t>(feature);
    }

    RID get_texture_rid(int atlas_id) const override {
        if (atlas_id >= 0 && atlas_id < static_cast<int>(m_atlases.size())) {
            return m_atlases[atlas_id]->get_rid();
        }
        return RID();
    }

    Ref<Texture2D> get_texture(int atlas_id) const override {
        if (atlas_id >= 0 && atlas_id < static_cast<int>(m_atlases.size())) {
            return m_atlases[atlas_id];
        }
        return Ref<Texture2D>();
    }

    std::vector<TextLine> shape_text(const String& text, TextDirection direction) override {
        return m_text_server->shape_text(text, Ref<Font>(this), m_size, direction);
    }

    float get_ascent(int size) const {
#ifdef XTU_USE_FREETYPE
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_face) {
            FT_Set_Pixel_Sizes(m_face, 0, size);
            return m_face->ascender * size / m_face->units_per_EM;
        }
#endif
        return size * 0.8f;
    }

    float get_descent(int size) const {
#ifdef XTU_USE_FREETYPE
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_face) {
            FT_Set_Pixel_Sizes(m_face, 0, size);
            return -m_face->descender * size / m_face->units_per_EM;
        }
#endif
        return size * 0.2f;
    }

    void get_glyph_metrics(uint32_t index, int size, GlyphInfo& glyph) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_glyph_cache.find(index);
        if (it != m_glyph_cache.end()) {
            glyph = it->second;
            return;
        }
#ifdef XTU_USE_FREETYPE
        if (m_face) {
            FT_Set_Pixel_Sizes(m_face, 0, size);
            FT_Load_Glyph(m_face, index, FT_LOAD_DEFAULT);
            FT_Render_Glyph(m_face->glyph, FT_RENDER_MODE_NORMAL);
            
            FT_Bitmap& bitmap = m_face->glyph->bitmap;
            if (bitmap.width > 0 && bitmap.rows > 0) {
                glyph.size = vec2f(bitmap.width, bitmap.rows);
                glyph.offset = vec2f(m_face->glyph->bitmap_left, -m_face->glyph->bitmap_top);
                
                // Pack into atlas
                pack_glyph(index, bitmap, glyph);
            }
            m_glyph_cache[index] = glyph;
        }
#endif
    }

#ifdef XTU_USE_HARFBUZZ
    hb_font_t* get_hb_font(int size) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_hb_font && m_hb_font_size == size) {
            return m_hb_font;
        }
        if (m_hb_font) {
            hb_font_destroy(m_hb_font);
        }
        if (m_face) {
            m_hb_font = hb_ft_font_create_referenced(m_face);
            hb_font_set_scale(m_hb_font, size * 64, size * 64);
            m_hb_font_size = size;
        }
        return m_hb_font;
    }
#endif

private:
    void load_font() {
        cleanup();
        if (m_font_path.empty() || !m_text_server) return;
        
        std::vector<uint8_t> data = m_text_server->load_font_data(m_font_path);
        if (data.empty()) return;
        m_font_data = std::move(data);
        
#ifdef XTU_USE_FREETYPE
        FT_New_Memory_Face(m_text_server->get_ft_library(), m_font_data.data(),
                           static_cast<FT_Long>(m_font_data.size()), 0, &m_face);
#endif
    }

    void cleanup() {
#ifdef XTU_USE_HARFBUZZ
        if (m_hb_font) {
            hb_font_destroy(m_hb_font);
            m_hb_font = nullptr;
        }
#endif
#ifdef XTU_USE_FREETYPE
        if (m_face) {
            FT_Done_Face(m_face);
            m_face = nullptr;
        }
#endif
        m_glyph_cache.clear();
        m_atlases.clear();
        m_current_atlas = 0;
        m_atlas_cursor = vec2i(0, 0);
        m_atlas_row_height = 0;
    }

    void pack_glyph(uint32_t index, const FT_Bitmap& bitmap, GlyphInfo& glyph) {
        int width = static_cast<int>(bitmap.width);
        int height = static_cast<int>(bitmap.rows);
        
        if (m_current_atlas >= static_cast<int>(m_atlases.size())) {
            create_new_atlas();
        }
        
        if (m_atlas_cursor.x() + width + 1 > m_atlas_size) {
            m_atlas_cursor.x() = 0;
            m_atlas_cursor.y() += m_atlas_row_height + 1;
            m_atlas_row_height = 0;
        }
        
        if (m_atlas_cursor.y() + height + 1 > m_atlas_size) {
            ++m_current_atlas;
            m_atlas_cursor = vec2i(0, 0);
            m_atlas_row_height = 0;
            if (m_current_atlas >= static_cast<int>(m_atlases.size())) {
                create_new_atlas();
            }
        }
        
        glyph.atlas_id = m_current_atlas;
        glyph.uv_tl = vec2f(
            static_cast<float>(m_atlas_cursor.x()) / m_atlas_size,
            static_cast<float>(m_atlas_cursor.y()) / m_atlas_size
        );
        glyph.uv_br = vec2f(
            static_cast<float>(m_atlas_cursor.x() + width) / m_atlas_size,
            static_cast<float>(m_atlas_cursor.y() + height) / m_atlas_size
        );
        
        // Update atlas texture
        Ref<Texture2D> atlas = m_atlases[m_current_atlas];
        std::vector<uint8_t> pixels(width * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                uint8_t val = bitmap.buffer[y * bitmap.pitch + x];
                pixels[y * width + x] = val;
            }
        }
        atlas->update_region(m_atlas_cursor.x(), m_atlas_cursor.y(), width, height, pixels);
        
        m_atlas_cursor.x() += width + 1;
        m_atlas_row_height = std::max(m_atlas_row_height, height);
    }

    void create_new_atlas() {
        Ref<Texture2D> atlas;
        atlas.instance();
        atlas->create(m_atlas_size, m_atlas_size, Image::FORMAT_R8, Texture2D::FLAG_FILTER);
        m_atlases.push_back(atlas);
    }
};

} // namespace godot

// Bring into main namespace
using godot::TextServer;
using godot::TextServerAdvanced;
using godot::Font;
using godot::DynamicFont;
using godot::FontHinting;
using godot::FontSubpixelPositioning;
using godot::TextDirection;
using godot::TextAlignment;
using godot::FontFeature;
using godot::GlyphInfo;
using godot::TextLine;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XTEXT_SERVER_HPP