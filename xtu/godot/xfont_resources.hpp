// include/xtu/godot/xfont_resources.hpp
// xtensor-unified - Font resources for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XFONT_RESOURCES_HPP
#define XTU_GODOT_XFONT_RESOURCES_HPP

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
#include "xtu/godot/xcore.hpp"
#include "xtu/godot/xrenderingserver.hpp"
#include "xtu/godot/xtext_server.hpp"

#ifdef XTU_USE_FREETYPE
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H
#include FT_BITMAP_H
#include FT_STROKER_H
#endif

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class Font;
class FontFile;
class DynamicFont;
class BitmapFont;
class SystemFont;
class FontVariation;

// #############################################################################
// Font spacing types
// #############################################################################
enum class FontSpacing : uint8_t {
    SPACING_GLYPH = 0,
    SPACING_SPACE = 1,
    SPACING_TOP = 2,
    SPACING_BOTTOM = 3
};

// #############################################################################
// Font features
// #############################################################################
struct FontFeatureTag {
    String tag;
    uint32_t value = 1;
};

// #############################################################################
// Glyph metrics
// #############################################################################
struct GlyphMetrics {
    vec2f advance;
    vec2f offset;
    vec2f size;
    vec2f uv_tl;
    vec2f uv_br;
    float ascent = 0.0f;
    float descent = 0.0f;
    int atlas_id = 0;
    bool valid = false;
};

// #############################################################################
// Font - Base class for all fonts
// #############################################################################
class Font : public Resource {
    XTU_GODOT_REGISTER_CLASS(Font, Resource)

protected:
    std::vector<Ref<Font>> m_fallbacks;
    std::unordered_map<FontSpacing, int> m_spacings;

public:
    static StringName get_class_static() { return StringName("Font"); }

    virtual void set_fallback(int idx, const Ref<Font>& fallback) {
        if (idx >= 0) {
            if (idx >= static_cast<int>(m_fallbacks.size())) {
                m_fallbacks.resize(idx + 1);
            }
            m_fallbacks[idx] = fallback;
        }
    }

    virtual Ref<Font> get_fallback(int idx) const {
        return idx >= 0 && idx < static_cast<int>(m_fallbacks.size()) ? m_fallbacks[idx] : Ref<Font>();
    }

    virtual int get_fallback_count() const {
        return static_cast<int>(m_fallbacks.size());
    }

    virtual void remove_fallback(int idx) {
        if (idx >= 0 && idx < static_cast<int>(m_fallbacks.size())) {
            m_fallbacks.erase(m_fallbacks.begin() + idx);
        }
    }

    virtual void set_spacing(FontSpacing spacing, int value) {
        m_spacings[spacing] = value;
    }

    virtual int get_spacing(FontSpacing spacing) const {
        auto it = m_spacings.find(spacing);
        return it != m_spacings.end() ? it->second : 0;
    }

    virtual float get_height(int size) const = 0;
    virtual float get_ascent(int size) const = 0;
    virtual float get_descent(int size) const = 0;
    virtual float get_underline_position(int size) const = 0;
    virtual float get_underline_thickness(int size) const = 0;

    virtual Ref<Texture2D> get_texture(int atlas_idx) const = 0;
    virtual RID get_rid() const = 0;

    virtual bool has_char(char32_t c) const = 0;
    virtual GlyphMetrics get_char_metrics(char32_t c, int size) const = 0;
    virtual vec2f get_string_size(const String& text, int size,
                                   TextDirection direction = TextDirection::DIRECTION_AUTO) const = 0;

    virtual void draw_string(RID canvas_item, const vec2f& pos, const String& text,
                             int size, const Color& color) const = 0;
};

// #############################################################################
// FontFile - TTF/OTF/WOFF font file loader
// #############################################################################
class FontFile : public Font {
    XTU_GODOT_REGISTER_CLASS(FontFile, Font)

private:
    std::vector<uint8_t> m_data;
    String m_font_path;
    String m_font_name;
    String m_font_style_name;
    int m_face_index = 0;
    FontHinting m_hinting = FontHinting::HINTING_NORMAL;
    FontSubpixelPositioning m_subpixel = FontSubpixelPositioning::SUBPIXEL_AUTO;
    float m_msdf_pixel_range = 8.0f;
    int m_msdf_size = 48;
    bool m_antialiased = true;
    bool m_fixed_size = false;
    int m_fixed_size_value = 0;
    bool m_force_autohinter = false;
    bool m_allow_system_fallback = true;
    bool m_oversampling = 1.0f;

    std::unordered_map<int, float> m_cache_ascent;
    std::unordered_map<int, float> m_cache_descent;
    std::unordered_map<int, float> m_cache_underline_position;
    std::unordered_map<int, float> m_cache_underline_thickness;

    std::vector<Ref<Texture2D>> m_atlases;
    std::unordered_map<uint64_t, GlyphMetrics> m_glyph_cache;
    int m_atlas_size = 512;
    int m_current_atlas = 0;
    vec2i m_atlas_cursor;
    int m_atlas_row_height = 0;

#ifdef XTU_USE_FREETYPE
    FT_Face m_face = nullptr;
#endif
    mutable std::mutex m_mutex;

    uint64_t make_glyph_key(char32_t c, int size) const {
        return (static_cast<uint64_t>(c) << 32) | static_cast<uint32_t>(size);
    }

    bool ensure_face() const {
#ifdef XTU_USE_FREETYPE
        if (m_face) return true;
        if (m_data.empty()) return false;

        FT_Library ft_library = TextServerAdvanced::get_singleton()->get_ft_library();
        if (!ft_library) return false;

        FT_Error err = FT_New_Memory_Face(ft_library, m_data.data(),
                                           static_cast<FT_Long>(m_data.size()),
                                           m_face_index, const_cast<FT_Face*>(&m_face));
        return err == 0;
#else
        return false;
#endif
    }

    void clear_cache() {
#ifdef XTU_USE_FREETYPE
        if (m_face) {
            FT_Done_Face(m_face);
            m_face = nullptr;
        }
#endif
        m_cache_ascent.clear();
        m_cache_descent.clear();
        m_cache_underline_position.clear();
        m_cache_underline_thickness.clear();
        m_glyph_cache.clear();
        m_atlases.clear();
        m_current_atlas = 0;
        m_atlas_cursor = vec2i(0, 0);
        m_atlas_row_height = 0;
    }

    void ensure_atlas() {
        if (m_current_atlas >= static_cast<int>(m_atlases.size())) {
            Ref<ImageTexture> tex;
            tex.instance();
            Ref<Image> img;
            img.instance();
            img->create(m_atlas_size, m_atlas_size, false, Image::FORMAT_R8);
            img->fill(Color(0, 0, 0, 0));
            tex->create_from_image(img);
            m_atlases.push_back(tex);
        }
    }

    bool pack_glyph(FT_Bitmap* bitmap, int width, int height, GlyphMetrics& metrics) {
        ensure_atlas();

        if (m_atlas_cursor.x + width + 1 > m_atlas_size) {
            m_atlas_cursor.x = 0;
            m_atlas_cursor.y += m_atlas_row_height + 1;
            m_atlas_row_height = 0;
        }

        if (m_atlas_cursor.y + height + 1 > m_atlas_size) {
            ++m_current_atlas;
            m_atlas_cursor = vec2i(0, 0);
            m_atlas_row_height = 0;
            ensure_atlas();
        }

        metrics.atlas_id = m_current_atlas;
        metrics.uv_tl = vec2f(
            static_cast<float>(m_atlas_cursor.x) / m_atlas_size,
            static_cast<float>(m_atlas_cursor.y) / m_atlas_size
        );
        metrics.uv_br = vec2f(
            static_cast<float>(m_atlas_cursor.x + width) / m_atlas_size,
            static_cast<float>(m_atlas_cursor.y + height) / m_atlas_size
        );

        // Copy bitmap to atlas
        Ref<Texture2D> atlas = m_atlases[m_current_atlas];
        Ref<Image> atlas_img = atlas->get_data();
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                uint8_t val = bitmap->buffer[y * bitmap->pitch + x];
                atlas_img->set_pixel(m_atlas_cursor.x + x, m_atlas_cursor.y + y, Color(val / 255.0f, 0, 0));
            }
        }
        atlas->update(atlas_img);

        m_atlas_cursor.x += width + 1;
        m_atlas_row_height = std::max(m_atlas_row_height, height);

        return true;
    }

public:
    static StringName get_class_static() { return StringName("FontFile"); }

    ~FontFile() { clear_cache(); }

    void set_data(const std::vector<uint8_t>& data) {
        std::lock_guard<std::mutex> lock(m_mutex);
        clear_cache();
        m_data = data;
        m_font_path.clear();
    }

    Error load_file(const String& path) {
        std::lock_guard<std::mutex> lock(m_mutex);
        Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
        if (!file.is_valid()) return ERR_FILE_CANT_OPEN;

        clear_cache();
        m_data = file->get_buffer(file->get_length());
        m_font_path = path;
        return OK;
    }

    void set_font_path(const String& path) { load_file(path); }
    String get_font_path() const { return m_font_path; }

    void set_face_index(int idx) { m_face_index = idx; clear_cache(); }
    int get_face_index() const { return m_face_index; }

    void set_hinting(FontHinting hinting) { m_hinting = hinting; clear_cache(); }
    FontHinting get_hinting() const { return m_hinting; }

    void set_subpixel_positioning(FontSubpixelPositioning subpixel) { m_subpixel = subpixel; clear_cache(); }
    FontSubpixelPositioning get_subpixel_positioning() const { return m_subpixel; }

    void set_antialiased(bool aa) { m_antialiased = aa; clear_cache(); }
    bool is_antialiased() const { return m_antialiased; }

    void set_msdf_pixel_range(float range) { m_msdf_pixel_range = range; clear_cache(); }
    float get_msdf_pixel_range() const { return m_msdf_pixel_range; }

    void set_msdf_size(int size) { m_msdf_size = size; clear_cache(); }
    int get_msdf_size() const { return m_msdf_size; }

    void set_fixed_size(int size) { m_fixed_size_value = size; m_fixed_size = size > 0; clear_cache(); }
    int get_fixed_size() const { return m_fixed_size_value; }

    String get_font_name() const {
        std::lock_guard<std::mutex> lock(m_mutex);
#ifdef XTU_USE_FREETYPE
        if (ensure_face() && m_face->family_name) {
            return String(m_face->family_name);
        }
#endif
        return m_font_name;
    }

    String get_font_style_name() const {
        std::lock_guard<std::mutex> lock(m_mutex);
#ifdef XTU_USE_FREETYPE
        if (ensure_face() && m_face->style_name) {
            return String(m_face->style_name);
        }
#endif
        return m_font_style_name;
    }

    float get_height(int size) const override {
        return get_ascent(size) + get_descent(size);
    }

    float get_ascent(int size) const override {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_cache_ascent.find(size);
        if (it != m_cache_ascent.end()) return it->second;

        float ascent = size * 0.8f;
#ifdef XTU_USE_FREETYPE
        if (ensure_face()) {
            FT_Set_Pixel_Sizes(m_face, 0, size);
            ascent = m_face->ascender * size / static_cast<float>(m_face->units_per_EM);
        }
#endif
        const_cast<FontFile*>(this)->m_cache_ascent[size] = ascent;
        return ascent;
    }

    float get_descent(int size) const override {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_cache_descent.find(size);
        if (it != m_cache_descent.end()) return it->second;

        float descent = size * 0.2f;
#ifdef XTU_USE_FREETYPE
        if (ensure_face()) {
            FT_Set_Pixel_Sizes(m_face, 0, size);
            descent = -m_face->descender * size / static_cast<float>(m_face->units_per_EM);
        }
#endif
        const_cast<FontFile*>(this)->m_cache_descent[size] = descent;
        return descent;
    }

    float get_underline_position(int size) const override {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_cache_underline_position.find(size);
        if (it != m_cache_underline_position.end()) return it->second;

        float pos = -size * 0.1f;
#ifdef XTU_USE_FREETYPE
        if (ensure_face()) {
            FT_Set_Pixel_Sizes(m_face, 0, size);
            pos = -m_face->underline_position * size / static_cast<float>(m_face->units_per_EM);
        }
#endif
        const_cast<FontFile*>(this)->m_cache_underline_position[size] = pos;
        return pos;
    }

    float get_underline_thickness(int size) const override {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_cache_underline_thickness.find(size);
        if (it != m_cache_underline_thickness.end()) return it->second;

        float thickness = size * 0.05f;
#ifdef XTU_USE_FREETYPE
        if (ensure_face()) {
            FT_Set_Pixel_Sizes(m_face, 0, size);
            thickness = m_face->underline_thickness * size / static_cast<float>(m_face->units_per_EM);
        }
#endif
        const_cast<FontFile*>(this)->m_cache_underline_thickness[size] = thickness;
        return thickness;
    }

    Ref<Texture2D> get_texture(int atlas_idx) const override {
        std::lock_guard<std::mutex> lock(m_mutex);
        return atlas_idx >= 0 && atlas_idx < static_cast<int>(m_atlases.size()) ?
               m_atlases[atlas_idx] : Ref<Texture2D>();
    }

    RID get_rid() const override {
        return m_atlases.empty() ? RID() : m_atlases[0]->get_rid();
    }

    bool has_char(char32_t c) const override {
        std::lock_guard<std::mutex> lock(m_mutex);
#ifdef XTU_USE_FREETYPE
        if (ensure_face()) {
            return FT_Get_Char_Index(m_face, c) != 0;
        }
#endif
        return false;
    }

    GlyphMetrics get_char_metrics(char32_t c, int size) const override {
        std::lock_guard<std::mutex> lock(m_mutex);
        uint64_t key = make_glyph_key(c, size);
        auto it = m_glyph_cache.find(key);
        if (it != m_glyph_cache.end()) {
            return it->second;
        }

        GlyphMetrics metrics;
        metrics.valid = false;

#ifdef XTU_USE_FREETYPE
        if (ensure_face()) {
            FT_Set_Pixel_Sizes(m_face, 0, size);
            FT_UInt glyph_index = FT_Get_Char_Index(m_face, c);
            if (glyph_index == 0) return metrics;

            FT_Error err = FT_Load_Glyph(m_face, glyph_index, FT_LOAD_DEFAULT);
            if (err != 0) return metrics;

            FT_Render_Glyph(m_face->glyph, FT_RENDER_MODE_NORMAL);
            FT_Bitmap* bitmap = &m_face->glyph->bitmap;

            metrics.advance = vec2f(m_face->glyph->advance.x / 64.0f, m_face->glyph->advance.y / 64.0f);
            metrics.offset = vec2f(m_face->glyph->bitmap_left, -m_face->glyph->bitmap_top);
            metrics.size = vec2f(bitmap->width, bitmap->rows);

            if (bitmap->width > 0 && bitmap->rows > 0) {
                pack_glyph(bitmap, bitmap->width, bitmap->rows, metrics);
            }

            metrics.valid = true;
            const_cast<FontFile*>(this)->m_glyph_cache[key] = metrics;
        }
#endif
        return metrics;
    }

    vec2f get_string_size(const String& text, int size, TextDirection direction) const override {
        float width = 0.0f;
        float height = get_height(size);
        std::string utf8 = text.to_std_string();
        const char* ptr = utf8.c_str();

        while (*ptr) {
            char32_t c = 0;
            if ((*ptr & 0x80) == 0) {
                c = *ptr++;
            } else if ((*ptr & 0xE0) == 0xC0) {
                c = ((*ptr++ & 0x1F) << 6);
                c |= (*ptr++ & 0x3F);
            } else if ((*ptr & 0xF0) == 0xE0) {
                c = ((*ptr++ & 0x0F) << 12);
                c |= ((*ptr++ & 0x3F) << 6);
                c |= (*ptr++ & 0x3F);
            } else if ((*ptr & 0xF8) == 0xF0) {
                c = ((*ptr++ & 0x07) << 18);
                c |= ((*ptr++ & 0x3F) << 12);
                c |= ((*ptr++ & 0x3F) << 6);
                c |= (*ptr++ & 0x3F);
            } else {
                ++ptr;
                continue;
            }

            GlyphMetrics metrics = get_char_metrics(c, size);
            width += metrics.advance.x();
        }

        return vec2f(width, height);
    }

    void draw_string(RID canvas_item, const vec2f& pos, const String& text,
                     int size, const Color& color) const override {
        vec2f cursor = pos;
        std::string utf8 = text.to_std_string();
        const char* ptr = utf8.c_str();

        while (*ptr) {
            char32_t c = 0;
            if ((*ptr & 0x80) == 0) {
                c = *ptr++;
            } else if ((*ptr & 0xE0) == 0xC0) {
                c = ((*ptr++ & 0x1F) << 6);
                c |= (*ptr++ & 0x3F);
            } else if ((*ptr & 0xF0) == 0xE0) {
                c = ((*ptr++ & 0x0F) << 12);
                c |= ((*ptr++ & 0x3F) << 6);
                c |= (*ptr++ & 0x3F);
            } else if ((*ptr & 0xF8) == 0xF0) {
                c = ((*ptr++ & 0x07) << 18);
                c |= ((*ptr++ & 0x3F) << 12);
                c |= ((*ptr++ & 0x3F) << 6);
                c |= (*ptr++ & 0x3F);
            } else {
                ++ptr;
                continue;
            }

            GlyphMetrics metrics = get_char_metrics(c, size);
            if (metrics.valid) {
                vec2f glyph_pos = cursor + metrics.offset;
                Rect2 src_rect(metrics.uv_tl, metrics.uv_br - metrics.uv_tl);
                Rect2 dst_rect(glyph_pos, metrics.size);

                Ref<Texture2D> tex = get_texture(metrics.atlas_id);
                if (tex.is_valid()) {
                    RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(
                        canvas_item, dst_rect, tex->get_rid(), src_rect, color);
                }
                cursor.x() += metrics.advance.x();
            }
        }
    }
};

// #############################################################################
// BitmapFont - BMFont format loader
// #############################################################################
class BitmapFont : public Font {
    XTU_GODOT_REGISTER_CLASS(BitmapFont, Font)

private:
    struct CharInfo {
        int x = 0, y = 0;
        int width = 0, height = 0;
        int x_offset = 0, y_offset = 0;
        int x_advance = 0;
        int page = 0;
    };

    std::unordered_map<char32_t, CharInfo> m_chars;
    std::vector<Ref<Texture2D>> m_textures;
    int m_height = 0;
    int m_ascent = 0;
    int m_descent = 0;

public:
    static StringName get_class_static() { return StringName("BitmapFont"); }

    Error load_from_file(const String& path) {
        Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
        if (!file.is_valid()) return ERR_FILE_CANT_OPEN;

        String base_path = path.get_base_dir();
        String line;

        while (!(line = file->get_line()).empty()) {
            auto parts = line.split(" ");
            if (parts.empty()) continue;

            if (parts[0] == "info") {
                for (size_t i = 1; i < parts.size(); ++i) {
                    auto kv = parts[i].split("=");
                    // Parse info
                }
            } else if (parts[0] == "common") {
                for (size_t i = 1; i < parts.size(); ++i) {
                    auto kv = parts[i].split("=");
                    if (kv.size() >= 2) {
                        if (kv[0] == "lineHeight") m_height = kv[1].to_int();
                        else if (kv[0] == "base") m_ascent = kv[1].to_int();
                    }
                }
            } else if (parts[0] == "page") {
                int id = 0;
                String file_name;
                for (size_t i = 1; i < parts.size(); ++i) {
                    auto kv = parts[i].split("=");
                    if (kv.size() >= 2) {
                        if (kv[0] == "id") id = kv[1].to_int();
                        else if (kv[0] == "file") file_name = kv[1].unquote();
                    }
                }
                if (!file_name.empty()) {
                    String tex_path = base_path + "/" + file_name;
                    Ref<Texture2D> tex = ResourceLoader::load(tex_path);
                    if (tex.is_valid()) {
                        if (id >= static_cast<int>(m_textures.size())) {
                            m_textures.resize(id + 1);
                        }
                        m_textures[id] = tex;
                    }
                }
            } else if (parts[0] == "char") {
                CharInfo info;
                char32_t id = 0;
                for (size_t i = 1; i < parts.size(); ++i) {
                    auto kv = parts[i].split("=");
                    if (kv.size() >= 2) {
                        int val = kv[1].to_int();
                        if (kv[0] == "id") id = static_cast<char32_t>(val);
                        else if (kv[0] == "x") info.x = val;
                        else if (kv[0] == "y") info.y = val;
                        else if (kv[0] == "width") info.width = val;
                        else if (kv[0] == "height") info.height = val;
                        else if (kv[0] == "xoffset") info.x_offset = val;
                        else if (kv[0] == "yoffset") info.y_offset = val;
                        else if (kv[0] == "xadvance") info.x_advance = val;
                        else if (kv[0] == "page") info.page = val;
                    }
                }
                m_chars[id] = info;
            }
        }

        m_descent = m_height - m_ascent;
        return OK;
    }

    float get_height(int size) const override { return static_cast<float>(m_height); }
    float get_ascent(int size) const override { return static_cast<float>(m_ascent); }
    float get_descent(int size) const override { return static_cast<float>(m_descent); }
    float get_underline_position(int size) const override { return -2.0f; }
    float get_underline_thickness(int size) const override { return 1.0f; }

    Ref<Texture2D> get_texture(int atlas_idx) const override {
        return atlas_idx >= 0 && atlas_idx < static_cast<int>(m_textures.size()) ?
               m_textures[atlas_idx] : Ref<Texture2D>();
    }

    RID get_rid() const override {
        return m_textures.empty() ? RID() : m_textures[0]->get_rid();
    }

    bool has_char(char32_t c) const override {
        return m_chars.find(c) != m_chars.end();
    }

    GlyphMetrics get_char_metrics(char32_t c, int size) const override {
        GlyphMetrics metrics;
        auto it = m_chars.find(c);
        if (it == m_chars.end()) return metrics;

        const CharInfo& info = it->second;
        metrics.advance = vec2f(static_cast<float>(info.x_advance), 0);
        metrics.offset = vec2f(static_cast<float>(info.x_offset), static_cast<float>(info.y_offset));
        metrics.size = vec2f(static_cast<float>(info.width), static_cast<float>(info.height));

        Ref<Texture2D> tex = get_texture(info.page);
        if (tex.is_valid()) {
            vec2 tex_size = tex->get_size();
            metrics.uv_tl = vec2f(info.x / tex_size.x(), info.y / tex_size.y());
            metrics.uv_br = vec2f((info.x + info.width) / tex_size.x(),
                                  (info.y + info.height) / tex_size.y());
            metrics.atlas_id = info.page;
            metrics.valid = true;
        }

        return metrics;
    }

    vec2f get_string_size(const String& text, int size, TextDirection direction) const override {
        float width = 0.0f;
        std::string utf8 = text.to_std_string();
        const char* ptr = utf8.c_str();

        while (*ptr) {
            char32_t c = 0;
            if ((*ptr & 0x80) == 0) {
                c = *ptr++;
            } else if ((*ptr & 0xE0) == 0xC0) {
                c = ((*ptr++ & 0x1F) << 6) | (*ptr++ & 0x3F);
            } else if ((*ptr & 0xF0) == 0xE0) {
                c = ((*ptr++ & 0x0F) << 12) | ((*ptr++ & 0x3F) << 6) | (*ptr++ & 0x3F);
            } else if ((*ptr & 0xF8) == 0xF0) {
                c = ((*ptr++ & 0x07) << 18) | ((*ptr++ & 0x3F) << 12) | ((*ptr++ & 0x3F) << 6) | (*ptr++ & 0x3F);
            } else {
                ++ptr;
                continue;
            }

            auto it = m_chars.find(c);
            if (it != m_chars.end()) {
                width += it->second.x_advance;
            }
        }

        return vec2f(width, static_cast<float>(m_height));
    }

    void draw_string(RID canvas_item, const vec2f& pos, const String& text,
                     int size, const Color& color) const override {
        vec2f cursor = pos;
        std::string utf8 = text.to_std_string();
        const char* ptr = utf8.c_str();

        while (*ptr) {
            char32_t c = 0;
            if ((*ptr & 0x80) == 0) {
                c = *ptr++;
            } else if ((*ptr & 0xE0) == 0xC0) {
                c = ((*ptr++ & 0x1F) << 6) | (*ptr++ & 0x3F);
            } else if ((*ptr & 0xF0) == 0xE0) {
                c = ((*ptr++ & 0x0F) << 12) | ((*ptr++ & 0x3F) << 6) | (*ptr++ & 0x3F);
            } else if ((*ptr & 0xF8) == 0xF0) {
                c = ((*ptr++ & 0x07) << 18) | ((*ptr++ & 0x3F) << 12) | ((*ptr++ & 0x3F) << 6) | (*ptr++ & 0x3F);
            } else {
                ++ptr;
                continue;
            }

            GlyphMetrics metrics = get_char_metrics(c, size);
            if (metrics.valid) {
                vec2f glyph_pos = cursor + metrics.offset;
                Rect2 src_rect(metrics.uv_tl, metrics.uv_br - metrics.uv_tl);
                Rect2 dst_rect(glyph_pos, metrics.size);

                Ref<Texture2D> tex = get_texture(metrics.atlas_id);
                if (tex.is_valid()) {
                    RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(
                        canvas_item, dst_rect, tex->get_rid(), src_rect, color);
                }
                cursor.x() += metrics.advance.x();
            }
        }
    }
};

// #############################################################################
// SystemFont - Native system font loader
// #############################################################################
class SystemFont : public Font {
    XTU_GODOT_REGISTER_CLASS(SystemFont, Font)

private:
    String m_font_name;
    String m_font_style;
    int m_weight = 400;
    bool m_italic = false;
    Ref<FontFile> m_fallback_font;

    Ref<FontFile> find_system_font() const {
        // Platform-specific system font enumeration
        // Returns a FontFile with the matched font data
        Ref<FontFile> font;
        font.instance();
        return font;
    }

public:
    static StringName get_class_static() { return StringName("SystemFont"); }

    void set_font_name(const String& name) { m_font_name = name; }
    String get_font_name() const { return m_font_name; }

    void set_font_style(const String& style) { m_font_style = style; }
    String get_font_style() const { return m_font_style; }

    void set_weight(int weight) { m_weight = weight; }
    int get_weight() const { return m_weight; }

    void set_italic(bool italic) { m_italic = italic; }
    bool is_italic() const { return m_italic; }

    float get_height(int size) const override {
        Ref<FontFile> font = find_system_font();
        return font.is_valid() ? font->get_height(size) : size;
    }

    float get_ascent(int size) const override {
        Ref<FontFile> font = find_system_font();
        return font.is_valid() ? font->get_ascent(size) : size * 0.8f;
    }

    float get_descent(int size) const override {
        Ref<FontFile> font = find_system_font();
        return font.is_valid() ? font->get_descent(size) : size * 0.2f;
    }

    float get_underline_position(int size) const override {
        Ref<FontFile> font = find_system_font();
        return font.is_valid() ? font->get_underline_position(size) : -size * 0.1f;
    }

    float get_underline_thickness(int size) const override {
        Ref<FontFile> font = find_system_font();
        return font.is_valid() ? font->get_underline_thickness(size) : size * 0.05f;
    }

    Ref<Texture2D> get_texture(int atlas_idx) const override {
        Ref<FontFile> font = find_system_font();
        return font.is_valid() ? font->get_texture(atlas_idx) : Ref<Texture2D>();
    }

    RID get_rid() const override {
        Ref<FontFile> font = find_system_font();
        return font.is_valid() ? font->get_rid() : RID();
    }

    bool has_char(char32_t c) const override {
        Ref<FontFile> font = find_system_font();
        return font.is_valid() && font->has_char(c);
    }

    GlyphMetrics get_char_metrics(char32_t c, int size) const override {
        Ref<FontFile> font = find_system_font();
        return font.is_valid() ? font->get_char_metrics(c, size) : GlyphMetrics();
    }

    vec2f get_string_size(const String& text, int size, TextDirection direction) const override {
        Ref<FontFile> font = find_system_font();
        return font.is_valid() ? font->get_string_size(text, size, direction) : vec2f(0, size);
    }

    void draw_string(RID canvas_item, const vec2f& pos, const String& text,
                     int size, const Color& color) const override {
        Ref<FontFile> font = find_system_font();
        if (font.is_valid()) {
            font->draw_string(canvas_item, pos, text, size, color);
        }
    }
};

} // namespace godot

// Bring into main namespace
using godot::Font;
using godot::FontFile;
using godot::BitmapFont;
using godot::SystemFont;
using godot::FontSpacing;
using godot::FontFeatureTag;
using godot::GlyphMetrics;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XFONT_RESOURCES_HPP