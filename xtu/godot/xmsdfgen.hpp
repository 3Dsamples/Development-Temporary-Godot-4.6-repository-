// include/xtu/godot/xmsdfgen.hpp
// xtensor-unified - MSDF Font Generation for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XMSDFGEN_HPP
#define XTU_GODOT_XMSDFGEN_HPP

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
#include "xtu/godot/xfont_resources.hpp"
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/parallel/xparallel.hpp"

#ifdef XTU_USE_MSDFGEN
#include <msdfgen.h>
#include <msdfgen-ext.h>
#endif

XTU_NAMESPACE_BEGIN
namespace godot {
namespace msdf {

// #############################################################################
// Forward declarations
// #############################################################################
class MSDFGenerator;
class MSDFGlyph;
class MSDFFont;

// #############################################################################
// MSDF output type
// #############################################################################
enum class MSDFOutputType : uint8_t {
    OUTPUT_SDF = 0,
    OUTPUT_PSDF = 1,
    OUTPUT_MSDF = 2,
    OUTPUT_MTSDF = 3
};

// #############################################################################
// MSDF generation parameters
// #############################################################################
struct MSDFParams {
    int width = 32;
    int height = 32;
    float range = 2.0f;
    float scale = 1.0f;
    float translation_x = 0.0f;
    float translation_y = 0.0f;
    bool edge_coloring = true;
    bool overlap_support = true;
    float angle_threshold = 3.0f;
    float corner_threshold = 0.25f;
    MSDFOutputType output_type = MSDFOutputType::OUTPUT_MSDF;
};

// #############################################################################
// MSDF glyph data
// #############################################################################
struct MSDFGlyphData {
    int width = 0;
    int height = 0;
    std::vector<float> distance_field;
    float advance_x = 0.0f;
    float advance_y = 0.0f;
    float bearing_x = 0.0f;
    float bearing_y = 0.0f;
    int unicode = 0;
    bool valid = false;
};

// #############################################################################
// MSDFGenerator - Main MSDF generation class
// #############################################################################
class MSDFGenerator : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(MSDFGenerator, RefCounted)

private:
    static MSDFGenerator* s_singleton;
    MSDFParams m_default_params;
    mutable std::mutex m_mutex;
    std::unordered_map<uint64_t, MSDFGlyphData> m_cache;
    bool m_cache_enabled = true;
    size_t m_max_cache_size = 1024;

public:
    static MSDFGenerator* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("MSDFGenerator"); }

    MSDFGenerator() { s_singleton = this; }
    ~MSDFGenerator() { s_singleton = nullptr; }

    void set_default_params(const MSDFParams& params) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_default_params = params;
    }

    MSDFParams get_default_params() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_default_params;
    }

    void set_cache_enabled(bool enabled) { m_cache_enabled = enabled; }
    bool is_cache_enabled() const { return m_cache_enabled; }

    void clear_cache() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_cache.clear();
    }

    // #########################################################################
    // Generate MSDF for a single glyph
    // #########################################################################
    MSDFGlyphData generate_glyph(const Ref<FontFile>& font, int unicode,
                                  const MSDFParams& params = {}) {
        if (!font.is_valid() || !font->has_char(unicode)) {
            MSDFGlyphData result;
            result.unicode = unicode;
            return result;
        }

        MSDFParams use_params = params.width == 0 ? m_default_params : params;
        uint64_t cache_key = compute_glyph_hash(font, unicode, use_params);

        if (m_cache_enabled) {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto it = m_cache.find(cache_key);
            if (it != m_cache.end()) {
                return it->second;
            }
        }

        MSDFGlyphData result = generate_msdf_impl(font, unicode, use_params);

        if (result.valid && m_cache_enabled) {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_cache.size() >= m_max_cache_size) {
                m_cache.erase(m_cache.begin());
            }
            m_cache[cache_key] = result;
        }

        return result;
    }

    // #########################################################################
    // Generate MSDF for ASCII range (32-126)
    // #########################################################################
    std::map<int, MSDFGlyphData> generate_ascii(const Ref<FontFile>& font,
                                                 const MSDFParams& params = {}) {
        std::map<int, MSDFGlyphData> result;
        for (int c = 32; c <= 126; ++c) {
            MSDFGlyphData glyph = generate_glyph(font, c, params);
            if (glyph.valid) {
                result[c] = glyph;
            }
        }
        return result;
    }

    // #########################################################################
    // Generate MSDF for a string of characters
    // #########################################################################
    std::map<int, MSDFGlyphData> generate_string(const Ref<FontFile>& font,
                                                  const String& characters,
                                                  const MSDFParams& params = {}) {
        std::map<int, MSDFGlyphData> result;
        std::string utf8 = characters.to_std_string();
        const char* ptr = utf8.c_str();

        while (*ptr) {
            int unicode = 0;
            if ((*ptr & 0x80) == 0) {
                unicode = *ptr++;
            } else if ((*ptr & 0xE0) == 0xC0) {
                unicode = ((*ptr++ & 0x1F) << 6) | (*ptr++ & 0x3F);
            } else if ((*ptr & 0xF0) == 0xE0) {
                unicode = ((*ptr++ & 0x0F) << 12) | ((*ptr++ & 0x3F) << 6) | (*ptr++ & 0x3F);
            } else if ((*ptr & 0xF8) == 0xF0) {
                unicode = ((*ptr++ & 0x07) << 18) | ((*ptr++ & 0x3F) << 12) | ((*ptr++ & 0x3F) << 6) | (*ptr++ & 0x3F);
            }

            if (unicode > 0 && result.find(unicode) == result.end()) {
                MSDFGlyphData glyph = generate_glyph(font, unicode, params);
                if (glyph.valid) {
                    result[unicode] = glyph;
                }
            }
        }
        return result;
    }

    // #########################################################################
    // Create texture atlas from multiple glyphs
    // #########################################################################
    Ref<Texture2D> create_atlas(const std::map<int, MSDFGlyphData>& glyphs,
                                 std::map<int, Rect2>& out_rects,
                                 vec2i& out_atlas_size) {
        // Simple rectangle packing
        std::vector<std::pair<int, vec2i>> sizes;
        for (const auto& kv : glyphs) {
            sizes.push_back({kv.first, vec2i(kv.second.width, kv.second.height)});
        }

        std::sort(sizes.begin(), sizes.end(),
                  [](const auto& a, const auto& b) { return a.second.y() > b.second.y(); });

        int atlas_width = 256;
        int atlas_height = 256;
        bool packed = false;

        while (!packed) {
            packed = pack_rectangles(sizes, atlas_width, atlas_height, out_rects);
            if (!packed) {
                atlas_width = std::min(atlas_width * 2, 4096);
                atlas_height = std::min(atlas_height * 2, 4096);
            }
        }

        out_atlas_size = vec2i(atlas_width, atlas_height);

        Ref<Image> atlas_img;
        atlas_img.instance();
        atlas_img->create(atlas_width, atlas_height, false, Image::FORMAT_RGBAF);

        for (const auto& kv : glyphs) {
            const MSDFGlyphData& glyph = kv.second;
            Rect2 rect = out_rects[kv.first];

            for (int y = 0; y < glyph.height; ++y) {
                for (int x = 0; x < glyph.width; ++x) {
                    int idx = (y * glyph.width + x) * 4;
                    float r = glyph.distance_field[idx];
                    float g = glyph.distance_field[idx + 1];
                    float b = glyph.distance_field[idx + 2];
                    float a = glyph.distance_field[idx + 3];

                    atlas_img->set_pixel(static_cast<int>(rect.position.x()) + x,
                                         static_cast<int>(rect.position.y()) + y,
                                         Color(r, g, b, a));
                }
            }
        }

        Ref<ImageTexture> tex;
        tex.instance();
        tex->create_from_image(atlas_img);
        return tex;
    }

private:
    uint64_t compute_glyph_hash(const Ref<FontFile>& font, int unicode, const MSDFParams& params) {
        uint64_t h = std::hash<int>{}(unicode);
        h = hash_combine(h, std::hash<String>{}(font->get_font_path()));
        h = hash_combine(h, params.width);
        h = hash_combine(h, params.height);
        h = hash_combine(h, static_cast<uint32_t>(std::hash<float>{}(params.range)));
        h = hash_combine(h, static_cast<uint32_t>(params.output_type));
        return h;
    }

    uint64_t hash_combine(uint64_t seed, uint64_t v) const {
        return seed ^ (v + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    }

    MSDFGlyphData generate_msdf_impl(const Ref<FontFile>& font, int unicode,
                                      const MSDFParams& params) {
        MSDFGlyphData result;
        result.unicode = unicode;

#ifdef XTU_USE_MSDFGEN
        // Get glyph metrics from font
        GlyphMetrics metrics = font->get_char_metrics(unicode, params.height);
        if (!metrics.valid) return result;

        // Use FreeType to get outline
        FT_Face face = font->get_ft_face();
        if (!face) return result;

        FT_Load_Glyph(face, FT_Get_Char_Index(face, unicode), FT_LOAD_NO_SCALE);
        FT_Outline& outline = face->glyph->outline;

        // Convert FreeType outline to MSDFgen shape
        msdfgen::Shape shape;
        msdfgen::Point2 last_point;
        int contour_start = 0;

        for (int c = 0; c < outline.n_contours; ++c) {
            int contour_end = outline.contours[c];
            msdfgen::Contour contour;

            for (int p = contour_start; p <= contour_end; ++p) {
                FT_Vector& pt = outline.points[p];
                msdfgen::Point2 point(pt.x, pt.y);

                if (p == contour_start) {
                    contour.addEdge(msdfgen::EdgeHolder(point, point));
                } else {
                    if (outline.tags[p] & FT_CURVE_TAG_ON) {
                        if (outline.tags[p-1] & FT_CURVE_TAG_ON) {
                            contour.addEdge(msdfgen::EdgeHolder(last_point, point));
                        } else {
                            FT_Vector& cp = outline.points[p-1];
                            contour.addEdge(msdfgen::EdgeHolder(last_point,
                                msdfgen::Point2(cp.x, cp.y), point));
                        }
                    }
                }
                last_point = point;
            }

            if (outline.tags[contour_start] & FT_CURVE_TAG_ON) {
                if (outline.tags[contour_end] & FT_CURVE_TAG_ON) {
                    contour.addEdge(msdfgen::EdgeHolder(last_point,
                        msdfgen::Point2(outline.points[contour_start].x,
                                        outline.points[contour_start].y)));
                }
            }

            shape.addContour(contour);
            contour_start = contour_end + 1;
        }

        shape.normalize();
        shape.inverseYAxis = true;

        // Generate MSDF
        msdfgen::Projection projection;
        msdfgen::Range range(params.range);
        msdfgen::MSDFGeneratorConfig config;
        config.overlapSupport = params.overlap_support;

        msdfgen::Bitmap<float, 4> msdf(params.width, params.height);
        msdfgen::generateMSDF(msdf, shape, projection, range, config);

        // Edge coloring
        if (params.edge_coloring) {
            msdfgen::edgeColoringSimple(shape, params.angle_threshold);
        }

        // Copy to result
        result.width = params.width;
        result.height = params.height;
        result.distance_field.resize(params.width * params.height * 4);

        for (int y = 0; y < params.height; ++y) {
            for (int x = 0; x < params.width; ++x) {
                int idx = (y * params.width + x) * 4;
                float* pixel = msdf(x, y);
                result.distance_field[idx] = pixel[0];
                result.distance_field[idx + 1] = pixel[1];
                result.distance_field[idx + 2] = pixel[2];
                result.distance_field[idx + 3] = 1.0f;
            }
        }

        result.advance_x = metrics.advance.x();
        result.advance_y = metrics.advance.y();
        result.bearing_x = metrics.offset.x();
        result.bearing_y = metrics.offset.y();
        result.valid = true;
#endif

        return result;
    }

    bool pack_rectangles(std::vector<std::pair<int, vec2i>>& sizes,
                         int width, int height,
                         std::map<int, Rect2>& out_rects) {
        out_rects.clear();

        // Simple shelf packing algorithm
        int x = 0, y = 0;
        int row_height = 0;

        for (const auto& item : sizes) {
            int w = item.second.x();
            int h = item.second.y();

            if (x + w > width) {
                x = 0;
                y += row_height;
                row_height = 0;
            }

            if (y + h > height) {
                return false;
            }

            out_rects[item.first] = Rect2(x, y, w, h);
            x += w + 1;
            row_height = std::max(row_height, h);
        }
        return true;
    }
};

// #############################################################################
// MSDFFont - Complete MSDF font resource
// #############################################################################
class MSDFFont : public Font {
    XTU_GODOT_REGISTER_CLASS(MSDFFont, Font)

private:
    Ref<FontFile> m_source_font;
    MSDFParams m_params;
    Ref<Texture2D> m_atlas_texture;
    std::map<int, MSDFGlyphData> m_glyphs;
    std::map<int, Rect2> m_glyph_rects;
    vec2i m_atlas_size;
    bool m_generated = false;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("MSDFFont"); }

    void set_source_font(const Ref<FontFile>& font) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_source_font = font;
        m_generated = false;
    }

    Ref<FontFile> get_source_font() const { return m_source_font; }

    void set_params(const MSDFParams& params) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_params = params;
        m_generated = false;
    }

    MSDFParams get_params() const { return m_params; }

    void set_range(float range) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_params.range = range;
        m_generated = false;
    }

    float get_range() const { return m_params.range; }

    void set_size(int size) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_params.width = size;
        m_params.height = size;
        m_generated = false;
    }

    void generate() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_source_font.is_valid()) return;

        Ref<MSDFGenerator> generator = MSDFGenerator::get_singleton();
        m_glyphs = generator->generate_ascii(m_source_font, m_params);
        m_atlas_texture = generator->create_atlas(m_glyphs, m_glyph_rects, m_atlas_size);
        m_generated = true;
    }

    void generate_async(std::function<void()> callback = nullptr) {
        std::thread([this, callback]() {
            generate();
            if (callback) callback();
            emit_signal("generated");
        }).detach();
    }

    bool is_generated() const { return m_generated; }

    float get_height(int size) const override {
        return static_cast<float>(m_params.height);
    }

    float get_ascent(int size) const override {
        return m_source_font.is_valid() ? m_source_font->get_ascent(size) : size * 0.8f;
    }

    float get_descent(int size) const override {
        return m_source_font.is_valid() ? m_source_font->get_descent(size) : size * 0.2f;
    }

    float get_underline_position(int size) const override {
        return m_source_font.is_valid() ? m_source_font->get_underline_position(size) : -2.0f;
    }

    float get_underline_thickness(int size) const override {
        return m_source_font.is_valid() ? m_source_font->get_underline_thickness(size) : 1.0f;
    }

    Ref<Texture2D> get_texture(int atlas_idx) const override {
        return m_atlas_texture;
    }

    RID get_rid() const override {
        return m_atlas_texture.is_valid() ? m_atlas_texture->get_rid() : RID();
    }

    bool has_char(char32_t c) const override {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_glyphs.find(static_cast<int>(c)) != m_glyphs.end();
    }

    GlyphMetrics get_char_metrics(char32_t c, int size) const override {
        std::lock_guard<std::mutex> lock(m_mutex);
        GlyphMetrics metrics;

        auto it = m_glyphs.find(static_cast<int>(c));
        if (it == m_glyphs.end()) return metrics;

        const MSDFGlyphData& glyph = it->second;
        const Rect2& rect = m_glyph_rects.at(static_cast<int>(c));

        metrics.advance = vec2f(glyph.advance_x, glyph.advance_y);
        metrics.offset = vec2f(glyph.bearing_x, glyph.bearing_y);
        metrics.size = vec2f(glyph.width, glyph.height);
        metrics.uv_tl = vec2f(rect.position.x() / m_atlas_size.x(),
                               rect.position.y() / m_atlas_size.y());
        metrics.uv_br = vec2f((rect.position.x() + rect.size.x()) / m_atlas_size.x(),
                               (rect.position.y() + rect.size.y()) / m_atlas_size.y());
        metrics.valid = true;

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
            }

            auto it = m_glyphs.find(static_cast<int>(c));
            if (it != m_glyphs.end()) {
                width += it->second.advance_x;
            }
        }

        return vec2f(width, static_cast<float>(m_params.height));
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
            }

            GlyphMetrics metrics = get_char_metrics(c, size);
            if (metrics.valid) {
                vec2f glyph_pos = cursor + metrics.offset;
                Rect2 src_rect(metrics.uv_tl, metrics.uv_br - metrics.uv_tl);
                Rect2 dst_rect(glyph_pos, metrics.size);

                RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(
                    canvas_item, dst_rect, m_atlas_texture->get_rid(), src_rect, color);
                cursor.x() += metrics.advance.x();
            }
        }
    }
};

} // namespace msdf

// Bring into main namespace
using msdf::MSDFGenerator;
using msdf::MSDFFont;
using msdf::MSDFOutputType;
using msdf::MSDFParams;
using msdf::MSDFGlyphData;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XMSDFGEN_HPP