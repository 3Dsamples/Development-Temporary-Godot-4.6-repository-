// include/xtu/godot/xtheme_advanced.hpp
// xtensor-unified - Advanced Theme and StyleBox system for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XTHEME_ADVANCED_HPP
#define XTU_GODOT_XTHEME_ADVANCED_HPP

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
#include "xtu/godot/xgui.hpp"
#include "xtu/godot/xrenderingserver.hpp"
#include "xtu/graphics/xgraphics.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Part 1: StyleBox - Base class for UI style boxes
// #############################################################################

class StyleBox : public Resource {
    XTU_GODOT_REGISTER_CLASS(StyleBox, Resource)

protected:
    float m_content_margin_left = 0.0f;
    float m_content_margin_top = 0.0f;
    float m_content_margin_right = 0.0f;
    float m_content_margin_bottom = 0.0f;

public:
    static StringName get_class_static() { return StringName("StyleBox"); }

    virtual void draw(RID canvas_item, const Rect2& rect) const = 0;
    virtual Rect2 get_draw_rect(const Rect2& rect) const { return rect; }

    void set_content_margin(Side side, float margin) {
        switch (side) {
            case SIDE_LEFT: m_content_margin_left = margin; break;
            case SIDE_TOP: m_content_margin_top = margin; break;
            case SIDE_RIGHT: m_content_margin_right = margin; break;
            case SIDE_BOTTOM: m_content_margin_bottom = margin; break;
            default: break;
        }
    }

    void set_content_margin_all(float margin) {
        m_content_margin_left = margin;
        m_content_margin_top = margin;
        m_content_margin_right = margin;
        m_content_margin_bottom = margin;
    }

    float get_content_margin(Side side) const {
        switch (side) {
            case SIDE_LEFT: return m_content_margin_left;
            case SIDE_TOP: return m_content_margin_top;
            case SIDE_RIGHT: return m_content_margin_right;
            case SIDE_BOTTOM: return m_content_margin_bottom;
            default: return 0.0f;
        }
    }

    Rect2 get_content_rect(const Rect2& rect) const {
        return Rect2(
            rect.position.x() + m_content_margin_left,
            rect.position.y() + m_content_margin_top,
            rect.size.x() - m_content_margin_left - m_content_margin_right,
            rect.size.y() - m_content_margin_top - m_content_margin_bottom
        );
    }

    virtual float get_style_margin(Side side) const { return 0.0f; }
    virtual vec2f get_center_size() const { return vec2f(0, 0); }
};

// #############################################################################
// Part 2: StyleBoxFlat - Flat style with borders and shadows
// #############################################################################

class StyleBoxFlat : public StyleBox {
    XTU_GODOT_REGISTER_CLASS(StyleBoxFlat, StyleBox)

public:
    enum AxisStretchMode {
        AXIS_STRETCH_MODE_STRETCH,
        AXIS_STRETCH_MODE_TILE,
        AXIS_STRETCH_MODE_TILE_FIT
    };

private:
    Color m_bg_color = Color(0.2f, 0.2f, 0.2f, 1.0f);
    Color m_border_color = Color(0.5f, 0.5f, 0.5f, 1.0f);
    float m_border_width_left = 1.0f;
    float m_border_width_top = 1.0f;
    float m_border_width_right = 1.0f;
    float m_border_width_bottom = 1.0f;
    float m_corner_radius_top_left = 0.0f;
    float m_corner_radius_top_right = 0.0f;
    float m_corner_radius_bottom_right = 0.0f;
    float m_corner_radius_bottom_left = 0.0f;
    bool m_anti_aliased = true;
    float m_aa_size = 1.0f;
    bool m_draw_center = true;
    bool m_draw_border = true;
    Color m_shadow_color = Color(0, 0, 0, 0.3f);
    vec2f m_shadow_offset = vec2f(2, 2);
    float m_shadow_size = 4.0f;
    bool m_expand_margin_left = false;
    bool m_expand_margin_top = false;
    bool m_expand_margin_right = false;
    bool m_expand_margin_bottom = false;

public:
    static StringName get_class_static() { return StringName("StyleBoxFlat"); }

    void set_bg_color(const Color& color) { m_bg_color = color; }
    Color get_bg_color() const { return m_bg_color; }

    void set_border_color(const Color& color) { m_border_color = color; }
    Color get_border_color() const { return m_border_color; }

    void set_border_width_all(float width) {
        m_border_width_left = width;
        m_border_width_top = width;
        m_border_width_right = width;
        m_border_width_bottom = width;
    }

    void set_border_width(Side side, float width) {
        switch (side) {
            case SIDE_LEFT: m_border_width_left = width; break;
            case SIDE_TOP: m_border_width_top = width; break;
            case SIDE_RIGHT: m_border_width_right = width; break;
            case SIDE_BOTTOM: m_border_width_bottom = width; break;
            default: break;
        }
    }

    float get_border_width(Side side) const {
        switch (side) {
            case SIDE_LEFT: return m_border_width_left;
            case SIDE_TOP: return m_border_width_top;
            case SIDE_RIGHT: return m_border_width_right;
            case SIDE_BOTTOM: return m_border_width_bottom;
            default: return 0.0f;
        }
    }

    void set_corner_radius_all(float radius) {
        m_corner_radius_top_left = radius;
        m_corner_radius_top_right = radius;
        m_corner_radius_bottom_right = radius;
        m_corner_radius_bottom_left = radius;
    }

    void set_corner_radius(Corner corner, float radius) {
        switch (corner) {
            case CORNER_TOP_LEFT: m_corner_radius_top_left = radius; break;
            case CORNER_TOP_RIGHT: m_corner_radius_top_right = radius; break;
            case CORNER_BOTTOM_RIGHT: m_corner_radius_bottom_right = radius; break;
            case CORNER_BOTTOM_LEFT: m_corner_radius_bottom_left = radius; break;
            default: break;
        }
    }

    float get_corner_radius(Corner corner) const {
        switch (corner) {
            case CORNER_TOP_LEFT: return m_corner_radius_top_left;
            case CORNER_TOP_RIGHT: return m_corner_radius_top_right;
            case CORNER_BOTTOM_RIGHT: return m_corner_radius_bottom_right;
            case CORNER_BOTTOM_LEFT: return m_corner_radius_bottom_left;
            default: return 0.0f;
        }
    }

    void set_anti_aliased(bool aa) { m_anti_aliased = aa; }
    bool is_anti_aliased() const { return m_anti_aliased; }

    void set_draw_center(bool draw) { m_draw_center = draw; }
    bool is_draw_center() const { return m_draw_center; }

    void set_shadow_color(const Color& color) { m_shadow_color = color; }
    Color get_shadow_color() const { return m_shadow_color; }

    void set_shadow_offset(const vec2f& offset) { m_shadow_offset = offset; }
    vec2f get_shadow_offset() const { return m_shadow_offset; }

    void set_shadow_size(float size) { m_shadow_size = size; }
    float get_shadow_size() const { return m_shadow_size; }

    void draw(RID canvas_item, const Rect2& rect) const override {
        RenderingServer* rs = RenderingServer::get_singleton();

        // Draw shadow if enabled
        if (m_shadow_color.a() > 0.0f && m_shadow_size > 0.0f) {
            Rect2 shadow_rect = rect;
            shadow_rect.position += m_shadow_offset;
            draw_rounded_rect(canvas_item, shadow_rect, m_shadow_color, true);
        }

        // Draw background
        if (m_draw_center && m_bg_color.a() > 0.0f) {
            draw_rounded_rect(canvas_item, rect, m_bg_color, false);
        }

        // Draw border
        if (m_draw_border && m_border_color.a() > 0.0f) {
            draw_rounded_border(canvas_item, rect);
        }
    }

    float get_style_margin(Side side) const override {
        float margin = 0.0f;
        if (side == SIDE_LEFT && m_expand_margin_left) margin += m_shadow_size + std::abs(m_shadow_offset.x());
        if (side == SIDE_TOP && m_expand_margin_top) margin += m_shadow_size + std::abs(m_shadow_offset.y());
        if (side == SIDE_RIGHT && m_expand_margin_right) margin += m_shadow_size + std::abs(m_shadow_offset.x());
        if (side == SIDE_BOTTOM && m_expand_margin_bottom) margin += m_shadow_size + std::abs(m_shadow_offset.y());
        return margin;
    }

private:
    void draw_rounded_rect(RID canvas_item, const Rect2& rect, const Color& color, bool is_shadow) const {
        // Simplified: use antialiased polygon rendering
        RenderingServer::get_singleton()->canvas_item_add_rect(canvas_item, rect, color);
    }

    void draw_rounded_border(RID canvas_item, const Rect2& rect) const {
        // Draw border as separate rectangles or use polygon
    }
};

// #############################################################################
// Part 3: StyleBoxTexture - Texture-based style box with 9-slice
// #############################################################################

class StyleBoxTexture : public StyleBox {
    XTU_GODOT_REGISTER_CLASS(StyleBoxTexture, StyleBox)

public:
    enum AxisStretchMode {
        AXIS_STRETCH_MODE_STRETCH,
        AXIS_STRETCH_MODE_TILE,
        AXIS_STRETCH_MODE_TILE_FIT
    };

private:
    Ref<Texture2D> m_texture;
    Rect2 m_region_rect;
    Rect2 m_margin;
    AxisStretchMode m_h_axis_stretch_mode = AXIS_STRETCH_MODE_STRETCH;
    AxisStretchMode m_v_axis_stretch_mode = AXIS_STRETCH_MODE_STRETCH;
    Color m_modulate_color = Color(1, 1, 1, 1);
    bool m_draw_center = true;

public:
    static StringName get_class_static() { return StringName("StyleBoxTexture"); }

    void set_texture(const Ref<Texture2D>& texture) { m_texture = texture; }
    Ref<Texture2D> get_texture() const { return m_texture; }

    void set_region_rect(const Rect2& rect) { m_region_rect = rect; }
    Rect2 get_region_rect() const { return m_region_rect; }

    void set_margin_size(Side side, float margin) {
        switch (side) {
            case SIDE_LEFT: m_margin.position.x() = margin; break;
            case SIDE_TOP: m_margin.position.y() = margin; break;
            case SIDE_RIGHT: m_margin.size.x() = margin; break;
            case SIDE_BOTTOM: m_margin.size.y() = margin; break;
            default: break;
        }
    }

    float get_margin_size(Side side) const {
        switch (side) {
            case SIDE_LEFT: return m_margin.position.x();
            case SIDE_TOP: return m_margin.position.y();
            case SIDE_RIGHT: return m_margin.size.x();
            case SIDE_BOTTOM: return m_margin.size.y();
            default: return 0.0f;
        }
    }

    void set_h_axis_stretch_mode(AxisStretchMode mode) { m_h_axis_stretch_mode = mode; }
    AxisStretchMode get_h_axis_stretch_mode() const { return m_h_axis_stretch_mode; }

    void set_v_axis_stretch_mode(AxisStretchMode mode) { m_v_axis_stretch_mode = mode; }
    AxisStretchMode get_v_axis_stretch_mode() const { return m_v_axis_stretch_mode; }

    void set_modulate_color(const Color& color) { m_modulate_color = color; }
    Color get_modulate_color() const { return m_modulate_color; }

    void set_draw_center(bool draw) { m_draw_center = draw; }
    bool is_draw_center() const { return m_draw_center; }

    void draw(RID canvas_item, const Rect2& rect) const override {
        if (!m_texture.is_valid()) return;

        float left = m_margin.position.x();
        float top = m_margin.position.y();
        float right = m_margin.size.x();
        float bottom = m_margin.size.y();

        vec2f tex_size = m_texture->get_size();
        if (m_region_rect.size.x() > 0 && m_region_rect.size.y() > 0) {
            tex_size = m_region_rect.size;
        }

        // Draw 9-slice patches
        draw_patch(canvas_item, rect, 0, 0, left, top, tex_size);
        draw_patch(canvas_item, rect, left, 0, rect.size.x() - left - right, top, tex_size);
        draw_patch(canvas_item, rect, rect.size.x() - right, 0, right, top, tex_size);

        draw_patch(canvas_item, rect, 0, top, left, rect.size.y() - top - bottom, tex_size);
        if (m_draw_center) {
            draw_patch(canvas_item, rect, left, top, rect.size.x() - left - right, rect.size.y() - top - bottom, tex_size);
        }
        draw_patch(canvas_item, rect, rect.size.x() - right, top, right, rect.size.y() - top - bottom, tex_size);

        draw_patch(canvas_item, rect, 0, rect.size.y() - bottom, left, bottom, tex_size);
        draw_patch(canvas_item, rect, left, rect.size.y() - bottom, rect.size.x() - left - right, bottom, tex_size);
        draw_patch(canvas_item, rect, rect.size.x() - right, rect.size.y() - bottom, right, bottom, tex_size);
    }

private:
    void draw_patch(RID canvas_item, const Rect2& rect, float x, float y, float w, float h, const vec2f& tex_size) const {
        if (w <= 0 || h <= 0) return;

        Rect2 dst(x, y, w, h);
        Rect2 src;
        // Calculate source UV based on 9-slice
        RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(
            canvas_item, dst, m_texture->get_rid(), src, m_modulate_color);
    }
};

// #############################################################################
// Part 4: Theme - UI theme resource with inheritance
// #############################################################################

class Theme : public Resource {
    XTU_GODOT_REGISTER_CLASS(Theme, Resource)

public:
    enum DataType {
        DATA_TYPE_COLOR,
        DATA_TYPE_CONSTANT,
        DATA_TYPE_FONT,
        DATA_TYPE_FONT_SIZE,
        DATA_TYPE_ICON,
        DATA_TYPE_STYLEBOX
    };

private:
    struct ThemeItem {
        DataType type;
        Variant value;
    };

    Ref<Theme> m_base_theme;
    std::unordered_map<String, std::unordered_map<String, ThemeItem>> m_items;
    std::unordered_map<String, std::unordered_map<String, Variant>> m_defaults;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("Theme"); }

    void set_base_theme(const Ref<Theme>& theme) { m_base_theme = theme; }
    Ref<Theme> get_base_theme() const { return m_base_theme; }

    void set_color(const String& name, const String& theme_type, const Color& color) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_items[theme_type][name] = {DATA_TYPE_COLOR, color};
    }

    Color get_color(const String& name, const String& theme_type, const Color& default_val = Color()) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto type_it = m_items.find(theme_type);
        if (type_it != m_items.end()) {
            auto item_it = type_it->second.find(name);
            if (item_it != type_it->second.end() && item_it->second.type == DATA_TYPE_COLOR) {
                return item_it->second.value.as<Color>();
            }
        }
        if (m_base_theme.is_valid()) {
            return m_base_theme->get_color(name, theme_type, default_val);
        }
        return default_val;
    }

    void set_constant(const String& name, const String& theme_type, int constant) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_items[theme_type][name] = {DATA_TYPE_CONSTANT, constant};
    }

    int get_constant(const String& name, const String& theme_type, int default_val = 0) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto type_it = m_items.find(theme_type);
        if (type_it != m_items.end()) {
            auto item_it = type_it->second.find(name);
            if (item_it != type_it->second.end() && item_it->second.type == DATA_TYPE_CONSTANT) {
                return item_it->second.value.as<int>();
            }
        }
        if (m_base_theme.is_valid()) {
            return m_base_theme->get_constant(name, theme_type, default_val);
        }
        return default_val;
    }

    void set_font(const String& name, const String& theme_type, const Ref<Font>& font) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_items[theme_type][name] = {DATA_TYPE_FONT, font};
    }

    Ref<Font> get_font(const String& name, const String& theme_type) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto type_it = m_items.find(theme_type);
        if (type_it != m_items.end()) {
            auto item_it = type_it->second.find(name);
            if (item_it != type_it->second.end() && item_it->second.type == DATA_TYPE_FONT) {
                return item_it->second.value.as<Ref<Font>>();
            }
        }
        if (m_base_theme.is_valid()) {
            return m_base_theme->get_font(name, theme_type);
        }
        return Ref<Font>();
    }

    void set_font_size(const String& name, const String& theme_type, int size) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_items[theme_type][name] = {DATA_TYPE_FONT_SIZE, size};
    }

    int get_font_size(const String& name, const String& theme_type, int default_val = 14) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto type_it = m_items.find(theme_type);
        if (type_it != m_items.end()) {
            auto item_it = type_it->second.find(name);
            if (item_it != type_it->second.end() && item_it->second.type == DATA_TYPE_FONT_SIZE) {
                return item_it->second.value.as<int>();
            }
        }
        if (m_base_theme.is_valid()) {
            return m_base_theme->get_font_size(name, theme_type, default_val);
        }
        return default_val;
    }

    void set_icon(const String& name, const String& theme_type, const Ref<Texture2D>& icon) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_items[theme_type][name] = {DATA_TYPE_ICON, icon};
    }

    Ref<Texture2D> get_icon(const String& name, const String& theme_type) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto type_it = m_items.find(theme_type);
        if (type_it != m_items.end()) {
            auto item_it = type_it->second.find(name);
            if (item_it != type_it->second.end() && item_it->second.type == DATA_TYPE_ICON) {
                return item_it->second.value.as<Ref<Texture2D>>();
            }
        }
        if (m_base_theme.is_valid()) {
            return m_base_theme->get_icon(name, theme_type);
        }
        return Ref<Texture2D>();
    }

    void set_stylebox(const String& name, const String& theme_type, const Ref<StyleBox>& stylebox) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_items[theme_type][name] = {DATA_TYPE_STYLEBOX, stylebox};
    }

    Ref<StyleBox> get_stylebox(const String& name, const String& theme_type) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto type_it = m_items.find(theme_type);
        if (type_it != m_items.end()) {
            auto item_it = type_it->second.find(name);
            if (item_it != type_it->second.end() && item_it->second.type == DATA_TYPE_STYLEBOX) {
                return item_it->second.value.as<Ref<StyleBox>>();
            }
        }
        if (m_base_theme.is_valid()) {
            return m_base_theme->get_stylebox(name, theme_type);
        }
        return Ref<StyleBox>();
    }

    bool has_color(const String& name, const String& theme_type) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto type_it = m_items.find(theme_type);
        if (type_it != m_items.end()) {
            return type_it->second.find(name) != type_it->second.end();
        }
        return m_base_theme.is_valid() && m_base_theme->has_color(name, theme_type);
    }

    bool has_stylebox(const String& name, const String& theme_type) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto type_it = m_items.find(theme_type);
        if (type_it != m_items.end()) {
            return type_it->second.find(name) != type_it->second.end();
        }
        return m_base_theme.is_valid() && m_base_theme->has_stylebox(name, theme_type);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_items.clear();
    }

    void clear_theme_item(const String& name, const String& theme_type) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto type_it = m_items.find(theme_type);
        if (type_it != m_items.end()) {
            type_it->second.erase(name);
        }
    }

    std::vector<String> get_type_list() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<String> types;
        for (const auto& kv : m_items) {
            types.push_back(kv.first);
        }
        return types;
    }

    std::vector<String> get_item_list(const String& theme_type) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<String> items;
        auto type_it = m_items.find(theme_type);
        if (type_it != m_items.end()) {
            for (const auto& kv : type_it->second) {
                items.push_back(kv.first);
            }
        }
        return items;
    }

    void set_default_font(const Ref<Font>& font) { m_defaults[""][""] = font; }
    Ref<Font> get_default_font() const { return m_defaults.count("") ? m_defaults.at("").at("").as<Ref<Font>>() : Ref<Font>(); }
};

// #############################################################################
// Part 5: ThemeDB - Global theme database
// #############################################################################

class ThemeDB : public Object {
    XTU_GODOT_REGISTER_CLASS(ThemeDB, Object)

private:
    static ThemeDB* s_singleton;
    Ref<Theme> m_default_theme;
    Ref<Theme> m_project_theme;
    Ref<Theme> m_editor_theme;
    std::unordered_map<String, Ref<Theme>> m_named_themes;
    float m_fallback_base_scale = 1.0f;
    mutable std::mutex m_mutex;

public:
    static ThemeDB* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("ThemeDB"); }

    ThemeDB() {
        s_singleton = this;
        m_default_theme.instance();
        initialize_default_theme();
    }

    ~ThemeDB() { s_singleton = nullptr; }

    void set_default_theme(const Ref<Theme>& theme) { m_default_theme = theme; }
    Ref<Theme> get_default_theme() const { return m_default_theme; }

    void set_project_theme(const Ref<Theme>& theme) { m_project_theme = theme; }
    Ref<Theme> get_project_theme() const { return m_project_theme; }

    void set_editor_theme(const Ref<Theme>& theme) { m_editor_theme = theme; }
    Ref<Theme> get_editor_theme() const { return m_editor_theme; }

    void add_named_theme(const String& name, const Ref<Theme>& theme) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_named_themes[name] = theme;
    }

    Ref<Theme> get_named_theme(const String& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_named_themes.find(name);
        return it != m_named_themes.end() ? it->second : Ref<Theme>();
    }

    void set_fallback_base_scale(float scale) { m_fallback_base_scale = scale; }
    float get_fallback_base_scale() const { return m_fallback_base_scale; }

private:
    void initialize_default_theme() {
        // Set up basic default theme values
        m_default_theme->set_color("font_color", "Label", Color(0.9f, 0.9f, 0.9f, 1.0f));
        m_default_theme->set_color("font_color_disabled", "Label", Color(0.5f, 0.5f, 0.5f, 1.0f));
        m_default_theme->set_constant("line_spacing", "Label", 2);
        m_default_theme->set_font_size("font_size", "Label", 14);

        m_default_theme->set_color("font_color", "Button", Color(0.9f, 0.9f, 0.9f, 1.0f));
        m_default_theme->set_color("font_color_pressed", "Button", Color(1.0f, 1.0f, 1.0f, 1.0f));
        m_default_theme->set_color("font_color_hover", "Button", Color(1.0f, 1.0f, 1.0f, 1.0f));
        m_default_theme->set_color("font_color_disabled", "Button", Color(0.5f, 0.5f, 0.5f, 1.0f));
        m_default_theme->set_font_size("font_size", "Button", 14);

        m_default_theme->set_color("font_color", "LineEdit", Color(0.9f, 0.9f, 0.9f, 1.0f));
        m_default_theme->set_color("font_color_selected", "LineEdit", Color(1.0f, 1.0f, 1.0f, 1.0f));
        m_default_theme->set_font_size("font_size", "LineEdit", 14);

        // Create default StyleBoxes
        Ref<StyleBoxFlat> normal_style;
        normal_style.instance();
        normal_style->set_bg_color(Color(0.2f, 0.2f, 0.2f, 1.0f));
        normal_style->set_border_color(Color(0.3f, 0.3f, 0.3f, 1.0f));
        normal_style->set_border_width_all(1);
        normal_style->set_corner_radius_all(3);
        m_default_theme->set_stylebox("normal", "Button", normal_style);
        m_default_theme->set_stylebox("normal", "LineEdit", normal_style);
    }
};

} // namespace godot

// Bring into main namespace
using godot::StyleBox;
using godot::StyleBoxFlat;
using godot::StyleBoxTexture;
using godot::Theme;
using godot::ThemeDB;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XTHEME_ADVANCED_HPP