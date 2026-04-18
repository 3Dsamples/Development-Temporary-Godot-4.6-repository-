// include/xtu/godot/xeditor_theme.hpp
// xtensor-unified - Editor theme and styling for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XEDITOR_THEME_HPP
#define XTU_GODOT_XEDITOR_THEME_HPP

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
#include "xtu/io/xio_json.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace editor {

// #############################################################################
// Forward declarations
// #############################################################################
class EditorTheme;
class EditorScale;
class EditorFonts;
class EditorThemeManager;

// #############################################################################
// Theme color roles
// #############################################################################
enum class ThemeColorRole : uint8_t {
    COLOR_BACKGROUND = 0,
    COLOR_BASE = 1,
    COLOR_ALTERNATE_BASE = 2,
    COLOR_TEXT = 3,
    COLOR_TEXT_DISABLED = 4,
    COLOR_HIGHLIGHT = 5,
    COLOR_HIGHLIGHTED_TEXT = 6,
    COLOR_LINK = 7,
    COLOR_LINK_VISITED = 8,
    COLOR_SELECTION = 9,
    COLOR_SELECTION_TEXT = 10,
    COLOR_WARNING = 11,
    COLOR_ERROR = 12,
    COLOR_SUCCESS = 13,
    COLOR_ACCENT = 14,
    COLOR_BORDER = 15,
    COLOR_SEPARATOR = 16,
    COLOR_TOOLTIP_BACKGROUND = 17,
    COLOR_TOOLTIP_TEXT = 18,
    COLOR_SCROLLBAR = 19,
    COLOR_SCROLLBAR_HOVER = 20,
    COLOR_SCROLLBAR_PRESSED = 21,
    COLOR_TAB_BACKGROUND = 22,
    COLOR_TAB_SELECTED = 23,
    COLOR_TAB_HOVER = 24,
    COLOR_TREE_GUIDE = 25,
    COLOR_MAX = 26
};

// #############################################################################
// Theme icon types
// #############################################################################
enum class ThemeIconType : uint16_t {
    ICON_NONE = 0,
    ICON_FILE = 1,
    ICON_FOLDER = 2,
    ICON_SCENE = 3,
    ICON_SCRIPT = 4,
    ICON_TEXTURE = 5,
    ICON_MATERIAL = 6,
    ICON_MESH = 7,
    ICON_AUDIO = 8,
    ICON_FONT = 9,
    ICON_PLUGIN = 10,
    ICON_PLAY = 11,
    ICON_STOP = 12,
    ICON_PAUSE = 13,
    ICON_SAVE = 14,
    ICON_SAVE_ALL = 15,
    ICON_UNDO = 16,
    ICON_REDO = 17,
    ICON_CUT = 18,
    ICON_COPY = 19,
    ICON_PASTE = 20,
    ICON_DELETE = 21,
    ICON_SEARCH = 22,
    ICON_SETTINGS = 23,
    ICON_HELP = 24,
    ICON_WARNING = 25,
    ICON_ERROR = 26,
    ICON_INFO = 27,
    ICON_LOCK = 28,
    ICON_UNLOCK = 29,
    ICON_VISIBLE = 30,
    ICON_HIDDEN = 31,
    ICON_CHECKED = 32,
    ICON_UNCHECKED = 33,
    ICON_RADIO_CHECKED = 34,
    ICON_RADIO_UNCHECKED = 35,
    ICON_ARROW_UP = 36,
    ICON_ARROW_DOWN = 37,
    ICON_ARROW_LEFT = 38,
    ICON_ARROW_RIGHT = 39,
    ICON_PLUS = 40,
    ICON_MINUS = 41,
    ICON_REFRESH = 42,
    ICON_CLOSE = 43,
    ICON_MAX = 44
};

// #############################################################################
// Theme variation
// #############################################################################
enum class ThemeVariation : uint8_t {
    VARIATION_DARK = 0,
    VARIATION_LIGHT = 1,
    VARIATION_CUSTOM = 2
};

// #############################################################################
// EditorTheme - Central theme manager
// #############################################################################
class EditorTheme : public Object {
    XTU_GODOT_REGISTER_CLASS(EditorTheme, Object)

private:
    static EditorTheme* s_singleton;
    std::unordered_map<ThemeColorRole, Color> m_colors;
    std::unordered_map<ThemeIconType, Ref<Texture2D>> m_icons;
    std::unordered_map<String, Ref<Texture2D>> m_custom_icons;
    ThemeVariation m_variation = ThemeVariation::VARIATION_DARK;
    float m_icon_saturation = 1.0f;
    float m_contrast = 1.0f;
    Color m_accent_color = Color(0.3f, 0.5f, 0.9f, 1.0f);
    std::mutex m_mutex;

public:
    static EditorTheme* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("EditorTheme"); }

    EditorTheme() {
        s_singleton = this;
        initialize_default_colors();
        initialize_default_icons();
    }

    ~EditorTheme() { s_singleton = nullptr; }

    void set_variation(ThemeVariation variation) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_variation = variation;
        generate_color_scheme();
        emit_signal("theme_changed");
    }

    ThemeVariation get_variation() const { return m_variation; }

    void set_accent_color(const Color& color) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_accent_color = color;
        generate_color_scheme();
        emit_signal("theme_changed");
    }

    Color get_accent_color() const { return m_accent_color; }

    void set_contrast(float contrast) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_contrast = std::clamp(contrast, 0.5f, 2.0f);
        generate_color_scheme();
        emit_signal("theme_changed");
    }

    float get_contrast() const { return m_contrast; }

    void set_icon_saturation(float saturation) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_icon_saturation = std::clamp(saturation, 0.0f, 2.0f);
        emit_signal("theme_changed");
    }

    float get_icon_saturation() const { return m_icon_saturation; }

    Color get_color(ThemeColorRole role) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_colors.find(role);
        return it != m_colors.end() ? it->second : Color(1, 1, 1, 1);
    }

    void set_color(ThemeColorRole role, const Color& color) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_colors[role] = color;
        emit_signal("theme_changed");
    }

    Ref<Texture2D> get_icon(ThemeIconType icon) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_icons.find(icon);
        return it != m_icons.end() ? it->second : Ref<Texture2D>();
    }

    void set_icon(ThemeIconType icon, const Ref<Texture2D>& texture) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_icons[icon] = texture;
        emit_signal("theme_changed");
    }

    Ref<Texture2D> get_custom_icon(const String& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_custom_icons.find(name);
        return it != m_custom_icons.end() ? it->second : Ref<Texture2D>();
    }

    void add_custom_icon(const String& name, const Ref<Texture2D>& texture) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_custom_icons[name] = texture;
    }

    void load_from_file(const String& path) {
        std::lock_guard<std::mutex> lock(m_mutex);
        String content = FileAccess::get_file_as_string(path);
        if (content.empty()) return;

        io::json::JsonValue json = io::json::JsonValue::parse(content.to_std_string());
        if (json.is_object()) {
            if (json["colors"].is_object()) {
                for (const auto& kv : json["colors"].as_object()) {
                    ThemeColorRole role = string_to_color_role(kv.first);
                    if (role != ThemeColorRole::COLOR_MAX) {
                        m_colors[role] = parse_color(kv.second);
                    }
                }
            }
            if (json["variation"].is_string()) {
                m_variation = string_to_variation(json["variation"].as_string());
            }
            if (json["accent"].is_string()) {
                m_accent_color = Color::from_string(json["accent"].as_string().c_str());
            }
            if (json["contrast"].is_number()) {
                m_contrast = static_cast<float>(json["contrast"].as_number());
            }
        }
        generate_color_scheme();
        emit_signal("theme_changed");
    }

    void save_to_file(const String& path) {
        std::lock_guard<std::mutex> lock(m_mutex);
        io::json::JsonValue json;
        io::json::JsonValue colors_obj;
        for (const auto& kv : m_colors) {
            colors_obj[color_role_to_string(kv.first)] = io::json::JsonValue(color_to_string(kv.second));
        }
        json["colors"] = colors_obj;
        json["variation"] = io::json::JsonValue(variation_to_string(m_variation));
        json["accent"] = io::json::JsonValue(m_accent_color.to_html().to_std_string());
        json["contrast"] = io::json::JsonValue(static_cast<double>(m_contrast));

        Ref<FileAccess> file = FileAccess::open(path, FileAccess::WRITE);
        if (file.is_valid()) {
            file->store_string(json.dump(2).c_str());
            file->close();
        }
    }

    void apply_to_control(Control* control) {
        if (!control) return;
        // Apply theme colors to control and its children
    }

    Ref<Theme> generate_theme() const {
        Ref<Theme> theme;
        theme.instance();
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const auto& kv : m_colors) {
            theme->set_color(color_role_to_string(kv.first), "Control", kv.second);
        }
        for (const auto& kv : m_icons) {
            if (kv.second.is_valid()) {
                theme->set_icon(icon_type_to_string(kv.first), "Control", kv.second);
            }
        }
        return theme;
    }

private:
    void initialize_default_colors() {
        // Dark theme defaults
        m_colors[ThemeColorRole::COLOR_BACKGROUND] = Color(0.12f, 0.12f, 0.12f, 1.0f);
        m_colors[ThemeColorRole::COLOR_BASE] = Color(0.18f, 0.18f, 0.18f, 1.0f);
        m_colors[ThemeColorRole::COLOR_ALTERNATE_BASE] = Color(0.15f, 0.15f, 0.15f, 1.0f);
        m_colors[ThemeColorRole::COLOR_TEXT] = Color(0.9f, 0.9f, 0.9f, 1.0f);
        m_colors[ThemeColorRole::COLOR_TEXT_DISABLED] = Color(0.5f, 0.5f, 0.5f, 1.0f);
        m_colors[ThemeColorRole::COLOR_HIGHLIGHT] = Color(0.3f, 0.5f, 0.9f, 1.0f);
        m_colors[ThemeColorRole::COLOR_HIGHLIGHTED_TEXT] = Color(1.0f, 1.0f, 1.0f, 1.0f);
        m_colors[ThemeColorRole::COLOR_LINK] = Color(0.3f, 0.6f, 1.0f, 1.0f);
        m_colors[ThemeColorRole::COLOR_SELECTION] = Color(0.3f, 0.5f, 0.9f, 0.5f);
        m_colors[ThemeColorRole::COLOR_SELECTION_TEXT] = Color(1.0f, 1.0f, 1.0f, 1.0f);
        m_colors[ThemeColorRole::COLOR_WARNING] = Color(0.9f, 0.7f, 0.1f, 1.0f);
        m_colors[ThemeColorRole::COLOR_ERROR] = Color(0.9f, 0.2f, 0.2f, 1.0f);
        m_colors[ThemeColorRole::COLOR_SUCCESS] = Color(0.2f, 0.8f, 0.3f, 1.0f);
        m_colors[ThemeColorRole::COLOR_ACCENT] = m_accent_color;
        m_colors[ThemeColorRole::COLOR_BORDER] = Color(0.25f, 0.25f, 0.25f, 1.0f);
        m_colors[ThemeColorRole::COLOR_SEPARATOR] = Color(0.2f, 0.2f, 0.2f, 1.0f);
        m_colors[ThemeColorRole::COLOR_TOOLTIP_BACKGROUND] = Color(0.1f, 0.1f, 0.1f, 0.95f);
        m_colors[ThemeColorRole::COLOR_TOOLTIP_TEXT] = Color(0.9f, 0.9f, 0.9f, 1.0f);
        m_colors[ThemeColorRole::COLOR_SCROLLBAR] = Color(0.3f, 0.3f, 0.3f, 1.0f);
        m_colors[ThemeColorRole::COLOR_SCROLLBAR_HOVER] = Color(0.4f, 0.4f, 0.4f, 1.0f);
        m_colors[ThemeColorRole::COLOR_TAB_BACKGROUND] = Color(0.15f, 0.15f, 0.15f, 1.0f);
        m_colors[ThemeColorRole::COLOR_TAB_SELECTED] = Color(0.2f, 0.2f, 0.2f, 1.0f);
        m_colors[ThemeColorRole::COLOR_TAB_HOVER] = Color(0.18f, 0.18f, 0.18f, 1.0f);
        m_colors[ThemeColorRole::COLOR_TREE_GUIDE] = Color(0.2f, 0.2f, 0.2f, 1.0f);
    }

    void initialize_default_icons() {
        // Icons would be loaded from built-in resources
    }

    void generate_color_scheme() {
        if (m_variation == ThemeVariation::VARIATION_LIGHT) {
            m_colors[ThemeColorRole::COLOR_BACKGROUND] = Color(0.95f, 0.95f, 0.95f, 1.0f);
            m_colors[ThemeColorRole::COLOR_BASE] = Color(1.0f, 1.0f, 1.0f, 1.0f);
            m_colors[ThemeColorRole::COLOR_ALTERNATE_BASE] = Color(0.9f, 0.9f, 0.9f, 1.0f);
            m_colors[ThemeColorRole::COLOR_TEXT] = Color(0.1f, 0.1f, 0.1f, 1.0f);
            m_colors[ThemeColorRole::COLOR_TEXT_DISABLED] = Color(0.5f, 0.5f, 0.5f, 1.0f);
            m_colors[ThemeColorRole::COLOR_HIGHLIGHT] = m_accent_color;
            m_colors[ThemeColorRole::COLOR_HIGHLIGHTED_TEXT] = Color(0.1f, 0.1f, 0.1f, 1.0f);
            m_colors[ThemeColorRole::COLOR_SELECTION] = Color(m_accent_color.r(), m_accent_color.g(), m_accent_color.b(), 0.3f);
            m_colors[ThemeColorRole::COLOR_BORDER] = Color(0.7f, 0.7f, 0.7f, 1.0f);
            m_colors[ThemeColorRole::COLOR_SEPARATOR] = Color(0.8f, 0.8f, 0.8f, 1.0f);
            m_colors[ThemeColorRole::COLOR_TOOLTIP_BACKGROUND] = Color(1.0f, 1.0f, 1.0f, 0.95f);
            m_colors[ThemeColorRole::COLOR_TOOLTIP_TEXT] = Color(0.1f, 0.1f, 0.1f, 1.0f);
            m_colors[ThemeColorRole::COLOR_SCROLLBAR] = Color(0.7f, 0.7f, 0.7f, 1.0f);
            m_colors[ThemeColorRole::COLOR_TAB_BACKGROUND] = Color(0.9f, 0.9f, 0.9f, 1.0f);
            m_colors[ThemeColorRole::COLOR_TAB_SELECTED] = Color(1.0f, 1.0f, 1.0f, 1.0f);
        } else {
            initialize_default_colors();
        }
        m_colors[ThemeColorRole::COLOR_ACCENT] = m_accent_color;

        // Apply contrast
        if (m_contrast != 1.0f) {
            for (auto& kv : m_colors) {
                kv.second = kv.second.linear_interpolate(
                    m_variation == ThemeVariation::VARIATION_DARK ? Color(0, 0, 0) : Color(1, 1, 1),
                    (m_contrast - 1.0f) * 0.5f);
            }
        }
    }

    static ThemeColorRole string_to_color_role(const String& str) {
        static std::unordered_map<String, ThemeColorRole> map = {
            {"background", ThemeColorRole::COLOR_BACKGROUND},
            {"base", ThemeColorRole::COLOR_BASE},
            {"alternate_base", ThemeColorRole::COLOR_ALTERNATE_BASE},
            {"text", ThemeColorRole::COLOR_TEXT},
            {"text_disabled", ThemeColorRole::COLOR_TEXT_DISABLED},
            {"highlight", ThemeColorRole::COLOR_HIGHLIGHT},
            {"highlighted_text", ThemeColorRole::COLOR_HIGHLIGHTED_TEXT},
            {"link", ThemeColorRole::COLOR_LINK},
            {"selection", ThemeColorRole::COLOR_SELECTION},
            {"selection_text", ThemeColorRole::COLOR_SELECTION_TEXT},
            {"warning", ThemeColorRole::COLOR_WARNING},
            {"error", ThemeColorRole::COLOR_ERROR},
            {"success", ThemeColorRole::COLOR_SUCCESS},
            {"accent", ThemeColorRole::COLOR_ACCENT},
            {"border", ThemeColorRole::COLOR_BORDER},
            {"separator", ThemeColorRole::COLOR_SEPARATOR},
            {"tooltip_background", ThemeColorRole::COLOR_TOOLTIP_BACKGROUND},
            {"tooltip_text", ThemeColorRole::COLOR_TOOLTIP_TEXT},
            {"scrollbar", ThemeColorRole::COLOR_SCROLLBAR},
            {"scrollbar_hover", ThemeColorRole::COLOR_SCROLLBAR_HOVER},
            {"tab_background", ThemeColorRole::COLOR_TAB_BACKGROUND},
            {"tab_selected", ThemeColorRole::COLOR_TAB_SELECTED},
            {"tab_hover", ThemeColorRole::COLOR_TAB_HOVER},
            {"tree_guide", ThemeColorRole::COLOR_TREE_GUIDE},
        };
        auto it = map.find(str);
        return it != map.end() ? it->second : ThemeColorRole::COLOR_MAX;
    }

    static String color_role_to_string(ThemeColorRole role) {
        switch (role) {
            case ThemeColorRole::COLOR_BACKGROUND: return "background";
            case ThemeColorRole::COLOR_BASE: return "base";
            case ThemeColorRole::COLOR_ALTERNATE_BASE: return "alternate_base";
            case ThemeColorRole::COLOR_TEXT: return "text";
            case ThemeColorRole::COLOR_TEXT_DISABLED: return "text_disabled";
            case ThemeColorRole::COLOR_HIGHLIGHT: return "highlight";
            case ThemeColorRole::COLOR_HIGHLIGHTED_TEXT: return "highlighted_text";
            case ThemeColorRole::COLOR_LINK: return "link";
            case ThemeColorRole::COLOR_SELECTION: return "selection";
            case ThemeColorRole::COLOR_SELECTION_TEXT: return "selection_text";
            case ThemeColorRole::COLOR_WARNING: return "warning";
            case ThemeColorRole::COLOR_ERROR: return "error";
            case ThemeColorRole::COLOR_SUCCESS: return "success";
            case ThemeColorRole::COLOR_ACCENT: return "accent";
            case ThemeColorRole::COLOR_BORDER: return "border";
            case ThemeColorRole::COLOR_SEPARATOR: return "separator";
            case ThemeColorRole::COLOR_TOOLTIP_BACKGROUND: return "tooltip_background";
            case ThemeColorRole::COLOR_TOOLTIP_TEXT: return "tooltip_text";
            case ThemeColorRole::COLOR_SCROLLBAR: return "scrollbar";
            case ThemeColorRole::COLOR_SCROLLBAR_HOVER: return "scrollbar_hover";
            case ThemeColorRole::COLOR_TAB_BACKGROUND: return "tab_background";
            case ThemeColorRole::COLOR_TAB_SELECTED: return "tab_selected";
            case ThemeColorRole::COLOR_TAB_HOVER: return "tab_hover";
            case ThemeColorRole::COLOR_TREE_GUIDE: return "tree_guide";
            default: return "";
        }
    }

    static String icon_type_to_string(ThemeIconType icon) {
        switch (icon) {
            case ThemeIconType::ICON_FILE: return "File";
            case ThemeIconType::ICON_FOLDER: return "Folder";
            case ThemeIconType::ICON_SCENE: return "Scene";
            case ThemeIconType::ICON_SCRIPT: return "Script";
            case ThemeIconType::ICON_PLAY: return "Play";
            case ThemeIconType::ICON_STOP: return "Stop";
            case ThemeIconType::ICON_SAVE: return "Save";
            case ThemeIconType::ICON_SEARCH: return "Search";
            case ThemeIconType::ICON_SETTINGS: return "Settings";
            default: return "";
        }
    }

    static ThemeVariation string_to_variation(const String& str) {
        if (str == "light") return ThemeVariation::VARIATION_LIGHT;
        if (str == "custom") return ThemeVariation::VARIATION_CUSTOM;
        return ThemeVariation::VARIATION_DARK;
    }

    static String variation_to_string(ThemeVariation var) {
        switch (var) {
            case ThemeVariation::VARIATION_LIGHT: return "light";
            case ThemeVariation::VARIATION_CUSTOM: return "custom";
            default: return "dark";
        }
    }

    static Color parse_color(const io::json::JsonValue& json) {
        if (json.is_string()) {
            return Color::from_string(json.as_string().c_str());
        } else if (json.is_array() && json.as_array().size() >= 3) {
            float r = static_cast<float>(json[0].as_number());
            float g = static_cast<float>(json[1].as_number());
            float b = static_cast<float>(json[2].as_number());
            float a = json.as_array().size() >= 4 ? static_cast<float>(json[3].as_number()) : 1.0f;
            return Color(r, g, b, a);
        }
        return Color(1, 1, 1, 1);
    }

    static String color_to_string(const Color& color) {
        return color.to_html();
    }
};

// #############################################################################
// EditorScale - DPI and scaling management
// #############################################################################
class EditorScale : public Object {
    XTU_GODOT_REGISTER_CLASS(EditorScale, Object)

private:
    static EditorScale* s_singleton;
    float m_scale = 1.0f;
    bool m_auto_scale = true;
    float m_custom_scale = 1.0f;
    std::unordered_map<String, float> m_element_scales;

public:
    static EditorScale* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("EditorScale"); }

    EditorScale() { s_singleton = this; }
    ~EditorScale() { s_singleton = nullptr; }

    void set_scale(float scale) {
        m_custom_scale = std::clamp(scale, 0.5f, 3.0f);
        if (!m_auto_scale) {
            m_scale = m_custom_scale;
        }
        apply_scale();
    }

    float get_scale() const { return m_scale; }

    void set_auto_scale(bool enabled) {
        m_auto_scale = enabled;
        if (enabled) {
            m_scale = detect_system_scale();
        } else {
            m_scale = m_custom_scale;
        }
        apply_scale();
    }

    bool is_auto_scale() const { return m_auto_scale; }

    void set_element_scale(const String& element, float scale) {
        m_element_scales[element] = scale;
    }

    float get_element_scale(const String& element) const {
        auto it = m_element_scales.find(element);
        return it != m_element_scales.end() ? it->second : 1.0f;
    }

    int scale_int(int value) const {
        return static_cast<int>(std::round(static_cast<float>(value) * m_scale));
    }

    float scale_float(float value) const {
        return value * m_scale;
    }

    vec2f scale_vec2(const vec2f& value) const {
        return value * m_scale;
    }

    Rect2 scale_rect(const Rect2& rect) const {
        return Rect2(rect.position * m_scale, rect.size * m_scale);
    }

private:
    float detect_system_scale() const {
        // Platform-specific DPI detection
        return 1.0f;
    }

    void apply_scale() {
        emit_signal("scale_changed", m_scale);
    }
};

// #############################################################################
// EditorFonts - Font management
// #############################################################################
class EditorFonts : public Object {
    XTU_GODOT_REGISTER_CLASS(EditorFonts, Object)

private:
    static EditorFonts* s_singleton;
    String m_main_font_path;
    String m_code_font_path;
    String m_docs_font_path;
    int m_main_font_size = 14;
    int m_code_font_size = 14;
    int m_docs_font_size = 14;
    std::unordered_map<String, Ref<Font>> m_loaded_fonts;

public:
    static EditorFonts* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("EditorFonts"); }

    EditorFonts() { s_singleton = this; }
    ~EditorFonts() { s_singleton = nullptr; }

    void set_main_font(const String& path, int size = 14) {
        m_main_font_path = path;
        m_main_font_size = size;
        m_loaded_fonts.erase("main");
        emit_signal("fonts_changed");
    }

    Ref<Font> get_main_font() {
        return get_or_load_font("main", m_main_font_path, m_main_font_size);
    }

    void set_code_font(const String& path, int size = 14) {
        m_code_font_path = path;
        m_code_font_size = size;
        m_loaded_fonts.erase("code");
        emit_signal("fonts_changed");
    }

    Ref<Font> get_code_font() {
        return get_or_load_font("code", m_code_font_path, m_code_font_size);
    }

    void set_docs_font(const String& path, int size = 14) {
        m_docs_font_path = path;
        m_docs_font_size = size;
        m_loaded_fonts.erase("docs");
    }

    Ref<Font> get_docs_font() {
        return get_or_load_font("docs", m_docs_font_path, m_docs_font_size);
    }

    Ref<Font> get_custom_font(const String& path, int size) {
        String key = path + ":" + String::num(size);
        return get_or_load_font(key, path, size);
    }

private:
    Ref<Font> get_or_load_font(const String& key, const String& path, int size) {
        auto it = m_loaded_fonts.find(key);
        if (it != m_loaded_fonts.end()) {
            return it->second;
        }
        Ref<Font> font;
        if (!path.empty() && FileAccess::file_exists(path)) {
            font = ResourceLoader::load(path);
        }
        if (!font.is_valid()) {
            font.instance();
        }
        font->set_size(size);
        m_loaded_fonts[key] = font;
        return font;
    }
};

} // namespace editor

// Bring into main namespace
using editor::EditorTheme;
using editor::EditorScale;
using editor::EditorFonts;
using editor::ThemeColorRole;
using editor::ThemeIconType;
using editor::ThemeVariation;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XEDITOR_THEME_HPP