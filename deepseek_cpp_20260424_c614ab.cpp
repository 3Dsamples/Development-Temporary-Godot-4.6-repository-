// genesis/styles.h

#pragma once

#include <string>
#include <unordered_map>
#include <array>
#include <cstdint>

namespace genesis {
namespace styles {

//------------------------------------------------------------------------------
// Color structure (RGBA)
//------------------------------------------------------------------------------
struct Color {
    uint8_t r, g, b, a;

    constexpr Color() : r(255), g(255), b(255), a(255) {}
    constexpr Color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255)
        : r(r), g(g), b(b), a(a) {}

    // Convert to hex string (e.g., "#RRGGBB" or "#RRGGBBAA")
    std::string to_hex(bool include_alpha = false) const {
        char buf[16];
        if (include_alpha) {
            std::snprintf(buf, sizeof(buf), "#%02X%02X%02X%02X", r, g, b, a);
        } else {
            std::snprintf(buf, sizeof(buf), "#%02X%02X%02X", r, g, b);
        }
        return std::string(buf);
    }

    // Convert to uint32_t (RGBA packed)
    constexpr uint32_t to_uint32() const {
        return (static_cast<uint32_t>(r) << 24) |
               (static_cast<uint32_t>(g) << 16) |
               (static_cast<uint32_t>(b) << 8)  |
               static_cast<uint32_t>(a);
    }

    // Create from uint32_t (RGBA packed)
    static constexpr Color from_uint32(uint32_t rgba) {
        return Color(
            static_cast<uint8_t>((rgba >> 24) & 0xFF),
            static_cast<uint8_t>((rgba >> 16) & 0xFF),
            static_cast<uint8_t>((rgba >> 8) & 0xFF),
            static_cast<uint8_t>(rgba & 0xFF)
        );
    }

    // Common colors as static members
    static const Color WHITE;
    static const Color BLACK;
    static const Color RED;
    static const Color GREEN;
    static const Color BLUE;
    static const Color YELLOW;
    static const Color CYAN;
    static const Color MAGENTA;
    static const Color ORANGE;
    static const Color GRAY;
    static const Color DARK_GRAY;
    static const Color LIGHT_GRAY;
    static const Color TRANSPARENT;
};

//------------------------------------------------------------------------------
// Theme definitions
//------------------------------------------------------------------------------
enum class ThemeType : uint8_t {
    DARK = 0,
    LIGHT = 1,
    DUMB = 2
};

class Theme {
public:
    std::string name;
    ThemeType type;

    // Background colors
    Color background_primary;
    Color background_secondary;
    Color background_tertiary;

    // Text colors
    Color text_primary;
    Color text_secondary;
    Color text_disabled;
    Color text_inverse;

    // Accent colors
    Color accent_primary;
    Color accent_secondary;
    Color accent_tertiary;

    // Status colors
    Color success;
    Color warning;
    Color error;
    Color info;

    // UI element colors
    Color border;
    Color separator;
    Color overlay;
    Color shadow;

    // Visualizer colors
    Color grid;
    Color axis_x;
    Color axis_y;
    Color axis_z;
    Color trajectory;
    Color contact_point;
    Color force_vector;
    Color velocity_vector;

    // Entity default colors
    Color rigid_body_color;
    Color soft_body_color;
    Color fluid_color;
    Color cloth_color;
    Color rope_color;

    // Constructor
    Theme(ThemeType t = ThemeType::DARK);

    // Get a pre-defined theme
    static const Theme& get_theme(ThemeType type);
    static const Theme& get_theme(const std::string& name);

    // Register a custom theme
    static void register_theme(const std::string& name, const Theme& theme);
};

//------------------------------------------------------------------------------
// Style manager (global style settings)
//------------------------------------------------------------------------------
class StyleManager {
public:
    static StyleManager& instance();

    // Set the active theme
    void set_theme(ThemeType type);
    void set_theme(const std::string& name);
    const Theme& current_theme() const;

    // Global scaling factors
    void set_ui_scale(float scale);
    float ui_scale() const;

    void set_font_scale(float scale);
    float font_scale() const;

    void set_line_width(float width);
    float line_width() const;

    void set_point_size(float size);
    float point_size() const;

    // Opacity settings
    void set_overlay_opacity(float opacity);
    float overlay_opacity() const;

    void set_shadow_opacity(float opacity);
    float shadow_opacity() const;

    // Visualization flags
    void set_show_grid(bool show);
    bool show_grid() const;

    void set_show_axes(bool show);
    bool show_axes() const;

    void set_show_contacts(bool show);
    bool show_contacts() const;

    void set_show_forces(bool show);
    bool show_forces() const;

    void set_show_velocities(bool show);
    bool show_velocities() const;

    void set_show_trajectories(bool show);
    bool show_trajectories() const;

    void set_wireframe_mode(bool wireframe);
    bool wireframe_mode() const;

    void set_lighting_enabled(bool enabled);
    bool lighting_enabled() const;

    // Reset to defaults
    void reset_to_defaults();

private:
    StyleManager();
    ~StyleManager() = default;

    ThemeType current_theme_type_;
    std::string custom_theme_name_;
    float ui_scale_;
    float font_scale_;
    float line_width_;
    float point_size_;
    float overlay_opacity_;
    float shadow_opacity_;
    bool show_grid_;
    bool show_axes_;
    bool show_contacts_;
    bool show_forces_;
    bool show_velocities_;
    bool show_trajectories_;
    bool wireframe_mode_;
    bool lighting_enabled_;

    static std::unordered_map<std::string, Theme> custom_themes_;
};

//------------------------------------------------------------------------------
// ANSI terminal color codes (for console output)
//------------------------------------------------------------------------------
namespace ansi {
    // Foreground colors
    constexpr const char* RESET   = "\033[0m";
    constexpr const char* BLACK   = "\033[30m";
    constexpr const char* RED     = "\033[31m";
    constexpr const char* GREEN   = "\033[32m";
    constexpr const char* YELLOW  = "\033[33m";
    constexpr const char* BLUE    = "\033[34m";
    constexpr const char* MAGENTA = "\033[35m";
    constexpr const char* CYAN    = "\033[36m";
    constexpr const char* WHITE   = "\033[37m";
    constexpr const char* GRAY    = "\033[90m";

    // Bright foreground
    constexpr const char* BRIGHT_BLACK   = "\033[90m";
    constexpr const char* BRIGHT_RED     = "\033[91m";
    constexpr const char* BRIGHT_GREEN   = "\033[92m";
    constexpr const char* BRIGHT_YELLOW  = "\033[93m";
    constexpr const char* BRIGHT_BLUE    = "\033[94m";
    constexpr const char* BRIGHT_MAGENTA = "\033[95m";
    constexpr const char* BRIGHT_CYAN    = "\033[96m";
    constexpr const char* BRIGHT_WHITE   = "\033[97m";

    // Background colors
    constexpr const char* BG_BLACK   = "\033[40m";
    constexpr const char* BG_RED     = "\033[41m";
    constexpr const char* BG_GREEN   = "\033[42m";
    constexpr const char* BG_YELLOW  = "\033[43m";
    constexpr const char* BG_BLUE    = "\033[44m";
    constexpr const char* BG_MAGENTA = "\033[45m";
    constexpr const char* BG_CYAN    = "\033[46m";
    constexpr const char* BG_WHITE   = "\033[47m";

    // Text styles
    constexpr const char* BOLD      = "\033[1m";
    constexpr const char* UNDERLINE = "\033[4m";
    constexpr const char* INVERT    = "\033[7m";

    // Helper to colorize a string
    inline std::string colorize(const std::string& text, const char* color_code) {
        return std::string(color_code) + text + RESET;
    }

    inline std::string bold(const std::string& text) {
        return std::string(BOLD) + text + RESET;
    }

    inline std::string underline(const std::string& text) {
        return std::string(UNDERLINE) + text + RESET;
    }
}

//------------------------------------------------------------------------------
// Pre-defined color palettes for various purposes
//------------------------------------------------------------------------------
namespace palette {
    // Material colors
    extern const std::array<Color, 10> MATERIAL_COLORS;

    // Entity type colors
    extern const Color RIGID_BODY;
    extern const Color SOFT_BODY;
    extern const Color FLUID;
    extern const Color CLOTH;
    extern const Color ROPE;
    extern const Color GRANULAR;
    extern const Color EMPTY;

    // Force visualization
    extern const Color FORCE_NORMAL;
    extern const Color FORCE_FRICTION;
    extern const Color FORCE_SPRING;
    extern const Color FORCE_DAMPING;
    extern const Color FORCE_EXTERNAL;

    // Get a color from a gradient (t in [0,1])
    Color heatmap_gradient(float t);
    Color coolwarm_gradient(float t);
    Color viridis_gradient(float t);
    Color plasma_gradient(float t);
}

} // namespace styles
} // namespace genesis