// genesis/styles.cpp

#include "genesis/styles.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace genesis {
namespace styles {

//------------------------------------------------------------------------------
// Static Color members
//------------------------------------------------------------------------------
const Color Color::WHITE(255, 255, 255);
const Color Color::BLACK(0, 0, 0);
const Color Color::RED(255, 0, 0);
const Color Color::GREEN(0, 255, 0);
const Color Color::BLUE(0, 0, 255);
const Color Color::YELLOW(255, 255, 0);
const Color Color::CYAN(0, 255, 255);
const Color Color::MAGENTA(255, 0, 255);
const Color Color::ORANGE(255, 165, 0);
const Color Color::GRAY(128, 128, 128);
const Color Color::DARK_GRAY(64, 64, 64);
const Color Color::LIGHT_GRAY(192, 192, 192);
const Color Color::TRANSPARENT(0, 0, 0, 0);

//------------------------------------------------------------------------------
// Predefined Themes
//------------------------------------------------------------------------------
namespace {
    Theme dark_theme;
    Theme light_theme;
    Theme dumb_theme;
    bool themes_initialized = false;

    void initialize_themes() {
        if (themes_initialized) return;

        // Dark theme (default)
        dark_theme.name = "dark";
        dark_theme.type = ThemeType::DARK;
        dark_theme.background_primary = Color(30, 30, 30);
        dark_theme.background_secondary = Color(45, 45, 45);
        dark_theme.background_tertiary = Color(60, 60, 60);
        dark_theme.text_primary = Color(240, 240, 240);
        dark_theme.text_secondary = Color(180, 180, 180);
        dark_theme.text_disabled = Color(100, 100, 100);
        dark_theme.text_inverse = Color(30, 30, 30);
        dark_theme.accent_primary = Color(0, 120, 215);
        dark_theme.accent_secondary = Color(0, 90, 158);
        dark_theme.accent_tertiary = Color(0, 60, 105);
        dark_theme.success = Color(80, 200, 120);
        dark_theme.warning = Color(255, 185, 0);
        dark_theme.error = Color(232, 17, 35);
        dark_theme.info = Color(0, 120, 215);
        dark_theme.border = Color(80, 80, 80);
        dark_theme.separator = Color(60, 60, 60);
        dark_theme.overlay = Color(0, 0, 0, 180);
        dark_theme.shadow = Color(0, 0, 0, 100);
        dark_theme.grid = Color(100, 100, 100, 80);
        dark_theme.axis_x = Color(255, 80, 80);
        dark_theme.axis_y = Color(80, 255, 80);
        dark_theme.axis_z = Color(80, 80, 255);
        dark_theme.trajectory = Color(255, 255, 0, 200);
        dark_theme.contact_point = Color(255, 0, 0);
        dark_theme.force_vector = Color(255, 0, 255);
        dark_theme.velocity_vector = Color(0, 255, 255);
        dark_theme.rigid_body_color = Color(100, 150, 200);
        dark_theme.soft_body_color = Color(200, 120, 100);
        dark_theme.fluid_color = Color(0, 150, 200);
        dark_theme.cloth_color = Color(150, 150, 200);
        dark_theme.rope_color = Color(200, 200, 100);

        // Light theme
        light_theme.name = "light";
        light_theme.type = ThemeType::LIGHT;
        light_theme.background_primary = Color(245, 245, 245);
        light_theme.background_secondary = Color(255, 255, 255);
        light_theme.background_tertiary = Color(230, 230, 230);
        light_theme.text_primary = Color(20, 20, 20);
        light_theme.text_secondary = Color(80, 80, 80);
        light_theme.text_disabled = Color(160, 160, 160);
        light_theme.text_inverse = Color(240, 240, 240);
        light_theme.accent_primary = Color(0, 120, 215);
        light_theme.accent_secondary = Color(0, 90, 158);
        light_theme.accent_tertiary = Color(0, 60, 105);
        light_theme.success = Color(0, 150, 0);
        light_theme.warning = Color(255, 140, 0);
        light_theme.error = Color(200, 0, 0);
        light_theme.info = Color(0, 100, 200);
        light_theme.border = Color(200, 200, 200);
        light_theme.separator = Color(220, 220, 220);
        light_theme.overlay = Color(0, 0, 0, 50);
        light_theme.shadow = Color(0, 0, 0, 30);
        light_theme.grid = Color(150, 150, 150, 80);
        light_theme.axis_x = Color(200, 0, 0);
        light_theme.axis_y = Color(0, 150, 0);
        light_theme.axis_z = Color(0, 0, 200);
        light_theme.trajectory = Color(200, 200, 0, 180);
        light_theme.contact_point = Color(200, 0, 0);
        light_theme.force_vector = Color(200, 0, 200);
        light_theme.velocity_vector = Color(0, 150, 150);
        light_theme.rigid_body_color = Color(100, 120, 180);
        light_theme.soft_body_color = Color(180, 100, 80);
        light_theme.fluid_color = Color(0, 120, 180);
        light_theme.cloth_color = Color(120, 120, 180);
        light_theme.rope_color = Color(180, 180, 80);

        // Dumb theme (no colors / high contrast)
        dumb_theme.name = "dumb";
        dumb_theme.type = ThemeType::DUMB;
        dumb_theme.background_primary = Color(0, 0, 0);
        dumb_theme.background_secondary = Color(0, 0, 0);
        dumb_theme.background_tertiary = Color(0, 0, 0);
        dumb_theme.text_primary = Color(255, 255, 255);
        dumb_theme.text_secondary = Color(255, 255, 255);
        dumb_theme.text_disabled = Color(128, 128, 128);
        dumb_theme.text_inverse = Color(0, 0, 0);
        dumb_theme.accent_primary = Color(255, 255, 255);
        dumb_theme.accent_secondary = Color(255, 255, 255);
        dumb_theme.accent_tertiary = Color(255, 255, 255);
        dumb_theme.success = Color(255, 255, 255);
        dumb_theme.warning = Color(255, 255, 255);
        dumb_theme.error = Color(255, 255, 255);
        dumb_theme.info = Color(255, 255, 255);
        dumb_theme.border = Color(255, 255, 255);
        dumb_theme.separator = Color(255, 255, 255);
        dumb_theme.overlay = Color(0, 0, 0);
        dumb_theme.shadow = Color(0, 0, 0);
        dumb_theme.grid = Color(255, 255, 255, 80);
        dumb_theme.axis_x = Color(255, 255, 255);
        dumb_theme.axis_y = Color(255, 255, 255);
        dumb_theme.axis_z = Color(255, 255, 255);
        dumb_theme.trajectory = Color(255, 255, 255);
        dumb_theme.contact_point = Color(255, 255, 255);
        dumb_theme.force_vector = Color(255, 255, 255);
        dumb_theme.velocity_vector = Color(255, 255, 255);
        dumb_theme.rigid_body_color = Color(255, 255, 255);
        dumb_theme.soft_body_color = Color(255, 255, 255);
        dumb_theme.fluid_color = Color(255, 255, 255);
        dumb_theme.cloth_color = Color(255, 255, 255);
        dumb_theme.rope_color = Color(255, 255, 255);

        themes_initialized = true;
    }
}

// Theme constructor - initializes with dark theme values
Theme::Theme(ThemeType t) : Theme() {
    initialize_themes();
    if (t == ThemeType::DARK) {
        *this = dark_theme;
    } else if (t == ThemeType::LIGHT) {
        *this = light_theme;
    } else {
        *this = dumb_theme;
    }
}

const Theme& Theme::get_theme(ThemeType type) {
    initialize_themes();
    switch (type) {
        case ThemeType::DARK:  return dark_theme;
        case ThemeType::LIGHT: return light_theme;
        default:               return dumb_theme;
    }
}

const Theme& Theme::get_theme(const std::string& name) {
    initialize_themes();
    if (name == "dark" || name == "Dark") {
        return dark_theme;
    } else if (name == "light" || name == "Light") {
        return light_theme;
    } else if (name == "dumb" || name == "Dumb") {
        return dumb_theme;
    }
    // Check custom themes
    auto it = StyleManager::custom_themes_.find(name);
    if (it != StyleManager::custom_themes_.end()) {
        return it->second;
    }
    // Default to dark
    return dark_theme;
}

void Theme::register_theme(const std::string& name, const Theme& theme) {
    StyleManager::custom_themes_[name] = theme;
}

//------------------------------------------------------------------------------
// StyleManager implementation
//------------------------------------------------------------------------------
std::unordered_map<std::string, Theme> StyleManager::custom_themes_;

StyleManager::StyleManager()
    : current_theme_type_(ThemeType::DARK)
    , ui_scale_(1.0f)
    , font_scale_(1.0f)
    , line_width_(1.5f)
    , point_size_(3.0f)
    , overlay_opacity_(0.7f)
    , shadow_opacity_(0.4f)
    , show_grid_(true)
    , show_axes_(true)
    , show_contacts_(false)
    , show_forces_(false)
    , show_velocities_(false)
    , show_trajectories_(false)
    , wireframe_mode_(false)
    , lighting_enabled_(true)
{
}

StyleManager& StyleManager::instance() {
    static StyleManager instance;
    return instance;
}

void StyleManager::set_theme(ThemeType type) {
    current_theme_type_ = type;
    custom_theme_name_.clear();
}

void StyleManager::set_theme(const std::string& name) {
    const Theme& theme = Theme::get_theme(name);
    current_theme_type_ = theme.type;
    custom_theme_name_ = name;
}

const Theme& StyleManager::current_theme() const {
    if (!custom_theme_name_.empty()) {
        return Theme::get_theme(custom_theme_name_);
    }
    return Theme::get_theme(current_theme_type_);
}

void StyleManager::set_ui_scale(float scale) { ui_scale_ = std::max(0.5f, std::min(scale, 3.0f)); }
float StyleManager::ui_scale() const { return ui_scale_; }

void StyleManager::set_font_scale(float scale) { font_scale_ = std::max(0.5f, std::min(scale, 3.0f)); }
float StyleManager::font_scale() const { return font_scale_; }

void StyleManager::set_line_width(float width) { line_width_ = std::max(0.1f, width); }
float StyleManager::line_width() const { return line_width_; }

void StyleManager::set_point_size(float size) { point_size_ = std::max(0.1f, size); }
float StyleManager::point_size() const { return point_size_; }

void StyleManager::set_overlay_opacity(float opacity) { overlay_opacity_ = std::clamp(opacity, 0.0f, 1.0f); }
float StyleManager::overlay_opacity() const { return overlay_opacity_; }

void StyleManager::set_shadow_opacity(float opacity) { shadow_opacity_ = std::clamp(opacity, 0.0f, 1.0f); }
float StyleManager::shadow_opacity() const { return shadow_opacity_; }

void StyleManager::set_show_grid(bool show) { show_grid_ = show; }
bool StyleManager::show_grid() const { return show_grid_; }

void StyleManager::set_show_axes(bool show) { show_axes_ = show; }
bool StyleManager::show_axes() const { return show_axes_; }

void StyleManager::set_show_contacts(bool show) { show_contacts_ = show; }
bool StyleManager::show_contacts() const { return show_contacts_; }

void StyleManager::set_show_forces(bool show) { show_forces_ = show; }
bool StyleManager::show_forces() const { return show_forces_; }

void StyleManager::set_show_velocities(bool show) { show_velocities_ = show; }
bool StyleManager::show_velocities() const { return show_velocities_; }

void StyleManager::set_show_trajectories(bool show) { show_trajectories_ = show; }
bool StyleManager::show_trajectories() const { return show_trajectories_; }

void StyleManager::set_wireframe_mode(bool wireframe) { wireframe_mode_ = wireframe; }
bool StyleManager::wireframe_mode() const { return wireframe_mode_; }

void StyleManager::set_lighting_enabled(bool enabled) { lighting_enabled_ = enabled; }
bool StyleManager::lighting_enabled() const { return lighting_enabled_; }

void StyleManager::reset_to_defaults() {
    *this = StyleManager();
}

//------------------------------------------------------------------------------
// Palette definitions
//------------------------------------------------------------------------------
namespace palette {
    const std::array<Color, 10> MATERIAL_COLORS = {
        Color(200, 100, 100), // Material 0
        Color(100, 200, 100),
        Color(100, 100, 200),
        Color(200, 200, 100),
        Color(200, 100, 200),
        Color(100, 200, 200),
        Color(150, 150, 150),
        Color(200, 150, 100),
        Color(150, 100, 200),
        Color(200, 200, 200)
    };

    const Color RIGID_BODY(100, 150, 200);
    const Color SOFT_BODY(200, 120, 100);
    const Color FLUID(0, 150, 200);
    const Color CLOTH(150, 150, 200);
    const Color ROPE(200, 200, 100);
    const Color GRANULAR(200, 180, 100);
    const Color EMPTY(128, 128, 128, 128);

    const Color FORCE_NORMAL(255, 0, 0);
    const Color FORCE_FRICTION(0, 255, 0);
    const Color FORCE_SPRING(0, 0, 255);
    const Color FORCE_DAMPING(255, 255, 0);
    const Color FORCE_EXTERNAL(255, 0, 255);

    Color heatmap_gradient(float t) {
        t = std::clamp(t, 0.0f, 1.0f);
        float r, g, b;
        if (t < 0.5f) {
            r = 0.0f;
            g = t * 2.0f;
            b = 1.0f - t * 2.0f;
        } else {
            r = (t - 0.5f) * 2.0f;
            g = 1.0f - (t - 0.5f) * 2.0f;
            b = 0.0f;
        }
        return Color(static_cast<uint8_t>(r * 255),
                     static_cast<uint8_t>(g * 255),
                     static_cast<uint8_t>(b * 255));
    }

    Color coolwarm_gradient(float t) {
        t = std::clamp(t, 0.0f, 1.0f);
        float r, g, b;
        if (t < 0.5f) {
            float s = t * 2.0f;
            r = 0.0f;
            g = s;
            b = 1.0f;
        } else {
            float s = (t - 0.5f) * 2.0f;
            r = s;
            g = 1.0f - s;
            b = 1.0f - s;
        }
        return Color(static_cast<uint8_t>(r * 255),
                     static_cast<uint8_t>(g * 255),
                     static_cast<uint8_t>(b * 255));
    }

    Color viridis_gradient(float t) {
        t = std::clamp(t, 0.0f, 1.0f);
        // Simplified Viridis approximation
        float r = std::max(0.0f, std::min(1.0f, 0.28f + t * 0.5f));
        float g = std::max(0.0f, std::min(1.0f, 0.3f + t * 0.7f));
        float b = std::max(0.0f, std::min(1.0f, 0.5f - t * 0.2f));
        return Color(static_cast<uint8_t>(r * 255),
                     static_cast<uint8_t>(g * 255),
                     static_cast<uint8_t>(b * 255));
    }

    Color plasma_gradient(float t) {
        t = std::clamp(t, 0.0f, 1.0f);
        float r = std::min(1.0f, t * 1.2f);
        float g = std::max(0.0f, std::min(1.0f, t * 0.8f));
        float b = std::max(0.0f, std::min(1.0f, 1.0f - t * 0.5f));
        return Color(static_cast<uint8_t>(r * 255),
                     static_cast<uint8_t>(g * 255),
                     static_cast<uint8_t>(b * 255));
    }
}

} // namespace styles
} // namespace genesis