// label_3d.h
#pragma once

#include "geometry_instance_3d.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lighting {

// ============================================================================
// Label3D – 3D text label with font support, billboard modes, and full lighting.
// Uses signed‑distance field (SDF) or cached bitmap glyphs for efficient rendering.
// Supports dynamic text changes, alignment, and per‑glyph color modulation.
// ============================================================================

enum class BillboardMode : uint8_t {
    DISABLED,           // fixed orientation in world space
    ENABLED,            // always face camera (full billboard)
    FIXED_Y,            // rotate around Y axis only, keep vertical alignment
    FIXED_X
};

enum class HorizontalAlignment : uint8_t {
    LEFT,
    CENTER,
    RIGHT
};

enum class VerticalAlignment : uint8_t {
    TOP,
    CENTER,
    BOTTOM
};

class Font; // forward declaration (resource)

class Label3D : public GeometryInstance3D {
public:
    Label3D();
    ~Label3D();

    // ------------------------------------------------------------------------
    // Text content
    // ------------------------------------------------------------------------
    void set_text(const char* text);
    const char* get_text() const;
    void set_font(Font* font);
    Font* get_font() const;
    void set_font_size(int size);   // pixels (in texture space)
    int get_font_size() const;

    // ------------------------------------------------------------------------
    // Appearance
    // ------------------------------------------------------------------------
    void set_color(const float* rgba);    // modulate color (RGB + alpha)
    void get_color(float* out_rgba) const;
    void set_outline_modulate(const float* rgba);
    void get_outline_modulate(float* out_rgba) const;
    void set_outline_size(float size);    // pixels
    float get_outline_size() const;
    void set_shadow_enabled(bool enabled);
    bool is_shadow_enabled() const;
    void set_shadow_offset(const double* offset); // in 3D world units
    void get_shadow_offset(double* out_offset) const;
    void set_shadow_color(const float* rgba);
    void get_shadow_color(float* out_rgba) const;

    // ------------------------------------------------------------------------
    // Alignment and extents
    // ------------------------------------------------------------------------
    void set_horizontal_alignment(HorizontalAlignment align);
    HorizontalAlignment get_horizontal_alignment() const;
    void set_vertical_alignment(VerticalAlignment align);
    VerticalAlignment get_vertical_alignment() const;
    void set_line_spacing(float spacing);   // multiplier (1.0 = normal)
    float get_line_spacing() const;
    void set_autowrap_enabled(bool enabled);
    bool is_autowrap_enabled() const;
    void set_autowrap_width(double width);  // in world units (pixel equivalent depends on font size)
    double get_autowrap_width() const;

    // ------------------------------------------------------------------------
    // Billboard and orientation
    // ------------------------------------------------------------------------
    void set_billboard_mode(BillboardMode mode);
    BillboardMode get_billboard_mode() const;
    void set_fixed_orientation(const double* forward, const double* up); // for billboard disabled
    void get_fixed_orientation(double* out_forward, double* out_up) const;
    void set_pixel_offset(const double* offset); // offset in screen pixels (for billboard only)
    void get_pixel_offset(double* out_offset) const;

    // ------------------------------------------------------------------------
    // Depth and transparency
    // ------------------------------------------------------------------------
    void set_double_sided(bool double_sided);
    bool is_double_sided() const;
    void set_depth_test_enabled(bool enabled);
    bool is_depth_test_enabled() const;

    // ------------------------------------------------------------------------
    // Lighting & GI (same as GeometryInstance3D)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;
    void set_gi_contribution(float amount) override;
    void set_emissive(const float* color, float intensity);
    void get_emissive(float* out_color, float& out_intensity) const;

    // ------------------------------------------------------------------------
    // Material override (advanced: per‑character material)
    // ------------------------------------------------------------------------
    void set_material_override(int material_id);
    int get_material_override() const;
    void set_vertex_color_enabled(bool enabled);
    bool is_vertex_color_enabled() const;

    // ------------------------------------------------------------------------
    // Force update (call after changing text / font / size)
    // ------------------------------------------------------------------------
    void update_label();

    // ------------------------------------------------------------------------
    // Node overrides
    // ------------------------------------------------------------------------
    void ready() override;
    void process(double delta) override;
    void synchronize_render_server(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting