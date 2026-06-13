// label_3d.h
#pragma once

#include "geometry_instance_3d.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lighting {

// ============================================================================
// Label3D – 3D text label with dynamic font, alignment, and shading.
// Supports bitmap fonts (BMFont) or dynamic signed distance field (SDF) fonts.
// Integrated with lighting: casts shadows (as a texture billboard or mesh),
// receives GI, can be emissive (glowing text). Optimized for many labels
// using GPU instancing and texture atlases.
// ============================================================================

class Label3D : public GeometryInstance3D {
public:
    Label3D();
    ~Label3D();

    // ------------------------------------------------------------------------
    // Text content
    // ------------------------------------------------------------------------
    void set_text(const char* text);
    const char* get_text() const;
    void set_font(const char* font_path); // BMFont .fnt or TTF with SDF
    const char* get_font() const;

    // ------------------------------------------------------------------------
    // Font size and resolution
    // ------------------------------------------------------------------------
    void set_font_size(int size_pixels);
    int get_font_size() const;
    void set_outline_size(int pixels);
    int get_outline_size() const;

    // ------------------------------------------------------------------------
    // Alignment and layout
    // ------------------------------------------------------------------------
    void set_horizontal_alignment(int align); // 0=left,1=center,2=right
    int get_horizontal_alignment() const;
    void set_vertical_alignment(int align);   // 0=top,1=center,2=bottom
    int get_vertical_alignment() const;
    void set_width(float width); // wrap width (0 = no wrap)
    float get_width() const;
    void set_line_spacing(float spacing);
    float get_line_spacing() const;

    // ------------------------------------------------------------------------
    // Color and material
    // ------------------------------------------------------------------------
    void set_color(float r, float g, float b, float a = 1.0f);
    void get_color(float* out_rgba) const;
    void set_outline_color(float r, float g, float b, float a = 1.0f);
    void get_outline_color(float* out_rgba) const;
    void set_material(const char* material_path); // custom material override
    const char* get_material() const;

    // ------------------------------------------------------------------------
    // Billboard mode (always face camera)
    // ------------------------------------------------------------------------
    void set_billboard_enabled(bool enabled);
    bool is_billboard_enabled() const;
    void set_billboard_axis(int axis); // 0=all, 1=Y, 2=XZ
    int get_billboard_axis() const;

    // ------------------------------------------------------------------------
    // Pixel size (world size per font pixel)
    // ------------------------------------------------------------------------
    void set_pixel_size(double size);
    double get_pixel_size() const;

    // ------------------------------------------------------------------------
    // Lighting & GI (text can be affected by lights)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;
    void set_gi_contribution(float amount) override;
    void set_emissive(const float* color, float intensity) override;
    void get_emissive(float* out_color, float& out_intensity) const;

    // ------------------------------------------------------------------------
    // Force rebuild (after text or font changes)
    // ------------------------------------------------------------------------
    void update_label();

    // ------------------------------------------------------------------------
    // Node overrides
    // ------------------------------------------------------------------------
    void process(double delta) override;
    void synchronize_render_server(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting