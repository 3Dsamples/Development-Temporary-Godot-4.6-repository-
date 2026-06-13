// label_3d.h
#pragma once

#include "geometry_instance_3d.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lighting {

// ============================================================================
// Label3D – renders dynamic text in 3D space.
// Supports TrueType/bitmap fonts, color, outline, shadow, and full lighting
// (shadows, GI, emissive). Optimized by updating mesh only on text change.
// ============================================================================

class Label3D : public GeometryInstance3D {
public:
    Label3D();
    ~Label3D();

    // ------------------------------------------------------------------------
    // Text content
    // ------------------------------------------------------------------------
    void set_text(const char* utf8_text);
    const char* get_text() const;

    // ------------------------------------------------------------------------
    // Font and size
    // ------------------------------------------------------------------------
    void set_font(const char* font_path, int size); // loads from file
    void set_font_size(int size);
    int get_font_size() const;
    void set_font_antialiased(bool antialiased);
    bool is_font_antialiased() const;

    // ------------------------------------------------------------------------
    // Appearance (color, outline, shadow)
    // ------------------------------------------------------------------------
    void set_color(const float* rgba);        // RGBA
    void get_color(float* out_rgba) const;
    void set_outline_enabled(bool enabled);
    bool is_outline_enabled() const;
    void set_outline_color(const float* rgba);
    void get_outline_color(float* out_rgba) const;
    void set_outline_size(float size);
    float get_outline_size() const;
    void set_shadow_enabled(bool enabled);
    bool is_shadow_enabled() const;
    void set_shadow_color(const float* rgba);
    void get_shadow_color(float* out_rgba) const;
    void set_shadow_offset(const double* offset); // 2D offset (x,y)
    void get_shadow_offset(double* out_offset) const;

    // ------------------------------------------------------------------------
    // Layout
    // ------------------------------------------------------------------------
    void set_alignment(int mode);      // 0=left, 1=center, 2=right
    int get_alignment() const;
    void set_line_spacing(float spacing);
    float get_line_spacing() const;
    void set_width(float max_width);   // 0 = no wrap
    float get_width() const;
    void set_auto_align(bool auto_align);
    bool get_auto_align() const;

    // ------------------------------------------------------------------------
    // Billboard mode (always face camera)
    // ------------------------------------------------------------------------
    void set_billboard(bool enabled);
    bool is_billboard() const;
    void set_billboard_axis(int axis); // 0=free, 1=Y axis only
    int get_billboard_axis() const;

    // ------------------------------------------------------------------------
    // Depth test / transparency
    // ------------------------------------------------------------------------
    void set_depth_test_enabled(bool enabled);
    bool is_depth_test_enabled() const;
    void set_transparent(bool transparent);
    bool is_transparent() const;

    // ------------------------------------------------------------------------
    // Lighting & GI (inherited)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;
    void set_gi_contribution(float amount) override;
    void set_emissive(const float* color, float intensity);
    void get_emissive(float* out_color, float& out_intensity) const;

    // ------------------------------------------------------------------------
    // Force mesh update (call after changing any property)
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