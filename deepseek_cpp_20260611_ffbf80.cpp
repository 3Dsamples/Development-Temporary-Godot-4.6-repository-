// texture_rect_3d.h
#pragma once

#include "geometry_instance_3d.h"
#include <cstdint>
#include <memory>

namespace lighting {

// ============================================================================
// TextureRect3D – a 3D rectangle (plane) that displays a 2D texture.
// Supports texture repeat, flip, alpha blending, billboard, and full lighting
// (can be unlit or affected by scene lights, shadows, GI, and emissive).
// ============================================================================

enum class TextureRectBillboard : uint8_t {
    DISABLED,
    ENABLED,        // always face camera (full billboard)
    FIXED_Y,        // rotate only around Y axis
    FIXED_X
};

enum class TextureRectMode : uint8_t {
    STRETCH,        // fill the rectangle (stretch texture)
    CENTER,         // keep original size, centered
    TILE,           // repeat texture to fill rectangle
    FIT             // scale to fit inside rectangle (maintain aspect)
};

class TextureRect3D : public GeometryInstance3D {
public:
    TextureRect3D();
    ~TextureRect3D();

    // ------------------------------------------------------------------------
    // Texture & material
    // ------------------------------------------------------------------------
    void set_texture(int64_t texture_rid);   // RenderingServer texture ID
    int64_t get_texture_rid() const;
    void set_material(int64_t material_rid); // custom material (override)
    int64_t get_material_rid() const;
    void set_texture_filter(int filter);     // 0 = nearest, 1 = linear, 2 = mipmap
    int get_texture_filter() const;
    void set_texture_repeat(bool repeat_u, bool repeat_v);
    void get_texture_repeat(bool& repeat_u, bool& repeat_v) const;

    // ------------------------------------------------------------------------
    // Geometry & transform
    // ------------------------------------------------------------------------
    void set_size(double width, double height);
    void get_size(double& width, double& height) const;
    void set_offset(const double* offset);   // local offset from origin
    void get_offset(double* out_offset) const;
    void set_flip(bool flip_h, bool flip_v);
    void get_flip(bool& flip_h, bool& flip_v) const;

    // ------------------------------------------------------------------------
    // Billboard / face camera
    // ------------------------------------------------------------------------
    void set_billboard_mode(TextureRectBillboard mode);
    TextureRectBillboard get_billboard_mode() const;
    void set_pixel_offset(const double* offset); // screen pixels offset (only for billboard)
    void get_pixel_offset(double* out_offset) const;

    // ------------------------------------------------------------------------
    // Color & transparency
    // ------------------------------------------------------------------------
    void set_modulate(const float* rgba);
    void get_modulate(float* out_rgba) const;
    void set_opacity(float opacity);
    float get_opacity() const;
    void set_transparent(bool transparent);
    bool is_transparent() const;

    // ------------------------------------------------------------------------
    // Lighting & shadows (geometry instance overrides)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;
    void set_gi_contribution(float amount) override;
    void set_emissive(const float* color, float intensity) override;
    void get_emissive(float* out_color, float& out_intensity) const override;

    // ------------------------------------------------------------------------
    // Force update (if texture or geometry changed)
    // ------------------------------------------------------------------------
    void update_rect();

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