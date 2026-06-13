// Name : lighting enhancement
// File : scene/3d/texture_rect_3d_ext.h 55 of 60
// Description : Extended 3D texture rectangle node with texture filtering, billboard,
//               modulate color, transparency, flip, and full RenderingServer sync.
#pragma once

#include "scene/3d/texture_rect_3d.h"
#include "servers/rendering_server.h"

class TextureRect3DExt : public TextureRect3D {
    GDCLASS(TextureRect3DExt, TextureRect3D);

public:
    TextureRect3DExt();
    ~TextureRect3DExt();

    // ------------------------------------------------------------------------
    // Texture and material
    // ------------------------------------------------------------------------
    void set_texture(const RID &p_texture);
    RID get_texture() const;
    void set_material(const RID &p_material);
    RID get_material() const;
    void set_texture_filter(int p_filter);   // 0 = nearest, 1 = linear, 2 = mipmap
    int get_texture_filter() const;
    void set_texture_repeat(bool p_repeat_u, bool p_repeat_v);
    void get_texture_repeat(bool &r_repeat_u, bool &r_repeat_v) const;

    // ------------------------------------------------------------------------
    // Geometry & transform
    // ------------------------------------------------------------------------
    void set_size(float p_width, float p_height);
    void get_size(float &r_width, float &r_height) const;
    void set_offset(const Vector3 &p_offset);
    Vector3 get_offset() const;
    void set_flip(bool p_flip_h, bool p_flip_v);
    void get_flip(bool &r_flip_h, bool &r_flip_v) const;

    // ------------------------------------------------------------------------
    // Billboard / face camera
    // ------------------------------------------------------------------------
    void set_billboard_mode(int p_mode);     // 0=disabled,1=enabled,2=fixed_y,3=fixed_x
    int get_billboard_mode() const;
    void set_pixel_offset(const Vector2 &p_offset);
    Vector2 get_pixel_offset() const;

    // ------------------------------------------------------------------------
    // Color & transparency
    // ------------------------------------------------------------------------
    void set_modulate(const Color &p_color);
    Color get_modulate() const;
    void set_opacity(float p_opacity);
    float get_opacity() const;
    void set_transparent(bool p_transparent);
    bool is_transparent() const;

    // ------------------------------------------------------------------------
    // Lighting & shadows (geometry instance overrides)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool p_cast) override;
    void set_receive_shadow(bool p_receive) override;
    void set_gi_mode(int p_mode) override;
    void set_gi_contribution(float p_amount) override;
    void set_emissive(const Color &p_color, float p_intensity) override;
    Color get_emissive() const override;
    float get_emissive_intensity() const override;

    // ------------------------------------------------------------------------
    // Rendering server synchronization
    // ------------------------------------------------------------------------
    void sync_rect();

private:
    struct Impl;
    Impl *pimpl;
};