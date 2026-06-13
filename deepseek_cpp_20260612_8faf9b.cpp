// Name : lighting enhancement
// File : scene/3d/decal_ext.h 19 of 60
// Description : Extended decal node with projection size, texture sets,
//               distance/angle fade, emissive GI contribution, and full RenderingServer sync.
#pragma once

#include "scene/3d/decal.h"
#include "servers/rendering_server.h"

class DecalExt : public Decal {
    GDCLASS(DecalExt, Decal);

public:
    DecalExt();
    ~DecalExt();

    // ------------------------------------------------------------------------
    // Decal size and projection
    // ------------------------------------------------------------------------
    void set_size(const Vector3 &p_size);
    Vector3 get_size() const;

    // ------------------------------------------------------------------------
    // Textures (albedo, normal, ORM, emissive)
    // ------------------------------------------------------------------------
    void set_albedo_texture(const RID &p_texture);
    RID get_albedo_texture() const;
    void set_normal_texture(const RID &p_texture);
    RID get_normal_texture() const;
    void set_orm_texture(const RID &p_texture);    // occlusion, roughness, metalness
    RID get_orm_texture() const;
    void set_emissive_texture(const RID &p_texture);
    RID get_emissive_texture() const;

    // ------------------------------------------------------------------------
    // Material properties (fallback if no texture)
    // ------------------------------------------------------------------------
    void set_albedo(const Color &p_color);
    Color get_albedo() const;
    void set_emissive(const Color &p_color, float p_intensity);
    Color get_emissive() const;
    float get_emissive_intensity() const;
    void set_roughness(float p_roughness);
    float get_roughness() const;
    void set_metalness(float p_metalness);
    float get_metalness() const;
    void set_occlusion(float p_occlusion);
    float get_occlusion() const;
    void set_normal_strength(float p_strength);
    float get_normal_strength() const;

    // ------------------------------------------------------------------------
    // Distance and angle fade
    // ------------------------------------------------------------------------
    void set_distance_fade_enabled(bool p_enabled);
    bool is_distance_fade_enabled() const;
    void set_distance_fade_range(float p_begin, float p_end);
    void get_distance_fade_range(float &p_begin, float &p_end) const;
    void set_angle_fade_enabled(bool p_enabled);
    bool is_angle_fade_enabled() const;
    void set_angle_fade_threshold(float p_angle_deg);
    float get_angle_fade_threshold() const;

    // ------------------------------------------------------------------------
    // Culling mask (which objects are affected)
    // ------------------------------------------------------------------------
    void set_cull_mask(uint32_t p_mask);
    uint32_t get_cull_mask() const;

    // ------------------------------------------------------------------------
    // Global illumination contribution (decal can be emissive and affect GI)
    // ------------------------------------------------------------------------
    void set_gi_mode(int p_mode);
    int get_gi_mode() const;
    void set_gi_contribution(float p_amount);
    float get_gi_contribution() const;

    // ------------------------------------------------------------------------
    // Rendering server synchronization
    // ------------------------------------------------------------------------
    void sync_decal();

private:
    struct Impl;
    Impl *pimpl;
};