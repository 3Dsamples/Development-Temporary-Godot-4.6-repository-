// decal_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// Decal3D – projects a texture onto surfaces in 3D space.
// Supports albedo, normal, roughness/ORM maps, emissive, and distance fade.
// Casts no shadows but can affect GI via albedo/emissive contribution.
// Optimized for many decals using GPU instancing and atlas textures.
// ============================================================================

class Decal3D : public Node3D {
public:
    Decal3D();
    ~Decal3D();

    // ------------------------------------------------------------------------
    // Texture assignment
    // ------------------------------------------------------------------------
    void set_albedo_texture(int64_t texture_rid);
    int64_t get_albedo_texture() const;
    void set_normal_texture(int64_t texture_rid);
    int64_t get_normal_texture() const;
    void set_orm_texture(int64_t texture_rid);   // Occlusion, Roughness, Metalness
    int64_t get_orm_texture() const;
    void set_emissive_texture(int64_t texture_rid);
    int64_t get_emissive_texture() const;

    // ------------------------------------------------------------------------
    // Decal size and projection
    // ------------------------------------------------------------------------
    void set_size(const double* extents);  // half extents in local space (x,y,z)
    void get_size(double* out_extents) const;
    void set_distance_fade_enabled(bool enabled);
    bool is_distance_fade_enabled() const;
    void set_distance_fade_range(double begin, double end);
    void get_distance_fade_range(double& begin, double& end) const;
    void set_angle_fade_enabled(bool enabled);
    bool is_angle_fade_enabled() const;
    void set_angle_fade_threshold(float angle_deg); // beyond this angle, fade to 0
    float get_angle_fade_threshold() const;

    // ------------------------------------------------------------------------
    // Material properties (constant values if no texture)
    // ------------------------------------------------------------------------
    void set_albedo(const float* color);
    void get_albedo(float* out_color) const;
    void set_emissive(const float* color, float intensity);
    void get_emissive(float* out_color, float& out_intensity) const;
    void set_roughness(float roughness);
    float get_roughness() const;
    void set_metalness(float metalness);
    float get_metalness() const;
    void set_occlusion(float occlusion);
    float get_occlusion() const;

    // ------------------------------------------------------------------------
    // Normal mapping intensity
    // ------------------------------------------------------------------------
    void set_normal_strength(float strength);
    float get_normal_strength() const;

    // ------------------------------------------------------------------------
    // Culling (decal culling frustum)
    // ------------------------------------------------------------------------
    void set_cull_mask(uint32_t mask);
    uint32_t get_cull_mask() const;
    void set_cull_angle(float angle_deg);
    float get_cull_angle() const;

    // ------------------------------------------------------------------------
    // GI contribution (decals can affect light probes and VCT)
    // ------------------------------------------------------------------------
    void set_gi_mode(int mode);   // 0=off,1=static,baked,2=dynamic
    int get_gi_mode() const;
    void set_gi_contribution(float amount);
    float get_gi_contribution() const;

    // ------------------------------------------------------------------------
    // Rendering server sync
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;
    void ready() override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting