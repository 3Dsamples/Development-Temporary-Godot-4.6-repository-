// static_body_3d.h
#pragma once

#include "physics_body_3d.h"
#include <memory>

namespace lighting {

// ============================================================================
// StaticBody3D – non-moving physics body for world geometry, walls, floors.
// Optimized for static lighting (lightmap, baked GI) and low memory footprint.
// ============================================================================

class StaticBody3D : public PhysicsBody3D {
public:
    StaticBody3D();
    ~StaticBody3D();

    // ------------------------------------------------------------------------
    // Static body specifics
    // ------------------------------------------------------------------------
    void set_constant_linear_velocity(const double* velocity);
    void get_constant_linear_velocity(double* out_velocity) const;
    void set_constant_angular_velocity(const double* velocity);
    void get_constant_angular_velocity(double* out_velocity) const;

    // ------------------------------------------------------------------------
    // Lightmap & baked lighting (static bodies are prime candidates)
    // ------------------------------------------------------------------------
    void set_lightmap_index(int index);
    int get_lightmap_index() const;
    void set_lightmap_uv_scale(const double* scale);
    void get_lightmap_uv_scale(double* out_scale) const;
    void set_emissive_lighting(const float* emissive_color, float intensity);
    void get_emissive_lighting(float* out_color, float& out_intensity) const;

    // ------------------------------------------------------------------------
    // Global illumination contribution (static geometry can bounce light)
    // ------------------------------------------------------------------------
    void set_gi_mode(int mode); // 0=off, 1=static, 2=dynamic (but static is best)
    int get_gi_mode() const;

    // ------------------------------------------------------------------------
    // Rendering & shadow optimization
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_lightmap_shadow_receiver(bool receive);
    bool is_lightmap_shadow_receiver() const;

    // ------------------------------------------------------------------------
    // Physics server integration
    // ------------------------------------------------------------------------
    void update_physics(double delta_time) override;
    void synchronize_render_server(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting