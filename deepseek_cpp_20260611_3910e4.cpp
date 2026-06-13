// character_body_3d.h
#pragma once

#include "physics_body_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// CharacterBody3D – kinematic physics body specialized for character movement.
// Provides move_and_slide(), floor/wall detection, and velocity smoothing.
// Supports lighting & GI through base class (shadows, global illumination).
// ============================================================================

class CharacterBody3D : public PhysicsBody3D {
public:
    CharacterBody3D();
    ~CharacterBody3D();

    // ------------------------------------------------------------------------
    // Movement parameters
    // ------------------------------------------------------------------------
    void set_velocity(const double* velocity);
    const double* get_velocity() const;
    void set_max_slides(int max_slides);
    int get_max_slides() const;
    void set_floor_max_angle(float radians);
    float get_floor_max_angle() const;
    void set_floor_stop_on_slope(bool enabled);
    bool get_floor_stop_on_slope() const;
    void set_up_direction(const double* up);
    const double* get_up_direction() const;
    void set_wall_min_angle(float radians);
    float get_wall_min_angle() const;

    // ------------------------------------------------------------------------
    // Move and slide (call once per physics frame)
    // ------------------------------------------------------------------------
    void move_and_slide();
    void move_and_slide_with_step(double delta);
    int get_slide_count() const;
    void get_slide_collision(int idx, double* out_position, double* out_normal, double* out_velocity) const;

    // ------------------------------------------------------------------------
    // Collision state queries
    // ------------------------------------------------------------------------
    bool is_on_floor() const;
    bool is_on_wall() const;
    bool is_on_ceiling() const;
    bool is_on_floor_only() const;
    bool is_on_wall_only() const;
    bool is_on_ceiling_only() const;

    // ------------------------------------------------------------------------
    // Floor velocity (moving platform)
    // ------------------------------------------------------------------------
    void set_floor_velocity(const double* velocity);
    const double* get_floor_velocity() const;

    // ------------------------------------------------------------------------
    // Lighting & GI (overrides from PhysicsBody3D)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;
    void set_gi_contribution(float amount) override;
    void set_emissive(const float* color, float intensity) override;
    void get_emissive(float* out_color, float& out_intensity) const override;

    // ------------------------------------------------------------------------
    // Node overrides
    // ------------------------------------------------------------------------
    void update_physics(double delta_time) override;
    void synchronize_render_server(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting