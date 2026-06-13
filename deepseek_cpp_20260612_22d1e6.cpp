// Name : lighting enhancement
// File : scene/3d/character_body_3d_ext.h 35 of 60
// Description : Extended character body with move_and_slide, floor/wall detection,
//               max slides, up direction, and full lighting integration.
#pragma once

#include "scene/3d/physics_body_3d_ext.h"
#include "servers/rendering_server.h"

class CharacterBody3DExt : public PhysicsBody3DExt {
    GDCLASS(CharacterBody3DExt, PhysicsBody3DExt);

public:
    CharacterBody3DExt();
    ~CharacterBody3DExt();

    // ------------------------------------------------------------------------
    // Movement parameters
    // ------------------------------------------------------------------------
    void set_velocity(const Vector3 &p_velocity);
    Vector3 get_velocity() const;
    void set_max_slides(int p_max_slides);
    int get_max_slides() const;
    void set_floor_max_angle(float p_radians);
    float get_floor_max_angle() const;
    void set_floor_stop_on_slope(bool p_enabled);
    bool get_floor_stop_on_slope() const;
    void set_up_direction(const Vector3 &p_up);
    Vector3 get_up_direction() const;
    void set_wall_min_angle(float p_radians);
    float get_wall_min_angle() const;

    // ------------------------------------------------------------------------
    // Move and slide (main kinematic movement)
    // ------------------------------------------------------------------------
    void move_and_slide();
    void move_and_slide_with_step(double p_delta);
    int get_slide_count() const;
    void get_slide_collision(int p_idx, Vector3 &r_position, Vector3 &r_normal, Vector3 &r_velocity) const;

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
    // Floor velocity (moving platforms)
    // ------------------------------------------------------------------------
    void set_floor_velocity(const Vector3 &p_velocity);
    Vector3 get_floor_velocity() const;

    // ------------------------------------------------------------------------
    // Lighting & shadows (overrides from base)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool p_cast) override;
    void set_gi_mode(int p_mode) override;
    void set_gi_contribution(float p_amount) override;
    void set_emissive(const Color &p_color, float p_intensity) override;

    // ------------------------------------------------------------------------
    // Rendering server synchronization
    // ------------------------------------------------------------------------
    void sync_character_body();

private:
    struct Impl;
    Impl *pimpl;
};