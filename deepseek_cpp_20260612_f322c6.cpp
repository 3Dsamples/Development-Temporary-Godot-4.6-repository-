// Name : lighting enhancement
// File : scene/3d/spring_arm_3d_ext.h 53 of 60
// Description : Extended spring arm node with collision avoidance, spring physics,
//               distance smoothing, and optional debug visualization with emissive.
#pragma once

#include "scene/3d/spring_arm_3d.h"
#include "servers/rendering_server.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"

class SpringArm3DExt : public SpringArm3D {
    GDCLASS(SpringArm3DExt, SpringArm3D);

public:
    SpringArm3DExt();
    ~SpringArm3DExt();

    // ------------------------------------------------------------------------
    // Spring length and collision
    // ------------------------------------------------------------------------
    void set_spring_length(float p_length);
    float get_spring_length() const;
    void set_collision_enabled(bool p_enabled);
    bool is_collision_enabled() const;
    void set_collision_mask(uint32_t p_mask);
    uint32_t get_collision_mask() const;
    void set_collision_margin(float p_margin);
    float get_collision_margin() const;

    // ------------------------------------------------------------------------
    // Spring physics (smoothing)
    // ------------------------------------------------------------------------
    void set_spring_stiffness(float p_stiffness);
    float get_spring_stiffness() const;
    void set_spring_damping(float p_damping);
    float get_spring_damping() const;
    void set_angular_stiffness(float p_stiffness);
    float get_angular_stiffness() const;
    void set_angular_damping(float p_damping);
    float get_angular_damping() const;

    // ------------------------------------------------------------------------
    // Clipping and avoidance
    // ------------------------------------------------------------------------
    void set_clip_far(bool p_clip);
    bool get_clip_far() const;
    void set_avoidance_radius(float p_radius);
    float get_avoidance_radius() const;

    // ------------------------------------------------------------------------
    // Current arm end position (world)
    // ------------------------------------------------------------------------
    Vector3 get_arm_end_position() const;

    // ------------------------------------------------------------------------
    // Debug visualization (line from origin to end, can be emissive)
    // ------------------------------------------------------------------------
    void set_debug_visible(bool p_visible);
    bool is_debug_visible() const;
    void set_debug_color(const Color &p_color);
    Color get_debug_color() const;
    void set_debug_emissive(const Color &p_color, float p_intensity);
    void get_debug_emissive(Color &r_color, float &r_intensity) const;

    // ------------------------------------------------------------------------
    // Global illumination (debug line can contribute to GI if emissive)
    // ------------------------------------------------------------------------
    void set_gi_mode(int p_mode);
    int get_gi_mode() const;
    void set_gi_contribution(float p_amount);
    float get_gi_contribution() const;

    // ------------------------------------------------------------------------
    // Force update (re‑evaluate arm length and smoothing)
    // ------------------------------------------------------------------------
    void update_arm();

private:
    struct Impl;
    Impl *pimpl;
};