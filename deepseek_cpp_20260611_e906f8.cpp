// spring_arm_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>

namespace lighting {

// ============================================================================
// SpringArm3D – a node that extends a spring arm to keep a target (e.g., camera)
// at a desired length while avoiding collisions with geometry.
// Uses raycasting to shorten the arm when obstacles are detected.
// Supports spring damping, linear/angular motion smoothing, and debug visualization.
// ============================================================================

class SpringArm3D : public Node3D {
public:
    SpringArm3D();
    ~SpringArm3D();

    // ------------------------------------------------------------------------
    // Spring length and collision
    // ------------------------------------------------------------------------
    void set_spring_length(double length);
    double get_spring_length() const;
    void set_collision_enabled(bool enabled);
    bool is_collision_enabled() const;
    void set_collision_mask(uint32_t mask);
    uint32_t get_collision_mask() const;
    void set_collision_margin(double margin);
    double get_collision_margin() const;

    // ------------------------------------------------------------------------
    // Spring physics (smoothing)
    // ------------------------------------------------------------------------
    void set_spring_stiffness(float stiffness);   // 0..1
    float get_spring_stiffness() const;
    void set_spring_damping(float damping);
    float get_spring_damping() const;
    void set_angular_stiffness(float stiffness);
    float get_angular_stiffness() const;
    void set_angular_damping(float damping);
    float get_angular_damping() const;

    // ------------------------------------------------------------------------
    // Clipping and avoidance
    // ------------------------------------------------------------------------
    void set_clip_far(bool clip);   // if true, limit arm length to avoid clipping through objects
    bool get_clip_far() const;
    void set_avoidance_radius(float radius);   // sphere cast radius
    float get_avoidance_radius() const;

    // ------------------------------------------------------------------------
    // Debug visualization (draw the spring arm line with optional emissive color)
    // ------------------------------------------------------------------------
    void set_debug_visible(bool visible);
    bool is_debug_visible() const;
    void set_debug_color(const float* rgb);
    void get_debug_color(float* out_rgb) const;
    void set_emissive_debug(bool enable, float intensity = 0.2f);
    bool is_emissive_debug() const;

    // ------------------------------------------------------------------------
    // Current arm target position (world) – after collision adjustment
    // ------------------------------------------------------------------------
    void get_arm_end_position(double* out_pos) const;

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