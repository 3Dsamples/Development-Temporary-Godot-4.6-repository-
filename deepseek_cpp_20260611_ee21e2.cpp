// navigation_link_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>

namespace lighting {

// ============================================================================
// NavigationLink3D – connects two navigation meshes (or two points) to allow
// agents to traverse between them (e.g., teleporter, jump link, gap crossing).
// Supports both global positions and region‑relative links.
// ============================================================================

class NavigationLink3D : public Node3D {
public:
    NavigationLink3D();
    ~NavigationLink3D();

    // ------------------------------------------------------------------------
    // Link endpoints (world positions)
    // ------------------------------------------------------------------------
    void set_start_position(const double* position);
    const double* get_start_position() const;
    void set_end_position(const double* position);
    const double* get_end_position() const;

    // ------------------------------------------------------------------------
    // Link geometry (optional radius for area‑based link)
    // ------------------------------------------------------------------------
    void set_radius(double radius);
    double get_radius() const;
    void set_bidirectional(bool bidir);
    bool is_bidirectional() const;

    // ------------------------------------------------------------------------
    // Navigation cost (higher cost = less desirable)
    // ------------------------------------------------------------------------
    void set_enter_cost(float cost);
    float get_enter_cost() const;
    void set_traversal_cost(float cost);
    float get_traversal_cost() const;

    // ------------------------------------------------------------------------
    // Link state (enabled/disabled)
    // ------------------------------------------------------------------------
    void set_enabled(bool enabled);
    bool is_enabled() const;

    // ------------------------------------------------------------------------
    // Lighting & GI (links are not visible by default, but can be debug‑drawn)
    // ------------------------------------------------------------------------
    void set_debug_visible(bool visible);
    bool is_debug_visible() const;
    void set_debug_color(const float* rgb);
    void get_debug_color(float* out_rgb) const;

    // ------------------------------------------------------------------------
    // Navigation server synchronization
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;
    void process(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting