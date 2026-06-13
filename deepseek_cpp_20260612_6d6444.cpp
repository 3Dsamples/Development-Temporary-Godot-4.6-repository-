// Name : lighting enhancement
// File : scene/3d/navigation_link_3d_ext.h 49 of 60
// Description : Extended navigation link connecting two points on a navigation mesh,
//               with radius, bidirection, costs, and debug visualization.
#pragma once

#include "scene/3d/navigation_link_3d.h"
#include "servers/navigation_server_3d.h"
#include "servers/rendering_server.h"

class NavigationLink3DExt : public NavigationLink3D {
    GDCLASS(NavigationLink3DExt, NavigationLink3D);

public:
    NavigationLink3DExt();
    ~NavigationLink3DExt();

    // ------------------------------------------------------------------------
    // Link endpoints (world positions)
    // ------------------------------------------------------------------------
    void set_start_position(const Vector3 &p_position);
    Vector3 get_start_position() const;
    void set_end_position(const Vector3 &p_position);
    Vector3 get_end_position() const;

    // ------------------------------------------------------------------------
    // Link geometry (radius for area‑based link, bidirection)
    // ------------------------------------------------------------------------
    void set_radius(float p_radius);
    float get_radius() const;
    void set_bidirectional(bool p_bidirectional);
    bool is_bidirectional() const;

    // ------------------------------------------------------------------------
    // Navigation costs (enter cost, traversal cost)
    // ------------------------------------------------------------------------
    void set_enter_cost(float p_cost);
    float get_enter_cost() const;
    void set_traversal_cost(float p_cost);
    float get_traversal_cost() const;

    // ------------------------------------------------------------------------
    // Enable / disable link
    // ------------------------------------------------------------------------
    void set_enabled(bool p_enabled);
    bool is_enabled() const;

    // ------------------------------------------------------------------------
    // Debug visualization (line with arrow, can be emissive)
    // ------------------------------------------------------------------------
    void set_debug_visible(bool p_visible);
    bool is_debug_visible() const;
    void set_debug_color(const Color &p_color);
    Color get_debug_color() const;
    void set_debug_emissive(const Color &p_color, float p_intensity);
    void get_debug_emissive(Color &r_color, float &r_intensity) const;

    // ------------------------------------------------------------------------
    // Global illumination (debug mesh can contribute to GI)
    // ------------------------------------------------------------------------
    void set_gi_mode(int p_mode);
    int get_gi_mode() const;
    void set_gi_contribution(float p_amount);
    float get_gi_contribution() const;

    // ------------------------------------------------------------------------
    // Synchronize with NavigationServer3D
    // ------------------------------------------------------------------------
    void sync_link();

private:
    struct Impl;
    Impl *pimpl;
};