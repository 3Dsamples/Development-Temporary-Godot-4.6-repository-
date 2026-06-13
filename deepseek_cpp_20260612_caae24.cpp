// Name : lighting enhancement
// File : scene/3d/navigation_obstacle_3d_ext.h 51 of 60
// Description : Extended navigation obstacle with shape (box, sphere, cylinder),
//               carving margin, avoidance radius, priority, debug visualization,
//               and full NavigationServer3D + RenderingServer sync.
#pragma once

#include "scene/3d/navigation_obstacle_3d.h"
#include "servers/navigation_server_3d.h"
#include "servers/rendering_server.h"

class NavigationObstacle3DExt : public NavigationObstacle3D {
    GDCLASS(NavigationObstacle3DExt, NavigationObstacle3D);

public:
    NavigationObstacle3DExt();
    ~NavigationObstacle3DExt();

    // ------------------------------------------------------------------------
    // Obstacle shape and size
    // ------------------------------------------------------------------------
    enum ShapeType {
        SHAPE_BOX,
        SHAPE_SPHERE,
        SHAPE_CYLINDER
    };
    void set_shape(ShapeType p_shape);
    ShapeType get_shape() const;
    void set_size(const Vector3 &p_size);      // half extents for box, radius for sphere/cylinder
    Vector3 get_size() const;
    void set_height(float p_height);            // for cylinder (total height)
    float get_height() const;

    // ------------------------------------------------------------------------
    // Carving (dynamic modification of navigation mesh)
    // ------------------------------------------------------------------------
    void set_carving_enabled(bool p_enabled);
    bool is_carving_enabled() const;
    void set_carving_margin(float p_margin);
    float get_carving_margin() const;

    // ------------------------------------------------------------------------
    // Agent avoidance (RVO)
    // ------------------------------------------------------------------------
    void set_avoidance_enabled(bool p_enabled);
    bool is_avoidance_enabled() const;
    void set_avoidance_radius(float p_radius);
    float get_avoidance_radius() const;
    void set_avoidance_priority(float p_priority);
    float get_avoidance_priority() const;

    // ------------------------------------------------------------------------
    // Enable / disable obstacle (temporarily)
    // ------------------------------------------------------------------------
    void set_enabled(bool p_enabled);
    bool is_enabled() const;

    // ------------------------------------------------------------------------
    // Debug visualization (wireframe shape, can be emissive)
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
    // Synchronize with NavigationServer3D (call after changes)
    // ------------------------------------------------------------------------
    void sync_obstacle();

private:
    struct Impl;
    Impl *pimpl;
};