// Name : lighting enhancement
// File : scene/3d/navigation_agent_3d_ext.h 47 of 60
// Description : Extended navigation agent with target following, pathfinding,
//               dynamic avoidance, velocity control, and optional debug visualization.
#pragma once

#include "scene/3d/navigation_agent_3d.h"
#include "servers/navigation_server_3d.h"
#include "servers/rendering_server.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"

class NavigationAgent3DExt : public NavigationAgent3D {
    GDCLASS(NavigationAgent3DExt, NavigationAgent3D);

public:
    NavigationAgent3DExt();
    ~NavigationAgent3DExt();

    // ------------------------------------------------------------------------
    // Target and pathfinding
    // ------------------------------------------------------------------------
    void set_target_position(const Vector3 &p_target);
    Vector3 get_target_position() const;
    void set_target_reached(bool p_reached);
    bool is_target_reached() const;
    void set_path_desired_distance(float p_distance);
    float get_path_desired_distance() const;
    void set_path_max_distance(float p_distance);
    float get_path_max_distance() const;

    // ------------------------------------------------------------------------
    // Movement and velocity
    // ------------------------------------------------------------------------
    void set_velocity(const Vector3 &p_velocity);
    Vector3 get_velocity() const;
    void set_max_speed(float p_speed);
    float get_max_speed() const;
    void set_max_accel(float p_accel);
    float get_max_accel() const;
    void set_avoidance_enabled(bool p_enabled);
    bool is_avoidance_enabled() const;
    void set_avoidance_radius(float p_radius);
    float get_avoidance_radius() const;

    // ------------------------------------------------------------------------
    // Path query and navigation
    // ------------------------------------------------------------------------
    void update_navigation();                       // Request path update
    Vector3 get_next_path_position() const;         // Next corner to move towards
    Vector<Vector3> get_current_path() const;       // Full path (array of points)
    void set_debug_path_enabled(bool p_enabled);
    bool is_debug_path_enabled() const;

    // ------------------------------------------------------------------------
    // Callbacks (handled via signals, but exposed as Callable)
    // ------------------------------------------------------------------------
    void set_path_update_callback(const Callable &p_callback);
    void set_target_reached_callback(const Callable &p_callback);

    // ------------------------------------------------------------------------
    // Debug visualization (path line)
    // ------------------------------------------------------------------------
    void set_debug_color(const Color &p_color);
    Color get_debug_color() const;
    void set_debug_emissive(const Color &p_color, float p_intensity);
    void get_debug_emissive(Color &r_color, float &r_intensity) const;

    // ------------------------------------------------------------------------
    // Global illumination (debug mesh can be emissive)
    // ------------------------------------------------------------------------
    void set_gi_mode(int p_mode);
    int get_gi_mode() const;
    void set_gi_contribution(float p_amount);
    float get_gi_contribution() const;

    // ------------------------------------------------------------------------
    // Synchronization (update navigation server and debug mesh)
    // ------------------------------------------------------------------------
    void sync_agent();

private:
    struct Impl;
    Impl *pimpl;
};