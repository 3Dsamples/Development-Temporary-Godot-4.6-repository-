// navigation_agent_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <functional>

namespace lighting {

// ============================================================================
// NavigationAgent3D – provides pathfinding and movement for dynamic objects.
// The agent requests paths from NavigationServer3D and moves along them using
// avoidance and velocity calculation. Can be used for AI characters.
// Integrated with lighting: agents can cast shadows and affect GI when moving
// (dynamic GI contribution).
// ============================================================================

class NavigationAgent3D : public Node3D {
public:
    NavigationAgent3D();
    ~NavigationAgent3D();

    // ------------------------------------------------------------------------
    // Target and pathfinding
    // ------------------------------------------------------------------------
    void set_target_position(const double* target); // world position
    void get_target_position(double* out_target) const;
    void set_target_reached(bool reached);
    bool is_target_reached() const;
    void set_navigation_map(int64_t map_rid); // NavigationServer map
    int64_t get_navigation_map() const;

    // ------------------------------------------------------------------------
    // Movement parameters
    // ------------------------------------------------------------------------
    void set_max_speed(float speed);
    float get_max_speed() const;
    void set_max_acceleration(float accel);
    float get_max_acceleration() const;
    void set_max_angular_speed(float rad_per_sec);
    float get_max_angular_speed() const;
    void set_path_desired_distance(float distance);
    float get_path_desired_distance() const;
    void set_path_max_distance(float distance);
    float get_path_max_distance() const;
    void set_avoidance_enabled(bool enable);
    bool is_avoidance_enabled() const;
    void set_avoidance_radius(float radius);
    float get_avoidance_radius() const;
    void set_avoidance_layers(uint32_t layers);
    uint32_t get_avoidance_layers() const;

    // ------------------------------------------------------------------------
    // Callbacks (for movement integration)
    // ------------------------------------------------------------------------
    void set_velocity_callback(std::function<void(const double* velocity)> callback);
    void set_path_updated_callback(std::function<void()> callback);

    // ------------------------------------------------------------------------
    // Current state (velocity, next position)
    // ------------------------------------------------------------------------
    void get_current_velocity(double* out_velocity) const;
    void set_velocity(const double* velocity);
    void get_next_position(double* out_position) const; // predicted position after delta

    // ------------------------------------------------------------------------
    // Path query (get current waypoints)
    // ------------------------------------------------------------------------
    int get_current_path_index() const;
    int get_path_waypoint_count() const;
    void get_path_waypoint(int idx, double* out_position) const;

    // ------------------------------------------------------------------------
    // Lighting & GI (agents can be dynamic GI contributors)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast);
    bool get_cast_shadow() const;
    void set_gi_mode(int mode);   // 0=off,1=static,2=dynamic
    int get_gi_mode() const;
    void set_gi_contribution(float amount);
    float get_gi_contribution() const;

    // ------------------------------------------------------------------------
    // Physics integration (called by game loop)
    // ------------------------------------------------------------------------
    void update_agent(double delta_time);
    void synchronize_render_server(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting