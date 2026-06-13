// navigation_agent_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <functional>

namespace lighting {

// ============================================================================
// NavigationAgent3D – moves a dynamic object along a navigation mesh path.
// Handles pathfinding, avoidance, and velocity calculation.
// Can be lit (casts shadows, receives GI) and optionally emits light.
// Optimized for real‑time with asynchronous path updates.
// ============================================================================

class NavigationAgent3D : public Node3D {
public:
    NavigationAgent3D();
    ~NavigationAgent3D();

    // ------------------------------------------------------------------------
    // Target and navigation
    // ------------------------------------------------------------------------
    void set_target_position(const double* pos);
    void get_target_position(double* out_pos) const;
    void set_target_node(Node3D* target_node);   // follow dynamic target
    Node3D* get_target_node() const;
    void set_navigation_map(int64_t nav_map_rid); // from NavigationServer
    int64_t get_navigation_map() const;
    void set_navigation_layer(int layer);
    int get_navigation_layer() const;
    void set_max_speed(float speed);
    float get_max_speed() const;
    void set_max_acceleration(float accel);
    float get_max_acceleration() const;
    void set_path_desired_distance(float dist);
    float get_path_desired_distance() const;
    void set_target_desired_distance(float dist);
    float get_target_desired_distance() const;
    void set_radius(float radius);
    float get_radius() const;

    // ------------------------------------------------------------------------
    // Pathfinding and avoidance
    // ------------------------------------------------------------------------
    void set_avoidance_enabled(bool enabled);
    bool is_avoidance_enabled() const;
    void set_avoidance_radius(float radius);
    float get_avoidance_radius() const;
    void set_avoidance_layers(uint32_t layers);
    uint32_t get_avoidance_layers() const;
    void set_avoidance_mask(uint32_t mask);
    uint32_t get_avoidance_mask() const;
    void set_avoidance_priority(float priority);
    float get_avoidance_priority() const;
    void set_time_horizon(float horizon);
    float get_time_horizon() const;
    void set_max_speed_avoidance(float speed);
    float get_max_speed_avoidance() const;

    // ------------------------------------------------------------------------
    // Runtime path state
    // ------------------------------------------------------------------------
    void update_navigation(double delta_time);
    const std::vector<double>& get_current_path() const; // world positions (x,y,z)
    double get_next_position(double* out_pos) const;    // returns distance to next point
    bool is_navigation_finished() const;
    bool is_target_reached() const;
    double get_remaining_distance() const;
    double get_current_velocity(double* out_vel) const;
    void set_velocity(const double* vel);
    void warp_to_position(const double* pos);

    // ------------------------------------------------------------------------
    // Signals (callbacks)
    // ------------------------------------------------------------------------
    using NavigationCallback = std::function<void()>;
    void set_path_changed_callback(NavigationCallback cb);
    void set_target_reached_callback(NavigationCallback cb);
    void set_waypoint_reached_callback(NavigationCallback cb);

    // ------------------------------------------------------------------------
    // Lighting & GI (agent can be lit and cast shadows)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast);
    bool get_cast_shadow() const;
    void set_receive_shadow(bool receive);
    bool get_receive_shadow() const;
    void set_gi_mode(int mode);  // 0=off,1=static,2=dynamic
    int get_gi_mode() const;
    void set_gi_contribution(float amount);
    float get_gi_contribution() const;
    void set_emissive(const float* color, float intensity);
    void get_emissive(float* out_color, float& out_intensity) const;

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