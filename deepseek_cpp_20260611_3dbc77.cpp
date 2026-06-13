// navigation_agent_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <functional>

namespace lighting {

// ============================================================================
// NavigationAgent3D – assists a moving object (e.g., character) to navigate
// through a 3D navigation mesh. Provides pathfinding, avoidance, and dynamic
// path following. Integrated with lighting: can be used for moving characters
// that cast shadows and receive GI, but the agent itself does not affect light.
// ============================================================================

enum class NavigationPathQueryMode : uint8_t {
    SIMPLE,
    OPTIMAL
};

class NavigationAgent3D : public Node3D {
public:
    NavigationAgent3D();
    ~NavigationAgent3D();

    // ------------------------------------------------------------------------
    // Target & pathfinding
    // ------------------------------------------------------------------------
    void set_target_position(const double* position);
    const double* get_target_position() const;
    void set_target_reached(bool reached);
    bool is_target_reached() const;
    void set_path_desired_distance(double distance);
    double get_path_desired_distance() const;
    void set_path_max_distance(double distance);
    double get_path_max_distance() const;
    void set_path_query_mode(NavigationPathQueryMode mode);
    NavigationPathQueryMode get_path_query_mode() const;

    // ------------------------------------------------------------------------
    // Movement parameters (used by calling code to adjust velocity)
    // ------------------------------------------------------------------------
    void set_velocity(const double* velocity);
    void get_velocity(double* out_velocity) const;
    void set_max_speed(float speed);
    float get_max_speed() const;
    void set_max_accel(float accel);
    float get_max_accel() const;
    void set_avoidance_enabled(bool enabled);
    bool is_avoidance_enabled() const;
    void set_avoidance_radius(float radius);
    float get_avoidance_radius() const;

    // ------------------------------------------------------------------------
    // Path query & update
    // ------------------------------------------------------------------------
    void update_navigation();               // call when target or map changes
    std::vector<double> get_next_path_position() const; // returns (x,y,z)
    std::vector<std::vector<double>> get_current_path() const;
    void set_debug_path_enabled(bool enabled);
    bool is_debug_path_enabled() const;

    // ------------------------------------------------------------------------
    // Callbacks (for path update or target reached)
    // ------------------------------------------------------------------------
    void set_path_update_callback(std::function<void()> callback);
    void set_target_reached_callback(std::function<void()> callback);

    // ------------------------------------------------------------------------
    // Lighting & GI (agent itself does not render, but its owning entity may)
    // These flags are passed to the associated visual instance (if any).
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast);
    bool get_cast_shadow() const;
    void set_receive_shadow(bool receive);
    bool get_receive_shadow() const;
    void set_gi_mode(int mode);
    int get_gi_mode() const;
    void set_gi_contribution(float amount);
    float get_gi_contribution() const;

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