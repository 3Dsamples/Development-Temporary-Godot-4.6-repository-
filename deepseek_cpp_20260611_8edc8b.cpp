// navigation_agent_3d.cpp
#include "navigation_agent_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <functional>

namespace lighting {

// ============================================================================
// Internal: Simple navigation map placeholder (real engine uses NavigationServer)
// ============================================================================
struct NavigationMap {
    // For demo, we simulate a plane with obstacles. No actual navigation mesh.
    // In a real implementation, we would query the NavigationServer.
    std::vector<double> path(const double* from, const double* to) {
        // Return a straight line path (simplified)
        std::vector<double> path;
        path.push_back(from[0]); path.push_back(from[1]); path.push_back(from[2]);
        path.push_back(to[0]);   path.push_back(to[1]);   path.push_back(to[2]);
        return path;
    }
};

// ============================================================================
// NavigationAgent3D implementation
// ============================================================================
struct NavigationAgent3D::Impl {
    double target_position[3] = {0,0,0};
    bool target_reached = true;
    double path_desired_distance = 0.5;
    double path_max_distance = 3.0;
    NavigationPathQueryMode query_mode = NavigationPathQueryMode::SIMPLE;

    double velocity[3] = {0,0,0};
    float max_speed = 5.0f;
    float max_accel = 10.0f;
    bool avoidance_enabled = false;
    float avoidance_radius = 0.5f;

    bool debug_path_enabled = false;
    std::vector<std::vector<double>> current_path;
    std::function<void()> path_update_callback;
    std::function<void()> target_reached_callback;

    // Navigation map (from server)
    std::shared_ptr<NavigationMap> nav_map;

    // Lighting flags (for associated entity)
    bool cast_shadow = true;
    bool receive_shadow = true;
    int gi_mode = 2;            // dynamic
    float gi_contribution = 1.0f;

    bool dirty_path = true;
    double last_path_update_time = 0.0;

    void update_path();
};

NavigationAgent3D::NavigationAgent3D() : pimpl(std::make_unique<Impl>()) {
    // Create a default navigation map (in real engine, get from scene)
    pimpl->nav_map = std::make_shared<NavigationMap>();
}
NavigationAgent3D::~NavigationAgent3D() = default;

void NavigationAgent3D::set_target_position(const double* position) {
    memcpy(pimpl->target_position, position, 3*sizeof(double));
    pimpl->target_reached = false;
    pimpl->dirty_path = true;
}
const double* NavigationAgent3D::get_target_position() const { return pimpl->target_position; }
void NavigationAgent3D::set_target_reached(bool reached) { pimpl->target_reached = reached; }
bool NavigationAgent3D::is_target_reached() const { return pimpl->target_reached; }
void NavigationAgent3D::set_path_desired_distance(double distance) {
    pimpl->path_desired_distance = std::max(0.1, distance);
}
double NavigationAgent3D::get_path_desired_distance() const { return pimpl->path_desired_distance; }
void NavigationAgent3D::set_path_max_distance(double distance) {
    pimpl->path_max_distance = std::max(0.1, distance);
}
double NavigationAgent3D::get_path_max_distance() const { return pimpl->path_max_distance; }
void NavigationAgent3D::set_path_query_mode(NavigationPathQueryMode mode) { pimpl->query_mode = mode; }
NavigationPathQueryMode NavigationAgent3D::get_path_query_mode() const { return pimpl->query_mode; }

void NavigationAgent3D::set_velocity(const double* velocity) { memcpy(pimpl->velocity, velocity, 3*sizeof(double)); }
void NavigationAgent3D::get_velocity(double* out_velocity) const { memcpy(out_velocity, pimpl->velocity, 3*sizeof(double)); }
void NavigationAgent3D::set_max_speed(float speed) { pimpl->max_speed = std::max(0.0f, speed); }
float NavigationAgent3D::get_max_speed() const { return pimpl->max_speed; }
void NavigationAgent3D::set_max_accel(float accel) { pimpl->max_accel = std::max(0.0f, accel); }
float NavigationAgent3D::get_max_accel() const { return pimpl->max_accel; }
void NavigationAgent3D::set_avoidance_enabled(bool enabled) { pimpl->avoidance_enabled = enabled; }
bool NavigationAgent3D::is_avoidance_enabled() const { return pimpl->avoidance_enabled; }
void NavigationAgent3D::set_avoidance_radius(float radius) { pimpl->avoidance_radius = std::max(0.01f, radius); }
float NavigationAgent3D::get_avoidance_radius() const { return pimpl->avoidance_radius; }

void NavigationAgent3D::Impl::update_path() {
    if (!dirty_path) return;
    if (!nav_map) return;
    Transform3D global = get_global_transform();
    double from[3] = {global.origin[0], global.origin[1], global.origin[2]};
    // Check distance to target: if close enough, mark target reached
    double dx = target_position[0] - from[0];
    double dy = target_position[1] - from[1];
    double dz = target_position[2] - from[2];
    double dist_to_target = sqrt(dx*dx + dy*dy + dz*dz);
    if (dist_to_target <= path_desired_distance) {
        target_reached = true;
        if (target_reached_callback) target_reached_callback();
        current_path.clear();
        dirty_path = false;
        return;
    }
    // Query path from navigation server
    auto raw_path = nav_map->path(from, target_position);
    // Convert to vector of points (3 doubles each)
    current_path.clear();
    for (size_t i = 0; i < raw_path.size(); i += 3) {
        std::vector<double> point = {raw_path[i], raw_path[i+1], raw_path[i+2]};
        current_path.push_back(point);
    }
    if (path_update_callback) path_update_callback();
    dirty_path = false;
}

std::vector<double> NavigationAgent3D::get_next_path_position() const {
    if (pimpl->current_path.empty()) {
        double zero[3] = {0,0,0};
        return {zero[0], zero[1], zero[2]};
    }
    // Return the first point (or the second if first is current position)
    Transform3D global = get_global_transform();
    double cur[3] = {global.origin[0], global.origin[1], global.origin[2]};
    size_t idx = 0;
    if (pimpl->current_path.size() > 1) {
        double dx = pimpl->current_path[0][0] - cur[0];
        double dy = pimpl->current_path[0][1] - cur[1];
        double dz = pimpl->current_path[0][2] - cur[2];
        if (sqrt(dx*dx+dy*dy+dz*dz) < 0.1) idx = 1;
    }
    if (idx >= pimpl->current_path.size()) idx = pimpl->current_path.size()-1;
    return pimpl->current_path[idx];
}

std::vector<std::vector<double>> NavigationAgent3D::get_current_path() const {
    return pimpl->current_path;
}
void NavigationAgent3D::set_debug_path_enabled(bool enabled) { pimpl->debug_path_enabled = enabled; }
bool NavigationAgent3D::is_debug_path_enabled() const { return pimpl->debug_path_enabled; }

void NavigationAgent3D::set_path_update_callback(std::function<void()> callback) {
    pimpl->path_update_callback = callback;
}
void NavigationAgent3D::set_target_reached_callback(std::function<void()> callback) {
    pimpl->target_reached_callback = callback;
}

void NavigationAgent3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; }
bool NavigationAgent3D::get_cast_shadow() const { return pimpl->cast_shadow; }
void NavigationAgent3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
bool NavigationAgent3D::get_receive_shadow() const { return pimpl->receive_shadow; }
void NavigationAgent3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; }
int NavigationAgent3D::get_gi_mode() const { return pimpl->gi_mode; }
void NavigationAgent3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
float NavigationAgent3D::get_gi_contribution() const { return pimpl->gi_contribution; }

void NavigationAgent3D::process(double delta) {
    Node3D::process(delta);
    pimpl->update_path();
    // If avoidance enabled, we would update velocity based on nearby agents (not implemented)
}

void NavigationAgent3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    // If debug path enabled, draw path using ImmediateMesh or debug draw
    if (pimpl->debug_path_enabled && !pimpl->current_path.empty()) {
        // Draw line strip for debug (omitted – would call rendering server)
    }
}

} // namespace lighting