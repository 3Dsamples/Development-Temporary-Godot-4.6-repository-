// navigation_agent_3d.cpp
#include "navigation_agent_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <queue>
#include <vector>
#include <unordered_map>
#include <limits>

namespace lighting {

// ============================================================================
// Simplified navigation server stub (in real engine, this is a low‑level service)
// ============================================================================
class NavigationServer3D {
public:
    struct PathQuery {
        double start[3];
        double target[3];
        int nav_layer;
        float max_speed;
        std::vector<double> result_path;
    };
    static NavigationServer3D& instance() { static NavigationServer3D s; return s; }
    void request_path(const PathQuery& query) {
        // Dummy: just a straight line path
        query.result_path.clear();
        query.result_path.push_back(query.start[0]); query.result_path.push_back(query.start[1]); query.result_path.push_back(query.start[2]);
        query.result_path.push_back(query.target[0]); query.result_path.push_back(query.target[1]); query.result_path.push_back(query.target[2]);
    }
    // Other methods for avoidance, etc.
};

// ============================================================================
// Agent implementation
// ============================================================================
struct NavigationAgent3D::Impl {
    // Target
    double target_pos[3] = {0,0,0};
    Node3D* target_node = nullptr;
    int64_t navigation_map = 0;
    int navigation_layer = 1;
    float max_speed = 5.0f;
    float max_acceleration = 10.0f;
    float path_desired_distance = 0.5f;
    float target_desired_distance = 0.2f;
    float radius = 0.5f;

    // Avoidance
    bool avoidance_enabled = true;
    float avoidance_radius = 1.0f;
    uint32_t avoidance_layers = 0xFFFFFFFF;
    uint32_t avoidance_mask = 0xFFFFFFFF;
    float avoidance_priority = 1.0f;
    float time_horizon = 2.0f;
    float max_speed_avoidance = 5.0f;

    // Runtime state
    std::vector<double> current_path;      // world positions, interleaved (x,y,z)
    size_t current_waypoint_idx = 0;
    double current_velocity[3] = {0,0,0};
    double desired_velocity[3] = {0,0,0};
    double avoidance_velocity[3] = {0,0,0};
    bool navigation_finished = true;
    bool target_reached = false;
    double remaining_distance = 0.0;
    float update_timer = 0.0f;
    float update_interval = 0.2f;          // refresh path every 200ms

    // Lighting
    bool cast_shadow = false;
    bool receive_shadow = true;
    int gi_mode = 2;                       // dynamic by default
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    // Callbacks
    NavigationAgent3D::NavigationCallback path_changed_cb;
    NavigationAgent3D::NavigationCallback target_reached_cb;
    NavigationAgent3D::NavigationCallback waypoint_reached_cb;

    // Render server handles (optional visual representation)
    int64_t instance_rid = -1;

    void request_new_path();
    void update_avoidance(double delta);
    void integrate_movement(double delta);
};

NavigationAgent3D::NavigationAgent3D() : pimpl(std::make_unique<Impl>()) {}
NavigationAgent3D::~NavigationAgent3D() = default;

void NavigationAgent3D::set_target_position(const double* pos) {
    memcpy(pimpl->target_pos, pos, 3*sizeof(double));
    pimpl->target_node = nullptr;
    pimpl->request_new_path();
}
void NavigationAgent3D::get_target_position(double* out_pos) const { memcpy(out_pos, pimpl->target_pos, 3*sizeof(double)); }
void NavigationAgent3D::set_target_node(Node3D* target_node) {
    pimpl->target_node = target_node;
    if (target_node) {
        Transform3D t = target_node->get_global_transform();
        memcpy(pimpl->target_pos, t.origin, 3*sizeof(double));
    }
    pimpl->request_new_path();
}
Node3D* NavigationAgent3D::get_target_node() const { return pimpl->target_node; }
void NavigationAgent3D::set_navigation_map(int64_t nav_map_rid) { pimpl->navigation_map = nav_map_rid; }
int64_t NavigationAgent3D::get_navigation_map() const { return pimpl->navigation_map; }
void NavigationAgent3D::set_navigation_layer(int layer) { pimpl->navigation_layer = layer; pimpl->request_new_path(); }
int NavigationAgent3D::get_navigation_layer() const { return pimpl->navigation_layer; }
void NavigationAgent3D::set_max_speed(float speed) { pimpl->max_speed = speed; }
float NavigationAgent3D::get_max_speed() const { return pimpl->max_speed; }
void NavigationAgent3D::set_max_acceleration(float accel) { pimpl->max_acceleration = accel; }
float NavigationAgent3D::get_max_acceleration() const { return pimpl->max_acceleration; }
void NavigationAgent3D::set_path_desired_distance(float dist) { pimpl->path_desired_distance = dist; }
float NavigationAgent3D::get_path_desired_distance() const { return pimpl->path_desired_distance; }
void NavigationAgent3D::set_target_desired_distance(float dist) { pimpl->target_desired_distance = dist; }
float NavigationAgent3D::get_target_desired_distance() const { return pimpl->target_desired_distance; }
void NavigationAgent3D::set_radius(float radius) { pimpl->radius = radius; }
float NavigationAgent3D::get_radius() const { return pimpl->radius; }

void NavigationAgent3D::set_avoidance_enabled(bool enabled) { pimpl->avoidance_enabled = enabled; }
bool NavigationAgent3D::is_avoidance_enabled() const { return pimpl->avoidance_enabled; }
void NavigationAgent3D::set_avoidance_radius(float radius) { pimpl->avoidance_radius = radius; }
float NavigationAgent3D::get_avoidance_radius() const { return pimpl->avoidance_radius; }
void NavigationAgent3D::set_avoidance_layers(uint32_t layers) { pimpl->avoidance_layers = layers; }
uint32_t NavigationAgent3D::get_avoidance_layers() const { return pimpl->avoidance_layers; }
void NavigationAgent3D::set_avoidance_mask(uint32_t mask) { pimpl->avoidance_mask = mask; }
uint32_t NavigationAgent3D::get_avoidance_mask() const { return pimpl->avoidance_mask; }
void NavigationAgent3D::set_avoidance_priority(float priority) { pimpl->avoidance_priority = priority; }
float NavigationAgent3D::get_avoidance_priority() const { return pimpl->avoidance_priority; }
void NavigationAgent3D::set_time_horizon(float horizon) { pimpl->time_horizon = horizon; }
float NavigationAgent3D::get_time_horizon() const { return pimpl->time_horizon; }
void NavigationAgent3D::set_max_speed_avoidance(float speed) { pimpl->max_speed_avoidance = speed; }
float NavigationAgent3D::get_max_speed_avoidance() const { return pimpl->max_speed_avoidance; }

const std::vector<double>& NavigationAgent3D::get_current_path() const { return pimpl->current_path; }
double NavigationAgent3D::get_next_position(double* out_pos) const {
    if (pimpl->current_waypoint_idx < pimpl->current_path.size() / 3) {
        memcpy(out_pos, &pimpl->current_path[pimpl->current_waypoint_idx*3], 3*sizeof(double));
        double dx = out_pos[0] - get_global_transform().origin[0];
        double dy = out_pos[1] - get_global_transform().origin[1];
        double dz = out_pos[2] - get_global_transform().origin[2];
        return std::sqrt(dx*dx+dy*dy+dz*dz);
    }
    out_pos[0]=out_pos[1]=out_pos[2]=0;
    return 0.0;
}
bool NavigationAgent3D::is_navigation_finished() const { return pimpl->navigation_finished; }
bool NavigationAgent3D::is_target_reached() const { return pimpl->target_reached; }
double NavigationAgent3D::get_remaining_distance() const { return pimpl->remaining_distance; }
double NavigationAgent3D::get_current_velocity(double* out_vel) const {
    memcpy(out_vel, pimpl->current_velocity, 3*sizeof(double));
    return std::sqrt(out_vel[0]*out_vel[0] + out_vel[1]*out_vel[1] + out_vel[2]*out_vel[2]);
}
void NavigationAgent3D::set_velocity(const double* vel) {
    memcpy(pimpl->current_velocity, vel, 3*sizeof(double));
}
void NavigationAgent3D::warp_to_position(const double* pos) {
    Transform3D t = get_global_transform();
    memcpy(t.origin, pos, 3*sizeof(double));
    set_global_transform(t);
}

void NavigationAgent3D::set_path_changed_callback(NavigationCallback cb) { pimpl->path_changed_cb = cb; }
void NavigationAgent3D::set_target_reached_callback(NavigationCallback cb) { pimpl->target_reached_cb = cb; }
void NavigationAgent3D::set_waypoint_reached_callback(NavigationCallback cb) { pimpl->waypoint_reached_cb = cb; }

void NavigationAgent3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; }
bool NavigationAgent3D::get_cast_shadow() const { return pimpl->cast_shadow; }
void NavigationAgent3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
bool NavigationAgent3D::get_receive_shadow() const { return pimpl->receive_shadow; }
void NavigationAgent3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; }
int NavigationAgent3D::get_gi_mode() const { return pimpl->gi_mode; }
void NavigationAgent3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
float NavigationAgent3D::get_gi_contribution() const { return pimpl->gi_contribution; }
void NavigationAgent3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void NavigationAgent3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}

void NavigationAgent3D::Impl::request_new_path() {
    NavigationServer3D::PathQuery query;
    Transform3D my_pos = get_global_transform();
    memcpy(query.start, my_pos.origin, 3*sizeof(double));
    memcpy(query.target, target_pos, 3*sizeof(double));
    query.nav_layer = navigation_layer;
    query.max_speed = max_speed;
    NavigationServer3D::instance().request_path(query);
    current_path = query.result_path;
    current_waypoint_idx = (current_path.size() >= 3) ? 1 : 0; // skip first (current position)
    navigation_finished = (current_path.size() < 6);
    remaining_distance = 0.0;
    if (path_changed_cb) path_changed_cb();
}

void NavigationAgent3D::Impl::update_avoidance(double delta) {
    if (!avoidance_enabled) {
        desired_velocity[0] = avoidance_velocity[0];
        desired_velocity[1] = avoidance_velocity[1];
        desired_velocity[2] = avoidance_velocity[2];
        return;
    }
    // Simplified agent‑based avoidance: just use desired velocity from path.
    // In full engine, we would query NavigationServer for velocity obstacles.
    avoidance_velocity[0] = desired_velocity[0];
    avoidance_velocity[1] = desired_velocity[1];
    avoidance_velocity[2] = desired_velocity[2];
    // clamp to max_speed_avoidance
    double spd = std::sqrt(avoidance_velocity[0]*avoidance_velocity[0] +
                           avoidance_velocity[1]*avoidance_velocity[1] +
                           avoidance_velocity[2]*avoidance_velocity[2]);
    if (spd > max_speed_avoidance) {
        avoidance_velocity[0] *= max_speed_avoidance / spd;
        avoidance_velocity[1] *= max_speed_avoidance / spd;
        avoidance_velocity[2] *= max_speed_avoidance / spd;
    }
}

void NavigationAgent3D::Impl::integrate_movement(double delta) {
    double current_pos[3];
    memcpy(current_pos, get_global_transform().origin, 3*sizeof(double));
    double new_pos[3] = {current_pos[0] + avoidance_velocity[0] * delta,
                         current_pos[1] + avoidance_velocity[1] * delta,
                         current_pos[2] + avoidance_velocity[2] * delta};
    Transform3D new_transform = get_global_transform();
    memcpy(new_transform.origin, new_pos, 3*sizeof(double));
    set_global_transform(new_transform);
    // Update remaining distance
    remaining_distance = 0.0;
    for (size_t i = current_waypoint_idx; i < current_path.size()/3; ++i) {
        double* p = &current_path[i*3];
        double dx = p[0] - new_pos[0];
        double dy = p[1] - new_pos[1];
        double dz = p[2] - new_pos[2];
        remaining_distance += std::sqrt(dx*dx+dy*dy+dz*dz);
    }
}

void NavigationAgent3D::update_navigation(double delta_time) {
    if (delta_time > 0.033) delta_time = 0.033;
    pimpl->update_timer += delta_time;
    // Refresh path periodically
    if (pimpl->update_timer >= pimpl->update_interval) {
        pimpl->update_timer = 0.0f;
        if (pimpl->target_node) {
            Transform3D t = pimpl->target_node->get_global_transform();
            memcpy(pimpl->target_pos, t.origin, 3*sizeof(double));
        }
        pimpl->request_new_path();
    }

    // If no path, stop
    if (pimpl->current_path.empty() || pimpl->current_waypoint_idx >= pimpl->current_path.size()/3) {
        pimpl->navigation_finished = true;
        pimpl->desired_velocity[0] = pimpl->desired_velocity[1] = pimpl->desired_velocity[2] = 0;
        if (!pimpl->target_reached && pimpl->target_reached_cb) pimpl->target_reached_cb();
        pimpl->target_reached = true;
        pimpl->update_avoidance(delta_time);
        pimpl->integrate_movement(delta_time);
        return;
    }
    pimpl->target_reached = false;
    pimpl->navigation_finished = false;

    // Get current and next waypoint
    double current_pos[3];
    memcpy(current_pos, get_global_transform().origin, 3*sizeof(double));
    double* next_waypoint = &pimpl->current_path[pimpl->current_waypoint_idx*3];
    double dx = next_waypoint[0] - current_pos[0];
    double dy = next_waypoint[1] - current_pos[1];
    double dz = next_waypoint[2] - current_pos[2];
    double dist_to_waypoint = std::sqrt(dx*dx+dy*dy+dz*dz);

    // If close enough to current waypoint, advance
    if (dist_to_waypoint < pimpl->path_desired_distance) {
        ++pimpl->current_waypoint_idx;
        if (pimpl->waypoint_reached_cb) pimpl->waypoint_reached_cb();
        // Recurse to handle next waypoint immediately
        update_navigation(0.0);
        return;
    }

    // Compute desired velocity towards next waypoint
    double desired_dir[3] = {dx / dist_to_waypoint, dy / dist_to_waypoint, dz / dist_to_waypoint};
    double desired_speed = pimpl->max_speed;
    pimpl->desired_velocity[0] = desired_dir[0] * desired_speed;
    pimpl->desired_velocity[1] = desired_dir[1] * desired_speed;
    pimpl->desired_velocity[2] = desired_dir[2] * desired_speed;

    // Apply acceleration limit
    double accel_limit = pimpl->max_acceleration * delta_time;
    double vel_change[3] = {
        pimpl->desired_velocity[0] - pimpl->current_velocity[0],
        pimpl->desired_velocity[1] - pimpl->current_velocity[1],
        pimpl->desired_velocity[2] - pimpl->current_velocity[2]
    };
    double change_mag = std::sqrt(vel_change[0]*vel_change[0] + vel_change[1]*vel_change[1] + vel_change[2]*vel_change[2]);
    if (change_mag > accel_limit) {
        vel_change[0] *= accel_limit / change_mag;
        vel_change[1] *= accel_limit / change_mag;
        vel_change[2] *= accel_limit / change_mag;
    }
    pimpl->current_velocity[0] += vel_change[0];
    pimpl->current_velocity[1] += vel_change[1];
    pimpl->current_velocity[2] += vel_change[2];

    // Clamp to max speed
    double spd = std::sqrt(pimpl->current_velocity[0]*pimpl->current_velocity[0] +
                           pimpl->current_velocity[1]*pimpl->current_velocity[1] +
                           pimpl->current_velocity[2]*pimpl->current_velocity[2]);
    if (spd > pimpl->max_speed) {
        pimpl->current_velocity[0] *= pimpl->max_speed / spd;
        pimpl->current_velocity[1] *= pimpl->max_speed / spd;
        pimpl->current_velocity[2] *= pimpl->max_speed / spd;
    }

    pimpl->update_avoidance(delta_time);
    pimpl->integrate_movement(delta_time);
}

void NavigationAgent3D::process(double delta) {
    Node3D::process(delta);
    update_navigation(delta);
}

void NavigationAgent3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    // If we have a visual representation (e.g., a MeshInstance3D child), we update its transform.
    // For lighting: if emissive, register as dynamic GI source.
    if (pimpl->emissive_intensity > 0.0f && pimpl->gi_mode > 0) {
        // Register for GI (e.g., inject into lightprobe or VCT)
    }
}

} // namespace lighting