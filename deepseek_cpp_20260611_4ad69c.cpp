// navigation_agent_3d.cpp
#include "navigation_agent_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <vector>

namespace lighting {

// ============================================================================
// NavigationServer3D stub – in real engine, this would be a singleton.
// ============================================================================
class NavigationServer3D {
public:
    static NavigationServer3D& get() { static NavigationServer3D instance; return instance; }
    int64_t map_create() { return ++next_map; }
    void map_set_up(int64_t map, const double* up) {}
    int64_t agent_create() { return ++next_agent; }
    void agent_set_map(int64_t agent, int64_t map) {}
    void agent_set_target_position(int64_t agent, const double* target) {}
    void agent_set_radius(int64_t agent, float radius) {}
    void agent_set_max_speed(int64_t agent, float speed) {}
    void agent_set_velocity(int64_t agent, const double* velocity) {}
    void agent_get_next_position(int64_t agent, double* out_pos) const { out_pos[0]=0;out_pos[1]=0;out_pos[2]=0; }
    void agent_set_avoidance_enabled(int64_t agent, bool enable) {}
    void agent_set_avoidance_layers(int64_t agent, uint32_t layers) {}
    void agent_set_max_acceleration(int64_t agent, float accel) {}
    void agent_set_path_desired_distance(int64_t agent, float dist) {}
    void agent_set_path_max_distance(int64_t agent, float dist) {}
    void agent_set_angular_speed(int64_t agent, float rad) {}
    void agent_get_current_velocity(int64_t agent, double* out_vel) const { out_vel[0]=out_vel[1]=out_vel[2]=0.0; }
    void agent_update(int64_t agent, double delta) {}
private:
    int64_t next_map = 1000;
    int64_t next_agent = 2000;
};

// ============================================================================
// Implementation
// ============================================================================
struct NavigationAgent3D::Impl {
    // Target
    bool target_reached = true;
    double target_position[3] = {0,0,0};

    // Navigation map
    int64_t navigation_map = -1;
    int64_t agent_rid = -1;

    // Parameters
    float max_speed = 1.0f;
    float max_acceleration = 4.0f;
    float max_angular_speed = 1.5f;
    float path_desired_distance = 0.5f;
    float path_max_distance = 1.0f;
    bool avoidance_enabled = true;
    float avoidance_radius = 0.5f;
    uint32_t avoidance_layers = 0xFFFFFFFF;

    // Current velocity
    double current_velocity[3] = {0,0,0};
    double next_position[3] = {0,0,0};

    // Path (cached from navigation server)
    std::vector<double> path_points; // x,y,z interleaved
    int current_path_index = -1;

    // Callbacks
    std::function<void(const double*)> velocity_callback;
    std::function<void()> path_updated_callback;

    // Lighting flags
    bool cast_shadow = true;
    int gi_mode = 2;        // dynamic
    float gi_contribution = 1.0f;

    bool dirty = true;

    void update_path();
    void compute_velocity(double delta_time);
};

NavigationAgent3D::NavigationAgent3D() : pimpl(std::make_unique<Impl>()) {
    // Create navigation agent after map is set
}
NavigationAgent3D::~NavigationAgent3D() = default;

void NavigationAgent3D::set_target_position(const double* target) {
    memcpy(pimpl->target_position, target, 3*sizeof(double));
    pimpl->target_reached = false;
    if (pimpl->agent_rid != -1) {
        NavigationServer3D::get().agent_set_target_position(pimpl->agent_rid, target);
    }
    pimpl->update_path();
}
void NavigationAgent3D::get_target_position(double* out_target) const {
    memcpy(out_target, pimpl->target_position, 3*sizeof(double));
}
void NavigationAgent3D::set_target_reached(bool reached) { pimpl->target_reached = reached; }
bool NavigationAgent3D::is_target_reached() const { return pimpl->target_reached; }

void NavigationAgent3D::set_navigation_map(int64_t map_rid) {
    pimpl->navigation_map = map_rid;
    if (pimpl->agent_rid == -1) {
        pimpl->agent_rid = NavigationServer3D::get().agent_create();
    }
    NavigationServer3D::get().agent_set_map(pimpl->agent_rid, pimpl->navigation_map);
    pimpl->dirty = true;
}
int64_t NavigationAgent3D::get_navigation_map() const { return pimpl->navigation_map; }

void NavigationAgent3D::set_max_speed(float speed) { pimpl->max_speed = speed; if(pimpl->agent_rid!=-1) NavigationServer3D::get().agent_set_max_speed(pimpl->agent_rid, speed); }
float NavigationAgent3D::get_max_speed() const { return pimpl->max_speed; }
void NavigationAgent3D::set_max_acceleration(float accel) { pimpl->max_acceleration = accel; if(pimpl->agent_rid!=-1) NavigationServer3D::get().agent_set_max_acceleration(pimpl->agent_rid, accel); }
float NavigationAgent3D::get_max_acceleration() const { return pimpl->max_acceleration; }
void NavigationAgent3D::set_max_angular_speed(float rad_per_sec) { pimpl->max_angular_speed = rad_per_sec; if(pimpl->agent_rid!=-1) NavigationServer3D::get().agent_set_angular_speed(pimpl->agent_rid, rad_per_sec); }
float NavigationAgent3D::get_max_angular_speed() const { return pimpl->max_angular_speed; }
void NavigationAgent3D::set_path_desired_distance(float distance) { pimpl->path_desired_distance = distance; if(pimpl->agent_rid!=-1) NavigationServer3D::get().agent_set_path_desired_distance(pimpl->agent_rid, distance); }
float NavigationAgent3D::get_path_desired_distance() const { return pimpl->path_desired_distance; }
void NavigationAgent3D::set_path_max_distance(float distance) { pimpl->path_max_distance = distance; if(pimpl->agent_rid!=-1) NavigationServer3D::get().agent_set_path_max_distance(pimpl->agent_rid, distance); }
float NavigationAgent3D::get_path_max_distance() const { return pimpl->path_max_distance; }
void NavigationAgent3D::set_avoidance_enabled(bool enable) { pimpl->avoidance_enabled = enable; if(pimpl->agent_rid!=-1) NavigationServer3D::get().agent_set_avoidance_enabled(pimpl->agent_rid, enable); }
bool NavigationAgent3D::is_avoidance_enabled() const { return pimpl->avoidance_enabled; }
void NavigationAgent3D::set_avoidance_radius(float radius) { pimpl->avoidance_radius = radius; if(pimpl->agent_rid!=-1) NavigationServer3D::get().agent_set_radius(pimpl->agent_rid, radius); }
float NavigationAgent3D::get_avoidance_radius() const { return pimpl->avoidance_radius; }
void NavigationAgent3D::set_avoidance_layers(uint32_t layers) { pimpl->avoidance_layers = layers; if(pimpl->agent_rid!=-1) NavigationServer3D::get().agent_set_avoidance_layers(pimpl->agent_rid, layers); }
uint32_t NavigationAgent3D::get_avoidance_layers() const { return pimpl->avoidance_layers; }

void NavigationAgent3D::set_velocity_callback(std::function<void(const double*)> callback) { pimpl->velocity_callback = callback; }
void NavigationAgent3D::set_path_updated_callback(std::function<void()> callback) { pimpl->path_updated_callback = callback; }

void NavigationAgent3D::get_current_velocity(double* out_velocity) const {
    if (pimpl->agent_rid != -1) {
        NavigationServer3D::get().agent_get_current_velocity(pimpl->agent_rid, out_velocity);
    } else {
        memcpy(out_velocity, pimpl->current_velocity, 3*sizeof(double));
    }
}
void NavigationAgent3D::set_velocity(const double* velocity) {
    memcpy(pimpl->current_velocity, velocity, 3*sizeof(double));
    if (pimpl->agent_rid != -1) {
        NavigationServer3D::get().agent_set_velocity(pimpl->agent_rid, velocity);
    }
    if (pimpl->velocity_callback) pimpl->velocity_callback(velocity);
}
void NavigationAgent3D::get_next_position(double* out_position) const {
    if (pimpl->agent_rid != -1) {
        NavigationServer3D::get().agent_get_next_position(pimpl->agent_rid, out_position);
    } else {
        memcpy(out_position, pimpl->next_position, 3*sizeof(double));
    }
}

int NavigationAgent3D::get_current_path_index() const { return pimpl->current_path_index; }
int NavigationAgent3D::get_path_waypoint_count() const { return (int)pimpl->path_points.size() / 3; }
void NavigationAgent3D::get_path_waypoint(int idx, double* out_position) const {
    if (idx >= 0 && idx < (int)pimpl->path_points.size()/3) {
        out_position[0] = pimpl->path_points[idx*3];
        out_position[1] = pimpl->path_points[idx*3+1];
        out_position[2] = pimpl->path_points[idx*3+2];
    }
}

void NavigationAgent3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; }
bool NavigationAgent3D::get_cast_shadow() const { return pimpl->cast_shadow; }
void NavigationAgent3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; }
int NavigationAgent3D::get_gi_mode() const { return pimpl->gi_mode; }
void NavigationAgent3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
float NavigationAgent3D::get_gi_contribution() const { return pimpl->gi_contribution; }

void NavigationAgent3D::Impl::update_path() {
    // Query path from navigation server (simplified – direct line)
    path_points.clear();
    if (agent_rid == -1) return;
    // In real engine, call NavigationServer3D::agent_get_path()
    // For demo, create straight line from current position to target.
    double current_pos[3];
    get_global_transform().origin[0]; // we need current world pos of agent node
    // We'll use node's transform as current position.
    // For simplicity, we add just target point as single waypoint.
    path_points.push_back(target_position[0]);
    path_points.push_back(target_position[1]);
    path_points.push_back(target_position[2]);
    current_path_index = 0;
    if (path_updated_callback) path_updated_callback();
}

void NavigationAgent3D::Impl::compute_velocity(double delta_time) {
    // Simplified: move towards next waypoint with max speed
    if (path_points.empty() || current_path_index >= (int)path_points.size()/3) {
        target_reached = true;
        current_velocity[0]=current_velocity[1]=current_velocity[2]=0.0;
        return;
    }
    double* next_wp = &path_points[current_path_index*3];
    double current_pos[3];
    // Get node's current position
    // For this demo, we assume the agent node's transform is used by external movement.
    // We just compute desired velocity.
    double dx = next_wp[0] - current_pos[0];
    double dy = next_wp[1] - current_pos[1];
    double dz = next_wp[2] - current_pos[2];
    double dist = sqrt(dx*dx+dy*dy+dz*dz);
    if (dist < path_desired_distance) {
        // advance to next waypoint
        current_path_index++;
        if (current_path_index >= (int)path_points.size()/3) {
            target_reached = true;
            current_velocity[0]=current_velocity[1]=current_velocity[2]=0.0;
        } else {
            compute_velocity(delta_time); // recursive
        }
        return;
    }
    double desired_vel[3] = {dx / dist * max_speed, dy / dist * max_speed, dz / dist * max_speed};
    // acceleration limit
    double accel_limit = max_acceleration * delta_time;
    double vel_diff[3] = {desired_vel[0] - current_velocity[0],
                          desired_vel[1] - current_velocity[1],
                          desired_vel[2] - current_velocity[2]};
    double mag = sqrt(vel_diff[0]*vel_diff[0] + vel_diff[1]*vel_diff[1] + vel_diff[2]*vel_diff[2]);
    if (mag > accel_limit) {
        vel_diff[0] = vel_diff[0] / mag * accel_limit;
        vel_diff[1] = vel_diff[1] / mag * accel_limit;
        vel_diff[2] = vel_diff[2] / mag * accel_limit;
    }
    current_velocity[0] += vel_diff[0];
    current_velocity[1] += vel_diff[1];
    current_velocity[2] += vel_diff[2];
    // clamp to max speed
    double speed = sqrt(current_velocity[0]*current_velocity[0] + current_velocity[1]*current_velocity[1] + current_velocity[2]*current_velocity[2]);
    if (speed > max_speed) {
        current_velocity[0] = current_velocity[0] / speed * max_speed;
        current_velocity[1] = current_velocity[1] / speed * max_speed;
        current_velocity[2] = current_velocity[2] / speed * max_speed;
    }
    // Update next position for prediction
    next_position[0] = current_pos[0] + current_velocity[0] * delta_time;
    next_position[1] = current_pos[1] + current_velocity[1] * delta_time;
    next_position[2] = current_pos[2] + current_velocity[2] * delta_time;
}

void NavigationAgent3D::update_agent(double delta_time) {
    if (pimpl->agent_rid != -1) {
        NavigationServer3D::get().agent_update(pimpl->agent_rid, delta_time);
        // Retrieve current velocity from server
        NavigationServer3D::get().agent_get_current_velocity(pimpl->agent_rid, pimpl->current_velocity);
        NavigationServer3D::get().agent_get_next_position(pimpl->agent_rid, pimpl->next_position);
    } else {
        pimpl->compute_velocity(delta_time);
    }
    // Optionally apply velocity to node's transform
    Transform3D trans = get_global_transform();
    trans.origin[0] += pimpl->current_velocity[0] * delta_time;
    trans.origin[1] += pimpl->current_velocity[1] * delta_time;
    trans.origin[2] += pimpl->current_velocity[2] * delta_time;
    set_global_transform(trans);
}

void NavigationAgent3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    // Update lighting flags for the agent's visual representation (if any)
    // For GI: if dynamic, mark as emissive contributor if needed.
    // Also update shadow settings.
    // No mesh by itself, but the node that inherits from this may have mesh.
}

} // namespace lighting