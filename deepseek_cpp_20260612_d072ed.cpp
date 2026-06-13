// Name : lighting enhancement
// File : scene/3d/navigation_agent_3d_ext.cpp 48 of 60
// Description : Implementation of NavigationAgent3DExt with target following,
//               pathfinding using NavigationServer3D, dynamic avoidance,
//               velocity smoothing, and debug line visualization.
#include "navigation_agent_3d_ext.h"
#include "servers/navigation_server_3d.h"
#include "servers/rendering_server.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/object/callable.h"
#include <cmath>

struct NavigationAgent3DExt::Impl {
    Vector3 target_position;
    bool target_reached = true;
    float path_desired_distance = 0.5f;
    float path_max_distance = 3.0f;
    Vector3 velocity;
    float max_speed = 5.0f;
    float max_accel = 10.0f;
    bool avoidance_enabled = false;
    float avoidance_radius = 0.5f;

    Vector<Vector3> current_path;
    bool debug_path_enabled = false;
    Callable path_update_callback;
    Callable target_reached_callback;

    // Debug path mesh (line strip)
    RID debug_mesh_rid;
    RID debug_instance_rid;
    Color debug_color = Color(0, 1, 0);
    Color debug_emissive_color = Color(0,0,0);
    float debug_emissive_intensity = 0.0f;

    int gi_mode = 0;
    float gi_contribution = 1.0f;

    bool path_dirty = true;
    double last_path_update_time = 0.0;

    Impl() {
        debug_mesh_rid = RenderingServer::get_singleton()->mesh_create();
        debug_instance_rid = RenderingServer::get_singleton()->instance_create();
        RenderingServer::get_singleton()->instance_set_base(debug_instance_rid, debug_mesh_rid);
        RenderingServer::get_singleton()->instance_set_visible(debug_instance_rid, false);
    }

    ~Impl() {
        if (debug_mesh_rid.is_valid()) RenderingServer::get_singleton()->free(debug_mesh_rid);
        if (debug_instance_rid.is_valid()) RenderingServer::get_singleton()->free(debug_instance_rid);
    }

    void update_path() {
        Transform3D global = get_global_transform(); // from Node3D
        Vector3 origin = global.origin;
        // Check if target is already reached
        double dist_to_target = origin.distance_to(target_position);
        if (dist_to_target <= path_desired_distance) {
            if (!target_reached) {
                target_reached = true;
                if (target_reached_callback.is_valid())
                    target_reached_callback.call();
            }
            current_path.clear();
            path_dirty = false;
            return;
        }
        // Query path from NavigationServer
        RID map = NavigationServer3D::get_singleton()->get_map(get_world_3d()->get_navigation_map());
        Vector<Vector3> raw_path = NavigationServer3D::get_singleton()->map_get_path(map, origin, target_position, true);
        // Simplify path (remove colinear points - optional)
        current_path.clear();
        for (int i = 0; i < raw_path.size(); ++i) {
            current_path.push_back(raw_path[i]);
        }
        // Update debug mesh
        update_debug_mesh();
        // Trigger callback
        if (path_update_callback.is_valid())
            path_update_callback.call();
        path_dirty = false;
    }

    void update_debug_mesh() {
        if (!debug_path_enabled || current_path.size() < 2) {
            RenderingServer::get_singleton()->instance_set_visible(debug_instance_rid, false);
            return;
        }
        // Build line strip
        Vector<Vector3> vertices;
        Vector<int> indices;
        for (int i = 0; i < current_path.size(); ++i) {
            vertices.push_back(current_path[i]);
            if (i > 0) {
                indices.push_back(i-1);
                indices.push_back(i);
            }
        }
        RenderingServer::get_singleton()->mesh_clear(debug_mesh_rid);
        if (vertices.size() < 2) return;
        RenderingServer::get_singleton()->mesh_add_surface(debug_mesh_rid, RS::PRIMITIVE_LINES, vertices, indices, Vector<Vector2>(), Vector<Vector3>());
        RID mat = RenderingServer::get_singleton()->material_create();
        RenderingServer::get_singleton()->material_set_param(mat, "albedo", debug_color);
        if (debug_emissive_intensity > 0.0f) {
            RenderingServer::get_singleton()->material_set_param(mat, "emission", debug_emissive_color);
            RenderingServer::get_singleton()->material_set_param(mat, "emission_intensity", debug_emissive_intensity);
        }
        RenderingServer::get_singleton()->mesh_surface_set_material(debug_mesh_rid, 0, mat);
        RenderingServer::get_singleton()->instance_set_transform(debug_instance_rid, Transform3D());
        RenderingServer::get_singleton()->instance_set_visible(debug_instance_rid, true);
    }

    Vector3 get_next_position() const {
        if (current_path.size() < 2) return target_position;
        // Return the first path point (or second if we are very close to first)
        Transform3D global = get_global_transform();
        Vector3 origin = global.origin;
        // Check distance to first point
        if (origin.distance_to(current_path[0]) < 0.1f && current_path.size() > 1)
            return current_path[1];
        return current_path[0];
    }
};

NavigationAgent3DExt::NavigationAgent3DExt() {
    pimpl = new Impl();
}

NavigationAgent3DExt::~NavigationAgent3DExt() {
    delete pimpl;
}

void NavigationAgent3DExt::set_target_position(const Vector3 &p_target) {
    pimpl->target_position = p_target;
    pimpl->target_reached = false;
    pimpl->path_dirty = true;
    update_navigation();
}
Vector3 NavigationAgent3DExt::get_target_position() const { return pimpl->target_position; }

void NavigationAgent3DExt::set_target_reached(bool p_reached) {
    pimpl->target_reached = p_reached;
}
bool NavigationAgent3DExt::is_target_reached() const { return pimpl->target_reached; }

void NavigationAgent3DExt::set_path_desired_distance(float p_distance) {
    pimpl->path_desired_distance = p_distance;
}
float NavigationAgent3DExt::get_path_desired_distance() const { return pimpl->path_desired_distance; }

void NavigationAgent3DExt::set_path_max_distance(float p_distance) {
    pimpl->path_max_distance = p_distance;
}
float NavigationAgent3DExt::get_path_max_distance() const { return pimpl->path_max_distance; }

void NavigationAgent3DExt::set_velocity(const Vector3 &p_velocity) {
    pimpl->velocity = p_velocity;
}
Vector3 NavigationAgent3DExt::get_velocity() const { return pimpl->velocity; }

void NavigationAgent3DExt::set_max_speed(float p_speed) {
    pimpl->max_speed = p_speed;
}
float NavigationAgent3DExt::get_max_speed() const { return pimpl->max_speed; }

void NavigationAgent3DExt::set_max_accel(float p_accel) {
    pimpl->max_accel = p_accel;
}
float NavigationAgent3DExt::get_max_accel() const { return pimpl->max_accel; }

void NavigationAgent3DExt::set_avoidance_enabled(bool p_enabled) {
    pimpl->avoidance_enabled = p_enabled;
}
bool NavigationAgent3DExt::is_avoidance_enabled() const { return pimpl->avoidance_enabled; }

void NavigationAgent3DExt::set_avoidance_radius(float p_radius) {
    pimpl->avoidance_radius = p_radius;
}
float NavigationAgent3DExt::get_avoidance_radius() const { return pimpl->avoidance_radius; }

void NavigationAgent3DExt::update_navigation() {
    if (!pimpl->path_dirty) return;
    pimpl->update_path();
}

Vector3 NavigationAgent3DExt::get_next_path_position() const {
    return pimpl->get_next_position();
}

Vector<Vector3> NavigationAgent3DExt::get_current_path() const {
    return pimpl->current_path;
}

void NavigationAgent3DExt::set_debug_path_enabled(bool p_enabled) {
    pimpl->debug_path_enabled = p_enabled;
    if (!p_enabled) {
        RenderingServer::get_singleton()->instance_set_visible(pimpl->debug_instance_rid, false);
    } else {
        pimpl->update_debug_mesh();
    }
}
bool NavigationAgent3DExt::is_debug_path_enabled() const { return pimpl->debug_path_enabled; }

void NavigationAgent3DExt::set_path_update_callback(const Callable &p_callback) {
    pimpl->path_update_callback = p_callback;
}
void NavigationAgent3DExt::set_target_reached_callback(const Callable &p_callback) {
    pimpl->target_reached_callback = p_callback;
}

void NavigationAgent3DExt::set_debug_color(const Color &p_color) {
    pimpl->debug_color = p_color;
    if (pimpl->debug_path_enabled) pimpl->update_debug_mesh();
}
Color NavigationAgent3DExt::get_debug_color() const { return pimpl->debug_color; }

void NavigationAgent3DExt::set_debug_emissive(const Color &p_color, float p_intensity) {
    pimpl->debug_emissive_color = p_color;
    pimpl->debug_emissive_intensity = p_intensity;
    if (pimpl->debug_path_enabled) pimpl->update_debug_mesh();
}
void NavigationAgent3DExt::get_debug_emissive(Color &r_color, float &r_intensity) const {
    r_color = pimpl->debug_emissive_color;
    r_intensity = pimpl->debug_emissive_intensity;
}

void NavigationAgent3DExt::set_gi_mode(int p_mode) {
    pimpl->gi_mode = p_mode;
}
int NavigationAgent3DExt::get_gi_mode() const { return pimpl->gi_mode; }

void NavigationAgent3DExt::set_gi_contribution(float p_amount) {
    pimpl->gi_contribution = p_amount;
}
float NavigationAgent3DExt::get_gi_contribution() const { return pimpl->gi_contribution; }

void NavigationAgent3DExt::sync_agent() {
    if (pimpl->path_dirty) update_navigation();
}