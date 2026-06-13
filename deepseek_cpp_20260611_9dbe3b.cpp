// navigation_obstacle_3d.cpp
#include "navigation_obstacle_3d.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

namespace lighting {

struct NavigationObstacle3D::Impl {
    ObstacleShape shape = ObstacleShape::BOX;
    double size[3] = {1.0, 1.0, 1.0}; // half extents for box, radius for sphere/cylinder
    double height = 2.0;               // cylinder height
    std::vector<double> mesh_vertices; // convex mesh vertices (3 per vertex)

    bool carving_enabled = true;
    double carving_margin = 0.2;
    bool affects_navigation = true;

    bool avoidance_enabled = false;
    double avoidance_radius = 1.0;
    float avoidance_priority = 1.0f;

    bool debug_visible = true;
    float debug_color[3] = {1.0f, 0.0f, 0.0f}; // red
    bool emissive_debug = false;
    float emissive_intensity = 0.2f;

    int64_t obstacle_rid = -1; // NavigationServer obstacle ID
    bool dirty = true;

    ~Impl() {
        if (obstacle_rid != -1) {
            // NavigationServer3D::obstacle_free(obstacle_rid);
        }
    }

    void update_navigation_server();
    void update_debug_mesh();
};

NavigationObstacle3D::NavigationObstacle3D() : pimpl(std::make_unique<Impl>()) {}
NavigationObstacle3D::~NavigationObstacle3D() = default;

void NavigationObstacle3D::set_shape(ObstacleShape shape) {
    pimpl->shape = shape;
    pimpl->dirty = true;
}
ObstacleShape NavigationObstacle3D::get_shape() const { return pimpl->shape; }

void NavigationObstacle3D::set_size(const double* size) {
    memcpy(pimpl->size, size, 3*sizeof(double));
    if (pimpl->shape == ObstacleShape::SPHERE) {
        // ensure sphere radius uses first component
        pimpl->size[1] = pimpl->size[2] = pimpl->size[0];
    }
    pimpl->dirty = true;
}
void NavigationObstacle3D::get_size(double* out_size) const {
    memcpy(out_size, pimpl->size, 3*sizeof(double));
}
void NavigationObstacle3D::set_height(double height) {
    pimpl->height = std::max(0.0, height);
    pimpl->dirty = true;
}
double NavigationObstacle3D::get_height() const { return pimpl->height; }
void NavigationObstacle3D::set_mesh_vertices(const std::vector<double>& vertices) {
    pimpl->mesh_vertices = vertices;
    pimpl->dirty = true;
}
std::vector<double> NavigationObstacle3D::get_mesh_vertices() const { return pimpl->mesh_vertices; }

void NavigationObstacle3D::set_carving_enabled(bool enabled) {
    pimpl->carving_enabled = enabled;
    pimpl->dirty = true;
}
bool NavigationObstacle3D::is_carving_enabled() const { return pimpl->carving_enabled; }
void NavigationObstacle3D::set_carving_margin(double margin) {
    pimpl->carving_margin = std::max(0.0, margin);
    pimpl->dirty = true;
}
double NavigationObstacle3D::get_carving_margin() const { return pimpl->carving_margin; }
void NavigationObstacle3D::set_affects_navigation(bool affects) {
    pimpl->affects_navigation = affects;
    pimpl->dirty = true;
}
bool NavigationObstacle3D::get_affects_navigation() const { return pimpl->affects_navigation; }

void NavigationObstacle3D::set_avoidance_enabled(bool enabled) {
    pimpl->avoidance_enabled = enabled;
    pimpl->dirty = true;
}
bool NavigationObstacle3D::is_avoidance_enabled() const { return pimpl->avoidance_enabled; }
void NavigationObstacle3D::set_avoidance_radius(double radius) {
    pimpl->avoidance_radius = std::max(0.0, radius);
    pimpl->dirty = true;
}
double NavigationObstacle3D::get_avoidance_radius() const { return pimpl->avoidance_radius; }
void NavigationObstacle3D::set_avoidance_priority(float priority) {
    pimpl->avoidance_priority = priority;
    pimpl->dirty = true;
}
float NavigationObstacle3D::get_avoidance_priority() const { return pimpl->avoidance_priority; }

void NavigationObstacle3D::set_debug_visible(bool visible) { pimpl->debug_visible = visible; }
bool NavigationObstacle3D::is_debug_visible() const { return pimpl->debug_visible; }
void NavigationObstacle3D::set_debug_color(const float* rgb) {
    memcpy(pimpl->debug_color, rgb, 3*sizeof(float));
}
void NavigationObstacle3D::get_debug_color(float* out_rgb) const {
    memcpy(out_rgb, pimpl->debug_color, 3*sizeof(float));
}
void NavigationObstacle3D::set_emissive_debug(bool enable, float intensity) {
    pimpl->emissive_debug = enable;
    pimpl->emissive_intensity = intensity;
}
bool NavigationObstacle3D::is_emissive_debug() const { return pimpl->emissive_debug; }

void NavigationObstacle3D::Impl::update_navigation_server() {
    if (obstacle_rid == -1) {
        // obstacle_rid = NavigationServer3D::obstacle_create();
    }
    // Set shape parameters
    // NavigationServer3D::obstacle_set_shape(obstacle_rid, shape, size, height, mesh_vertices);
    // Set carving, avoidance, etc.
    // NavigationServer3D::obstacle_set_carving(obstacle_rid, carving_enabled, carving_margin);
    // NavigationServer3D::obstacle_set_avoidance(obstacle_rid, avoidance_enabled, avoidance_radius, avoidance_priority);
    // Set position/orientation (global transform)
    Transform3D global = get_global_transform();
    // NavigationServer3D::obstacle_set_transform(obstacle_rid, global);
    // Enable/disable
    // NavigationServer3D::obstacle_set_enabled(obstacle_rid, affects_navigation && (carving_enabled || avoidance_enabled));
}

void NavigationObstacle3D::Impl::update_debug_mesh() {
    if (!debug_visible) return;
    // Create a temporary debug mesh (wireframe or solid) for visualization
    // using the obstacle shape parameters. This mesh can be rendered by the
    // rendering server with optional emissive lighting.
    // For performance, we would reuse a mesh resource and update instance transform.
    // Implementation omitted for brevity (would involve generating vertices/indices
    // for box, sphere, cylinder, or convex mesh and sending to RenderingServer).
    // The mesh should respect the debug color and emissive flag (if emissive_debug).
}

void NavigationObstacle3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    if (pimpl->dirty) {
        pimpl->update_navigation_server();
        pimpl->update_debug_mesh();
        pimpl->dirty = false;
    }
}

void NavigationObstacle3D::process(double delta) {
    Node3D::process(delta);
    // If obstacle is dynamic (moving), we need to update navigation server every frame.
    // We can check if transform changed and mark dirty.
    if (is_transform_dirty()) {
        pimpl->dirty = true;
    }
}

} // namespace lighting