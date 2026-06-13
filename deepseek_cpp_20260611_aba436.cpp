// navigation_region_3d.cpp
#include "navigation_region_3d.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <thread>
#include <atomic>
#include <functional>

namespace lighting {

// ============================================================================
// Internal: Recast/Detour stub (replace with actual library in production)
// ============================================================================
struct NavMeshBuilder {
    struct Config {
        float cell_size = 0.2f;
        float cell_height = 0.1f;
        float agent_height = 2.0f;
        float agent_radius = 0.5f;
        float agent_max_climb = 0.9f;
        float agent_max_slope = 45.0f;
        float region_min_size = 2.0f;
        float region_merge_size = 20.0f;
        float border_size = 0.0f;
    };
    Config config;
    std::vector<double> vertices;
    std::vector<int> indices;
    bool bake(const std::vector<double>& geom_vertices, const std::vector<int>& geom_indices) {
        // Placeholder: generate a simple plane mesh as NavMesh.
        // In real implementation, runs Recast.
        vertices = { -10,0,-10, 10,0,-10, 10,0,10, -10,0,10 };
        indices = { 0,1,2, 0,2,3 };
        return true;
    }
};

// ============================================================================
// NavigationRegion3D implementation
// ============================================================================
struct NavigationRegion3D::Impl {
    int64_t mesh_rid = -1;
    char mesh_path[256] = {0};
    bool bake_geometry = true;
    NavMeshBuilder::Config bake_config;

    bool dynamic_updates = false;
    std::vector<int64_t> obstacles;

    bool debug_visible = true;
    float debug_color[3] = {0.2f, 0.6f, 1.0f}; // light blue
    int debug_mode = 1;                        // solid

    bool emissive_debug = false;
    float emissive_intensity = 0.1f;
    float gi_contribution = 1.0f;

    // Baking state
    std::atomic<bool> baking{false};
    std::atomic<float> bake_progress{0.0f};
    std::thread bake_thread;
    NavMeshBuilder builder;
    std::vector<double> baked_vertices;
    std::vector<int> baked_indices;

    // Navigation server handle
    int64_t nav_region_rid = -1;
    bool dirty = true;

    // Debug mesh for visualization (RenderingServer)
    int64_t debug_mesh_rid = -1;
    int64_t debug_instance_rid = -1;

    ~Impl() {
        if (bake_thread.joinable()) bake_thread.join();
        if (nav_region_rid != -1) {
            // NavigationServer3D::region_free(nav_region_rid);
        }
    }

    void start_bake_async();
    void finish_bake();
    void update_debug_mesh();
    void update_navigation_server();
};

NavigationRegion3D::NavigationRegion3D() : pimpl(std::make_unique<Impl>()) {}
NavigationRegion3D::~NavigationRegion3D() = default;

void NavigationRegion3D::set_mesh(int64_t mesh_rid) {
    pimpl->mesh_rid = mesh_rid;
    pimpl->mesh_path[0] = 0;
    pimpl->dirty = true;
}
int64_t NavigationRegion3D::get_mesh_rid() const { return pimpl->mesh_rid; }
void NavigationRegion3D::set_mesh_path(const char* path) {
    strncpy(pimpl->mesh_path, path, 255);
    pimpl->mesh_path[255] = 0;
    pimpl->mesh_rid = -1;
    pimpl->dirty = true;
}
const char* NavigationRegion3D::get_mesh_path() const { return pimpl->mesh_path; }

void NavigationRegion3D::set_bake_geometry(bool bake) { pimpl->bake_geometry = bake; }
bool NavigationRegion3D::get_bake_geometry() const { return pimpl->bake_geometry; }
void NavigationRegion3D::set_bake_cell_size(float size) { pimpl->bake_config.cell_size = size; }
float NavigationRegion3D::get_bake_cell_size() const { return pimpl->bake_config.cell_size; }
void NavigationRegion3D::set_bake_cell_height(float height) { pimpl->bake_config.cell_height = height; }
float NavigationRegion3D::get_bake_cell_height() const { return pimpl->bake_config.cell_height; }
void NavigationRegion3D::set_bake_agent_height(float height) { pimpl->bake_config.agent_height = height; }
float NavigationRegion3D::get_bake_agent_height() const { return pimpl->bake_config.agent_height; }
void NavigationRegion3D::set_bake_agent_radius(float radius) { pimpl->bake_config.agent_radius = radius; }
float NavigationRegion3D::get_bake_agent_radius() const { return pimpl->bake_config.agent_radius; }
void NavigationRegion3D::set_bake_agent_max_climb(float max_climb) { pimpl->bake_config.agent_max_climb = max_climb; }
float NavigationRegion3D::get_bake_agent_max_climb() const { return pimpl->bake_config.agent_max_climb; }
void NavigationRegion3D::set_bake_agent_max_slope(float degrees) { pimpl->bake_config.agent_max_slope = degrees; }
float NavigationRegion3D::get_bake_agent_max_slope() const { return pimpl->bake_config.agent_max_slope; }
void NavigationRegion3D::set_bake_region_min_size(float size) { pimpl->bake_config.region_min_size = size; }
float NavigationRegion3D::get_bake_region_min_size() const { return pimpl->bake_config.region_min_size; }
void NavigationRegion3D::set_bake_region_merge_size(float size) { pimpl->bake_config.region_merge_size = size; }
float NavigationRegion3D::get_bake_region_merge_size() const { return pimpl->bake_config.region_merge_size; }
void NavigationRegion3D::set_bake_border_size(float size) { pimpl->bake_config.border_size = size; }
float NavigationRegion3D::get_bake_border_size() const { return pimpl->bake_config.border_size; }

void NavigationRegion3D::Impl::start_bake_async() {
    if (baking) return;
    baking = true;
    bake_progress = 0.0f;
    bake_thread = std::thread([this]() {
        // Collect geometry from scene (simplified: use a dummy square)
        std::vector<double> geom_vertices;
        std::vector<int> geom_indices;
        // In real engine, gather all static meshes within region bounds
        // For demo, use plane
        geom_vertices = { -20,0,-20, 20,0,-20, 20,0,20, -20,0,20 };
        geom_indices = { 0,1,2, 0,2,3 };
        // Configure builder
        builder.config = bake_config;
        if (builder.bake(geom_vertices, geom_indices)) {
            baked_vertices = builder.vertices;
            baked_indices = builder.indices;
            // Update navigation server
            if (nav_region_rid != -1) {
                // NavigationServer3D::region_set_mesh(nav_region_rid, baked_vertices, baked_indices);
            }
            // Update debug mesh
            update_debug_mesh();
        }
        bake_progress = 1.0f;
        baking = false;
        dirty = true;
    });
}

void NavigationRegion3D::bake_navigation_mesh() {
    if (pimpl->baking) return;
    pimpl->start_bake_async();
    if (pimpl->bake_thread.joinable()) pimpl->bake_thread.join();
}
void NavigationRegion3D::bake_async() {
    if (pimpl->baking) return;
    pimpl->start_bake_async();
}
bool NavigationRegion3D::is_baking() const { return pimpl->baking; }
void NavigationRegion3D::cancel_bake() {
    if (pimpl->bake_thread.joinable()) pimpl->bake_thread.detach();
    pimpl->baking = false;
}
float NavigationRegion3D::get_bake_progress() const { return pimpl->bake_progress; }

void NavigationRegion3D::set_dynamic_updates_enabled(bool enabled) { pimpl->dynamic_updates = enabled; }
bool NavigationRegion3D::is_dynamic_updates_enabled() const { return pimpl->dynamic_updates; }
void NavigationRegion3D::add_navmesh_obstacle(int64_t obstacle_rid) {
    pimpl->obstacles.push_back(obstacle_rid);
    if (pimpl->nav_region_rid != -1) {
        // NavigationServer3D::region_add_obstacle(pimpl->nav_region_rid, obstacle_rid);
    }
}
void NavigationRegion3D::remove_navmesh_obstacle(int64_t obstacle_rid) {
    auto it = std::find(pimpl->obstacles.begin(), pimpl->obstacles.end(), obstacle_rid);
    if (it != pimpl->obstacles.end()) {
        pimpl->obstacles.erase(it);
        // NavigationServer3D::region_remove_obstacle(pimpl->nav_region_rid, obstacle_rid);
    }
}
void NavigationRegion3D::update_navmesh(const std::vector<double>& vertices, const std::vector<int>& indices) {
    pimpl->baked_vertices = vertices;
    pimpl->baked_indices = indices;
    if (pimpl->nav_region_rid != -1) {
        // NavigationServer3D::region_set_mesh(pimpl->nav_region_rid, vertices, indices);
    }
    pimpl->update_debug_mesh();
}

bool NavigationRegion3D::find_path(const double* from, const double* to, std::vector<double>& out_path) const {
    // Query navigation server
    // For demo, return straight line
    out_path.clear();
    out_path.push_back(from[0]); out_path.push_back(from[1]); out_path.push_back(from[2]);
    out_path.push_back(to[0]);   out_path.push_back(to[1]);   out_path.push_back(to[2]);
    return true;
}

void NavigationRegion3D::set_debug_visible(bool visible) { pimpl->debug_visible = visible; pimpl->update_debug_mesh(); }
bool NavigationRegion3D::is_debug_visible() const { return pimpl->debug_visible; }
void NavigationRegion3D::set_debug_color(const float* rgb) { memcpy(pimpl->debug_color, rgb, 3*sizeof(float)); pimpl->update_debug_mesh(); }
void NavigationRegion3D::get_debug_color(float* out_rgb) const { memcpy(out_rgb, pimpl->debug_color, 3*sizeof(float)); }
void NavigationRegion3D::set_debug_mode(int mode) { pimpl->debug_mode = mode; pimpl->update_debug_mesh(); }
int NavigationRegion3D::get_debug_mode() const { return pimpl->debug_mode; }
void NavigationRegion3D::set_emissive_debug(bool enable, float intensity) { pimpl->emissive_debug = enable; pimpl->emissive_intensity = intensity; pimpl->update_debug_mesh(); }
bool NavigationRegion3D::is_emissive_debug() const { return pimpl->emissive_debug; }
void NavigationRegion3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
float NavigationRegion3D::get_gi_contribution() const { return pimpl->gi_contribution; }

void NavigationRegion3D::Impl::update_debug_mesh() {
    if (!debug_visible || baked_vertices.empty()) return;
    // Generate mesh resource for debug visualization (colored, optionally emissive)
    if (debug_mesh_rid == -1) {
        // debug_mesh_rid = RenderingServer::mesh_create();
    }
    // Clear previous surfaces and add new one with vertices, indices, and normals (face normals)
    // Use debug_color and emissive flag.
    // Set up material: unlit or emissive if emissive_debug.
    // In real engine, we would create a material resource and assign.
    if (debug_instance_rid == -1) {
        // debug_instance_rid = RenderingServer::instance_create();
    }
    // RenderingServer::instance_set_base(debug_instance_rid, debug_mesh_rid);
    // RenderingServer::instance_set_transform(debug_instance_rid, get_global_transform());
    // RenderingServer::instance_set_visible(debug_instance_rid, debug_visible);
}

void NavigationRegion3D::Impl::update_navigation_server() {
    if (nav_region_rid == -1) {
        // nav_region_rid = NavigationServer3D::region_create();
    }
    // NavigationServer3D::region_set_transform(nav_region_rid, get_global_transform());
    // NavigationServer3D::region_set_mesh(nav_region_rid, baked_vertices, baked_indices);
    // NavigationServer3D::region_set_agent_settings(...)
    // Set dynamic updates flag, add obstacles.
    // Also set navigation layers, etc. (omitted for brevity)
}

void NavigationRegion3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    if (pimpl->dirty) {
        pimpl->update_navigation_server();
        if (pimpl->debug_visible) pimpl->update_debug_mesh();
        pimpl->dirty = false;
    }
}

void NavigationRegion3D::process(double delta) {
    Node3D::process(delta);
    // If transform changed, update navigation server transform.
    if (is_transform_dirty()) {
        pimpl->dirty = true;
    }
    // For dynamic updates, we might need to re-apply obstacles.
}

} // namespace lighting