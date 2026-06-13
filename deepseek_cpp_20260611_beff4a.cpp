// navigation_region_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// NavigationRegion3D – defines a navigable area using a mesh (NavMesh)
// or grid. Supports runtime generation, dynamic obstacles, and baking.
// The region can be used for pathfinding and agent movement.
// ============================================================================

class NavigationRegion3D : public Node3D {
public:
    NavigationRegion3D();
    ~NavigationRegion3D();

    // ------------------------------------------------------------------------
    // Navigation mesh source
    // ------------------------------------------------------------------------
    void set_mesh(int64_t mesh_rid);         // from RenderingServer
    int64_t get_mesh_rid() const;
    void set_mesh_path(const char* path);    // load from file (.navmesh)
    const char* get_mesh_path() const;

    // ------------------------------------------------------------------------
    // Baking (generate NavMesh from scene geometry)
    // ------------------------------------------------------------------------
    void set_bake_geometry(bool bake);
    bool get_bake_geometry() const;
    void set_bake_cell_size(float size);
    float get_bake_cell_size() const;
    void set_bake_cell_height(float height);
    float get_bake_cell_height() const;
    void set_bake_agent_height(float height);
    float get_bake_agent_height() const;
    void set_bake_agent_radius(float radius);
    float get_bake_agent_radius() const;
    void set_bake_agent_max_climb(float max_climb);
    float get_bake_agent_max_climb() const;
    void set_bake_agent_max_slope(float degrees);
    float get_bake_agent_max_slope() const;
    void set_bake_region_min_size(float size);
    float get_bake_region_min_size() const;
    void set_bake_region_merge_size(float size);
    float get_bake_region_merge_size() const;
    void set_bake_border_size(float size);
    float get_bake_border_size() const;

    void bake_navigation_mesh();            // synchronous
    void bake_async();                      // non‑blocking
    bool is_baking() const;
    void cancel_bake();
    float get_bake_progress() const;

    // ------------------------------------------------------------------------
    // Dynamic updates (modify mesh at runtime)
    // ------------------------------------------------------------------------
    void set_dynamic_updates_enabled(bool enabled);
    bool is_dynamic_updates_enabled() const;
    void add_navmesh_obstacle(int64_t obstacle_rid);
    void remove_navmesh_obstacle(int64_t obstacle_rid);
    void update_navmesh(const std::vector<double>& vertices, const std::vector<int>& indices);

    // ------------------------------------------------------------------------
    // Navigation query (direct, may be delegated to server)
    // ------------------------------------------------------------------------
    bool find_path(const double* from, const double* to, std::vector<double>& out_path) const;

    // ------------------------------------------------------------------------
    // Debug visualization (wireframe / solid, can be lit)
    // ------------------------------------------------------------------------
    void set_debug_visible(bool visible);
    bool is_debug_visible() const;
    void set_debug_color(const float* rgb);
    void get_debug_color(float* out_rgb) const;
    void set_debug_mode(int mode);   // 0 = wireframe, 1 = solid, 2 = both
    int get_debug_mode() const;

    // ------------------------------------------------------------------------
    // Lighting & GI (the debug mesh can be emissive to mark navigation areas)
    // ------------------------------------------------------------------------
    void set_emissive_debug(bool enable, float intensity = 0.1f);
    bool is_emissive_debug() const;
    void set_gi_contribution(float amount);
    float get_gi_contribution() const;

    // ------------------------------------------------------------------------
    // Navigation server synchronization
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;
    void process(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting