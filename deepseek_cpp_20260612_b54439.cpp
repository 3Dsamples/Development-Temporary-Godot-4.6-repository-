// Name : lighting enhancement
// File : scene/3d/navigation_region_3d_ext.h 45 of 60
// Description : Extended navigation region node with NavMesh baking, dynamic obstacles,
//               debug visualization, and full NavigationServer3D + RenderingServer sync.
#pragma once

#include "scene/3d/navigation_region_3d.h"
#include "servers/navigation_server_3d.h"
#include "servers/rendering_server.h"

class NavigationRegion3DExt : public NavigationRegion3D {
    GDCLASS(NavigationRegion3DExt, NavigationRegion3D);

public:
    NavigationRegion3DExt();
    ~NavigationRegion3DExt();

    // ------------------------------------------------------------------------
    // Navigation mesh (source geometry)
    // ------------------------------------------------------------------------
    void set_navigation_mesh(const RID &p_nav_mesh);
    RID get_navigation_mesh() const;

    // ------------------------------------------------------------------------
    // Baking parameters (using Recast/Detour)
    // ------------------------------------------------------------------------
    void set_cell_size(float p_size);
    float get_cell_size() const;
    void set_cell_height(float p_height);
    float get_cell_height() const;
    void set_agent_height(float p_height);
    float get_agent_height() const;
    void set_agent_radius(float p_radius);
    float get_agent_radius() const;
    void set_agent_max_climb(float p_climb);
    float get_agent_max_climb() const;
    void set_agent_max_slope(float p_slope_deg);
    float get_agent_max_slope() const;
    void set_region_min_size(float p_size);
    float get_region_min_size() const;
    void set_region_merge_size(float p_size);
    float get_region_merge_size() const;

    // ------------------------------------------------------------------------
    // Bake navigation mesh (asynchronous)
    // ------------------------------------------------------------------------
    void bake_navmesh();
    bool is_baking() const;
    float get_bake_progress() const;
    void cancel_bake();

    // ------------------------------------------------------------------------
    // Dynamic obstacles (for runtime updates)
    // ------------------------------------------------------------------------
    void add_obstacle(const RID &p_obstacle);
    void remove_obstacle(const RID &p_obstacle);
    void clear_obstacles();

    // ------------------------------------------------------------------------
    // Debug visualization (wireframe of navigation mesh)
    // ------------------------------------------------------------------------
    void set_debug_visible(bool p_visible);
    bool is_debug_visible() const;
    void set_debug_color(const Color &p_color);
    Color get_debug_color() const;
    void set_debug_emissive(const Color &p_color, float p_intensity);
    void get_debug_emissive(Color &r_color, float &r_intensity) const;

    // ------------------------------------------------------------------------
    // Global illumination (navmesh can contribute to GI if emissive debug)
    // ------------------------------------------------------------------------
    void set_gi_mode(int p_mode);        // 0=off,1=static,2=dynamic
    int get_gi_mode() const;
    void set_gi_contribution(float p_amount);
    float get_gi_contribution() const;

    // ------------------------------------------------------------------------
    // Synchronization (call after changing parameters)
    // ------------------------------------------------------------------------
    void sync_region();

private:
    struct Impl;
    Impl *pimpl;
};