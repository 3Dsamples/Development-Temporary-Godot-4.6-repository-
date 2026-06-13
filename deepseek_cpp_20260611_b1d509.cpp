// navigation_obstacle_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// NavigationObstacle3D – dynamic obstacle that can affect navigation meshes
// by carving holes or adjusting avoidance. Does not cast shadows but can be
// debug‑rendered with basic lighting/color.
// ============================================================================

enum class ObstacleShape : uint8_t {
    BOX,
    SPHERE,
    CYLINDER,
    CONVEX_MESH
};

class NavigationObstacle3D : public Node3D {
public:
    NavigationObstacle3D();
    ~NavigationObstacle3D();

    // ------------------------------------------------------------------------
    // Obstacle geometry
    // ------------------------------------------------------------------------
    void set_shape(ObstacleShape shape);
    ObstacleShape get_shape() const;
    void set_size(const double* size);      // half extents for box, radius for sphere/cylinder
    void get_size(double* out_size) const;
    void set_height(double height);          // for cylinder only
    double get_height() const;
    void set_mesh_vertices(const std::vector<double>& vertices); // for convex mesh
    std::vector<double> get_mesh_vertices() const;

    // ------------------------------------------------------------------------
    // Navigation carving (dynamic updates)
    // ------------------------------------------------------------------------
    void set_carving_enabled(bool enabled);
    bool is_carving_enabled() const;
    void set_carving_margin(double margin);  // extra space around obstacle
    double get_carving_margin() const;
    void set_affects_navigation(bool affects);
    bool get_affects_navigation() const;

    // ------------------------------------------------------------------------
    // Agent avoidance (for RVO)
    // ------------------------------------------------------------------------
    void set_avoidance_enabled(bool enabled);
    bool is_avoidance_enabled() const;
    void set_avoidance_radius(double radius);
    double get_avoidance_radius() const;
    void set_avoidance_priority(float priority);
    float get_avoidance_priority() const;

    // ------------------------------------------------------------------------
    // Debug visualization (optional, can be lit with emissive)
    // ------------------------------------------------------------------------
    void set_debug_visible(bool visible);
    bool is_debug_visible() const;
    void set_debug_color(const float* rgb);
    void get_debug_color(float* out_rgb) const;
    void set_emissive_debug(bool enable, float intensity = 0.2f);
    bool is_emissive_debug() const;

    // ------------------------------------------------------------------------
    // Navigation server sync
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;
    void process(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting