// Name : lighting enhancement
// File : scene/3d/collision_polygon_3d_ext.h 43 of 60
// Description : Extended 3D collision polygon node with extrusion depth, convex decomposition,
//               debug visualization (wireframe), and full PhysicsServer + RenderingServer sync.
#pragma once

#include "scene/3d/collision_polygon_3d.h"
#include "servers/rendering_server.h"
#include "servers/physics_server_3d.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"

class CollisionPolygon3DExt : public CollisionPolygon3D {
    GDCLASS(CollisionPolygon3DExt, CollisionPolygon3D);

public:
    CollisionPolygon3DExt();
    ~CollisionPolygon3DExt();

    // ------------------------------------------------------------------------
    // Polygon definition (2D points in local XY plane)
    // ------------------------------------------------------------------------
    void set_polygon(const Vector<Vector2> &p_polygon);
    Vector<Vector2> get_polygon() const;

    // ------------------------------------------------------------------------
    // Extrusion depth (along local Z axis)
    // ------------------------------------------------------------------------
    void set_depth(float p_depth);
    float get_depth() const;

    // ------------------------------------------------------------------------
    // Build mode: solid (prism), hollow (edges only), or convex hull decomposition
    // ------------------------------------------------------------------------
    enum BuildMode {
        BUILD_SOLID,
        BUILD_HOLLOW,
        BUILD_CONVEX_HULL
    };
    void set_build_mode(BuildMode p_mode);
    BuildMode get_build_mode() const;

    // ------------------------------------------------------------------------
    // Convex decomposition (max convex pieces, for concave polygons)
    // ------------------------------------------------------------------------
    void set_max_convex_pieces(int p_max);
    int get_max_convex_pieces() const;

    // ------------------------------------------------------------------------
    // Collision margin (for shape generation)
    // ------------------------------------------------------------------------
    void set_margin(float p_margin);
    float get_margin() const;

    // ------------------------------------------------------------------------
    // Transform relative to parent CollisionObject3D
    // ------------------------------------------------------------------------
    void set_local_transform(const Transform3D &p_transform);
    Transform3D get_local_transform() const;

    // ------------------------------------------------------------------------
    // Debug visualization (wireframe, can be emissive)
    // ------------------------------------------------------------------------
    void set_debug_visible(bool p_visible);
    bool is_debug_visible() const;
    void set_debug_color(const Color &p_color);
    Color get_debug_color() const;
    void set_debug_emissive(const Color &p_color, float p_intensity);
    void get_debug_emissive(Color &r_color, float &r_intensity) const;

    // ------------------------------------------------------------------------
    // Global illumination (polygon can contribute to GI if emissive)
    // ------------------------------------------------------------------------
    void set_gi_mode(int p_mode);        // 0=off,1=static,2=dynamic
    int get_gi_mode() const;
    void set_gi_contribution(float p_amount);
    float get_gi_contribution() const;

    // ------------------------------------------------------------------------
    // Synchronization with servers (call after changing polygon or depth)
    // ------------------------------------------------------------------------
    void sync_polygon();

private:
    struct Impl;
    Impl *pimpl;
};