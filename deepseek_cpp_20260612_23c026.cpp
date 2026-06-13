// Name : lighting enhancement
// File : scene/3d/collision_shape_3d_ext.h 41 of 60
// Description : Extended collision shape node supporting sphere, box, capsule, cylinder,
//               convex/concave meshes, heightfield, and plane shapes. Full PhysicsServer
//               and RenderingServer synchronization with debug visualization and GI flags.
#pragma once

#include "scene/3d/collision_shape_3d.h"
#include "servers/rendering_server.h"
#include "servers/physics_server_3d.h"

class CollisionShape3DExt : public CollisionShape3D {
    GDCLASS(CollisionShape3DExt, CollisionShape3D);

public:
    CollisionShape3DExt();
    ~CollisionShape3DExt();

    // ------------------------------------------------------------------------
    // Shape type and geometry
    // ------------------------------------------------------------------------
    enum ShapeType {
        SHAPE_SPHERE,
        SHAPE_BOX,
        SHAPE_CAPSULE,
        SHAPE_CYLINDER,
        SHAPE_CONVEX_POLYHEDRON,
        SHAPE_CONCAVE_MESH,
        SHAPE_HEIGHTFIELD,
        SHAPE_PLANE
    };
    void set_shape_type(ShapeType p_type);
    ShapeType get_shape_type() const;

    // Sphere
    void set_sphere_radius(float p_radius);
    float get_sphere_radius() const;

    // Box
    void set_box_half_extents(const Vector3 &p_half_extents);
    Vector3 get_box_half_extents() const;

    // Capsule
    void set_capsule_radius(float p_radius);
    float get_capsule_radius() const;
    void set_capsule_height(float p_height);
    float get_capsule_height() const;

    // Cylinder
    void set_cylinder_radius(float p_radius);
    float get_cylinder_radius() const;
    void set_cylinder_height(float p_height);
    float get_cylinder_height() const;

    // Convex / concave mesh
    void set_convex_mesh(const Vector<Vector3> &p_vertices, const Vector<int> &p_indices);
    void set_concave_mesh(const Vector<Vector3> &p_vertices, const Vector<int> &p_indices);
    void clear_mesh();

    // Heightfield (grid of heights)
    void set_heightfield(int p_width, int p_depth, const Vector<float> &p_heights,
                         float p_min_height, float p_max_height,
                         const Transform3D &p_transform = Transform3D(), float p_cell_size = 1.0f);
    void clear_heightfield();

    // Plane (infinite plane, defined by normal and distance from origin)
    void set_plane(const Plane &p_plane);
    Plane get_plane() const;

    // ------------------------------------------------------------------------
    // Transform relative to parent CollisionObject3D
    // ------------------------------------------------------------------------
    void set_local_transform(const Transform3D &p_transform);
    Transform3D get_local_transform() const;

    // ------------------------------------------------------------------------
    // Debug visualization (wireframe shape, can be lit/emissive)
    // ------------------------------------------------------------------------
    void set_debug_visible(bool p_visible);
    bool is_debug_visible() const;
    void set_debug_color(const Color &p_color);
    Color get_debug_color() const;
    void set_debug_emissive(const Color &p_color, float p_intensity);
    void get_debug_emissive(Color &r_color, float &r_intensity) const;

    // ------------------------------------------------------------------------
    // Global illumination (shape can contribute to GI if emissive)
    // ------------------------------------------------------------------------
    void set_gi_mode(int p_mode);        // 0=off,1=static,2=dynamic
    int get_gi_mode() const;
    void set_gi_contribution(float p_amount);
    float get_gi_contribution() const;

    // ------------------------------------------------------------------------
    // Synchronization with servers (called automatically on changes)
    // ------------------------------------------------------------------------
    void sync_shape();

private:
    struct Impl;
    Impl *pimpl;
};