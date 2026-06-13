// Name : lighting enhancement
// File : scene/3d/camera_3d.h file number : 5
// Description : 3D camera node with perspective/orthographic projection, frustum culling,
//               motion vectors, temporal anti-aliasing support, and physics interpolation.

#pragma once

#include "scene/3d/node_3d.h"
#include "servers/rendering_server.h"
#include "core/math/transform_3d.h"
#include "core/math/projection.h"

class Camera3D : public Node3D {
    GDCLASS(Camera3D, Node3D);

public:
    enum ProjectionType {
        PROJECTION_PERSPECTIVE,
        PROJECTION_ORTHOGONAL,
        PROJECTION_FRUSTUM
    };

    Camera3D();
    ~Camera3D();

    // Projection parameters
    void set_projection_type(ProjectionType p_type);
    ProjectionType get_projection_type() const;
    void set_fov(float p_fov_degrees);
    float get_fov() const;
    void set_near(float p_near);
    float get_near() const;
    void set_far(float p_far);
    float get_far() const;
    void set_orthogonal_size(float p_size);
    float get_orthogonal_size() const;
    void set_frustum(float p_left, float p_right, float p_bottom, float p_top, float p_near, float p_far);

    // Viewport and aspect
    void set_viewport_size(int p_width, int p_height);
    void get_viewport_size(int &r_width, int &r_height) const;
    void set_aspect(float p_aspect);
    float get_aspect() const;

    // Culling
    void set_culling_enabled(bool p_enabled);
    bool is_culling_enabled() const;
    void set_cull_mask(uint32_t p_mask);
    uint32_t get_cull_mask() const;

    // Motion vectors (for temporal effects)
    void set_motion_vectors_enabled(bool p_enabled);
    bool are_motion_vectors_enabled() const;
    void get_previous_view_projection(Projection &r_proj) const;
    void get_current_view_projection(Projection &r_proj) const;
    void update_motion_vectors();

    // Physics interpolation
    void set_physics_interpolation_enabled(bool p_enabled);
    bool is_physics_interpolation_enabled() const;
    void set_interpolation_delta(float p_delta);
    float get_interpolation_delta() const;

    // Matrix access
    Projection get_view_matrix() const;
    Projection get_projection_matrix() const;
    Projection get_view_projection_matrix() const;
    Projection get_inverse_view_projection() const;

    // Frustum planes (world space)
    void get_frustum_planes(Vector<Plane> &r_planes) const;
    bool is_sphere_visible(const Vector3 &p_center, float p_radius) const;
    bool is_aabb_visible(const AABB &p_aabb) const;

    // Rendering server synchronization
    void synchronize_render_server(double p_delta) override;

    // Get camera RID for rendering server
    RID get_camera_rid() const;

protected:
    void _transform_changed() override;
    void _update_render_server_transform() override;

private:
    struct Impl;
    Impl *pimpl;
};