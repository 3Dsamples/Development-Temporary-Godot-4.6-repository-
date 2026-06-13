// camera_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <array>

namespace lighting {

// ============================================================================
// Camera3D – view/projection, frustum culling, motion vectors
// ============================================================================

enum class ProjectionType : uint8_t {
    PERSPECTIVE,
    ORTHOGONAL,
    FRUSTUM
};

class Camera3D : public Node3D {
public:
    Camera3D();
    ~Camera3D();

    // ------------------------------------------------------------------------
    // Projection parameters
    // ------------------------------------------------------------------------
    void set_projection_type(ProjectionType type);
    ProjectionType get_projection_type() const;

    // Perspective
    void set_fov_degrees(float fov);
    float get_fov_degrees() const;
    void set_near_plane(float near);
    float get_near_plane() const;
    void set_far_plane(float far);
    float get_far_plane() const;

    // Orthographic
    void set_orthographic_size(float size);
    float get_orthographic_size() const;

    // Frustum (custom)
    void set_frustum(float left, float right, float bottom, float top, float near, float far);

    // ------------------------------------------------------------------------
    // Aspect ratio (usually set by viewport)
    // ------------------------------------------------------------------------
    void set_aspect(float aspect);
    float get_aspect() const;

    // ------------------------------------------------------------------------
    // Viewport / render target
    // ------------------------------------------------------------------------
    void set_viewport_size(int width, int height);
    void get_viewport_size(int& width, int& height) const;

    // ------------------------------------------------------------------------
    // Culling (frustum)
    // ------------------------------------------------------------------------
    void set_culling_enabled(bool enabled);
    bool is_culling_enabled() const;
    void set_cull_mask(uint32_t mask);
    uint32_t get_cull_mask() const;

    // ------------------------------------------------------------------------
    // Motion vectors (for temporal effects)
    // ------------------------------------------------------------------------
    void set_motion_vectors_enabled(bool enabled);
    bool are_motion_vectors_enabled() const;
    void get_previous_view_projection(double* out_matrix) const; // 4x4
    void get_current_view_projection(double* out_matrix) const;
    void update_motion_vectors();

    // ------------------------------------------------------------------------
    // Physics interpolation (smooth camera movement)
    // ------------------------------------------------------------------------
    void set_physics_interpolation_enabled(bool enabled);
    bool is_physics_interpolation_enabled() const;
    void set_interpolation_delta(float delta);
    float get_interpolation_delta() const;

    // ------------------------------------------------------------------------
    // Matrix access (GPU friendly)
    // ------------------------------------------------------------------------
    void get_view_matrix(double* out_matrix) const;     // world → view
    void get_projection_matrix(double* out_matrix) const;   // view → clip
    void get_view_projection_matrix(double* out_matrix) const;
    void get_inverse_view_projection(double* out_matrix) const;

    // ------------------------------------------------------------------------
    // Frustum planes (for culling)
    // ------------------------------------------------------------------------
    enum PlaneSide { INSIDE, INTERSECT, OUTSIDE };
    PlaneSide test_sphere_frustum(const double* center, double radius) const;
    PlaneSide test_aabb_frustum(const double* min, const double* max) const;
    void get_frustum_planes(double* out_planes) const; // 6 planes, each 4 doubles

    // ------------------------------------------------------------------------
    // Rendering callbacks
    // ------------------------------------------------------------------------
    void prepare_for_render(); // call before rendering frame
    void finalize_after_render();

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting