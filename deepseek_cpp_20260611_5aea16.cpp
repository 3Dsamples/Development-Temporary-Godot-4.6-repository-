// Name : lighting enhancement
// File : scene/3d/camera_3d_ext.h 5 of 60
// Description : Extended camera with frustum culling, motion vectors for temporal effects,
//               and real‑time rendering server updates for shadows and GI.
#pragma once

#include "scene/3d/camera_3d.h"
#include "servers/rendering_server.h"

class Camera3DExt : public Camera3D {
    GDCLASS(Camera3DExt, Camera3D);

public:
    Camera3DExt();
    ~Camera3DExt();

    // ------------------------------------------------------------------------
    // Projection parameters
    // ------------------------------------------------------------------------
    void set_fov(float p_fov);
    void set_near(float p_near);
    void set_far(float p_far);
    void set_orthogonal(bool p_orthogonal, float p_size = 1.0f);

    // ------------------------------------------------------------------------
    // Frustum culling (for light culling and occlusion)
    // ------------------------------------------------------------------------
    void set_cull_mask(uint32_t p_mask);
    uint32_t get_cull_mask() const;

    // ------------------------------------------------------------------------
    // Motion vectors for temporal anti‑aliasing and motion blur
    // ------------------------------------------------------------------------
    void set_motion_vectors_enabled(bool p_enabled);
    bool is_motion_vectors_enabled() const;

    // ------------------------------------------------------------------------
    // Real‑time rendering server sync (push matrices and culling parameters)
    // ------------------------------------------------------------------------
    void sync_camera();

    // ------------------------------------------------------------------------
    // GPU‑side camera data (uniform buffer)
    // ------------------------------------------------------------------------
    RID get_camera_uniform_buffer() const;

private:
    struct Impl;
    Impl *pimpl;
};