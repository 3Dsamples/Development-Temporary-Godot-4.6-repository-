// Name : lighting enhancement
// File : scene/3d/camera_3d_ext.cpp 6 of 60
// Description : Implementation of Camera3DExt with full projection math, uniform buffer,
//               motion vectors, and real‑time rendering server updates.
#include "camera_3d_ext.h"
#include "servers/rendering_server.h"
#include "core/math/projection.h"
#include "core/math/transform_3d.h"
#include "core/math/math_funcs.h"
#include <cstring>

struct Camera3DExt::Impl {
    RID camera_rid;
    RID uniform_buffer_rid;
    float fov = 75.0f;
    float near_plane = 0.05f;
    float far_plane = 1000.0f;
    bool orthogonal = false;
    float ortho_size = 1.0f;
    uint32_t cull_mask = 0xFFFFFFFF;
    bool motion_vectors_enabled = false;
    bool params_dirty = true;
    bool transform_dirty = true;
    bool uniform_dirty = true;

    // Cached matrices
    Transform3D cached_transform;
    Projection cached_projection;
    Projection cached_view_projection;
    Projection cached_prev_view_projection;

    // Uniform buffer data (CPU side)
    struct CameraUniforms {
        float view_matrix[16];
        float projection_matrix[16];
        float view_projection_matrix[16];
        float inv_view_projection_matrix[16];
        float prev_view_projection_matrix[16];
        float near_far[2]; // x = near, y = far
    } uniforms;

    float viewport_aspect = 16.0f / 9.0f; // default, can be updated from viewport

    Impl() {
        camera_rid = RenderingServer::get_singleton()->camera_create();
        uniform_buffer_rid = RenderingServer::get_singleton()->uniform_buffer_create(sizeof(CameraUniforms));
        cached_transform.set_identity();
        cached_projection = Projection();
        cached_view_projection = Projection();
        cached_prev_view_projection = Projection();
        memset(&uniforms, 0, sizeof(uniforms));
    }

    ~Impl() {
        if (camera_rid.is_valid()) {
            RenderingServer::get_singleton()->free(camera_rid);
        }
        if (uniform_buffer_rid.is_valid()) {
            RenderingServer::get_singleton()->free(uniform_buffer_rid);
        }
    }

    void update_projection() {
        if (orthogonal) {
            cached_projection = Projection::create_orthogonal(ortho_size, viewport_aspect, near_plane, far_plane);
        } else {
            cached_projection = Projection::create_perspective(fov, viewport_aspect, near_plane, far_plane);
        }
    }

    void update_view_projection() {
        cached_view_projection = cached_projection * cached_transform.inverse();
        // Store previous for motion vectors
        if (uniform_dirty) {
            // first frame: set previous same as current
            cached_prev_view_projection = cached_view_projection;
        }
    }

    void fill_uniforms() {
        // Convert matrices to float arrays (row-major, OpenGL style)
        // view matrix: world to camera
        Transform3D inv_transform = cached_transform.inverse();
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                uniforms.view_matrix[i*4 + j] = inv_transform[i][j];
            }
        }
        // projection matrix
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                uniforms.projection_matrix[i*4 + j] = cached_projection[i][j];
            }
        }
        // view_projection matrix
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                uniforms.view_projection_matrix[i*4 + j] = cached_view_projection[i][j];
            }
        }
        // inverse view_projection
        Projection inv_vp = cached_view_projection.inverse();
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                uniforms.inv_view_projection_matrix[i*4 + j] = inv_vp[i][j];
            }
        }
        // previous view_projection
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                uniforms.prev_view_projection_matrix[i*4 + j] = cached_prev_view_projection[i][j];
            }
        }
        uniforms.near_far[0] = near_plane;
        uniforms.near_far[1] = far_plane;

        // upload to GPU
        RenderingServer::get_singleton()->uniform_buffer_update(uniform_buffer_rid, 0, sizeof(CameraUniforms), &uniforms);
    }

    void update_camera_server() {
        if (!params_dirty && !transform_dirty && !uniform_dirty) return;
        if (params_dirty) {
            update_projection();
            RenderingServer::get_singleton()->camera_set_projection(camera_rid, orthogonal, fov, near_plane, far_plane, ortho_size);
            params_dirty = false;
        }
        if (transform_dirty) {
            RenderingServer::get_singleton()->camera_set_transform(camera_rid, cached_transform);
            transform_dirty = false;
        }
        if (params_dirty || transform_dirty) {
            update_view_projection();
            uniform_dirty = true;
        }
        if (uniform_dirty) {
            fill_uniforms();
            uniform_dirty = false;
        }
    }
};

Camera3DExt::Camera3DExt() {
    pimpl = new Impl();
}

Camera3DExt::~Camera3DExt() {
    delete pimpl;
}

void Camera3DExt::set_fov(float p_fov) {
    pimpl->fov = p_fov;
    pimpl->params_dirty = true;
    sync_camera();
}

void Camera3DExt::set_near(float p_near) {
    pimpl->near_plane = p_near;
    pimpl->params_dirty = true;
    sync_camera();
}

void Camera3DExt::set_far(float p_far) {
    pimpl->far_plane = p_far;
    pimpl->params_dirty = true;
    sync_camera();
}

void Camera3DExt::set_orthogonal(bool p_orthogonal, float p_size) {
    pimpl->orthogonal = p_orthogonal;
    pimpl->ortho_size = p_size;
    pimpl->params_dirty = true;
    sync_camera();
}

void Camera3DExt::set_cull_mask(uint32_t p_mask) {
    pimpl->cull_mask = p_mask;
    RenderingServer::get_singleton()->camera_set_cull_mask(pimpl->camera_rid, p_mask);
}

uint32_t Camera3DExt::get_cull_mask() const {
    return pimpl->cull_mask;
}

void Camera3DExt::set_motion_vectors_enabled(bool p_enabled) {
    pimpl->motion_vectors_enabled = p_enabled;
    // In rendering server, motion vectors are enabled via camera attributes.
    RenderingServer::get_singleton()->camera_set_motion_vectors(pimpl->camera_rid, p_enabled);
}

bool Camera3DExt::is_motion_vectors_enabled() const {
    return pimpl->motion_vectors_enabled;
}

void Camera3DExt::sync_camera() {
    // Capture current global transform
    Transform3D global = get_global_transform();
    if (global != pimpl->cached_transform) {
        pimpl->cached_transform = global;
        pimpl->transform_dirty = true;
    }
    pimpl->update_camera_server();
}

RID Camera3DExt::get_camera_uniform_buffer() const {
    return pimpl->uniform_buffer_rid;
}