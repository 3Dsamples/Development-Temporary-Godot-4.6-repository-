// Name : lighting enhancement updated
// File : gsplat_ext.h 70 of 63
// Description : Header for Gaussian splatting extension using Godot math types.
//               Declares forward/backward rasterization and camera conversion.
#pragma once

#include "core/math/vector3.h"
#include "core/math/quaternion.h"
#include "core/math/transform_3d.h"
#include "core/math/camera_matrix.h"
#include "core/math/color.h"
#include "servers/rendering_server.h"
#include <vector>

// ----------------------------------------------------------------------------
// Gaussian primitive structure (matching Godot's expected layout)
// ----------------------------------------------------------------------------
struct GaussianPrimitive {
    Vector3 position;
    Vector3 scale;       // ellipsoid radii
    Quaternion rotation; // stored as (x, y, z, w)
    Color color;         // RGB base color (alpha not used)
    float opacity;
    // Spherical harmonics coefficients would be here, but omitted for brevity
};

// ----------------------------------------------------------------------------
// Camera parameters extracted from Godot camera and projection
// ----------------------------------------------------------------------------
struct GSCameraParams {
    Vector2 focal;       // (fx, fy) in pixels
    Vector2 principal;   // (cx, cy)
    Vector2 screen_size; // (width, height)
    Transform3D view;    // world → camera transform
    Projection proj;     // camera → clip projection matrix
};

// ----------------------------------------------------------------------------
// Forward rasterization
// ----------------------------------------------------------------------------
void gsplat_forward(const std::vector<GaussianPrimitive> &gaussians,
                    const GSCameraParams &cam,
                    std::vector<uint8_t> &out_color,
                    std::vector<float> *out_depth = nullptr);

// ----------------------------------------------------------------------------
// Backward rasterization (computes gradients for each Gaussian)
// ----------------------------------------------------------------------------
struct GaussianGradients {
    Vector3 d_pos;
    Vector3 d_scale;
    Quaternion d_rot;
    float d_opacity;
    Color d_color;
};

void gsplat_backward(const std::vector<GaussianPrimitive> &gaussians,
                     const GSCameraParams &cam,
                     const std::vector<uint8_t> &grad_color,
                     std::vector<GaussianGradients> &out_grads);

// ----------------------------------------------------------------------------
// Convert Godot's camera and projection to our camera parameters
// ----------------------------------------------------------------------------
GSCameraParams gs_camera_from_godot(const Transform3D &view_matrix,
                                    const Projection &proj_matrix,
                                    int width, int height);