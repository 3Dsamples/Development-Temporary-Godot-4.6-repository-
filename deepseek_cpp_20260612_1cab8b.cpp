// Name : lighting enhancement updated
// File : gsplat_ext.h 70 of 63
// Description : Header for Gaussian splatting extension using Godot math types.
//               Declares forward/backward rasterization with full analytic gradients.
#pragma once

#include "core/math/vector3.h"
#include "core/math/quaternion.h"
#include "core/math/transform_3d.h"
#include "core/math/camera_matrix.h"
#include "core/math/color.h"
#include "core/templates/vector.h"
#include "servers/rendering_server.h"

// ----------------------------------------------------------------------------
// Gaussian primitive structure (matches Godot's expected layout)
// ----------------------------------------------------------------------------
struct GaussianPrimitive {
    Vector3 position;
    Vector3 scale;          // ellipsoid radii (sx, sy, sz)
    Quaternion rotation;    // unit quaternion (x, y, z, w)
    Color color;            // RGB base color
    float opacity;
    // Spherical harmonics would be added here, but omitted for brevity
};

// ----------------------------------------------------------------------------
// Camera parameters extracted from Godot camera and projection
// ----------------------------------------------------------------------------
struct GSCameraParams {
    Vector2 focal;          // (fx, fy) in pixels
    Vector2 principal;      // (cx, cy)
    Vector2 screen_size;    // (width, height)
    Transform3D view;       // world → camera transform
    Projection proj;        // camera → clip projection matrix
};

// ----------------------------------------------------------------------------
// Gradients for a single Gaussian
// ----------------------------------------------------------------------------
struct GaussianGradients {
    Vector3 d_pos;
    Vector3 d_scale;
    Quaternion d_rot;       // quaternion gradient (w, x, y, z)
    float d_opacity;
    Color d_color;
};

// ----------------------------------------------------------------------------
// Forward rasterization: produces RGB image and optionally depth.
// ----------------------------------------------------------------------------
void gsplat_forward(const Vector<GaussianPrimitive> &gaussians,
                    const GSCameraParams &cam,
                    Vector<uint8_t> &out_color,
                    Vector<float> *out_depth = nullptr);

// ----------------------------------------------------------------------------
// Backward rasterization: computes gradients w.r.t. each Gaussian.
//   grad_color: loss gradient w.r.t. output color (RGB, same size as out_color)
// ----------------------------------------------------------------------------
void gsplat_backward(const Vector<GaussianPrimitive> &gaussians,
                     const GSCameraParams &cam,
                     const Vector<uint8_t> &grad_color,
                     Vector<GaussianGradients> &out_grads);

// ----------------------------------------------------------------------------
// Convert Godot's view/projection matrices to our camera parameters.
// ----------------------------------------------------------------------------
GSCameraParams gs_camera_from_godot(const Transform3D &view_matrix,
                                    const Projection &proj_matrix,
                                    int width, int height);