// Name : lighting enhancement updated
// File : gsplat_backward.h 66 of 63
// Description : Header for backward pass of Gaussian splatting. Declares structures and
//               function to compute gradients w.r.t. position, scale, rotation (quaternion),
//               opacity, and color.
#pragma once

#include <cmath>
#include <vector>

// ----------------------------------------------------------------------------
// Basic math types
// ----------------------------------------------------------------------------
struct Vec3 { float x, y, z; };
struct Vec2 { float x, y; };
struct Quat { float w, x, y, z; };
struct Mat3 { float m[9]; };
struct Mat33 { float m[3][3]; };  // for easier indexing

// ----------------------------------------------------------------------------
// Camera parameters needed for projection and gradients
// ----------------------------------------------------------------------------
struct CameraParams {
    float fx, fy;      // focal lengths (pixels)
    float cx, cy;      // principal point
    float width, height;
    Mat3 view_rot;     // rotation matrix (world → camera) row‑major 3x3
    Vec3 view_trans;   // translation part
};

// ----------------------------------------------------------------------------
// Projection data computed during forward pass (to reuse in backward)
// ----------------------------------------------------------------------------
struct ProjData {
    float depth;                 // Z in camera space
    Vec2 screen_center;          // pixel coordinates
    float cov_xx, cov_xy, cov_yy; // 2D covariance (screen space)
    float radius;                // 3‑sigma bounding box (pixels)
    float weight;                // Gaussian weight at current pixel (optional)
};

// ----------------------------------------------------------------------------
// Gradients for a single Gaussian (to be accumulated)
// ----------------------------------------------------------------------------
struct GaussianGradients {
    Vec3 d_pos;
    Vec3 d_scale;
    Quat d_rot;      // quaternion gradient (w, x, y, z)
    float d_opacity;
    Vec3 d_color;
};

// ----------------------------------------------------------------------------
// Compute analytic gradients for a Gaussian given loss gradient at a pixel.
// Inputs:
//   pos, scale, rot, opacity, color – current Gaussian parameters
//   proj – precomputed projection data (from forward pass)
//   pixel – (px, py) pixel coordinates (float)
//   dL_dC – loss gradient w.r.t. pixel color (RGB)
//   transmittance_before – accumulated transmittance before this Gaussian
//   cam – camera parameters
// Returns gradients for this Gaussian (to be accumulated over all pixels)
// ----------------------------------------------------------------------------
GaussianGradients compute_gaussian_gradients(const Vec3 &pos, const Vec3 &scale,
                                             const Quat &rot, float opacity,
                                             const Vec3 &color, const ProjData &proj,
                                             float px, float py, const Vec3 &dL_dC,
                                             float transmittance_before,
                                             const CameraParams &cam);