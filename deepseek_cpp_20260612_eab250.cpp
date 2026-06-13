// Name : lighting enhancement updated
// File : projection_ext.h 68 of 63
// Description : Header for 3D Gaussian projection math: world to camera space,
//               computation of 2D covariance, screen space coordinates, bounding box.
#pragma once

#include <cmath>
#include "gsplat_backward.h"  // for Vec3, Quat, Mat3, CameraParams

// ----------------------------------------------------------------------------
// Projection result: screen space data for a single Gaussian
// ----------------------------------------------------------------------------
struct ProjectionResult {
    Vec2 screen_center;   // pixel coordinates
    float depth;          // camera space Z (positive)
    float cov_xx, cov_xy, cov_yy;  // 2D covariance matrix (screen space)
    float radius;         // 3‑sigma bounding box radius (pixels)
    bool visible;         // true if depth > 0 and covariance valid
};

// ----------------------------------------------------------------------------
// Project a Gaussian to screen space using camera parameters.
// Inputs:
//   pos, scale, rot – Gaussian parameters in world space
//   cam – camera intrinsics and extrinsics
// Output: ProjectionResult structure
// ----------------------------------------------------------------------------
ProjectionResult project_gaussian_full(const Vec3 &pos, const Vec3 &scale,
                                       const Quat &rot, const CameraParams &cam);

// ----------------------------------------------------------------------------
// Compute 2D covariance from camera‑space covariance and 3D point.
// Used internally.
// ----------------------------------------------------------------------------
void compute_2d_covariance(const float cov6_cam[6], float X, float Y, float Z,
                           float fx, float fy, float &cxx, float &cxy, float &cyy);

// ----------------------------------------------------------------------------
// Transform 3D covariance from world to camera space (rotation only)
// ----------------------------------------------------------------------------
void transform_covariance_to_camera(const float cov6_world[6], const Mat3 &R_cam,
                                    float cov6_cam[6]);

// ----------------------------------------------------------------------------
// Compute 3D world covariance from scale and rotation
// ----------------------------------------------------------------------------
void compute_3d_covariance_full(const Vec3 &scale, const Quat &rot, float cov6[6]);