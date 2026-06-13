// Name : lighting enhancement updated
// File : gsplat_ext.h 70 of 63
// Description : Header for the main Gaussian splatting extension interface.
//               Declares functions for forward and backward rasterization,
//               tensor binding, and CUDA kernel wrappers.
#pragma once

#include <cstdint>
#include <vector>
#include "gsplat_backward.h"   // for Vec3, Quat, CameraParams
#include "projection_ext.h"    // for ProjectionResult

// ----------------------------------------------------------------------------
// Forward rasterization: produces color and optionally depth images.
// Inputs:
//   gaussians – list of Gaussian parameters (position, scale, rotation, color, opacity)
//   cam – camera parameters (intrinsics, view matrix)
//   width, height – image dimensions
// Output:
//   out_color – flattened RGB image (3 * width * height bytes, uint8)
//   out_depth – optional depth buffer (float, width*height)
// ----------------------------------------------------------------------------
void gsplat_forward(const std::vector<Gaussian> &gaussians,
                    const CameraParams &cam,
                    int width, int height,
                    std::vector<uint8_t> &out_color,
                    std::vector<float> *out_depth = nullptr);

// ----------------------------------------------------------------------------
// Backward rasterization: compute gradients w.r.t. Gaussian parameters.
// Inputs:
//   gaussians – list of Gaussian parameters (forward pass values)
//   cam – camera parameters
//   width, height – image dimensions
//   grad_color – loss gradient w.r.t. output color (RGB, same size as out_color)
//   transmittance_map – optional per‑pixel transmittance from forward pass (for efficiency)
// Output:
//   grad_gaussians – list of gradients for each Gaussian (same order as input)
// ----------------------------------------------------------------------------
void gsplat_backward(const std::vector<Gaussian> &gaussians,
                     const CameraParams &cam,
                     int width, int height,
                     const std::vector<uint8_t> &grad_color,
                     std::vector<GaussianGradients> &grad_gaussians);

// ----------------------------------------------------------------------------
// Utility: convert camera parameters from Godot's projection matrix
// ----------------------------------------------------------------------------
CameraParams camera_from_godot(const float *view_matrix, const float *proj_matrix,
                               int width, int height);
