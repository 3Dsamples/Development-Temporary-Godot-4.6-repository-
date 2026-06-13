// Name : lighting enhancement updated
// File : projection_ext.cpp 69 of 63
// Description : Implementation of projection functions for 3D Gaussians.
//               Converts world space Gaussians to screen space with 2D covariance.
#include "projection_ext.h"
#include <cmath>

// ----------------------------------------------------------------------------
// Compute 3D world covariance from scale and rotation (6 components)
// ----------------------------------------------------------------------------
void compute_3d_covariance_full(const Vec3 &scale, const Quat &rot, float cov6[6]) {
    float sx2 = scale.x * scale.x;
    float sy2 = scale.y * scale.y;
    float sz2 = scale.z * scale.z;
    // Rotation matrix from quaternion
    Mat3 R = quat_to_mat3(rot);
    // RS2 = R * S^2
    float RS2[3][3];
    for (int i = 0; i < 3; ++i) {
        RS2[i][0] = R.m[i*3+0] * sx2;
        RS2[i][1] = R.m[i*3+1] * sy2;
        RS2[i][2] = R.m[i*3+2] * sz2;
    }
    // C = RS2 * R^T
    cov6[0] = RS2[0][0]*R.m[0] + RS2[0][1]*R.m[1] + RS2[0][2]*R.m[2];
    cov6[1] = RS2[0][0]*R.m[3] + RS2[0][1]*R.m[4] + RS2[0][2]*R.m[5];
    cov6[2] = RS2[0][0]*R.m[6] + RS2[0][1]*R.m[7] + RS2[0][2]*R.m[8];
    cov6[3] = RS2[1][0]*R.m[3] + RS2[1][1]*R.m[4] + RS2[1][2]*R.m[5];
    cov6[4] = RS2[1][0]*R.m[6] + RS2[1][1]*R.m[7] + RS2[1][2]*R.m[8];
    cov6[5] = RS2[2][0]*R.m[6] + RS2[2][1]*R.m[7] + RS2[2][2]*R.m[8];
}

// ----------------------------------------------------------------------------
// Transform 3D covariance from world to camera space (rotation only)
// ----------------------------------------------------------------------------
void transform_covariance_to_camera(const float cov6_world[6], const Mat3 &R_cam,
                                    float cov6_cam[6]) {
    float RC[3][3];
    RC[0][0] = R_cam.m[0]*cov6_world[0] + R_cam.m[1]*cov6_world[1] + R_cam.m[2]*cov6_world[2];
    RC[0][1] = R_cam.m[0]*cov6_world[1] + R_cam.m[1]*cov6_world[3] + R_cam.m[2]*cov6_world[4];
    RC[0][2] = R_cam.m[0]*cov6_world[2] + R_cam.m[1]*cov6_world[4] + R_cam.m[2]*cov6_world[5];
    RC[1][0] = R_cam.m[3]*cov6_world[0] + R_cam.m[4]*cov6_world[1] + R_cam.m[5]*cov6_world[2];
    RC[1][1] = R_cam.m[3]*cov6_world[1] + R_cam.m[4]*cov6_world[3] + R_cam.m[5]*cov6_world[4];
    RC[1][2] = R_cam.m[3]*cov6_world[2] + R_cam.m[4]*cov6_world[4] + R_cam.m[5]*cov6_world[5];
    RC[2][0] = R_cam.m[6]*cov6_world[0] + R_cam.m[7]*cov6_world[1] + R_cam.m[8]*cov6_world[2];
    RC[2][1] = R_cam.m[6]*cov6_world[1] + R_cam.m[7]*cov6_world[3] + R_cam.m[8]*cov6_world[4];
    RC[2][2] = R_cam.m[6]*cov6_world[2] + R_cam.m[7]*cov6_world[4] + R_cam.m[8]*cov6_world[5];

    cov6_cam[0] = RC[0][0]*R_cam.m[0] + RC[0][1]*R_cam.m[1] + RC[0][2]*R_cam.m[2];
    cov6_cam[1] = RC[0][0]*R_cam.m[3] + RC[0][1]*R_cam.m[4] + RC[0][2]*R_cam.m[5];
    cov6_cam[2] = RC[0][0]*R_cam.m[6] + RC[0][1]*R_cam.m[7] + RC[0][2]*R_cam.m[8];
    cov6_cam[3] = RC[1][0]*R_cam.m[3] + RC[1][1]*R_cam.m[4] + RC[1][2]*R_cam.m[5];
    cov6_cam[4] = RC[1][0]*R_cam.m[6] + RC[1][1]*R_cam.m[7] + RC[1][2]*R_cam.m[8];
    cov6_cam[5] = RC[2][0]*R_cam.m[6] + RC[2][1]*R_cam.m[7] + RC[2][2]*R_cam.m[8];
}

// ----------------------------------------------------------------------------
// Compute 2D covariance from camera‑space covariance and 3D point (X,Y,Z)
// ----------------------------------------------------------------------------
void compute_2d_covariance(const float cov6_cam[6], float X, float Y, float Z,
                           float fx, float fy, float &cxx, float &cxy, float &cyy) {
    float Z2 = Z * Z;
    float J00 = fx / Z;
    float J02 = -fx * X / Z2;
    float J11 = fy / Z;
    float J12 = -fy * Y / Z2;

    // J * C_cam (first two rows)
    float row0_c0 = J00 * cov6_cam[0] + J02 * cov6_cam[2];
    float row0_c1 = J00 * cov6_cam[1] + J02 * cov6_cam[4];
    float row0_c2 = J00 * cov6_cam[2] + J02 * cov6_cam[5];
    float row1_c0 = J11 * cov6_cam[1] + J12 * cov6_cam[2];
    float row1_c1 = J11 * cov6_cam[3] + J12 * cov6_cam[4];
    float row1_c2 = J11 * cov6_cam[4] + J12 * cov6_cam[5];

    // Multiply by J^T
    cxx = row0_c0 * J00 + row0_c2 * J02;
    cxy = row0_c0 * 0    + row0_c2 * J12;
    cyy = row1_c1 * J11 + row1_c2 * J12;

    // Stabilization
    cxx += 0.3f;
    cyy += 0.3f;
}

// ----------------------------------------------------------------------------
// Main projection function
// ----------------------------------------------------------------------------
ProjectionResult project_gaussian_full(const Vec3 &pos, const Vec3 &scale,
                                       const Quat &rot, const CameraParams &cam) {
    ProjectionResult out = {};
    // Transform to camera space
    Vec3 cam_pos;
    cam_pos.x = cam.view_rot.m[0]*pos.x + cam.view_rot.m[1]*pos.y + cam.view_rot.m[2]*pos.z + cam.view_trans.x;
    cam_pos.y = cam.view_rot.m[3]*pos.x + cam.view_rot.m[4]*pos.y + cam.view_rot.m[5]*pos.z + cam.view_trans.y;
    cam_pos.z = cam.view_rot.m[6]*pos.x + cam.view_rot.m[7]*pos.y + cam.view_rot.m[8]*pos.z + cam.view_trans.z;
    out.depth = cam_pos.z;
    if (out.depth <= 0.01f) {
        out.visible = false;
        return out;
    }

    // Compute 3D world covariance
    float cov6_world[6];
    compute_3d_covariance_full(scale, rot, cov6_world);
    // Transform to camera space
    float cov6_cam[6];
    transform_covariance_to_camera(cov6_world, cam.view_rot, cov6_cam);
    // Compute 2D covariance
    compute_2d_covariance(cov6_cam, cam_pos.x, cam_pos.y, cam_pos.z,
                          cam.fx, cam.fy, out.cov_xx, out.cov_xy, out.cov_yy);

    // Project center to screen
    out.screen_center.x = cam.fx * (cam_pos.x / out.depth) + cam.cx;
    out.screen_center.y = cam.fy * (cam_pos.y / out.depth) + cam.cy;

    // Compute bounding box radius (3 sigma)
    float det = out.cov_xx * out.cov_yy - out.cov_xy * out.cov_xy;
    if (det <= 0.0f) {
        out.radius = 0.0f;
        out.visible = false;
        return out;
    }
    float trace = out.cov_xx + out.cov_yy;
    float disc = trace * trace - 4.0f * det;
    if (disc < 0.0f) disc = 0.0f;
    float lambda_max = 0.5f * (trace + sqrtf(disc));
    out.radius = 3.0f * sqrtf(lambda_max);
    out.visible = true;
    return out;
}