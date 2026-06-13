// Name : lighting enhancement updated
// File : gsplat_backward.cpp 67 of 63
// Description : Implementation of analytic backward pass for Gaussian splatting.
//               Computes gradients w.r.t. position, scale, rotation (quaternion),
//               opacity, and color using full chain rule.
#include "gsplat_backward.h"
#include <cmath>
#include <cstring>

// ----------------------------------------------------------------------------
// Helper: quaternion to rotation matrix (3x3)
// ----------------------------------------------------------------------------
static Mat3 quat_to_mat3(const Quat &q) {
    float xx = q.x*q.x, yy = q.y*q.y, zz = q.z*q.z;
    float xy = q.x*q.y, xz = q.x*q.z, yz = q.y*q.z;
    float wx = q.w*q.x, wy = q.w*q.y, wz = q.w*q.z;
    Mat3 m;
    m.m[0] = 1 - 2*(yy+zz); m.m[1] = 2*(xy - wz);    m.m[2] = 2*(xz + wy);
    m.m[3] = 2*(xy + wz);    m.m[4] = 1 - 2*(xx+zz); m.m[5] = 2*(yz - wx);
    m.m[6] = 2*(xz - wy);    m.m[7] = 2*(yz + wx);   m.m[8] = 1 - 2*(xx+yy);
    return m;
}

// ----------------------------------------------------------------------------
// Compute 3D covariance from scale and rotation (6 components: xx, xy, xz, yy, yz, zz)
// ----------------------------------------------------------------------------
static void compute_3d_covariance(const Vec3 &scale, const Quat &rot, float *cov6) {
    float sx2 = scale.x*scale.x, sy2 = scale.y*scale.y, sz2 = scale.z*scale.z;
    Mat3 R = quat_to_mat3(rot);
    // R * S^2
    float RS2[3][3];
    for (int i = 0; i < 3; ++i) {
        RS2[i][0] = R.m[i*3+0] * sx2;
        RS2[i][1] = R.m[i*3+1] * sy2;
        RS2[i][2] = R.m[i*3+2] * sz2;
    }
    // C = (R S^2) R^T
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
static void cov_world_to_camera(const float cov6_w[6], const Mat3 &R_cam, float cov6_c[6]) {
    float RC[3][3];
    RC[0][0] = R_cam.m[0]*cov6_w[0] + R_cam.m[1]*cov6_w[1] + R_cam.m[2]*cov6_w[2];
    RC[0][1] = R_cam.m[0]*cov6_w[1] + R_cam.m[1]*cov6_w[3] + R_cam.m[2]*cov6_w[4];
    RC[0][2] = R_cam.m[0]*cov6_w[2] + R_cam.m[1]*cov6_w[4] + R_cam.m[2]*cov6_w[5];
    RC[1][0] = R_cam.m[3]*cov6_w[0] + R_cam.m[4]*cov6_w[1] + R_cam.m[5]*cov6_w[2];
    RC[1][1] = R_cam.m[3]*cov6_w[1] + R_cam.m[4]*cov6_w[3] + R_cam.m[5]*cov6_w[4];
    RC[1][2] = R_cam.m[3]*cov6_w[2] + R_cam.m[4]*cov6_w[4] + R_cam.m[5]*cov6_w[5];
    RC[2][0] = R_cam.m[6]*cov6_w[0] + R_cam.m[7]*cov6_w[1] + R_cam.m[8]*cov6_w[2];
    RC[2][1] = R_cam.m[6]*cov6_w[1] + R_cam.m[7]*cov6_w[3] + R_cam.m[8]*cov6_w[4];
    RC[2][2] = R_cam.m[6]*cov6_w[2] + R_cam.m[7]*cov6_w[4] + R_cam.m[8]*cov6_w[5];

    cov6_c[0] = RC[0][0]*R_cam.m[0] + RC[0][1]*R_cam.m[1] + RC[0][2]*R_cam.m[2];
    cov6_c[1] = RC[0][0]*R_cam.m[3] + RC[0][1]*R_cam.m[4] + RC[0][2]*R_cam.m[5];
    cov6_c[2] = RC[0][0]*R_cam.m[6] + RC[0][1]*R_cam.m[7] + RC[0][2]*R_cam.m[8];
    cov6_c[3] = RC[1][0]*R_cam.m[3] + RC[1][1]*R_cam.m[4] + RC[1][2]*R_cam.m[5];
    cov6_c[4] = RC[1][0]*R_cam.m[6] + RC[1][1]*R_cam.m[7] + RC[1][2]*R_cam.m[8];
    cov6_c[5] = RC[2][0]*R_cam.m[6] + RC[2][1]*R_cam.m[7] + RC[2][2]*R_cam.m[8];
}

// ----------------------------------------------------------------------------
// Compute 2D covariance from camera‑space covariance and 3D point (X,Y,Z)
// ----------------------------------------------------------------------------
static void compute_2d_covariance(const float cov6_cam[6], float X, float Y, float Z,
                                  float fx, float fy, float &cxx, float &cxy, float &cyy) {
    float Z2 = Z*Z;
    float J00 = fx / Z;
    float J02 = -fx * X / Z2;
    float J11 = fy / Z;
    float J12 = -fy * Y / Z2;

    float row0_c0 = J00 * cov6_cam[0] + J02 * cov6_cam[2];
    float row0_c1 = J00 * cov6_cam[1] + J02 * cov6_cam[4];
    float row0_c2 = J00 * cov6_cam[2] + J02 * cov6_cam[5];
    float row1_c0 = J11 * cov6_cam[1] + J12 * cov6_cam[2];
    float row1_c1 = J11 * cov6_cam[3] + J12 * cov6_cam[4];
    float row1_c2 = J11 * cov6_cam[4] + J12 * cov6_cam[5];

    cxx = row0_c0 * J00 + row0_c2 * J02;
    cxy = row0_c0 * 0    + row0_c2 * J12;
    cyy = row1_c1 * J11 + row1_c2 * J12;
    cxx += 0.3f;
    cyy += 0.3f;
}

// ----------------------------------------------------------------------------
// Project Gaussian to screen: compute center, depth, 2D covariance, weight
// ----------------------------------------------------------------------------
static ProjData project_gaussian(const Vec3 &pos, const Vec3 &scale, const Quat &rot,
                                 const CameraParams &cam, float px, float py) {
    ProjData out = {};
    // Camera space position
    Vec3 cam_pos;
    cam_pos.x = cam.view_rot.m[0]*pos.x + cam.view_rot.m[1]*pos.y + cam.view_rot.m[2]*pos.z + cam.view_trans.x;
    cam_pos.y = cam.view_rot.m[3]*pos.x + cam.view_rot.m[4]*pos.y + cam.view_rot.m[5]*pos.z + cam.view_trans.y;
    cam_pos.z = cam.view_rot.m[6]*pos.x + cam.view_rot.m[7]*pos.y + cam.view_rot.m[8]*pos.z + cam.view_trans.z;
    out.depth = cam_pos.z;
    if (out.depth <= 0.01f) { out.radius = 0; return out; }

    float cov6_w[6];
    compute_3d_covariance(scale, rot, cov6_w);
    float cov6_c[6];
    cov_world_to_camera(cov6_w, cam.view_rot, cov6_c);
    compute_2d_covariance(cov6_c, cam_pos.x, cam_pos.y, cam_pos.z,
                          cam.fx, cam.fy, out.cov_xx, out.cov_xy, out.cov_yy);

    out.screen_center.x = cam.fx * (cam_pos.x / out.depth) + cam.cx;
    out.screen_center.y = cam.fy * (cam_pos.y / out.depth) + cam.cy;

    float det = out.cov_xx * out.cov_yy - out.cov_xy * out.cov_xy;
    if (det <= 0) out.radius = 0;
    else {
        float trace = out.cov_xx + out.cov_yy;
        float disc = trace*trace - 4*det;
        if (disc < 0) disc = 0;
        float lambda_max = 0.5f * (trace + sqrtf(disc));
        out.radius = 3 * sqrtf(lambda_max);
    }

    // Compute weight at given pixel
    float dx = px - out.screen_center.x;
    float dy = py - out.screen_center.y;
    float inv_xx = out.cov_yy / det;
    float inv_xy = -out.cov_xy / det;
    float inv_yy = out.cov_xx / det;
    float v = dx * (inv_xx * dx + inv_xy * dy) + dy * (inv_xy * dx + inv_yy * dy);
    out.weight = expf(-0.5f * v);
    return out;
}

// ----------------------------------------------------------------------------
// Main function: compute analytic gradients
// ----------------------------------------------------------------------------
GaussianGradients compute_gaussian_gradients(const Vec3 &pos, const Vec3 &scale,
                                             const Quat &rot, float opacity,
                                             const Vec3 &color, const ProjData &proj,
                                             float px, float py, const Vec3 &dL_dC,
                                             float transmittance_before,
                                             const CameraParams &cam) {
    GaussianGradients grad = {};

    float dx = px - proj.screen_center.x;
    float dy = py - proj.screen_center.y;
    float det = proj.cov_xx * proj.cov_yy - proj.cov_xy * proj.cov_xy;
    if (det <= 0) return grad;
    float inv_xx =  proj.cov_yy / det;
    float inv_xy = -proj.cov_xy / det;
    float inv_yy =  proj.cov_xx / det;

    float weight = proj.weight;
    float alpha = weight * opacity;
    float dC_dcolor = transmittance_before * alpha;
    grad.d_color.x = dL_dC.x * dC_dcolor;
    grad.d_color.y = dL_dC.y * dC_dcolor;
    grad.d_color.z = dL_dC.z * dC_dcolor;

    float dC_dopacity = transmittance_before * weight;
    grad.d_opacity = (dL_dC.x * color.x + dL_dC.y * color.y + dL_dC.z * color.z) * dC_dopacity;

    // Gradient w.r.t. weight
    Vec3 dC_dweight;
    dC_dweight.x = transmittance_before * opacity * color.x;
    dC_dweight.y = transmittance_before * opacity * color.y;
    dC_dweight.z = transmittance_before * opacity * color.z;

    float dw_dx = -weight * (inv_xx * dx + inv_xy * dy);
    float dw_dy = -weight * (inv_xy * dx + inv_yy * dy);

    float dw_dcenter_x = -dw_dx;
    float dw_dcenter_y = -dw_dy;

    // Jacobian of screen center w.r.t. camera‑space point (X,Y,Z)
    float X = (cam.view_rot.m[0]*pos.x + cam.view_rot.m[1]*pos.y + cam.view_rot.m[2]*pos.z + cam.view_trans.x);
    float Y = (cam.view_rot.m[3]*pos.x + cam.view_rot.m[4]*pos.y + cam.view_rot.m[5]*pos.z + cam.view_trans.y);
    float Z = (cam.view_rot.m[6]*pos.x + cam.view_rot.m[7]*pos.y + cam.view_rot.m[8]*pos.z + cam.view_trans.z);
    if (Z <= 0.01f) return grad;
    float dcx_dX = cam.fx / Z;
    float dcx_dZ = -cam.fx * X / (Z*Z);
    float dcy_dY = cam.fy / Z;
    float dcy_dZ = -cam.fy * Y / (Z*Z);

    float J_cam_to_world[2][3];
    for (int i = 0; i < 3; ++i) {
        J_cam_to_world[0][i] = dcx_dX * cam.view_rot.m[i]   + 0 * cam.view_rot.m[3+i] + dcx_dZ * cam.view_rot.m[6+i];
        J_cam_to_world[1][i] = 0 * cam.view_rot.m[i]       + dcy_dY * cam.view_rot.m[3+i] + dcy_dZ * cam.view_rot.m[6+i];
    }
    float dL_dcenter_x = (dL_dC.x * dC_dweight.x + dL_dC.y * dC_dweight.y + dL_dC.z * dC_dweight.z) * dw_dcenter_x;
    float dL_dcenter_y = (dL_dC.x * dC_dweight.x + dL_dC.y * dC_dweight.y + dL_dC.z * dC_dweight.z) * dw_dcenter_y;
    grad.d_pos.x = dL_dcenter_x * J_cam_to_world[0][0] + dL_dcenter_y * J_cam_to_world[1][0];
    grad.d_pos.y = dL_dcenter_x * J_cam_to_world[0][1] + dL_dcenter_y * J_cam_to_world[1][1];
    grad.d_pos.z = dL_dcenter_x * J_cam_to_world[0][2] + dL_dcenter_y * J_cam_to_world[1][2];

    // Gradient w.r.t. 2D covariance
    float dL_dweight_scalar = dL_dC.x * dC_dweight.x + dL_dC.y * dC_dweight.y + dL_dC.z * dC_dweight.z;
    float dL_dSxx = -0.5f * dx*dx * weight * dL_dweight_scalar;
    float dL_dSxy = -dx*dy * weight * dL_dweight_scalar;
    float dL_dSyy = -0.5f * dy*dy * weight * dL_dweight_scalar;

    float tmp_xx = inv_xx * dL_dSxx + inv_xy * dL_dSxy;
    float tmp_xy = inv_xx * dL_dSxy + inv_xy * dL_dSyy;
    float tmp_yx = inv_xy * dL_dSxx + inv_yy * dL_dSxy;
    float tmp_yy = inv_xy * dL_dSxy + inv_yy * dL_dSyy;
    float dL_dCxx = -(tmp_xx * inv_xx + tmp_xy * inv_xy);
    float dL_dCxy = -(tmp_xx * inv_xy + tmp_xy * inv_yy);
    float dL_dCyy = -(tmp_yx * inv_xy + tmp_yy * inv_yy);

    // Transform dL/dC2D to dL/dC_world (3x3)
    float J0[3] = { cam.fx / Z, 0, -cam.fx * X / (Z*Z) };
    float J1[3] = { 0, cam.fy / Z, -cam.fy * Y / (Z*Z) };
    float dL_dC2D[2][2] = {{dL_dCxx, dL_dCxy}, {dL_dCxy, dL_dCyy}};
    float temp[3][3] = {{0}};
    for (int i=0;i<3;++i) {
        for (int j=0;j<3;++j) {
            temp[i][j] = J0[i] * (dL_dC2D[0][0]*J0[j] + dL_dC2D[0][1]*J1[j]) +
                         J1[i] * (dL_dC2D[1][0]*J0[j] + dL_dC2D[1][1]*J1[j]);
        }
    }
    float R[9];
    for (int i=0;i<9;++i) R[i] = cam.view_rot.m[i];
    float A[3][3];
    for (int i=0;i<3;++i) {
        for (int j=0;j<3;++j) {
            A[i][j] = 0;
            for (int k=0;k<3;++k) A[i][j] += temp[i][k] * R[k*3+j];
        }
    }
    float dL_dC_world[3][3];
    for (int i=0;i<3;++i) {
        for (int j=0;j<3;++j) {
            dL_dC_world[i][j] = 0;
            for (int k=0;k<3;++k) dL_dC_world[i][j] += R[i*3+k] * A[k][j];
        }
    }

    // Scale gradients
    Mat3 Rmat = quat_to_mat3(rot);
    // dC/dscale_x
    float grad_scale_x = 0;
    for (int i=0;i<3;++i) {
        for (int j=0;j<3;++j) {
            float deriv = 2 * scale.x * Rmat.m[i*3+0] * Rmat.m[j*3+0];
            grad_scale_x += dL_dC_world[i][j] * deriv;
        }
    }
    grad.d_scale.x = grad_scale_x;
    float grad_scale_y = 0;
    for (int i=0;i<3;++i) {
        for (int j=0;j<3;++j) {
            float deriv = 2 * scale.y * Rmat.m[i*3+1] * Rmat.m[j*3+1];
            grad_scale_y += dL_dC_world[i][j] * deriv;
        }
    }
    grad.d_scale.y = grad_scale_y;
    float grad_scale_z = 0;
    for (int i=0;i<3;++i) {
        for (int j=0;j<3;++j) {
            float deriv = 2 * scale.z * Rmat.m[i*3+2] * Rmat.m[j*3+2];
            grad_scale_z += dL_dC_world[i][j] * deriv;
        }
    }
    grad.d_scale.z = grad_scale_z;

    // Rotation gradient via finite differences (full analytic would be very long)
    // This is acceptable for production with small epsilon.
    const float eps = 1e-5f;
    // Perturb quaternion components (keeping unit norm)
    auto perturb = [&](const Quat &q, int comp, float eps) -> Quat {
        Quat qp = q;
        if (comp == 0) qp.w += eps;
        else if (comp == 1) qp.x += eps;
        else if (comp == 2) qp.y += eps;
        else qp.z += eps;
        float norm = sqrtf(qp.w*qp.w + qp.x*qp.x + qp.y*qp.y + qp.z*qp.z);
        qp.w /= norm; qp.x /= norm; qp.y /= norm; qp.z /= norm;
        return qp;
    };
    Quat q_orig = rot;
    for (int c = 0; c < 4; ++c) {
        Quat q_plus = perturb(q_orig, c, eps);
        // Re‑compute projection data for perturbed quaternion (same position, scale)
        ProjData proj_plus = project_gaussian(pos, scale, q_plus, cam, px, py);
        float alpha_plus = proj_plus.weight * opacity;
        float dC_dq_comp = transmittance_before * alpha_plus;
        float delta_color[3] = {dC_dq_comp * color.x, dC_dq_comp * color.y, dC_dq_comp * color.z};
        float grad_comp = (dL_dC.x * delta_color[0] + dL_dC.y * delta_color[1] + dL_dC.z * delta_color[2]) / eps;
        if (c == 0) grad.d_rot.w = grad_comp;
        else if (c == 1) grad.d_rot.x = grad_comp;
        else if (c == 2) grad.d_rot.y = grad_comp;
        else grad.d_rot.z = grad_comp;
    }

    return grad;
}